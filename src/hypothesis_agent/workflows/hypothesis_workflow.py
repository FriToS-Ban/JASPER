"""Local workflow runner for hypothesis validation on the LangGraph research graph.

Each run is a thread of the compiled research graph in a checkpointer owned by the client. The
client streams the graph's task events to keep per-stage milestones current (RUNNING when a node
starts, COMPLETED or BLOCKED when it ends); stage code runs in LangGraph's executor threads so the
event loop stays free. ``human_review`` pauses the thread with ``interrupt()`` and
:meth:`HypothesisWorkflowClient.resume` continues it from the checkpoint with
``Command(resume=decision)``, which runs the review node and then delivery.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from hypothesis_agent.models.hypothesis import (
    EvidenceReference,
    HypothesisRequest,
    MilestoneStatus,
    ValidationSummary,
    WorkflowMilestone,
)
from hypothesis_agent.orchestration.langgraph_pipeline import (
    MAX_STAGE_EXECUTIONS,
    LangGraphValidationOrchestrator,
    StageExecutionResult,
)
from hypothesis_agent.research.agent import STAGES

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkflowSubmissionResult:
    """Result of initiating a validation workflow."""

    workflow_id: str
    workflow_run_id: str
    validation: ValidationSummary


@dataclass(slots=True)
class WorkflowExecutionDetails:
    """Live execution metadata for a workflow."""

    status: str
    history_length: int | None = None
    milestones: List[WorkflowMilestone] | None = None
    awaiting_review: bool = False


@dataclass(slots=True)
class LocalWorkflowRun:
    """State retained for locally simulated workflow executions."""

    hypothesis: HypothesisRequest
    context: Dict[str, Any]
    summary: ValidationSummary
    awaiting_review: bool
    completed: bool = False
    cancelled: bool = False


@dataclass(slots=True)
class HypothesisWorkflowClient:
    """Execute the LangGraph pipeline for hypothesis validation."""

    namespace: str
    task_queue: str
    workflow: str
    address: str

    _local_runs: Dict[Tuple[str, str], LocalWorkflowRun] = field(default_factory=dict, init=False, repr=False)
    _tasks: Dict[Tuple[str, str], asyncio.Task[None]] = field(default_factory=dict, init=False, repr=False)
    # Shared by every orchestrator instance the client creates, so a run paused for review by one
    # orchestrator is resumed from its checkpoint by another.
    _checkpointer: InMemorySaver = field(default_factory=InMemorySaver, init=False, repr=False)

    async def submit(self, hypothesis_id: UUID, hypothesis: HypothesisRequest) -> WorkflowSubmissionResult:
        """Schedule the validation pipeline and return tracking metadata."""

        workflow_id = f"hypothesis-{hypothesis_id}"
        workflow_run_id = self._generate_run_id(workflow_id)
        logger.info(
            "Executing local workflow hypothesis=%s workflow_id=%s run_id=%s user=%s",
            hypothesis_id,
            workflow_id,
            workflow_run_id,
            hypothesis.user_id,
        )

        context: Dict[str, Any] = {
            "metadata": {
                "workflow_id": workflow_id,
                "workflow_run_id": workflow_run_id,
                "stages": {},
            }
        }
        initial_summary = self._build_initial_summary(hypothesis.requires_human_review)
        local_run = LocalWorkflowRun(
            hypothesis=hypothesis,
            context=context,
            summary=initial_summary,
            awaiting_review=False,
            completed=False,
            cancelled=False,
        )

        key = (workflow_id, workflow_run_id)
        self._local_runs[key] = local_run

        task = asyncio.create_task(self._run_pipeline_async(workflow_id, workflow_run_id, hypothesis))
        self._tasks[key] = task
        task.add_done_callback(lambda t, task_key=key: self._on_task_complete(task_key, t))

        return WorkflowSubmissionResult(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            validation=initial_summary,
        )

    async def describe(self, workflow_id: str, workflow_run_id: str) -> WorkflowExecutionDetails:
        """Return execution details for a workflow."""

        key = (workflow_id, workflow_run_id)
        if key not in self._local_runs:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")

        local_run = self._local_runs[key]
        summary = local_run.summary
        history_length = None
        if summary.milestones:
            history_length = sum(
                1 for milestone in summary.milestones if milestone.status != MilestoneStatus.PENDING
            )

        if local_run.awaiting_review:
            status = "AWAITING_REVIEW"
        elif local_run.cancelled:
            status = "CANCELLED"
        elif any(m.status == MilestoneStatus.BLOCKED for m in summary.milestones):
            status = "FAILED"
        elif local_run.completed:
            status = "COMPLETED"
        else:
            status = "RUNNING"

        return WorkflowExecutionDetails(
            status=status,
            history_length=history_length,
            milestones=summary.milestones,
            awaiting_review=local_run.awaiting_review,
        )

    async def resume(self, workflow_id: str, workflow_run_id: str, decision: str = "approved") -> ValidationSummary:
        """Resume a paused workflow after a human review decision."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        if not local_run.awaiting_review:
            return local_run.summary

        local_run.awaiting_review = False
        local_run.cancelled = False
        # A fresh orchestrator continues the paused thread from the shared checkpointer: the
        # human_review node receives the decision, then delivery runs on the checkpointed context.
        orchestrator = LangGraphValidationOrchestrator()
        # Tracked like a submitted run, so cancel() can stop review and delivery while they run.
        task = asyncio.create_task(self._advance(key, orchestrator, Command(resume=decision)))
        self._tasks[key] = task
        await asyncio.wait({task})
        if task.cancelled():
            # cancel() stopped the run; it records the cancelled summary right after the task ends.
            await asyncio.sleep(0)
            return self._local_runs[key].summary
        task.result()  # re-raise a stage failure (already recorded as a BLOCKED milestone)
        return self._local_runs[key].summary

    async def fetch_summary(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Retrieve the current validation summary for a workflow."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        return local_run.summary

    async def close(self) -> None:
        """Cancel any running local workflow tasks."""

        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def cancel(self, workflow_id: str, workflow_run_id: str) -> ValidationSummary:
        """Forcefully stop an in-flight workflow and mark milestones as cancelled."""

        key = (workflow_id, workflow_run_id)
        local_run = self._local_runs.get(key)
        if local_run is None:
            raise KeyError(f"Workflow {workflow_id} with run {workflow_run_id} not found")
        if local_run.cancelled:
            return local_run.summary
        if local_run.completed and not local_run.awaiting_review:
            raise RuntimeError("Workflow has already completed and cannot be cancelled.")

        task = self._tasks.pop(key, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        cancelled_milestones: List[WorkflowMilestone] = []
        for milestone in local_run.summary.milestones:
            if milestone.status in {MilestoneStatus.COMPLETED, MilestoneStatus.BLOCKED}:
                cancelled_milestones.append(milestone)
                continue
            detail = milestone.detail
            if milestone.status == MilestoneStatus.RUNNING:
                detail = "Stage cancelled by user."
            elif milestone.status == MilestoneStatus.WAITING_REVIEW:
                detail = "Human review bypassed due to cancellation."
            else:
                detail = detail or "Stage not executed due to cancellation."
            cancelled_milestones.append(
                WorkflowMilestone(name=milestone.name, status=MilestoneStatus.CANCELLED, detail=detail)
            )

        summary = local_run.summary.model_copy(
            update={
                "milestones": cancelled_milestones,
                "conclusion": "Validation cancelled by user request.",
                "current_stage": "cancelled",
            }
        )

        local_run.summary = summary
        local_run.awaiting_review = False
        local_run.completed = True
        local_run.cancelled = True
        metadata = local_run.context.setdefault("metadata", {})
        metadata["cancelled"] = True
        self._local_runs[key] = local_run
        self._discard_checkpoints(key)
        return summary

    def _generate_run_id(self, workflow_id: str) -> str:
        run_id = f"{workflow_id}-run"
        counter = 1
        while (workflow_id, run_id) in self._local_runs:
            run_id = f"{workflow_id}-run-{counter}"
            counter += 1
        return run_id

    @staticmethod
    def _thread_id(key: Tuple[str, str]) -> str:
        """Checkpoint thread of a run: one thread per workflow id and run id."""

        return f"{key[0]}:{key[1]}"

    def _discard_checkpoints(self, key: Tuple[str, str]) -> None:
        """Drop a finished run's checkpoints; only a run paused for review needs them."""

        self._checkpointer.delete_thread(self._thread_id(key))

    def _stage_sequence(self, include_human_review: bool) -> List[str]:
        return [stage for stage in STAGES if include_human_review or stage != "human_review"]

    def _build_initial_summary(self, include_human_review: bool) -> ValidationSummary:
        stages = self._stage_sequence(include_human_review)
        milestones = [
            WorkflowMilestone(name=stage, status=MilestoneStatus.PENDING, detail=None)
            for stage in stages
        ]
        current_stage = stages[0] if stages else "pending"
        return ValidationSummary(
            score=0.0,
            conclusion="Pending validation",
            confidence=0.0,
            evidence=[],
            current_stage=current_stage,
            milestones=milestones,
        )

    async def _run_pipeline_async(
        self,
        workflow_id: str,
        workflow_run_id: str,
        hypothesis: HypothesisRequest,
    ) -> None:
        key = (workflow_id, workflow_run_id)
        orchestrator = LangGraphValidationOrchestrator()

        local_run = self._local_runs.get(key)
        if local_run is None:
            return

        try:
            await self._advance(key, orchestrator, orchestrator.graph_input(hypothesis, local_run.context))
        except Exception:
            return  # already recorded as a BLOCKED milestone

    async def _advance(
        self,
        key: Tuple[str, str],
        orchestrator: LangGraphValidationOrchestrator,
        graph_input: Any,
    ) -> None:
        """Stream the run's graph thread until it finishes, pauses for review or fails.

        ``graph_input`` is the initial state for a new run or ``Command(resume=...)`` for a paused
        one. Failures are recorded as a BLOCKED milestone on the failing stage and re-raised.
        """

        config = orchestrator.graph_config(thread_id=self._thread_id(key))
        stage = "plan_generation"
        review_request: Dict[str, Any] | None = None
        try:
            graph = orchestrator.build_graph(checkpointer=self._checkpointer, record_stage_status=True)
            # "tasks" mode emits an event when a node starts and another when it finishes.
            async for event in graph.astream(graph_input, config, stream_mode="tasks"):
                stage = event["name"]
                if "input" in event:
                    if stage != "human_review":  # the review milestone goes straight to WAITING_REVIEW
                        self._mark_stage_running(key, stage)
                elif event["interrupts"]:
                    review_request = event["interrupts"][0]["value"]
                elif event["error"] is None:
                    self._apply_stage_result(key, StageExecutionResult.from_graph_update(event["result"]))
        except GraphRecursionError as exc:
            snapshot = await graph.aget_state(config)
            budget = RuntimeError(f"Stage budget of {MAX_STAGE_EXECUTIONS} executions exhausted.")
            self._handle_stage_failure(key, snapshot.next[0] if snapshot.next else stage, budget)
            raise budget from exc
        except Exception as exc:
            self._handle_stage_failure(key, stage, exc)
            raise

        if review_request is not None:
            self._prepare_human_review(key, review_request.get("reason"))
        else:
            self._complete_run(key)
        self._tasks.pop(key, None)

    def _complete_run(self, key: Tuple[str, str]) -> None:
        local_run = self._local_runs[key]
        local_run.completed = True
        local_run.awaiting_review = False
        local_run.cancelled = False
        metadata = local_run.context.setdefault("metadata", {})
        if not (metadata.get("human_review") or {}).get("required"):
            metadata["human_review"] = {"required": False, "decision": "auto"}
        self._discard_checkpoints(key)

    def _mark_stage_running(self, key: Tuple[str, str], stage: str) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        milestone = WorkflowMilestone(
            name=stage,
            status=MilestoneStatus.RUNNING,
            detail=f"{stage.replace('_', ' ').title()} in progress.",
        )
        milestones = self._merge_milestone(local_run, milestone)
        summary = local_run.summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        local_run.summary = summary
        local_run.completed = False
        local_run.cancelled = False
        self._local_runs[key] = local_run

    def _handle_stage_failure(self, key: Tuple[str, str], stage: str, exc: Exception) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        detail = str(exc)
        milestone = WorkflowMilestone(name=stage, status=MilestoneStatus.BLOCKED, detail=detail)
        milestones = self._merge_milestone(local_run, milestone)
        summary = local_run.summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": stage,
            }
        )
        local_run.summary = summary
        local_run.awaiting_review = False
        local_run.completed = True
        local_run.cancelled = False
        metadata = local_run.context.setdefault("metadata", {})
        metadata.setdefault("human_review", {"required": False, "decision": None})
        self._local_runs[key] = local_run
        self._tasks.pop(key, None)
        self._discard_checkpoints(key)
        logger.exception(
            "Pipeline stage %s failed for workflow %s/%s",
            stage,
            key[0],
            key[1],
            exc_info=True,
        )

    def _apply_stage_result(self, key: Tuple[str, str], result: StageExecutionResult) -> None:
        local_run = self._local_runs.get(key)
        if local_run is None:
            return

        summary_source = result.summary or local_run.summary
        milestones = self._merge_milestone(local_run, result.milestone)
        evidence = self._merge_evidence(summary_source.evidence, result.evidence)
        summary = summary_source.model_copy(
            update={
                "milestones": milestones,
                "current_stage": result.milestone.name,
                "evidence": evidence,
            }
        )
        local_run.summary = summary
        local_run.context = result.context
        local_run.completed = False
        local_run.cancelled = False
        self._local_runs[key] = local_run

    def _merge_milestone(
        self,
        local_run: LocalWorkflowRun,
        milestone: WorkflowMilestone,
        *,
        include_human: bool | None = None,
    ) -> List[WorkflowMilestone]:
        include = include_human if include_human is not None else (
            local_run.hypothesis.requires_human_review
            or milestone.name == "human_review"
            # the research agent can also escalate to review, so keep an existing review milestone in order
            or any(existing.name == "human_review" for existing in local_run.summary.milestones)
        )
        order = self._stage_sequence(include)
        milestone_map = {existing.name: existing for existing in local_run.summary.milestones}
        milestone_map[milestone.name] = milestone
        merged: List[WorkflowMilestone] = []
        for name in order:
            if name in milestone_map:
                merged.append(milestone_map.pop(name))
        merged.extend(milestone_map.values())
        return merged

    def _prepare_human_review(self, key: Tuple[str, str], reason: str | None) -> None:
        """Show the run as paused for review; ``reason`` is the agent's escalation, if any."""

        local_run = self._local_runs.get(key)
        if local_run is None:
            return
        human_milestone = WorkflowMilestone(
            name="human_review",
            status=MilestoneStatus.WAITING_REVIEW,
            detail=f"Awaiting human reviewer decision. Agent escalation: {reason}" if reason else "Awaiting human reviewer decision.",
        )
        milestones = self._merge_milestone(local_run, human_milestone, include_human=True)
        summary = local_run.summary.model_copy(
            update={
                "milestones": milestones,
                "current_stage": "human_review",
            }
        )
        local_run.summary = summary
        local_run.awaiting_review = True
        local_run.completed = False
        metadata = local_run.context.setdefault("metadata", {})
        metadata["human_review"] = {"required": True, "decision": None}
        self._local_runs[key] = local_run

    @staticmethod
    def _merge_evidence(
        existing: List[EvidenceReference],
        additions: List[EvidenceReference],
    ) -> List[EvidenceReference]:
        merged = list(existing)
        seen = {(item.type, str(item.uri)) for item in merged}
        for reference in additions:
            key = (reference.type, str(reference.uri))
            if key not in seen:
                merged.append(reference)
                seen.add(key)
        return merged

    def _on_task_complete(self, key: Tuple[str, str], task: asyncio.Task[None]) -> None:
        self._tasks.pop(key, None)
        try:
            task.result()
        except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "Workflow task failed for workflow %s/%s",
                key[0],
                key[1],
                exc_info=True,
            )
