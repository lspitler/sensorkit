# SPDX-License-Identifier: Apache-2.0
"""Lifecycle procedures that track controller tasks and manage program tasking."""

from __future__ import annotations

import asyncio
import contextlib
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

from loguru import logger

from sensorkit.backend.base import Entity, KeyNotFound
from sensorkit.backend.request import Call, CallError, CallRejected
from sensorkit.common.aio import cleanup_future, defer_cancellation, interruptible_wait
from sensorkit.common.keyword import KeywordDict
from sensorkit.core.controller import ControllerState, InternalControllerState
from sensorkit.core.entity import EntityClient
from sensorkit.core.executor import TaskCall, TaskFailed, TaskOutcomeUnknown
from sensorkit.core.program import (
    ProgramActiveStateRequest,
    ProgramClient,
    ProgramState,
    set_active_state_request,
)
from sensorkit.core.task import (
    InitTask,
    RecoverTask,
    ShutdownTask,
    StandbyTask,
    Task,
    TaskContextMap,
)

if TYPE_CHECKING:
    from sensorkit.auto.reactor import ControllerReactor

LIFECYCLE_TASK_TYPES: dict[InternalControllerState, str] = {
    InternalControllerState.OPERATE: "init",
    InternalControllerState.STANDBY: "standby",
    InternalControllerState.SHUTDOWN: "shutdown",
}
"""Lifecycle task types for the demandable controller states."""


class LifecycleStep(StrEnum):
    """The lifecycle operation currently in progress."""

    INIT = auto()
    TASKING = auto()
    STANDBY = auto()
    SHUTDOWN = auto()
    RECOVER = auto()
    WAIT = auto()


@dataclass(slots=True, frozen=True)
class DemandState:
    """Demanded controller state, task contexts, and program."""

    state: InternalControllerState | None
    contexts: TaskContextMap | None = None
    program: ProgramClient | None = None

    # Interruption flags wake the reactor but do not affect demand equality.
    interrupt: bool = field(default=False, compare=False)
    """Abort the current procedure when this demand replaces it."""
    program_interrupt: bool = field(default=False, compare=False)
    """Allow aborting program tasking at OPERATE with no transition or lifecycle call pending."""

    def lifecycle_context(self, state: InternalControllerState | None = None) -> KeywordDict:
        """Return the task context for a lifecycle state, defaulting to the demanded state.

        Return an empty context if no contexts or state are set.
        """
        state = self.state if state is None else state

        if self.contexts is None or state is None:
            return KeywordDict()

        return copy.deepcopy(self.contexts.get(LIFECYCLE_TASK_TYPES[state]))


class ProcedureError(Exception):
    """A procedure failed and requires a delayed retry."""


class ControllerFailure(ProcedureError):
    """A procedure failure that contributes to the controller's recovery threshold."""


class ControllerIndeterminate(Exception):
    """Controller task execution or program tasking could not be confirmed stopped."""


class ControllerOffline(Exception):
    """The controller does not hold its lease."""


class ControllerUnrecoverable(Exception):
    """Automatic recovery was exhausted and emergency shutdown succeeded.

    Operator intervention is required before lifecycle control resumes.
    """


class LifecycleProcedure(ABC):
    """Apply demand and track dispatched tasks through cleanup.

    A stop request or state logic failure triggers graceful cleanup. An abort request or failed
    graceful cleanup triggers abort cleanup. Dispatched tasks and registered futures remain
    tracked when the logic awaiting them is interrupted.

    Returning from state logic sends no cleanup requests. Canceling the procedure stops local
    cleanup without sending further controller or program requests. Work that cannot be
    confirmed stopped is marked unresolved, causing the reactor to restart.
    """

    graceful_budget: ClassVar[float] = 150.0
    """Time limit in seconds for each graceful cleanup stage before escalating to abort."""

    abort_budget: ClassVar[float] = 15.0
    """Time limit in seconds for each abort cleanup stage."""

    state_reads: ClassVar[int] = 3
    """Attempts to read the controller state after a task execution ends."""

    state_read_delay: ClassVar[float] = 0.5
    """Seconds between attempts to read the controller state."""

    def __init__(self, reactor: ControllerReactor, demand: DemandState):
        self.reactor = reactor
        self.demand = demand
        self.interrupt: Literal["stop", "abort"] | None = None
        """The most severe interruption requested."""
        self.achieved = False
        """Whether this procedure confirmed the required controller state and lifecycle context."""
        self.unresolved = False
        """Whether tracked tasks, cleanup futures, or program tasking may still be active."""
        self.cleanup_errors: list[BaseException] = []
        self._calls: list[TaskCall] = []
        self._cleanup_futures: set[asyncio.Future] = set()
        self._logic: asyncio.Task[None] | None = None
        self._stopping: asyncio.Task[bool] | None = None
        self._task: asyncio.Task[None] | None = None

    @final
    def start(self):
        """Schedule the procedure."""
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        logger.debug(f"{type(self).__name__} starting")

        controller = self.reactor.controller

        try:
            # Check the controller's lease before starting procedure logic.
            if self.interrupt is None and not await controller.is_online():
                raise ControllerOffline(f"{controller.entity} is offline")

            # A procedure interrupted before its first step has nothing to run.
            if self.interrupt is None:
                self._logic = asyncio.create_task(self.state_logic())
                await self._finish(self._logic)

                # Failed logic escalates to a graceful stop.
                if not self._logic.cancelled() and self._logic.exception() is not None:
                    self.request_stop()

            if self.interrupt == "stop":
                self._stopping = asyncio.create_task(self._stop())
                await self._finish(self._stopping)

                # Work that did not end gracefully is aborted.
                if self.interrupt != "abort" and not self._stopping.result():
                    self.request_abort()

            if self.interrupt == "abort":
                await self._abort()
        finally:
            self.unresolved |= any(not future.done() for future in self._owned()) or any(
                isinstance(call.error(), TaskOutcomeUnknown) for call in self._calls
            )

            for error in self.cleanup_errors:
                logger.opt(exception=error).warning(
                    f"{type(self).__name__} cleanup failed: {error}"
                )

        if (error := self._failure()) is not None:
            raise error

    @staticmethod
    async def _finish(stage: asyncio.Task):
        """Wait for a stage, canceling and joining it if this wait is canceled."""
        try:
            await asyncio.wait((stage,))
        except asyncio.CancelledError:
            stage.cancel()
            await defer_cancellation(asyncio.wait((stage,)))
            raise

    async def _stop(self) -> bool:
        """Wait for tracked tasks and futures to finish, then run the stop logic.

        Returns:
            Whether tracked work and stop logic finished within their time limits.
        """
        if await interruptible_wait(self._owned(), timeout=self.graceful_budget):
            return False

        await self._read_ended()

        try:
            async with asyncio.timeout(self.graceful_budget):
                await self.stop_logic()
        except Exception as e:
            self.cleanup_errors.append(e)
            return False

        return True

    async def _abort(self):
        """Abort tracked tasks, cancel registered futures, then run the abort logic."""
        for call in self._calls:
            call.abort()

        for future in self._cleanup_futures:
            future.cancel("Procedure abort")

        if await interruptible_wait(self._owned(), timeout=self.abort_budget):
            logger.error(f"{type(self).__name__} owned work did not end after abort")

        await self._read_ended()

        try:
            async with asyncio.timeout(self.abort_budget):
                await self.abort_logic()
        except Exception as e:
            self.cleanup_errors.append(e)

    def _owned(self) -> list[asyncio.Future]:
        """Return completion futures for tracked task calls and registered cleanup work."""
        return [*(call.future() for call in self._calls), *self._cleanup_futures]

    async def _read_ended(self):
        """Refresh controller state after an interrupted task call finishes.

        The task may have changed state before the monitor received the update.
        """
        if any(self._ended(call) for call in self._calls):
            await self._read_state()

    def _failure(self) -> BaseException | None:
        """Return the primary failure of a finished procedure."""
        logic = self._logic

        if logic is not None and not logic.cancelled() and (error := logic.exception()):
            return error

        # A request can fail after interruption ended the logic that sent it.
        for call in self._calls:
            match cause := call.error():
                case CallRejected():
                    error = ProcedureError(f"{call.task.task_type} task rejected")
                case TaskFailed(aborted=False):
                    error = ControllerFailure(f"{call.task.task_type} task failed")
                case _:
                    continue

            error.__cause__ = cause
            return error

        return next(iter(self.cleanup_errors), None)

    def request_stop(self):
        """Request a graceful end, letting sent requests and tasking finish."""
        if self.interrupt is None:
            self.interrupt = "stop"

            if self._logic is not None:
                self._logic.cancel()

    def request_abort(self):
        """Request abort cleanup, interrupting state logic or graceful cleanup."""
        if self.interrupt != "abort":
            self.interrupt = "abort"

            for stage in (self._logic, self._stopping):
                if stage is not None:
                    stage.cancel()

    def may_dispatch(self) -> bool:
        """Return whether the procedure may start another operation.

        Interruption, disabled control, or failed monitoring prevents further operations.
        A changed demand also prevents them unless this is a recovery procedure.
        """
        return self.interrupt is None and self.reactor.may_dispatch(self)

    def _check_dispatch(self):
        """Raise cancellation if the procedure may not start another operation."""
        if not self.may_dispatch():
            raise asyncio.CancelledError()

    async def _prepare(self, client: EntityClient):
        """Subscribe to client events and check that another operation is still permitted.

        Callers must send their request before the next suspension to preserve this check.
        """
        await client.subscribe_events()
        self._check_dispatch()

    def pending(self) -> bool:
        """Return True while a tracked lifecycle task call is awaiting a result."""
        return any(not call.future().done() for call in self._calls)

    @staticmethod
    def _ended(call: TaskCall) -> bool:
        """Return whether a call's execution is known to have ended."""
        return call.future().done() and isinstance(call.error(), TaskFailed | None)

    async def _read_state(self) -> ControllerState | None:
        """Refresh the reactor's controller state, retrying failed reads."""
        controller = self.reactor.controller

        for attempt in range(1, self.state_reads + 1):
            try:
                state = await controller.kv_get_model(ControllerState)
            except Exception as e:
                logger.warning(f"Could not read {controller.entity} state ({e!r})")
            else:
                self.reactor.observe(state.operating_state)
                return state

            # An abort request ends the retries early.
            if attempt < self.state_reads and self.interrupt != "abort":
                await asyncio.sleep(self.state_read_delay)
            else:
                break

        return None

    async def run_task(
        self,
        task: Task,
        *,
        context: KeywordDict | None = None,
        timeout: float | None = None,
    ) -> ControllerState:
        """Execute and track a lifecycle task, interrupting any controller task in progress.

        Args:
            task: The lifecycle task to execute.
            context: Optional keyword context to attach to the execution.
            timeout: Seconds before the execution is aborted, or None to wait for its result.

        Returns:
            The controller state read after the task succeeded.

        Raises:
            ProcedureError: If the request is rejected.
            ControllerFailure: If the task fails, times out, or the controller state cannot be
                read afterward.
            ControllerIndeterminate: If the execution may still be running.
        """
        controller = self.reactor.controller
        await self._prepare(controller)

        call = TaskCall(controller, task, context=context, interrupt=True)
        self._calls.append(call)

        # Send before yielding so demand cannot change between validation and dispatch.
        # Keep waiting for acceptance if procedure logic is interrupted.
        sent = asyncio.eager_task_factory(asyncio.get_running_loop(), call.send())
        sent.add_done_callback(cleanup_future)

        # Read dispatch failures from the call's shared result below.
        with contextlib.suppress(CallRejected, TaskOutcomeUnknown):
            await asyncio.shield(sent)

        if await interruptible_wait((call.future(),), timeout=timeout):
            call.abort()
            await interruptible_wait((call.future(),), timeout=self.abort_budget)

        state = await self._read_state() if self._ended(call) else None
        error = call.error()

        # Remove resolved calls before reporting their result to avoid duplicate cleanup errors.
        if call.future().done() and not isinstance(error, TaskOutcomeUnknown):
            self._calls.remove(call)

        match error:
            case None if not call.future().done():
                raise ControllerIndeterminate(f"{task.task_type} execution may still be running")
            case None if state is None:
                # An older report must not stand in for the state the task left.
                raise ControllerFailure(f"Controller state unread after {task.task_type}")
            case None:
                return state
            case CallRejected():
                raise ProcedureError(f"{task.task_type} task rejected") from error
            case TaskFailed(aborted=True):
                raise ControllerFailure(f"{task.task_type} task aborted") from error
            case TaskFailed():
                raise ControllerFailure(f"{task.task_type} task failed") from error
            case _:
                raise ControllerIndeterminate(
                    f"{task.task_type} execution may still be running"
                ) from error

    async def run_unless_achieved(self, task: Task):
        """Run a lifecycle task unless the controller already reports its target state.

        Both the state and demanded lifecycle context must match, with no transition pending.

        Raises:
            ProcedureError: If the task request is rejected.
            ControllerFailure: If the task fails or the resulting state or context differs.
            ControllerIndeterminate: If the task cannot be confirmed stopped.
        """
        state = task.target_state()
        context = self.demand.lifecycle_context()
        self._check_dispatch()

        # Controllers should also skip work that is already done, but the agent does not rely on it.
        if self.reactor.observed().matches(state, context):
            self.confirm_state(state, context, commanded=False)
            return

        reported = await self.run_task(task, context=context)

        if not reported.operating_state.matches(state, context):
            raise ControllerFailure(f"Controller did not report {state} after {task.task_type}")

        self.confirm_state(state, context, commanded=True)

    def confirm_state(self, state: InternalControllerState, context: KeywordDict, *, commanded: bool):
        """Record that the controller reports the required state and lifecycle context.

        Args:
            state: The reported state.
            context: The reported lifecycle context.
            commanded: Whether this procedure's own lifecycle task established the state.
        """
        self.achieved = True
        self.reactor.record_achieved_state(state, context, commanded=commanded)

    @abstractmethod
    async def state_logic(self):
        """Run logic for this procedure."""

    async def stop_logic(self):
        """Perform graceful cleanup specific to this procedure.

        Runs after tracked lifecycle tasks and registered futures finish. Overrides should stop
        any program tasking they started or adopted. Failure, timeout, or an abort request
        escalates to abort logic.
        """

    async def abort_logic(self):
        """Perform abort cleanup specific to this procedure.

        Runs after aborting tracked lifecycle tasks and canceling registered futures.
        Overrides should abort any program tasking they started or adopted.
        """

    async def shield_with_cleanup[T](self, fut: asyncio.Future[T]) -> T:
        """Await a future while shielding it from interruption.

        Graceful cleanup waits for the future, and abort cancels it.
        """
        self._cleanup_futures.add(fut)
        result = await asyncio.shield(fut)
        self._cleanup_futures.discard(fut)
        return result

    @final
    async def stop(self):
        """Request a graceful end and wait until cleanup has finished."""
        self.request_stop()
        await self.join()

    @final
    async def abort(self):
        """Request an immediate end and wait until cleanup has finished."""
        self.request_abort()
        await self.join()

    @final
    def cancel(self):
        """Cancel procedure logic and cleanup without sending further requests.

        Work that has not ended remains unresolved.
        """
        self.future().cancel()

    @final
    async def join(self):
        """Wait for the procedure and cleanup without propagating their exceptions."""
        await asyncio.wait((self.future(),))

    @final
    def done(self):
        """Return True if the procedure and its cleanup have finished."""
        return self.future().done()

    @final
    def error(self) -> BaseException | None:
        """Return the procedure's failure, or None if running, successful, or cancelled."""
        task = self.future()

        if not task.done() or task.cancelled():
            return None

        return task.exception()

    @final
    def future(self) -> asyncio.Task[None]:
        """Return the procedure task for use as an awaitable or in `asyncio.wait`."""
        assert self._task is not None, "procedure not started"
        return self._task


class OperateProcedure(LifecycleProcedure):
    """Achieve OPERATE and supervise program tasking.

    Cleanup stops or aborts program tasking that this procedure started or adopted.
    Returning from state logic leaves tasking for the next procedure to adopt or abort.
    """

    activation_timeout: ClassVar[float] = 5.0
    """Seconds to wait for program tasking to start."""

    def __init__(self, reactor: ControllerReactor, demand: DemandState):
        super().__init__(reactor, demand)
        self._activation: asyncio.Task[None] | None = None
        self._ends_tasking = False

    @override
    async def state_logic(self):
        controller = str(self.reactor.controller.entity)
        program = self.demand.program

        self.reactor.step = LifecycleStep.INIT
        await self._reconcile_existing_tasking()

        # Adoption skips activation because the program is already tasking with matching contexts.
        adopted = self._ends_tasking

        logger.info(f"Preparing to operate ({controller})")
        await self.run_unless_achieved(InitTask())

        # Maintain OPERATE while waiting for a program demand.
        if program is None:
            self.reactor.step = LifecycleStep.WAIT
            logger.info(f"Waiting for a program for tasking ({controller})")
            await asyncio.sleep(float("inf"))

        self.reactor.step = LifecycleStep.TASKING

        if adopted:
            logger.info(f"Resuming {program.entity} tasking ({controller})")
        else:
            logger.info(f"Activating {program.entity} program ({controller})")
            await self.activate_tasking()

        logger.info(f"Controller is operating ({controller})")

        # Tasking may end because of an error, an external stop request, or a program restart.
        stop_event = await program.wait_until_tasking_stops()

        match stop_event and stop_event.origin:
            case None:
                raise ProcedureError("Failed to start tasking")
            case "error":
                # Program errors count toward controller recovery.
                logger.error(f"Error reported ({controller})")
                raise ControllerFailure("Program reported error")
            case "init":
                # TODO: Bound tasking retries after program restarts.
                logger.warning(f"Active program restarted ({controller})")

                # Pause briefly, then return for replacement.
                self.reactor.step = LifecycleStep.WAIT
                await asyncio.sleep(3)
            case "request":
                logger.info(f"Tasking stopped by request ({controller})")
                raise ProcedureError("Program was stopped")

    @override
    async def stop_logic(self):
        # Wait for the start request to finish before sending stop.
        if self._activation is not None:
            await asyncio.wait((self._activation,))

        if self._ends_tasking:
            await self.demand.program.stop_tasking()
            self._ends_tasking = False

    @override
    async def abort_logic(self):
        if self._activation is not None:
            await asyncio.wait((self._activation,))

        try:
            if self._ends_tasking:
                await self.demand.program.abort_tasking()
                self._ends_tasking = False
        finally:
            # A failed abort leaves no confirmation that program tasking stopped.
            self.unresolved |= self._ends_tasking

    async def _reconcile_existing_tasking(self):
        """Adopt the demanded program's active tasking, or abort it.

        Adopt tasking only if its controller and task contexts match demand and the controller
        reports the required state and lifecycle context. Otherwise, abort the program's tasking.

        Raises:
            ProcedureError: If tasking could not be aborted.
        """
        program = self.demand.program

        if program is None:
            return

        try:
            tasking = await program.kv_get_model(ProgramState)
        except KeyNotFound:
            return

        if not tasking.active_state.active:
            return

        enabled_on = tasking.enable_state.controller

        if (
            not tasking.active_state.stopping
            and enabled_on is not None
            and Entity.at(enabled_on) == self.reactor.controller.entity
            and tasking.active_state.contexts == (self.demand.contexts or TaskContextMap())
            and self.reactor.observed().matches(self.demand.state, self.demand.lifecycle_context())
        ):
            self._ends_tasking = True
            return

        await self._prepare(program)

        try:
            async with asyncio.timeout(self.abort_budget):
                await program.abort_tasking()
        except Exception as e:
            raise ProcedureError(f"Could not abort {program.entity} tasking") from e

    async def activate_tasking(self):
        """Start program tasking and take responsibility for stopping it during cleanup.

        Abort any running controller task first so the controller can accept program tasks.
        Cleanup waits for the start request before stopping or aborting tasking. If the request
        result is unknown, tasking may start later and is marked unresolved.

        Raises:
            ControllerFailure: If a running execution could not be aborted.
            ProcedureError: If tasking did not start in time.
        """
        program = self.demand.program
        controller = self.reactor.controller

        if self._activation is not None:
            raise RuntimeError("Tasking was already activated")

        if (execution := self.reactor.running_execution()) is not None:
            await self._prepare(controller)

            try:
                async with asyncio.timeout(self.abort_budget):
                    await controller.abort_task(execution.task_id)
            except CallRejected:
                # The execution already ended. A later request is rejected if other work started.
                pass
            except Exception as e:
                raise ControllerFailure(f"Could not abort {execution.task.task_type} task") from e

        await self._prepare(program)
        call = program.call(
            set_active_state_request,
            ProgramActiveStateRequest(
                action="start", contexts=self.demand.contexts or TaskContextMap()
            ),
        )
        self._ends_tasking = True

        # Send before yielding so demand cannot change between validation and dispatch.
        # Keep waiting for the reply if procedure logic is interrupted.
        self._activation = asyncio.eager_task_factory(
            asyncio.get_running_loop(), self._activated(program, call)
        )
        self._activation.add_done_callback(cleanup_future)

        # A timeout leaves the start request running so cleanup can wait for its result.
        if await interruptible_wait((self._activation,), timeout=self.activation_timeout):
            raise ProcedureError("Tasking did not start in time")

        if (error := self._activation.exception()) is not None:
            raise ProcedureError("Failed to start tasking") from error

    async def _activated(self, program: ProgramClient, call: Call[None]):
        """Send an activation request and follow it to its result."""
        accepted = False

        # Read request failures from the shared result below.
        with contextlib.suppress(Exception):
            await call.invoke()
            accepted = True

        # Keep the shared result available if this wait is interrupted.
        result = call.get_future()
        await asyncio.wait((result,))

        error = asyncio.CancelledError() if result.cancelled() else result.exception()

        match error:
            case None:
                return
            case CallRejected():
                # Explicit rejection leaves tasking inactive.
                self._ends_tasking = False
                raise error
            case CallError() if accepted:
                # The program finished handling the request, so the stop logic can end tasking.
                raise error
            case _:
                self.unresolved = True
                raise ControllerIndeterminate(f"{program.entity} may start tasking later") from error


class StandbyProcedure(LifecycleProcedure):
    """Achieve STANDBY and wait for replacement."""

    @override
    async def state_logic(self):
        controller = str(self.reactor.controller.entity)

        self.reactor.step = LifecycleStep.STANDBY

        logger.info(f"Entering standby mode ({controller})")
        await self.run_unless_achieved(StandbyTask())

        self.reactor.step = LifecycleStep.WAIT

        logger.info(f"Controller is idle in standby ({controller})")
        await asyncio.sleep(float("inf"))


class ShutdownProcedure(LifecycleProcedure):
    """Achieve SHUTDOWN and wait for replacement."""

    @override
    async def state_logic(self):
        controller = str(self.reactor.controller.entity)

        self.reactor.step = LifecycleStep.SHUTDOWN

        logger.info(f"Commencing shutdown ({controller})")
        await self.run_unless_achieved(ShutdownTask())

        self.reactor.step = LifecycleStep.WAIT

        logger.info(f"Controller is shut down ({controller})")
        await asyncio.sleep(float("inf"))


class RecoverProcedure(LifecycleProcedure):
    """Run one recovery attempt, or an emergency shutdown once attempts are exhausted.

    Recovery can start from and leave any reported state. The reactor schedules attempts and
    evaluates demand again after each one.
    """

    recover_timeout: ClassVar[float] = 120.0
    """Seconds before a recovery attempt is aborted."""

    shutdown_timeout: ClassVar[float] = 240.0
    """Seconds before an emergency shutdown is aborted."""

    def __init__(self, reactor: ControllerReactor, demand: DemandState, *, emergency: bool):
        super().__init__(reactor, demand)
        self.emergency = emergency
        """Whether this attempt is an emergency shutdown."""
        self.recovered = False
        """Whether the recovery task succeeded."""

    @override
    async def state_logic(self):
        controller = str(self.reactor.controller.entity)

        if self.emergency:
            self.reactor.step = LifecycleStep.SHUTDOWN
            task = ShutdownTask()
            state = task.target_state()
            context = self.demand.lifecycle_context(state)

            logger.warning(f"Failed to recover, attempting shutdown ({controller})")
            reported = await self.run_task(task, context=context, timeout=self.shutdown_timeout)

            if not reported.operating_state.matches(state, context):
                raise ControllerFailure(
                    f"Controller did not report {state} after emergency shutdown"
                )

            # Emergency shutdown leaves control disabled pending operator intervention.
            logger.info(f"Emergency shutdown succeeded ({controller})")
            raise ControllerUnrecoverable("Emergency shutdown")

        self.reactor.step = LifecycleStep.RECOVER

        logger.info(f"Attempting recovery ({controller})")
        await self.run_task(RecoverTask(), timeout=self.recover_timeout)

        logger.info(f"Recovery succeeded ({controller})")
        self.recovered = True
