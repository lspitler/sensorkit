# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from loguru import logger

from sensorkit.backend.request import Call, CallError, CallRejected
from sensorkit.common.aio import cleanup_future, interruptible_wait
from sensorkit.common.keyword import KeywordDict
from sensorkit.core.controller import (
    ControllerClient,
    ExecuteResponseMessage,
    TaskExecutionResult,
)
from sensorkit.core.entity import EventsNotSubscribed
from sensorkit.core.task import Task, TaskContextMap, TaskExecution, TaskSubmission

type TaskFactoryResult = Task | TaskSubmission | None
type TaskFactoryFunc = Callable[
    [],
    Coroutine[Any, Any, TaskFactoryResult]
    | AsyncGenerator[TaskFactoryResult, TaskExecution]
    | TaskFactoryResult,
]
type TaskChangeCallback = Callable[[TaskExecution | None], Coroutine[Any, Any, None]]


class NoTaskAvailable(Exception):
    """There is no task available for execution."""


async def _invoke_factory(
    func: TaskFactoryFunc,
) -> tuple[AsyncGenerator[TaskFactoryResult, TaskExecution] | None, TaskSubmission]:
    """Invoke a task factory and normalize its output.

    The factory may be a plain function or coroutine returning a `Task`/`TaskSubmission`/`None`, or
    an async generator that yields one of those and is later resumed with the minted
    `TaskExecution` (see `_resume_factory`).

    Returns:
        A tuple of the generator to resume after dispatch (or `None` for non-generator factories)
        and the normalized `TaskSubmission`.

    Raises:
        NoTaskAvailable: If the factory offers no task this cycle.
    """
    obj = func()

    match obj:
        case AsyncGenerator():
            agen = obj
            try:
                result = await agen.asend(None)
            except StopAsyncIteration as e:
                # A generator that never yields is a no-task scenario rather than an error.
                raise NoTaskAvailable from e
        case Coroutine():
            agen = None
            result = await obj
        case Task() | TaskSubmission() | None:
            agen = None
            result = obj
        case _:
            raise RuntimeError("task factory returned invalid type")

    if result is None:
        # The factory had no task for us. Close any generator so its finally-blocks run.
        if agen is not None:
            await agen.aclose()

        raise NoTaskAvailable

    # Normalize a bare task into a submission with no execution parameters.
    submission = result if isinstance(result, TaskSubmission) else TaskSubmission(task=result)

    return agen, submission


async def _resume_factory(
    agen: AsyncGenerator[TaskFactoryResult, TaskExecution], execution: TaskExecution
) -> None:
    """Resume a factory generator once, handing it the minted execution.

    The generator may simply end (ignoring the result), or `await` the execution itself to consume
    the result before ending. Either way it is expected to stop after a single resumption.
    """
    try:
        await agen.asend(execution)
    except StopAsyncIteration:
        # Expected: the factory ran its post-dispatch logic (if any) and ended.
        return
    else:
        logger.warning(
            "Task factory generator yielded more than once! This indicates a programming error."
        )
        await agen.aclose()


class TaskOutcomeUnknown(Exception):
    """Task acceptance or execution completion could not be confirmed."""


class TaskFailed(CallError):
    """A task execution ended without success."""

    def __init__(self, *args, aborted: bool = False):
        super().__init__(*args)
        self.aborted = aborted
        """Whether this call requested an abort before receiving the failed result."""


class TaskCall:
    """Send one task to a controller and follow the execution it starts.

    Rejection confirms that the task did not run. After acceptance, the execution's end event
    confirms success or failure. Missing acceptance or completion leaves the call unresolved.
    """

    def __init__(
        self,
        controller: ControllerClient,
        task: Task,
        *,
        context: KeywordDict | None = None,
        expiry_time: datetime | None = None,
        interrupt: bool = False,
    ):
        """Create a call without sending it.

        Args:
            controller: The controller to execute the task.
            task: The task to execute.
            context: Optional keyword context to attach to the execution.
            expiry_time: Optional time after which the execution should be considered expired.
            interrupt: Whether to interrupt any task currently in progress.
        """
        self.controller = controller
        self.task = task
        self.context = context
        self.expiry_time = expiry_time
        self.interrupt = interrupt
        self.execution: TaskExecution | None = None
        """The execution the controller identified, if any."""
        loop = asyncio.get_running_loop()
        self._outcome: asyncio.Future[TaskExecutionResult] = loop.create_future()
        self._outcome.add_done_callback(cleanup_future)
        self._abort = asyncio.Event()
        self._sent = False
        self._follower: asyncio.Task[TaskExecutionResult] | None = None

    async def send(self) -> TaskExecution:
        """Send the request and wait until the controller accepts the task.

        The request goes out before the caller first suspends. Canceling the caller while it
        waits for the reply leaves the call unresolved. The returned execution can be awaited
        for the call's result.

        Raises:
            RuntimeError: If the call was already sent.
            EventsNotSubscribed: If the controller's events are not subscribed.
            BackendError: If the controller's event subscription failed to start.
            CallRejected: If the controller rejected the task without running it.
            TaskOutcomeUnknown: If acceptance could not be established.
        """
        if self._sent:
            raise RuntimeError("Task call already sent")

        if not self.controller.get_event_mux().ready():
            raise EventsNotSubscribed(f"{self.controller.entity} events are not subscribed")

        self._sent = True
        call = self.controller.execute_task(
            self.task, context=self.context, expiry_time=self.expiry_time, interrupt=self.interrupt
        )

        try:
            await call.invoke()
        except CallRejected as e:
            # A rejection reply can still identify the execution it declined.
            if call.response is not None:
                self.execution = call.response.execution

            self._outcome.set_exception(e)
            raise
        except asyncio.CancelledError as e:
            # Losing the reply leaves acceptance unknown even if the controller started the task.
            self._outcome.set_exception(self._outcome_unknown("acceptance is unknown", e))
            raise
        except Exception as e:
            error = self._outcome_unknown("acceptance is unknown", e)
            self._outcome.set_exception(error)
            raise error from e

        if call.response is None or call.response.execution is None:
            error = self._outcome_unknown("was accepted without an execution")
            self._outcome.set_exception(error)
            raise error

        execution = call.response.execution
        self.execution = execution

        # Holders of the execution await the outcome without canceling it.
        shared = asyncio.shield(self._outcome)
        shared.add_done_callback(cleanup_future)
        execution.bind_result(shared)

        self._follower = asyncio.create_task(self._follow(call, execution))
        self._follower.add_done_callback(self._settle)

        return execution

    def abort(self):
        """Request an abort of the execution once the controller accepts it.

        Repeated requests retry a failed abort until the controller accepts one.
        The execution's end event determines the result.
        """
        self._abort.set()

    def future(self) -> asyncio.Future[TaskExecutionResult]:
        """Return the shared future for task rejection or the execution result.

        The future raises `CallRejected` if the task never ran, `TaskFailed` if the execution
        ended without success, and `TaskOutcomeUnknown` if acceptance or completion is unknown.
        Use `asyncio.wait` or `asyncio.shield` to keep waiter cancellation from canceling it.
        """
        return self._outcome

    def error(self) -> BaseException | None:
        """Return why the call ended without success, or None if running or successful."""
        if not self._outcome.done():
            return None

        if self._outcome.cancelled():
            return self._outcome_unknown("outcome was cancelled")

        return self._outcome.exception()

    def _outcome_unknown(self, reason: str, cause: BaseException | None = None) -> TaskOutcomeUnknown:
        error = TaskOutcomeUnknown(f"{self.task.task_type} {reason}")
        error.__cause__ = cause
        return error

    def _settle(self, follower: asyncio.Task[TaskExecutionResult]):
        if self._outcome.done():
            return

        if follower.cancelled():
            self._outcome.set_exception(self._outcome_unknown("call was cancelled"))
        elif (error := follower.exception()) is not None:
            self._outcome.set_exception(error)
        else:
            self._outcome.set_result(follower.result())

    async def _follow(
        self,
        call: Call[ExecuteResponseMessage, TaskExecutionResult],
        execution: TaskExecution,
    ) -> TaskExecutionResult:
        result = call.get_future()
        aborted = False
        delivered = False

        while not result.done():
            if not await interruptible_wait((result,), interrupt=self._abort):
                break

            # Requests received during this send trigger another attempt if the send fails.
            self._abort.clear()
            aborted = True

            if not delivered:
                delivered = await self._send_abort(execution)

        error = asyncio.CancelledError() if result.cancelled() else result.exception()

        match error:
            case None:
                return result.result()
            case CallError():
                outcome = "aborted" if aborted else "failed"
                raise TaskFailed(f"{self.task.task_type} {outcome}", aborted=aborted) from error
            case _:
                # The end event was lost, so the execution may still be running.
                raise self._outcome_unknown("end is unknown", error)

    async def _send_abort(self, execution: TaskExecution) -> bool:
        """Send a targeted abort and return whether the controller accepted it."""
        try:
            await self.controller.abort_task(execution.task_id).invoke()
        except Exception as e:
            logger.warning(f"Could not abort {self.task.task_type} execution ({e!r})")
            return False

        return True


class TaskingLoop:
    """Executes a loop that sources tasks from a factory and runs them on a target Controller."""

    def __init__(
        self,
        controller: ControllerClient,
        factory_func: TaskFactoryFunc,
        contexts: TaskContextMap,
        task_group: asyncio.TaskGroup,
        *,
        on_task_change: TaskChangeCallback | None = None,
    ):
        self.controller = controller
        self.task_factory: TaskFactoryFunc = factory_func
        self.contexts = contexts
        self._task_group = task_group
        # Invoked with the live `TaskExecution` when a task begins running and with `None` when it
        # completes, so an owner (the program) can publish its current tasking state. Error/abort
        # exits are left to the owner's loop-teardown path rather than signalled here.
        self._on_task_change = on_task_change
        self._aio_task: asyncio.Task | None = None
        self._stop_requested = asyncio.Event()

    def start(self):
        """Activate the tasking loop."""
        if self._aio_task:
            raise RuntimeError("Tasking loop was already started")

        self._aio_task = self._task_group.create_task(self._tasking_loop())
        return self._aio_task

    async def stop(self, *, timeout=None):
        """Deactivate the tasking loop, invalidating the object."""
        try:
            # Wait for a graceful stop.
            self._stop_requested.set()

            async with asyncio.timeout(timeout):
                await self._aio_task
        except TimeoutError:
            # The loop task was cancelled and will call abort on the Controller.
            with contextlib.suppress(asyncio.CancelledError):
                await self._aio_task

    @property
    def stop_requested(self):
        """Return True if a stop has been requested for this tasking loop."""
        return self._stop_requested.is_set()

    def _resolve_context(self, submission: TaskSubmission) -> KeywordDict | None:
        """Merge the factory-supplied context with the per-task-type context bundle."""
        context = KeywordDict(submission.context or {})

        if type_context := self.contexts.get(submission.task.task_type):
            context.update(type_context)

        return context or None

    def _resolve_expiry(self, submission: TaskSubmission) -> datetime | None:
        """Resolve the absolute expiry time for a submission.

        Uses the factory-supplied `expiry_time` when present, otherwise the task's
        `default_expiry`. A `timedelta` default is anchored to the current time.
        """
        if submission.expiry_time is not None:
            return submission.expiry_time

        default = submission.task.default_expiry()

        if isinstance(default, timedelta):
            return datetime.now(UTC) + default

        return default

    async def _tasking_loop(self):
        while not self._stop_requested.is_set():
            # Source the next task from the factory.
            try:
                agen, submission = await _invoke_factory(self.task_factory)
            except NoTaskAvailable:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_requested.wait(), timeout=5.0)

                continue

            # Dispatch and run it. A fatal error ends the loop.
            if not await self._dispatch(agen, submission):
                return

    async def _dispatch(
        self,
        agen: AsyncGenerator[TaskFactoryResult, TaskExecution] | None,
        submission: TaskSubmission,
    ) -> bool:
        """Dispatch a single task and run it to completion.

        Returns:
            True if the task completed (success, or a failure the factory handled without
            re-raising); False if a fatal error occurred and the loop should stop.
        """
        task = submission.task
        context = self._resolve_context(submission)
        expiry_time = self._resolve_expiry(submission)
        ttl = (expiry_time - datetime.now(UTC)).total_seconds() if expiry_time else None

        logger.info(
            f"Dispatching {task.task_type} ({type(task).__name__}) to {self.controller.entity}"
        )

        # Instruct the controller to begin execution. The controller mints the identity and binds
        # the in-flight call as the execution's result future.
        try:
            execution = await self.controller.start_task(
                task, context=context, expiry_time=expiry_time
            )
        except (Exception, asyncio.CancelledError) as e:
            # Dispatch failed before the factory was resumed. Deliver the error into the still-
            # suspended generator so its except/finally blocks run, then abandon this task.
            logger.error(f"Failed to dispatch {task.task_type} to {self.controller.entity}: {e!r}")

            if agen is not None:
                with contextlib.suppress(BaseException):
                    await agen.athrow(e)

            return False

        await self._notify_task_change(execution)

        try:
            async with asyncio.timeout(ttl):
                # Resume the factory with the running execution (task_id now known). It may await
                # the execution to consume the result and run post-task logic.
                if agen is not None:
                    await _resume_factory(agen, execution)

                # Guarantee completion before the next iteration, regardless of whether the factory
                # awaited the execution itself. If it did, the call is already settled and this
                # returns (or re-raises) immediately; if it consumed an error without re-raising,
                # the execution is done and we leave that decision to the factory.
                if not execution.done():
                    await execution
        except (Exception, asyncio.CancelledError) as e:
            # On error/abort the loop terminates; clearing the tasking state is left to the owner's
            # teardown so we don't emit while unwinding (possibly under cancellation).
            await self._abort(execution, e)
            return False

        await self._notify_task_change(None)
        logger.info(f"Finished {task.task_type} ({type(task).__name__})")

        return True

    async def _notify_task_change(self, execution: TaskExecution | None):
        """Signal the owner that the currently running task changed, if a callback is registered."""
        if self._on_task_change is not None:
            await self._on_task_change(execution)

    async def _abort(self, execution: TaskExecution, e: BaseException):
        """Log and abort an in-flight task following a timeout, interrupt, or unhandled error."""
        msg = f"Aborting {execution.task.task_type} on {self.controller.entity}"

        match e:
            case asyncio.CancelledError():
                logger.error(f"{msg} due to interrupt")
            case _:
                logger.error(f"{msg} due to error")
                logger.opt(exception=e).debug(
                    f"{type(e).__name__} during execution of {execution.task_id}"
                )

        with contextlib.suppress(BaseException):
            # Kick off an abort here, but don't wait for it to complete.
            await self.controller.abort_task(execution.task_id).invoke()
