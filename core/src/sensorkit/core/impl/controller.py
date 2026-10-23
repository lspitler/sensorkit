# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import copy
from _contextvars import ContextVar
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, ClassVar, Collection, Mapping, override

import uuid_utils.compat as uuid
from loguru import logger

from sensorkit.backend.base import Entity
from sensorkit.backend.request import CallContext
from sensorkit.common.aio import defer_cancellation
from sensorkit.common.keyword import KeywordDict
from sensorkit.core.controller import (
    AbortRequestMessage,
    AbortResponseMessage,
    ControllerDevice,
    ControllerEnableState,
    ControllerEnableStateRequest,
    ControllerInterface,
    ControllerOperatingState,
    ControllerState,
    ExecuteRequestMessage,
    ExecuteResponseMessage,
    ExecutionHandlerCallback,
    InternalControllerState,
    TaskExecutionResult,
    TaskExecutionState,
    TaskFinishInfo,
    TaskHandlerCallback,
    abort_task_request,
    execute_task_request,
    set_enable_state_request,
)
from sensorkit.core.device import DeviceClient
from sensorkit.core.entity import ControllerDetails, EntityInfo
from sensorkit.core.impl.entity import EntityImpl
from sensorkit.core.task import RecoverTask, Task, TaskExecution, TaskInfo
from sensorkit.data.context import Context, ContextSubscription


class ControllerDeviceMap(Mapping[str, ControllerDevice]):
    """Ordered mapping of device names to their ControllerDevice instances, created on demand."""

    def __init__(self):
        self._dict: dict[str, ControllerDevice] = {}

    def get_or_add(self, name: str, *, get_client_func: Callable[[str], DeviceClient]):
        """Return the ControllerDevice for the given name, creating it if it does not yet exist."""
        if name not in self._dict:
            client = get_client_func(name)
            self._dict[name] = ControllerDevice(client, ContextSubscription(client))

        return self._dict[name]

    def __getitem__(self, key, /):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)


@dataclass(eq=False)
class TaskRun:
    """Own a task's runner and execution outcome."""

    execution: TaskExecution
    context: KeywordDict
    """The task's input context, captured before execution metadata is added."""

    entered: bool = field(default=False, init=False)
    cancel_requested: bool = field(default=False, init=False)
    started: bool = field(default=False, init=False)
    """Whether the start of this execution was published."""

    finish_info: TaskFinishInfo = field(
        default_factory=lambda: TaskFinishInfo(aborted=True),
        init=False,
    )
    outcome: InternalControllerState | None = field(
        default=InternalControllerState.UNKNOWN,
        init=False,
    )
    start_time: datetime | None = field(default=None, init=False)

    _handler: ExecutionHandlerCallback[Any] | None = field(default=None, init=False)
    _runner: asyncio.Task[TaskExecutionResult] | None = field(default=None, init=False)

    @classmethod
    def create(
        cls,
        controller_id: str,
        task: Task,
        *,
        context: KeywordDict | None,
        expiry_time: datetime | None,
    ) -> TaskRun:
        """Mint execution metadata without scheduling a runner."""
        execution = TaskExecution(
            task=task,
            task_id=uuid.uuid7(),
            controller_id=controller_id,
            context=context,
            expiry_time=expiry_time,
        )
        context = copy.deepcopy(execution.context)

        # Add task identity to the working context.
        execution.context.set(
            TaskInfo(task=task, task_id=execution.task_id, controller_id=controller_id)
        )

        return cls(execution, context)

    @property
    def handler(self) -> ExecutionHandlerCallback[Any]:
        assert self._handler is not None, "execution is not attached"
        return self._handler

    @property
    def runner(self) -> asyncio.Task[TaskExecutionResult]:
        assert self._runner is not None, "execution is not attached"
        return self._runner

    def attach(
        self,
        handler: ExecutionHandlerCallback[Any],
        run: Coroutine[Any, Any, TaskExecutionResult],
    ):
        """Bind the handler and schedule the runner of an admitted execution.

        Raises:
            RuntimeError: If the execution is already attached.
        """
        if self._runner is not None:
            raise RuntimeError("execution is already attached")

        self._handler = handler
        self._runner = asyncio.create_task(run)

    def done(self) -> bool:
        return self.runner.done()

    def cancel(self, msg: str | None = None):
        """Request cancellation once."""
        if self.cancel_requested:
            return

        self.cancel_requested = True

        # Canceling before the first step would skip the runner's cleanup.
        if self.entered and not self.runner.done():
            self.runner.cancel(msg)

    async def join(self):
        """Join the runner, propagating errors other than execution cancellation."""
        await asyncio.wait((self.runner,))

        if not self.runner.cancelled() and (error := self.runner.exception()):
            raise error

    async def wait(self):
        """Join the runner without propagating its outcome.

        Caller cancellation waits for completion, then propagates.
        """
        try:
            await defer_cancellation(asyncio.wait((self.runner,)))
        finally:
            # Observe failures before propagating caller cancellation.
            if self.runner.done() and not self.runner.cancelled():
                self.runner.exception()

    @contextlib.contextmanager
    def execute(self) -> Iterator[TaskExecution]:
        """Time execution and record success, failure, or cancellation.

        Raises:
            asyncio.CancelledError: If cancellation was requested before execution.
        """
        if self.cancel_requested:
            raise asyncio.CancelledError()

        self.start_time = datetime.now(UTC)

        try:
            yield self.execution
        except Exception as e:
            self.finish_info = TaskFinishInfo(error=str(e) or type(e).__name__)
            self.outcome = InternalControllerState.ERROR
            raise
        else:
            self.finish_info = TaskFinishInfo()
            self.outcome = self.execution.task.target_state()

    def result(self) -> TaskExecutionResult:
        """Build a successful result with the current time as its end."""
        assert self.start_time is not None, "execution has not started"

        return TaskExecutionResult(
            task_id=self.execution.task_id,
            start_time=self.start_time,
            end_time=datetime.now(UTC),
        )


def _receive_task[T: Task](callback: TaskHandlerCallback[T]) -> ExecutionHandlerCallback[T]:
    """Pass the envelope's task to the callback."""
    return lambda execution: callback(execution.task)


class ControllerImpl(EntityImpl, ControllerInterface):
    """Helper for implementing server-side functionality of a Controller.

    A controller reports UNKNOWN when its service starts. Attach hooks that can determine the
    hardware state report it with `set_internal_state`.
    """

    current: ClassVar[ContextVar[ControllerImpl | None]] = ContextVar("current_controller", default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._enable_hooks: list[Callable[[], None]] = []
        self._disable_hooks: list[Callable[[], None]] = []
        self._task_handlers: dict[type[Task], ExecutionHandlerCallback[Any]] = {}
        self._execution: TaskRun | None = None
        self._devices = ControllerDeviceMap()

    @override
    def task_running(self):
        return self._execution is not None and not self._execution.done()

    @override
    def on_enable(self, func: Callable[[], None]):
        self._enable_hooks.append(func)
        return func

    @override
    def on_disable(self, func: Callable[[], None]):
        self._disable_hooks.append(func)
        return func

    @override
    async def init_impl(self):
        self._state = await ControllerState.recover_or_init(
            self,
            enable_state=ControllerEnableState(enabled=True),
            operating_state=ControllerOperatingState(
                current=InternalControllerState.UNKNOWN,
                context=KeywordDict(),
            ),
            execution_state=TaskExecutionState(execution=None),
        )

        await self._normalize_startup_state()

    async def _normalize_startup_state(self):
        async with self._state.updating(self) as update:
            operating = self._state.operating_state
            current = self._state.execution_state

            # Persist legacy normalization before accepting work.
            legacy = "context" not in operating.model_fields_set

            if (
                operating.current == InternalControllerState.UNKNOWN
                and operating.target is None
                and not legacy
                and not current.executing
            ):
                return

            events = []

            # A pending target also identifies an interrupted lifecycle execution.
            if (
                operating.current == InternalControllerState.TRANSITION
                or operating.target is not None
            ):
                logger.warning(f"Recovering from interrupted transition to {operating.target}")

            if current.executing:
                events.append(
                    TaskExecutionState(
                        finished=TaskFinishInfo(aborted=True),
                        execution=current.execution,
                        context=current.context,
                    )
                )

            # The persisted state describes hardware before the restart.
            events.append(operating.derive(current=InternalControllerState.UNKNOWN, target=None))
            await update(*events)

    @override
    async def attach_impl(self):
        if self._state.enable_state.enabled:
            await self._call_with_context(self._enable_hooks)

        await self.handle_request(set_enable_state_request, self._set_enable_state)
        await self.handle_request(abort_task_request, self._abort_request)
        await self.handle_request(execute_task_request, self._execute_request)

        await self.start_device_subscriptions()

    @override
    async def detach_impl(self):
        await self.stop_device_subscriptions()

    @override
    def use_device(self, name: str, *, subscribe: list[type] | None = None) -> DeviceClient:
        """Register a controlled device and return a client for it.

        Registers a device that this controller interacts with. Optionally subscribes to
        specific keyword types published by the device. Subscribed keywords are automatically
        cached and made available in contexts via `update_context()`.

        Note:
            Subscriptions remain inactive until `start_device_subscriptions()` is called.
            When using the declarative API, this occurs automatically after initialization.

        Args:
            name: Entity identifier for the device (e.g., "mount", "camera").
            subscribe: Optional list of keyword model types (e.g., Pydantic models) to
                subscribe to. Each subscribed keyword will be monitored, and its latest
                value cached for inclusion in contexts.

        Returns:
            A client instance for interacting with the registered device.
                Access the subscription object via `get_device(name).subscription` to
                retrieve cached values directly.

        Raises:
            KeyError: If attempting to access a device via `get_device(name)` that
                hasn't been registered with `use_device()`.

        Examples:
            Basic device registration without subscriptions:

            >>> camera = self.use_device("camera")

            Register device and subscribe to specific keyword types:

            >>> mount = self.use_device("mount", subscribe=[AltAzPointing, RADecPointing])

            Access cached subscription values from the ControlledDevice instance:

            >>> pointing = self.get_device("mount").get(AltAzPointing)
        """
        device = self._devices.get_or_add(
            name,
            get_client_func=lambda _: self.sensorkit().device(Entity((name,))),
        )

        if subscribe:
            for kw_type in subscribe:
                device.subscription.add(kw_type)

        return device.client

    @override
    def get_device(self, name: str) -> ControllerDevice:
        return self._devices[name]

    @override
    def all_devices(self) -> Collection[ControllerDevice]:
        return self._devices.values()

    @override
    async def start_device_subscriptions(self):
        """Start all keyword subscriptions registered via `use_device`.

        This is called automatically by the declarative API after init
        callbacks have run.  It may also be called manually if the
        controller is not using the declarative API.
        """
        for device in self._devices.values():
            await device.subscription.start()

    @override
    async def stop_device_subscriptions(self):
        """Stop all keyword subscriptions registered via `use_device`.

        Called automatically by the declarative API before deinit
        callbacks run.
        """
        for device in self._devices.values():
            await device.subscription.stop()

    @override
    async def update_context(
        self,
        *args,
        **kwargs,
    ) -> Context:
        """Update the context of the currently executing task.

        Merges the base context with fresh snapshots from all device keyword
        subscriptions and additional provided values. The updated context is
        persisted to the execution state and returned.

        This method allows task handlers to refresh their context mid-execution,
        incorporating the latest device state.

        Args:
            **kwargs: Additional literal key-value pairs to include in the context.

        Returns:
            The newly updated `Context` containing (in precedence order,
            highest-last): current task context, fresh device snapshots,
            and *kwargs*.

        Raises:
            RuntimeError: If called when no task is currently executing.

        Examples:
            Update context with latest device data during task execution:

            >>> async def my_task_handler(self, task):
            ...     # ... some work ...
            ...     ctx = await self.update_context()
            ...     pointing = ctx.get(AltAzPointing)

            Add custom values while updating:

            >>> ctx = await self.update_context(iteration=5, timestamp=time.time())
        """
        async with self._state.updating(self) as update:
            current = self._state.execution_state

            if not current.executing or current.execution is None:
                raise RuntimeError("no task executing")

            ctx = Context(current.context)

            for device in self._devices.values():
                device.subscription.snapshot(into=ctx)

            ctx.set(*args)

            for key, value in kwargs.items():
                ctx.set_value(key, value)

            await update(
                TaskExecutionState(
                    executing=current.executing,
                    aborting=current.aborting,
                    execution=current.execution,
                    context=ctx,
                ),
            )

        return ctx

    async def _set_enable_state(self, request: ControllerEnableStateRequest):
        if request.enable == self._state.enable_state.enabled:
            return

        await self._state.update(self, ControllerEnableState(enabled=request.enable))

        if request.enable:
            await self._call_with_context(self._enable_hooks)
        else:
            #TODO: Abort. Need a nice way of calling a local "loopback" request.
            # await self.call_loopback(abort_task_request, AbortRequestMessage(task_id=None))

            await self._call_with_context(self._disable_hooks)

    async def _abort_request(
        self,
        requested: AbortRequestMessage,
        call: CallContext[None, AbortResponseMessage],
    ):
        async with self._state.updating(self) as update:
            target = self._execution if self.task_running() else None
            current = self._state.execution_state

            # A finished runner may still be completing its request.
            if target is None:
                call.reject(
                    response=AbortResponseMessage(aborting=False, task_id=None),
                    reason="No task is running",
                )
                return

            task_id = target.execution.task_id

            # If given, ensure the task_id matches the latest accepted execution.
            if requested.task_id and task_id != requested.task_id:
                call.reject(
                    response=AbortResponseMessage(aborting=False, task_id=None),
                    reason=f"Running task {task_id} does not match",
                )
                return

            logger.debug(f"aborting running task {task_id}")
            call.accept(response=AbortResponseMessage(aborting=True, task_id=task_id))

            # Pending work leaves its predecessor's published execution alone.
            if current.executing and current.execution.task_id == task_id:
                await update(
                    TaskExecutionState(
                        executing=True,
                        aborting=True,
                        execution=current.execution,
                        context=current.context,
                    ),
                )

            target.cancel("Aborted by remote request")

        # Wait for the execution and its predecessors to end.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await call.progress_from_task(target.runner, cadence=5, ttl=10)

        if target.done():
            # Success.
            await call.succeed(result=None)
        else:
            await call.fail()

    async def _admit(
        self,
        msg: ExecuteRequestMessage,
        call: CallContext[TaskExecutionResult, ExecuteResponseMessage],
    ) -> TaskRun | None:
        task = msg.task

        async with self._state.updating(self):
            predecessor = self._execution if self.task_running() else None
            running = TaskRun.create(
                str(self.entity),
                task,
                context=msg.context,
                expiry_time=msg.expiry_time,
            )

            response = ExecuteResponseMessage(execution=running.execution)
            handler = self._task_handlers.get(type(task))

            match self._state.enable_state.enabled, msg.interrupt or predecessor is None, handler:
                case False, _, _:
                    reason = "Controller is disabled"
                case _, False, _:
                    reason = "Task already in progress"
                case _, _, None:
                    reason = f"No handler registered for {task.task_type}"
                case _, _, handler:
                    if predecessor:
                        logger.info(f"Interrupting {predecessor.execution.task.task_type} task")

                    # Ownership is installed before acceptance becomes visible.
                    running.attach(handler, self._run_execution(running, predecessor))
                    self._execution = running
                    call.accept(response=response)

                    return running

        logger.warning(f"Rejecting {task.task_type} task: {reason}")
        call.reject(response=response, reason=reason)

        return None

    async def _execute_request(
        self,
        msg: ExecuteRequestMessage,
        call: CallContext[TaskExecutionResult, ExecuteResponseMessage],
    ):
        task = msg.task
        running = await self._admit(msg, call)

        if running is None:
            return

        execution = running.execution

        try:
            await call.progress_from_task(running.runner, cadence=6.0, ttl=10.0)
        except asyncio.CancelledError:
            with self.enter_context():
                logger.warning(f"Execution of {task.task_type} task cancelled")

            raise
        except Exception as e:
            with self.enter_context():
                logger.error(f"Error executing {task.task_type} task")
                logger.opt(exception=e).debug(f"{task.task_type} ({execution.task_id}) failed")
        else:
            with self.enter_context():
                logger.info(f"Finished {task.task_type} task")

            await call.succeed(result=running.runner.result())
        finally:
            # Join this request's execution even after a successor replaced it.
            if not running.done():
                running.cancel()

            try:
                await running.wait()
            finally:
                if self._execution is running:
                    self._execution = None

                # Clients treat the final event as proof the run ended, so it follows the join.
                if not call.finalized:
                    await defer_cancellation(self._fail_call(call, task))

    @staticmethod
    async def _fail_call(call: CallContext, task: Task):
        try:
            await call.fail()
        except Exception as e:
            logger.warning(f"Error reporting failed {task.task_type} task ({type(e).__name__})")

    async def _run_execution(
        self,
        running: TaskRun,
        predecessor: TaskRun | None,
    ) -> TaskExecutionResult:
        # Cancellation can target the runner once it has entered.
        running.entered = True

        with self.enter_context():
            try:
                if predecessor is not None:
                    predecessor.cancel("Interrupted")
                    await defer_cancellation(predecessor.join())
            finally:
                # Release the predecessor so repeated interruption cannot retain a chain.
                predecessor = None

            try:
                with running.execute() as execution:
                    logger.info(f"Executing {execution.task.task_type} task")
                    logger.debug(f"execution begun {execution=}")

                    await self._publish_start(running)
                    running.started = True
                    await running.handler(execution)
            finally:
                # Only an entered execution scope needs settlement.
                if running.start_time is not None:
                    await defer_cancellation(self._publish_finish(running))

            return running.result()

    async def _publish_start(self, running: TaskRun):
        execution = running.execution
        task = execution.task

        async with self._state.updating(self) as update:
            operating = self._state.operating_state

            # Execution context is the base for later device snapshots.
            events = [
                TaskExecutionState(executing=True, execution=execution, context=execution.context)
            ]

            match task:
                # Recovery invalidates the achieved state before hardware work.
                case RecoverTask():
                    events.append(
                        operating.derive(current=InternalControllerState.UNKNOWN, target=None)
                    )
                case _ if (target := task.target_state()) is not None:
                    events.append(
                        operating.derive(
                            current=InternalControllerState.TRANSITION,
                            target=target,
                            context=running.context,
                        )
                    )

            await update(*events)

    async def _publish_finish(self, running: TaskRun):
        execution = running.execution
        task = execution.task

        async with self._state.updating(self) as update:
            operating = self._state.operating_state
            published = self._state.execution_state.execution
            owned = published is not None and published.task_id == execution.task_id
            events = [TaskExecutionState(finished=running.finish_info, execution=execution)]

            match task:
                # Republish retained state when recovery start fails.
                case RecoverTask() if not running.started:
                    events.append(operating.derive())

                # Preserve state reported by a successful recovery handler.
                case RecoverTask():
                    if running.outcome is not None:
                        events.append(operating.derive(current=running.outcome))

                # Settle only a transition this execution started.
                case _ if (
                    task.target_state() is not None
                    and operating.current == InternalControllerState.TRANSITION
                    and owned
                ):
                    events.append(operating.derive(current=running.outcome, target=None))

            await update(*events)

    @override
    def add_task_handler[T: Task](
        self,
        task_model: type[T],
        *,
        task_callback: TaskHandlerCallback[T] | None = None,
        execution_callback: ExecutionHandlerCallback[T] | None = None,
    ) -> None:
        match task_callback, execution_callback:
            case None, None:
                raise TypeError("A task handler requires a task or execution callback")
            case _, None:
                handler = lambda execution: task_callback(execution.task)
            case None, _:
                handler = execution_callback
            case _:
                raise TypeError("A task handler takes one callback")

        if task_model in self._task_handlers:
            raise RuntimeError("multiple handlers for task type")

        self._task_handlers[task_model] = handler

    @override
    def task_handler[T: Task](self, task_model: type[T]):
        def decorator(func: TaskHandlerCallback[T]):
            self.add_task_handler(task_model, task_callback=func)
            return func

        return decorator

    @override
    def entity_info(self) -> EntityInfo:
        return EntityInfo(
            entity_type="controller",
            details=ControllerDetails(
                published_keywords=frozenset(self._published_keywords),
                supported_tasks=[t.__name__ for t in self._task_handlers.keys()],
                controlled_devices=list(self._devices.keys()),
            ),
        )

    @override
    async def set_internal_state(
        self,
        state: InternalControllerState,
        *,
        context: KeywordDict | None = None,
    ):
        if state == InternalControllerState.TRANSITION:
            raise ValueError("A transition requires a lifecycle task")

        match state, context:
            # Retain lifecycle input when reporting state loss.
            case InternalControllerState.UNKNOWN | InternalControllerState.ERROR, None:
                recorded = ControllerOperatingState.NO_CHANGE
            case _, None:
                recorded = KeywordDict()
            case _:
                recorded = copy.deepcopy(context)

        async with self._state.updating(self) as update:
            current = self._state.execution_state

            # A finished execution can leave a transition behind if publication failed.
            if current.executing and current.execution.task.target_state() is not None:
                raise ValueError("Cannot report state while a lifecycle task is running")

            await update(
                self._state.operating_state.derive(current=state, target=None, context=recorded)
            )
