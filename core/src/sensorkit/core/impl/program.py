# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
from _contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, ClassVar, override

from intervaltree import IntervalTree
from loguru import logger

from sensorkit.backend.base import Entity, KVError
from sensorkit.backend.event import Event
from sensorkit.backend.request import CallContext
from sensorkit.core.entity import EntityDetails, EntityInfo
from sensorkit.core.executor import TaskFactoryFunc, TaskingLoop, TaskOutcomeUnknown
from sensorkit.core.impl.entity import EntityImpl
from sensorkit.core.program import (
    OfferInterval,
    ProgramActiveState,
    ProgramActiveStateRequest,
    ProgramEnableState,
    ProgramEnableStateRequest,
    ProgramInterface,
    ProgramOffering,
    ProgramState,
    ProgramTaskingState,
    set_active_state_request,
    set_enable_state_request,
)
from sensorkit.core.task import TaskContextMap, TaskExecution


class ProgramOffers:
    """Manages an IntervalTree of offer windows and signals updates to waiters."""

    def __init__(self):
        self._tree = IntervalTree()
        self._updated = asyncio.Event()

    def wait(self):
        """Return an awaitable that resolves when the offer windows have been updated."""
        return self._updated.wait()

    def poll(self):
        """Return True and clear the update flag if a new update is pending, else False."""
        if self._updated.is_set():
            self._updated.clear()
            return True

        return False

    def get_offer_windows(self) -> list[OfferInterval]:
        """Return a sorted list of merged offer windows from the current interval tree."""
        # Create a tree that *references* the offer tree data and merge overlapping intervals. This
        # avoids an extra copy, and we know this is safe because `merge_overlaps` builds a new set
        # of intervals and does not modify `all_intervals`.
        assert_no_change = self._tree.all_intervals.copy()
        ref = IntervalTree(self._tree.all_intervals)
        ref.merge_overlaps()
        assert self._tree.all_intervals == assert_no_change
        return sorted(ref)

    def add(self, start: datetime, end: datetime, obj: Any = None):
        """Add an offer window to the interval tree and signal waiters."""
        self._tree.addi(
            begin=start,
            end=end,
            data=obj,
        )
        self._updated.set()

    def remove(self, start: datetime, end: datetime, obj: Any = None):
        """Remove an offer window from the interval tree, logging a warning if it does not exist."""
        try:
            self._tree.removei(
                begin=start,
                end=end,
                data=obj,
            )
            self._updated.set()
        except ValueError:
            logger.warning(f"Nonexistent offer window removed: {start} -> {end} {obj=}")

    def clear(self):
        if not self._tree.is_empty():
            self._tree.clear()
            self._updated.set()


class ProgramImpl(EntityImpl, ProgramInterface):
    """Helper for implementing server-side functionality of a Program."""

    current: ClassVar[ContextVar[ProgramImpl | None]] = ContextVar("current_program", default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store the user-supplied intervals that are to make up the offered operating windows.
        self._offers = ProgramOffers()

        self._enable_hooks: list[Callable[[], None]] = []
        self._disable_hooks: list[Callable[[], None]] = []
        self._state = ProgramState(
            enable_state=ProgramEnableState(enabled=False),
            active_state=ProgramActiveState(active=False, origin="init"),
            tasking_state=ProgramTaskingState(),
        )
        self._state_lock = asyncio.Lock()
        self._task_factory: TaskFactoryFunc | None = None
        self._task_loop: TaskingLoop | None = None
        self._task_finalizer: asyncio.Task | None = None
        self._control_lock = asyncio.Lock()

    @override
    async def init_impl(self):
        try:
            # Restore state.
            async with self._state_lock:
                self._state = await self.kv_get_model(ProgramState)
            logger.debug(f"restored program state: {self._state}")
        except KVError:
            logger.debug("initializing program state")
            await self.kv_put_model(self._state)

        if self._state.active_state.active:
            # Our last known state was tasking. We aren't going to automatically start tasking,
            # but we can emit an event indicating the previous tasking state stopped, giving
            # observers an opportunity to react themselves.
            logger.debug("ending lingering tasking active state")
            await self._update_state(
                "active_state",
                ProgramActiveState(active=False, origin="init"),
            )

    @override
    async def attach_impl(self):
        if self._state.enable_state.enabled:
            await self._call_with_context(self._enable_hooks)

        await self.handle_request(set_enable_state_request, self._set_enable_state)
        await self.handle_request(set_active_state_request, self._set_active_state)

    @override
    def on_enable(self, func: Callable[[], None]):
        self._enable_hooks.append(func)
        return func

    @override
    def on_disable(self, func: Callable[[], None]):
        self._disable_hooks.append(func)
        return func

    @override
    def task_factory(self, func: TaskFactoryFunc):
        self._task_factory = func
        return func

    async def _start_loop(self, contexts: TaskContextMap):
        logger.debug(f"starting tasking loop with {contexts=}")

        async def _on_task_change(execution: TaskExecution | None):
            # Publish the program's current tasking state as the loop starts/finishes each task.
            await self._update_state(
                "tasking_state", ProgramTaskingState(executing_task=execution)
            )

        # Create the tasking loop.
        self._task_loop = TaskingLoop(
            controller=self.sensorkit().controller(
                Entity.at(self._state.enable_state.controller)
            ),
            factory_func=self._task_factory,
            contexts=contexts,
            task_group=self.task_group,
            on_task_change=_on_task_change,
        )
        loop = self._task_loop

        # Create the finalizer before starting task production so even an immediate exit is handled.
        # Release it after publishing the active state to keep state events in order.
        queue = asyncio.Queue()

        async def _finalize_loop():
            aio_task = await queue.get()

            # Shutdown cancellation reaches the producer and starts abort cleanup.
            with contextlib.suppress(asyncio.CancelledError):
                await aio_task

            await loop.finalize()

            # Failure or abort may leave a published execution after task production ends.
            if self._state.tasking_state.executing_task is not None:
                await self._update_state("tasking_state", ProgramTaskingState())

            await self._update_state(
                "active_state",
                ProgramActiveState(active=False, origin="request")
                if loop.stop_requested
                else ProgramActiveState(active=False, origin="error"),
            )
            logger.info(f"Tasking loop exiting by {self._state.active_state.origin}")

            self._task_loop = None
            self._task_finalizer = None

        self._task_finalizer = self.task_group.create_task(_finalize_loop())

        # Publish the active state before allowing the finalizer to publish the inactive state.
        with self.enter_context():
            logger.info("Starting tasking loop")
            aio_task = self._task_loop.start()

        await self._update_state(
            "active_state",
            ProgramActiveState(active=True, origin="request", contexts=contexts),
        )

        # Tell the finalizer about the loop task.
        queue.put_nowait(aio_task)

    async def _stop_loop(self, loop: TaskingLoop | None, *, abort: bool = False):
        if loop is None or self._task_loop is not loop:
            return

        finalizer = self._task_finalizer
        loop.request_stop(abort=abort)
        active = self._state.active_state

        await self._update_state(
            "active_state",
            ProgramActiveState(
                active=True, origin=active.origin, stopping=True, contexts=active.contexts
            ),
            loop=loop,
        )

        # Wait until the finalizer has published the program's inactive state.
        await asyncio.shield(finalizer)

    @staticmethod
    def _require_ended(loop: TaskingLoop | None):
        """Raise if the loop could not confirm task acceptance or completion."""
        if loop is not None and loop.unresolved:
            raise TaskOutcomeUnknown(f"{loop.controller.entity} work may still be running")

    async def _set_enable_state(self, request: ProgramEnableStateRequest):
        async with self._control_lock:
            if not request.enable and not self._state.enable_state.enabled:
                return

            enablement_changed = request.enable ^ self._state.enable_state.enabled
            controller_changed = request.controller != self._state.enable_state.controller

            if not enablement_changed and not controller_changed:
                return

            if controller_changed or not request.enable:
                # Confirm the current task ended before changing controllers or disabling.
                loop = self._task_loop
                await self._stop_loop(loop, abort=True)
                self._require_ended(loop)

            logger.info(
                f"{'Enabling' if request.enable else 'Disabling'} "
                f"{self.entity} for target {request.controller}"
            )

            await self._update_state(
                "enable_state",
                ProgramEnableState(enabled=request.enable, controller=request.controller),
            )

            if request.enable:
                # Call the enable hook.
                await self._call_with_context(self._enable_hooks)
            else:
                # Call the disable hook.
                await self._call_with_context(self._disable_hooks)

    async def _set_active_state(
        self,
        request: ProgramActiveStateRequest,
        call: CallContext[None],
    ):
        with self.enter_context():
            logger.info(f"Requested to {request.action} the tasking loop")

        if request.action == "start":
            if self._control_lock.locked():
                call.reject(reason="Program enablement is changing")
                return

            async with self._control_lock:
                await self._start_tasking(request, call)

            return

        loop = self._task_loop
        call.accept()

        try:
            with self.enter_context():
                await call.progress_from_task(
                    self.task_group.create_task(
                        self._stop_loop(loop, abort=request.action == "abort")
                    ),
                    cadence=6,
                    ttl=10,
                )

            self._require_ended(loop)
        except Exception as e:
            with self.enter_context():
                logger.exception("Error setting active state")

            await call.fail(f"{type(e).__name__} setting active state ({e})")
        else:
            await call.succeed(result=None)

    async def _start_tasking(self, request: ProgramActiveStateRequest, call: CallContext[None]):
        loop = self._task_loop
        finalizer = self._task_finalizer

        if loop is not None and not loop.running:
            if not loop.finalized:
                call.reject(reason="Previous tasking is not finalized")
                return

            # Wait for the previous loop's inactive event before checking whether to start tasking.
            await asyncio.shield(finalizer)

        # Cannot activate tasking without an enabled target controller.
        if not self._state.enable_state.enabled or not self._state.enable_state.controller:
            call.reject(reason="Program is not enabled with a controller")
            return

        if (
            self._state.active_state.active
            and request.contexts != self._state.active_state.contexts
        ):
            call.reject(reason="Program is already active with different contexts")
            return

        call.accept()

        try:
            if not self._state.active_state.active:
                with self.enter_context():
                    await self._start_loop(request.contexts)
        except Exception as e:
            with self.enter_context():
                logger.exception("Error setting active state")

            await call.fail(f"{type(e).__name__} setting active state ({e})")
        else:
            await call.succeed(result=None)

    async def _update_state(self, key: str, event: Event, *, loop: TaskingLoop | None = None):
        async with self._state_lock:
            if loop is not None and self._task_loop is not loop:
                return

            setattr(self._state, key, event)
            await self.emit_event(event)
            await self.kv_put_model(self._state)

    @override
    def get_offers(self):
        return self._offers.get_offer_windows()

    @override
    async def publish_offers(self):
        """Publish the current set of offers if they have changed."""
        if self._offers.poll():
            await self.publish(
                ProgramOffering(offer_windows=self._offers.get_offer_windows())
            )

    @override
    def add_offer(self, start: datetime, end: datetime, obj: Any = None):
        self._offers.add(start, end, obj)

    @override
    def remove_offer(self, start: datetime, end: datetime, obj: Any = None):
        self._offers.remove(start, end, obj)

    @override
    def clear_offers(self):
        self._offers.clear()

    @override
    def entity_info(self) -> EntityInfo:
        return EntityInfo(
            entity_type="program",
            details=EntityDetails(
                published_keywords=frozenset(self._published_keywords),
            ),
        )
