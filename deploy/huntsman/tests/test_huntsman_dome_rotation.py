"""Dome rotation policy client against an in-process fake dome server (gRPC, no domepi).

Run from /opt/sensorkit:
    .venv/bin/python -m pytest -p no:cacheprovider deploy/huntsman/tests -q
"""

from __future__ import annotations

import asyncio
import sys

import pytest
import pytest_asyncio

grpc = pytest.importorskip("grpc")

hd = sys.modules["huntsman_dome_under_test"]  # loaded once by conftest.py
pb, pb_grpc = hd._load_hx2dome_stubs()


class FakeDomeServer(pb_grpc.HX2DomeServicer):
    """Answers the rotation policy calls like the domepi dome server does."""

    def __init__(self):
        self.policy = pb.RotationPolicy(allowed_direction="both", long_way_settle_tolerance=5)
        self.moving = False
        self.sets = 0

    async def dapiGetRotationPolicy(self, request, context):
        reply = pb.RotationPolicy()
        reply.CopyFrom(self.policy)
        return reply

    async def dapiSetRotationPolicy(self, request, context):
        self.sets += 1
        if request.allowed_direction not in ("both", "CW"):
            return pb.ReturnCode(return_code=2, error_message="bad allowed_direction")
        if self.moving:
            return pb.ReturnCode(return_code=1)
        self.policy.CopyFrom(request)
        return pb.ReturnCode(return_code=0)

    def reboot(self):
        """A restart reverts the server to its config file (shortest route)."""
        self.policy = pb.RotationPolicy(allowed_direction="both", long_way_settle_tolerance=5)


@pytest_asyncio.fixture
async def server():
    fake = FakeDomeServer()
    srv = grpc.aio.server()
    pb_grpc.add_HX2DomeServicer_to_server(fake, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    await srv.start()
    fake.address = f"127.0.0.1:{port}"
    yield fake
    await srv.stop(0)


def client_for(address: str, **policy) -> hd._RotationPolicyClient:
    config = hd.HuntsmanDomeConfig(
        dome_server=address,
        rotation_policy=hd.RotationPolicyConfig(**policy) if policy else None,
        rotation_policy_check_interval=0.05,
    )
    return hd._RotationPolicyClient(config)


@pytest.mark.asyncio
async def test_not_configured_never_contacts_server(server):
    client = client_for(server.address)
    client.start()
    await asyncio.sleep(0.2)
    await client.stop()
    assert client.state == "not configured"
    assert server.sets == 0


@pytest.mark.asyncio
async def test_pushes_configured_policy(server):
    client = client_for(server.address, allowed_direction="CW", max_reverse_degrees=5)
    await client.sync()
    assert client.state == "CW"
    assert server.policy.allowed_direction == "CW"
    assert server.policy.max_reverse_degrees == 5
    assert server.policy.long_way_settle_seconds == 60
    await client.stop()


@pytest.mark.asyncio
async def test_no_push_when_server_already_matches(server):
    client = client_for(server.address, allowed_direction="CW")
    await client.sync()
    await client.sync()
    assert server.sets == 1
    await client.stop()


@pytest.mark.asyncio
async def test_repushes_after_server_reboot(server):
    client = client_for(server.address, allowed_direction="CW")
    client.start()
    await asyncio.sleep(0.15)
    assert server.policy.allowed_direction == "CW"
    server.reboot()
    await asyncio.sleep(0.15)
    assert server.policy.allowed_direction == "CW"
    assert server.sets >= 2
    await client.stop()


@pytest.mark.asyncio
async def test_refused_while_moving_then_retried(server):
    server.moving = True
    client = client_for(server.address, allowed_direction="CW")
    client.start()
    await asyncio.sleep(0.15)
    assert client.state == "both"  # what is actually in effect
    server.moving = False
    await asyncio.sleep(0.15)
    assert client.state == "CW"
    await client.stop()


@pytest.mark.asyncio
async def test_switching_back_to_both(server):
    await client_for(server.address, allowed_direction="CW").sync()
    client = client_for(server.address, allowed_direction="both")
    await client.sync()
    assert client.state == "both"
    assert server.policy.allowed_direction == "both"
    await client.stop()


@pytest.mark.asyncio
async def test_unreachable_server_reports_unknown_without_raising():
    client = client_for("127.0.0.1:9", allowed_direction="CW")
    client.RPC_TIMEOUT = 0.3
    client.start()
    await asyncio.sleep(0.6)
    assert client.state == "unknown"
    await client.stop()


def test_config_rejects_bad_direction():
    with pytest.raises(ValueError):
        hd.RotationPolicyConfig(allowed_direction="CCW")
    with pytest.raises(ValueError):
        hd.RotationPolicyConfig(allowed_direction="CW", max_reverse_degrees=200)


def test_config_section_without_rotation_policy_is_unchanged():
    config = hd.HuntsmanDomeConfig()
    assert config.rotation_policy is None
    assert hd._RotationPolicyClient(config).state == "not configured"
