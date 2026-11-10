"""Local background import jobs; Redis tracks status and per-dataset leases."""
from __future__ import annotations

import asyncio
import json
import uuid

from src.agents.knowledge.public_datasets import ExistingDataError, import_dataset
from src.infra.redis_cache import get_redis_client

LEASE_SECONDS = 90
RECORD_SECONDS = 7 * 24 * 3600
PREFIX = "medflow:public-data:"
_RUNNING: set[asyncio.Task] = set()

_CREATE = """-- create_public_data_job
if not redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then return 0 end
redis.call('HSET', KEYS[2], 'task_id', ARGV[1], 'dataset', ARGV[3], 'status', 'pending', 'stage', 'queued', 'completed', '0', 'total', '0')
redis.call('EXPIRE', KEYS[2], ARGV[4])
return 1
"""
_RENEW = """-- renew_public_data_job
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""
_UPDATE = """-- update_public_data_job
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[2], unpack(ARGV, 2))
return 1
"""
_RELEASE = """-- release_public_data_job
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
_GET = """-- get_public_data_job
if redis.call('EXISTS', KEYS[1]) == 0 then return {} end
local dataset = redis.call('HGET', KEYS[1], 'dataset')
local status = redis.call('HGET', KEYS[1], 'status')
local task_id = redis.call('HGET', KEYS[1], 'task_id')
if (status == 'pending' or status == 'running') and redis.call('GET', ARGV[1] .. dataset) ~= task_id then
    redis.call('HSET', KEYS[1], 'status', 'interrupted', 'stage', 'interrupted', 'error', 'Worker stopped or lease expired; inspect partial data before retrying')
end
return redis.call('HGETALL', KEYS[1])
"""


def _lock(dataset: str) -> str:
    return f"{PREFIX}lock:{dataset}"


def _job(task_id: str) -> str:
    return f"{PREFIX}job:{task_id}"


class JobAlreadyRunning(Exception):
    pass


class LeaseLost(Exception):
    pass


async def submit(dataset: str) -> str:
    if dataset not in {"cmirb", "dialogue", "medqa"}:
        raise ValueError("Unknown dataset")
    redis = await get_redis_client()
    task_id = uuid.uuid4().hex
    created = await redis.eval(_CREATE, 2, _lock(dataset), _job(task_id), task_id,
                               LEASE_SECONDS, dataset, RECORD_SECONDS)
    if not created:
        raise JobAlreadyRunning(dataset)
    try:
        task = asyncio.create_task(_run(redis, dataset, task_id))
        _RUNNING.add(task)
        task.add_done_callback(_RUNNING.discard)
    except Exception:
        await redis.eval(_RELEASE, 1, _lock(dataset), task_id)
        raise
    return task_id


async def get_job(task_id: str) -> dict | None:
    redis = await get_redis_client()
    raw = await redis.eval(_GET, 1, _job(task_id), PREFIX + "lock:")
    if not raw:
        return None
    data = dict(zip(raw[::2], raw[1::2]))
    for field in ("completed", "total"):
        data[field] = int(data.get(field, 0))
    if "result" in data:
        data["result"] = json.loads(data["result"])
    return data


async def _run(redis, dataset: str, task_id: str) -> None:
    lock, job = _lock(dataset), _job(task_id)
    lost = asyncio.Event()

    async def update(**fields):
        args = [task_id]
        for key, value in fields.items():
            args.extend([key, str(value)])
        if not await redis.eval(_UPDATE, 2, lock, job, *args):
            lost.set()
            raise LeaseLost()

    async def check_owner():
        if lost.is_set() or await redis.get(lock) != task_id:
            lost.set()
            raise LeaseLost()

    async def report(stage, completed, total):
        await update(status="running", stage=stage, completed=completed, total=total)

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(LEASE_SECONDS / 3)
                if not await redis.eval(_RENEW, 1, lock, task_id, LEASE_SECONDS):
                    lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            lost.set()

    beat = asyncio.create_task(heartbeat())
    try:
        await update(status="running", stage="starting")
        result = await import_dataset(dataset, report, check_owner)
        await check_owner()
        await update(status="completed", stage="completed", result=json.dumps(result, ensure_ascii=False))
    except ExistingDataError:
        await update(status="failed", stage="failed", error="Existing data found; no overwrite performed")
    except LeaseLost:
        # Polling reports interruption once the expired lease is observed.
        pass
    except asyncio.CancelledError:
        try:
            await update(status="interrupted", stage="interrupted", error="Worker stopped; inspect partial data before retrying")
        finally:
            raise
    except Exception as exc:
        try:
            await update(status="failed", stage="failed", error=f"Import failed ({type(exc).__name__}); inspect partial data before retrying")
        except LeaseLost:
            pass
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        try:
            await redis.eval(_RELEASE, 1, lock, task_id)
        except Exception:
            pass