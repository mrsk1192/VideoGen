import copy
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TaskCancelledError(RuntimeError):
    pass


class TaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, task_type: str, message: str = "Queued") -> str:
        task_id = str(uuid.uuid4())
        now = utc_now()
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "task_type": task_type,
                "status": "queued",
                "step": "queued",
                "progress": 0.0,
                "message": message,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "downloaded_bytes": None,
                "total_bytes": None,
                "cancel_requested": False,
            }
        return task_id

    def update(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            if task_id not in self._tasks:
                return
            if self._tasks[task_id].get("status") == "cancelled" and "status" not in updates:
                return
            self._tasks[task_id].update(updates)
            status = str(self._tasks[task_id].get("status") or "")
            if status == "running" and not self._tasks[task_id].get("started_at"):
                self._tasks[task_id]["started_at"] = utc_now()
            if status in ("completed", "error", "cancelled"):
                self._tasks[task_id]["finished_at"] = self._tasks[task_id].get("finished_at") or utc_now()
            self._tasks[task_id]["updated_at"] = utc_now()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            return copy.deepcopy(task) if task else None

    def list(self, task_type: str = "", status: str = "all", limit: int = 30) -> list[Dict[str, Any]]:
        normalized_task_type = str(task_type or "").strip().lower()
        normalized_status = str(status or "all").strip().lower()
        capped_limit = min(max(int(limit), 1), 200)
        with self._lock:
            values = [copy.deepcopy(task) for task in self._tasks.values()]
        filtered: list[Dict[str, Any]] = []
        for task in values:
            if normalized_task_type and str(task.get("task_type") or "").strip().lower() != normalized_task_type:
                continue
            if normalized_status != "all" and str(task.get("status") or "").strip().lower() != normalized_status:
                continue
            filtered.append(task)
        filtered.sort(key=lambda task: str(task.get("updated_at") or task.get("created_at") or ""), reverse=True)
        return filtered[:capped_limit]

    def request_cancel(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(task_id)
            status = str(task.get("status") or "")
            if status in {"completed", "error", "cancelled"}:
                return copy.deepcopy(task)
            task["cancel_requested"] = True
            task["step"] = "cancel_requested"
            task["message"] = "Cancellation requested"
            task["updated_at"] = utc_now()
            return copy.deepcopy(task)

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            return bool(task.get("cancel_requested"))

    def check_cancelled(self, task_id: str) -> None:
        if self.is_cancel_requested(task_id):
            self.update(task_id, status="cancelled", progress=1.0, message="Cancelled")
            raise TaskCancelledError("Task was cancelled by user")


@contextmanager
def task_progress_heartbeat(
    manager: TaskManager,
    task_id: str,
    start_progress: float,
    end_progress: float,
    message: str,
    *,
    interval_sec: float = 0.5,
    estimated_duration_sec: float = 20.0,
) -> Any:
    start = max(0.0, min(1.0, float(start_progress)))
    end = max(start, min(1.0, float(end_progress)))
    span = max(0.0, end - start)
    if span <= 0.0:
        yield
        return

    stop_event = threading.Event()
    state = {"last_progress": start}

    def worker() -> None:
        started_at = time.perf_counter()
        expected = max(0.5, float(estimated_duration_sec))
        while not stop_event.wait(max(0.2, float(interval_sec))):
            if manager.is_cancel_requested(task_id):
                continue
            elapsed = time.perf_counter() - started_at
            ratio = min(elapsed / expected, 0.985)
            progress = start + span * ratio
            if progress <= state["last_progress"] + 1e-6:
                continue
            state["last_progress"] = progress
            manager.update(task_id, progress=progress, message=message)

    heartbeat = threading.Thread(target=worker, name=f"task-hb-{task_id[:8]}", daemon=True)
    heartbeat.start()
    try:
        yield
    finally:
        stop_event.set()
        heartbeat.join(timeout=2.0)
