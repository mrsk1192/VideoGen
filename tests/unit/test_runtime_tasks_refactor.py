import pytest

import main

pytestmark = pytest.mark.unit


def test_detect_runtime_exposes_dtype_and_env() -> None:
    runtime = main.detect_runtime()
    assert "rocm_aotriton_env" in runtime
    assert "require_gpu" in runtime
    assert "allow_cpu_fallback" in runtime
    assert "preferred_dtype" in runtime


def test_task_manager_cancel_flow() -> None:
    task_id = main.create_task("download", "queued")
    main.update_task(task_id, status="running", progress=0.2, message="working")
    cancelled = main.TASK_MANAGER.request_cancel(task_id)
    assert cancelled["cancel_requested"] is True
    with pytest.raises(main.TaskCancelledError):
        main.ensure_task_not_cancelled(task_id)
    task = main.get_task(task_id)
    assert task is not None
    assert task["status"] == "cancelled"

