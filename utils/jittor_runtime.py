import gc
import os
from contextlib import contextmanager

import jittor as jt


def configure_jittor_runtime():
    jt.flags.use_cuda = 1
    if hasattr(jt, "cudnn"):
        try:
            workspace_ratio = float(os.environ.get("JT_CUDNN_WORKSPACE_RATIO", "0.5"))
            jt.cudnn.set_max_workspace_ratio(workspace_ratio)
        except Exception:
            pass


def sync_gc():
    try:
        jt.sync_all(True)
    except TypeError:
        jt.sync_all()
    jt.gc()
    gc.collect()


def print_runtime_hints():
    print(
        "Jittor memory config: "
        f"JT_SAVE_MEM={os.environ.get('JT_SAVE_MEM')}, "
        f"cpu_mem_limit={os.environ.get('cpu_mem_limit')}, "
        f"device_mem_limit={os.environ.get('device_mem_limit')}, "
        f"JT_CUDNN_WORKSPACE_RATIO={os.environ.get('JT_CUDNN_WORKSPACE_RATIO', '0.5')}"
    )


@contextmanager
def optional_memory_profile(enabled=False):
    if enabled:
        with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
            yield
    else:
        yield


def maybe_print_memory_profile(enabled=False):
    if enabled:
        jt.get_max_memory_treemap()
