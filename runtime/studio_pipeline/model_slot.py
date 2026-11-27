"""One cross-process inference lease shared by the renderer and local readers."""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import gc
import os
from pathlib import Path
from functools import wraps
import sys
import time

RUNTIME=Path(__file__).resolve().parents[1]
_owner=ContextVar('studio_model_lease',default=None)
before_auxiliary=ContextVar('studio_unload_reader',default=None)


@contextmanager
def slot():
    if _owner.get()==os.getpid():
        yield 0.
        return
    with (RUNTIME/'vision-model-lock').open('a') as lock:
        started=time.monotonic()
        fcntl.flock(lock,fcntl.LOCK_EX)
        token=_owner.set(os.getpid())
        try:yield time.monotonic()-started
        finally:_owner.reset(token)


def release_memory():
    gc.collect()
    torch=sys.modules.get('torch')
    if torch and torch.backends.mps.is_available():torch.mps.empty_cache()
    if torch and torch.cuda.is_available():torch.cuda.empty_cache()


def auxiliary(fn):
    @wraps(fn)
    def call(*args,**kwargs):
        with slot():
            unload=before_auxiliary.get()
            if unload:unload()
            try:return fn(*args,**kwargs)
            finally:release_memory()
    return call
