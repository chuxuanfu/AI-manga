"""Admit one generation round at a time, in arrival order, across Studio jobs."""
from contextlib import contextmanager
import fcntl
import json
import os
import time


def first_live_ticket(queue, own):
    # The caller holds the guard during listing, liveness checks and removal.
    for prior in sorted(queue.glob('*.ticket')):
        if prior == own:
            return True
        with prior.open() as holder:
            try:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            prior.unlink()
    raise RuntimeError('The active job lost its queue ticket.')


@contextmanager
def round_slot(runtime, job):
    queue = runtime / 'round-queue'
    queue.mkdir(exist_ok=True)
    with (queue / 'guard').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        # Allocate under the guard so simultaneous arrivals and clock adjustments
        # cannot cause a newer job to jump ahead.
        positions = sorted(queue.glob('*.ticket'))
        index = int(positions[-1].stem) + 1 if positions else 1
        path = queue / f'{index:020d}.ticket'
        with path.open('x+') as ticket:
            try:
                fcntl.flock(ticket, fcntl.LOCK_EX)
                json.dump({'job': str(job), 'pid': os.getpid(), 'queued_at': time.time()}, ticket)
                ticket.flush()
                started = time.monotonic()
                while True:
                    if first_live_ticket(queue, path):
                        break
                    fcntl.flock(guard, fcntl.LOCK_UN)
                    time.sleep(0.5)
                    fcntl.flock(guard, fcntl.LOCK_EX)
                fcntl.flock(guard, fcntl.LOCK_UN)
                print(f'开始本地生成，等待调度 {time.monotonic()-started:.1f} 秒', flush=True)
                yield
            finally:
                fcntl.flock(guard, fcntl.LOCK_EX)
                path.unlink(missing_ok=True)
