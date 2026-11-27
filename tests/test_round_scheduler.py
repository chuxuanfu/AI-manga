import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

PIPELINE = Path(__file__).resolve().parents[1] / 'runtime/studio_pipeline'
CHILD = """
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
from scheduler import round_slot
folder = Path(sys.argv[2])
name = sys.argv[3]
with round_slot(folder, name):
    with (folder / 'admissions').open('a') as stream:
        print(name, file=stream)
    while not (folder / (name + '.finish')).exists():
        time.sleep(.05)
"""


class RoundSchedulerTests(unittest.TestCase):
    def wait_for(self, condition):
        deadline = time.monotonic()+5
        while not condition() and time.monotonic()<deadline:
            time.sleep(.03)
        self.assertTrue(condition())

    def test_fifo_and_abandoned_active_and_waiting_tickets(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            workers = {}
            def tickets():
                return [json.loads(path.read_text())['job'] for path in
                        sorted((folder/'round-queue').glob('*.ticket')) if path.stat().st_size]
            def admitted():
                record = folder/'admissions'
                return record.read_text().splitlines() if record.exists() else []
            try:
                for name in ('current','cancelled','next','last'):
                    workers[name] = subprocess.Popen(
                        [sys.executable,'-c',CHILD,str(PIPELINE),temporary,name],
                        stdout=subprocess.DEVNULL)
                    self.wait_for(lambda: name in tickets())
                self.assertEqual(admitted(), ['current'])
                for name in ('cancelled','current'):
                    workers[name].kill()
                    workers[name].wait(timeout=3)
                self.wait_for(lambda: admitted()==['current','next'])
                (folder/'next.finish').touch()
                self.wait_for(lambda: admitted()==['current','next','last'])
                (folder/'last.finish').touch()
                workers['last'].wait(timeout=3)
                self.assertFalse(tickets())
            finally:
                for worker in workers.values():
                    if worker.poll() is None:
                        worker.kill()
                    worker.wait(timeout=3)
