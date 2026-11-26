from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime/studio_pipeline'),str(ROOT/'extensions')]
from print_judge import store


class ParallelReceiptsTests(unittest.TestCase):
    def test_concurrent_writers_replace_only_fully_written_receipts(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            def write(i):
                store(path,{'index':i,'stamp':str(i)*4096})
                row=json.loads(path.read_text())
                self.assertEqual(row['stamp'],str(row['index'])*4096)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write,range(80)))
            self.assertEqual(list(path.parent.iterdir()),[path])
