import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_readings as audit


class DiagnosticTests(unittest.TestCase):
    def test_independent_quotes_and_positions_survive_a_review_decision(self):
        replies = ["smooth", "silver", "wood", "mesh", "paint", "glass"]
        for a, b in zip(replies[:3], replies[3:6]):
            replies.append(json.dumps({"parts": [
                {"part": "surface", "a": a, "b": b, "comparison": "incompatible"},
                {"part": "invented", "a": "no such phrase", "b": b, "comparison": "incompatible"},
            ]}))
        replies.extend([json.dumps({"indices": [0, 2]}),
                        json.dumps({"decision": "resolved", "reason": "visible join"}),
                        json.dumps({"decision": "ambiguous", "reason": "two readings"})])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            Image.new("RGB", (160, 100)).save(path)
            with patch.object(audit, "objects", return_value=[{"name": "fixture", "box": [100,100,900,900]}]), \
                 patch.object(audit, "model_identity", return_value="fixture"), \
                 patch.object(audit, "review_session", return_value=nullcontext()), \
                 patch.object(audit, "backend", return_value=nullcontext("loopback")), \
                 patch.object(audit, "ask", side_effect=replies):
                report = audit.audit(path)
        self.assertEqual([pair["a"] for pair in report["comparisons"]], ["smooth", "silver", "wood"])
        self.assertEqual([item["pair"] for item in report["checks"]], [0, 2])
        self.assertEqual(report["readings"][-1]["pixels"], (99, 10, 144, 90))
        self.assertEqual(report["status"], "diagnostic_complete")
