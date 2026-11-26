"""Offline regression checks for retaining or escalating a structural repair."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/studio_pipeline'), str(ROOT / 'extensions')]
from construction_review import LARGE, assess_local_edit
from masks import structural_composite
from print_judge import issue_category
from production import repeated_rejection


class LocalRepairTests(unittest.TestCase):
    def test_repair_preserves_a_new_contour_and_the_crop_boundary(self):
        original = Image.new('RGB', (180, 128), 'beige')
        proposal = original.copy()
        ImageDraw.Draw(proposal).line((58, 63, 119, 79), fill='navy', width=4)
        mask = Image.new('L', original.size)
        ImageDraw.Draw(mask).rectangle((47, 44, 127, 92), fill=255)
        result = structural_composite(original, proposal, mask, (False,) * 4)
        actual, expected = result.getpixel((88, 71)), proposal.getpixel((88, 71))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 2)
        diff = ImageChops.difference(original, result)
        self.assertIsNone(diff.crop((0, 0, 180, 1)).getbbox())
        self.assertIsNone(diff.crop((179, 0, 180, 128)).getbbox())

    def test_unrelated_or_accepted_edits_do_not_trigger_escalation(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            def receipt(index, box, verdict):
                (folder / f'{index}-patch-review.json').write_text(json.dumps(
                    {'target_box': box, 'verdict': verdict}))
            target = [200, 300, 315, 390]
            receipt(1, target, 'still_present')
            receipt(2, [400, 520, 625, 880], 'uncertain')
            receipt(3, target, 'resolved')
            self.assertFalse(repeated_rejection(folder, target))
            receipt(4, [201, 305, 318, 398], 'new_defect')
            self.assertTrue(repeated_rejection(folder, target))

    def test_conflicting_readings_need_a_separate_positive_and_negative_control(self):
        source = Image.new('RGB', (144, 144))
        candidate = source.copy()
        ImageDraw.Draw(candidate).line((40, 50, 100, 84), fill='white', width=5)
        for source_finding, candidate_finding, expected in [
                ('defective', 'coherent', 'resolved'), ('coherent', 'coherent', 'uncertain'),
                ('defective', 'ambiguous', 'uncertain'), ('ambiguous', 'coherent', 'uncertain')]:
            def read(url, images, prompt, schema, *, tag, **kwargs):
                if tag == LARGE:
                    finding = candidate_finding if images[0].getbbox() else source_finding
                    return json.dumps({'trace': 'Visible joint boundary.', 'construction': finding,
                                       'unrelated_damage': False})
                if 'better' in schema['properties']:
                    return json.dumps({'better': 'equal', 'evidence': 'Overlapping edges.'})
                return json.dumps({'verdict': 'resolved', 'evidence': 'Continuous contour.', 'next_edit': ''})
            with patch('construction_review.ask', side_effect=read), patch(
                    'construction_review.backend', return_value=nullcontext(None)), patch(
                    'construction_review.available', return_value=True):
                verdict = assess_local_edit(source, candidate, [0, 0, 1000, 1000], 'Joint is attached.')
            self.assertEqual(verdict['verdict'], expected)

    def test_ambiguous_geometry_does_not_request_a_resolution_rebuild(self):
        image = Image.new('RGB', (32, 32))
        with patch('print_judge.ask', side_effect=AssertionError('No optical symptom')):
            self.assertEqual(issue_category(None, image, 'The knot merges into the ring.', 'Woodcut'), 'other')
        read = {'issue': 'unresolved_focus', 'optical_loss_visible': True,
                'geometric_fault': False, 'pixel_trace': 'Fine boundaries are smeared.'}
        with patch('print_judge.ask', return_value=json.dumps(read)):
            self.assertEqual(issue_category(None, image, 'The edges appear smeared.', 'Woodcut'),
                             'unresolved_focus')


if __name__ == '__main__':
    unittest.main()
