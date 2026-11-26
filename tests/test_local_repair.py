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
from construction_review import LARGE, assess_local_edit, verify_interior_figure
from illustration import instruction_edit, reference_draft
from masks import structural_composite
from print_judge import actionable_edit, crop_brief, inspect_repaired_region, issue_category, verify_depicted_connection, verify_part_identity
from page_layout import retry_direction
from production import repeated_rejection, same_repair_region
from review import inspection_key
from structure import inspect_small_figure_support


class LocalRepairTests(unittest.TestCase):
    def test_occluded_figure_requires_a_traced_edge_and_the_requested_pose(self):
        box = [610, 290, 655, 415]
        for outcome, connection, pose, expected in [('lower_body_occluded', True, True, True),
                ('lower_body_occluded', True, False, False), ('physically_supported', False, True, False),
                ('unresolved', True, True, False), ('broken', True, True, False)]:
            reading = {'trace': 'Torso continues to the upper lip of the hull.', 'outcome': outcome,
                       'clear_visible_attachment': connection, 'requested_pose_visible': pose}
            with patch('structure.ask', return_value=json.dumps(reading)):
                result = inspect_small_figure_support(None, Image.new('RGB', (960, 540)), box, 'Sailor on deck')
            self.assertEqual(result['confirmed'], expected)

    def test_reuse_of_composition_review_binds_the_brief_focus_and_models(self):
        args = ('rules', 'A bird', [['bird', 1]], [10, 20, 50, 60], 'ref', ['reader-a'])
        baseline = inspection_key(*args)
        self.assertEqual(baseline, inspection_key(*args))
        for index, alternative in enumerate(('new rules', 'Two birds', [['bird', 2]], None, 'other ref', ['reader-b'])):
            changed = list(args)
            changed[index] = alternative
            self.assertNotEqual(baseline, inspection_key(*changed))

    def test_full_scene_repair_sees_the_native_detail_but_uses_the_full_canvas(self):
        graph = instruction_edit('Repair this connection', 77, 'whole.png', 'trial',
                                 detail_reference='detail.png').api
        encoder = next(n for n in graph.values() if n['class_type'] == 'TextEncodeQwenImageEditPlus')
        latent = next(n for n in graph.values() if n['class_type'] == 'VAEEncode')
        full = encoder['inputs']['image1'][0]
        detail = encoder['inputs']['image2'][0]
        self.assertEqual(graph[full]['inputs']['image'], 'whole.png')
        self.assertEqual(graph[detail]['inputs']['image'], 'detail.png')
        self.assertEqual(latent['inputs']['pixels'][0], full)

    def test_structural_edit_sampling_is_anchored_to_the_source_and_mask(self):
        editors = (reference_draft('Fix a seam', 77, 960, 960, 'trial', reference='guide.png',
                                   edit=True, masked_source='original-mask.png', preserve_layout=True),
                   instruction_edit('Fix a seam', 77, 'original-mask.png', 'trial', masked=True))
        for graph in editors:
            nodes = graph.api
            sampler = next(n for n in nodes.values() if n['class_type'] in ('KSampler', 'SamplerCustomAdvanced'))
            mask = nodes[sampler['inputs']['latent_image'][0]]
            self.assertEqual(mask['class_type'], 'SetLatentNoiseMask')
            source, channel = mask['inputs']['mask']
            self.assertEqual(channel, 1)
            self.assertEqual(nodes[source]['inputs']['image'], 'original-mask.png')
            original = nodes[mask['inputs']['samples'][0]]
            self.assertEqual(original['class_type'], 'VAEEncode')
            self.assertEqual(original['inputs']['pixels'], [source, 0])
        sampler = next(n for n in editors[0].api.values() if n['class_type'] == 'SamplerCustomAdvanced')
        split_id, selected_output = sampler['inputs']['sigmas']
        split = editors[0].api[split_id]
        self.assertEqual((split['class_type'], selected_output), ('SplitSigmas', 1))
        schedule = editors[0].api[split['inputs']['sigmas'][0]]
        self.assertEqual(schedule['inputs']['steps'] - split['inputs']['step'], 4)
        normal = reference_draft('A scene', 77, 960, 960, 'trial')
        self.assertTrue(any(n['class_type'] == 'EmptyFlux2LatentImage' for n in normal.api.values()))

    def test_review_reads_current_pixels_before_receiving_the_past_diagnosis(self):
        image = Image.new('RGB', (64, 64))
        diagnosis = 'Historic blurred pattern on the plate.'
        inventory = {'visible': 'Ceramic surface', 'condition': 'Even finish.'}
        for state, expected in [('resolved_in_view', 'clear'), ('present', 'uncertain'), ('indeterminate', 'uncertain')]:
            answer = {'trace': 'Uniform plate surface.', 'defect_now': state}
            with patch('print_judge.ask', side_effect=[json.dumps(inventory), json.dumps(answer)]) as ask, patch(
                    'print_judge.judge_view', return_value={'decision': 'clear', 'evidence': 'Even surface.'}):
                row, record = inspect_repaired_region(None, image, diagnosis, 'Ceramic still life')
            self.assertNotIn(diagnosis, ask.call_args_list[0].args[2])
            self.assertIn(diagnosis, ask.call_args_list[1].args[2])
            self.assertEqual(row['decision'], expected)
            self.assertEqual(record['inventory'], inventory)

    def test_hidden_hardware_check_preserves_broken_or_explicitly_required_connections(self):
        field = Image.new('RGB', (400, 300))
        claim = 'The socket has no visible bolt.'
        for condition, basis, expected in [
                ('visible_break', None, False), ('unresolved', None, False),
                ('visible_coherent_contact', 'visible_fault', False),
                ('visible_coherent_contact', 'required_visible_detail_missing', False),
                ('visible_coherent_contact', 'only_unseen_internal_hardware', True)]:
            values = [json.dumps({'condition': condition, 'trace': 'The edges meet at the collar.'})]
            if basis:values.append(json.dumps({'basis': basis, 'trace': 'Crossing traced in the final crop.'}))
            with patch('print_judge.ask', side_effect=values), patch('print_judge.model_identity', return_value='installed'):
                audit = verify_depicted_connection(None, field, [200, 350, 370, 530], claim, 'An assembled mechanism')
            self.assertEqual(audit['disproved'], expected)
        with patch('print_judge.ask', side_effect=AssertionError('Anatomy needs a direct review')):
            self.assertIsNone(verify_depicted_connection(None, field, [5, 5, 80, 80],
                                                        'Two fingers are fused together.', 'A hand holding bread'))

    def test_mistaken_part_requires_coherent_geometry_and_two_readers(self):
        image = Image.new('RGB', (512, 288))
        finding = {'operation': 'redraw', 'target': 'paw', 'symptom': 'This outline belongs to a paw.'}
        identified = {'feature': 'scroll', 'material': 'metal', 'attachment': 'armrest',
                      'image_evidence': 'Continuous polished metal around the curve.'}
        for outcome, intact, expected in [('no', True, True), ('yes', True, False),
                                          ('no', False, False), ('unresolved', True, False)]:
            audit = {'is_claimed_part': outcome, 'blind_identification_visible': True, 'intact_contour': intact, 'trace': 'Continuous metal.'}
            with patch('print_judge.model_identity', return_value='installed'), patch(
                    'print_judge.ask', side_effect=[json.dumps(identified), json.dumps(audit)]) as ask:
                result = verify_part_identity(None, image, [170, 280, 230, 350], finding)
            self.assertEqual(result['disproved'], expected)
            self.assertNotEqual(ask.call_args_list[0].kwargs['tag'], ask.call_args_list[1].kwargs['tag'])
            self.assertNotIn(finding['symptom'], ask.call_args_list[0].args[2])

    def test_identifying_a_visible_neighbor_cannot_disprove_a_missing_required_subject(self):
        image = Image.new('RGB', (512, 288))
        finding = {'operation': 'redraw', 'target': 'leg', 'symptom': 'A limb is confused with the railing.'}
        with patch('print_judge.ask', side_effect=AssertionError('Needs composition/framing review')):
            result = verify_part_identity(None, image, [100, 810, 200, 999], finding,
                                          'The required feet are missing from the frame.')
        self.assertIsNone(result)

    def test_retry_brief_cannot_copy_objects_from_a_discarded_composition(self):
        for note, unexpected in [('The strands around the unrequested cleat have tangled crossings.', 'cleat'),
                                 ('An extra red suitcase appears.', 'suitcase'),
                                 ('多余的蓝色雨伞遮挡了河岸。', '雨伞')]:
            direction = retry_direction(note)
            self.assertTrue(direction)
            self.assertNotIn(unexpected, direction)
        self.assertEqual(retry_direction(''), '')
        self.assertEqual(retry_direction('An unfamiliar artifact in the picture.'), '')

    def test_native_detail_review_keeps_the_authors_material_and_color_directions(self):
        for brief in ('A red boat in an indigo woodblock print with crisp carved contours.',
                      'A plain cotton scarf in warm reflected light. · Japanese anime key visual.',
                      '青花瓷碗，釉下钴蓝纹样，有微小开片。'):
            with self.subTest(brief=brief):
                self.assertIn(brief, crop_brief(brief))
                self.assertIn('only on the full composition', crop_brief(brief))

    def test_negative_edit_instructions_cannot_trigger_a_redraw(self):
        for text in (None, '', 'none', 'No correction needed; the contour is continuous.',
                     'No repair is necessary.', '无需修改。'):
            self.assertFalse(actionable_edit(text))
        self.assertTrue(actionable_edit('Redraw the joint where it meets the handle.'))

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

    def test_new_layout_requires_failure_in_the_same_part_after_scene_reconstruction(self):
        self.assertTrue(same_repair_region([588, 684, 693, 907], [647, 650, 750, 850]))
        self.assertFalse(same_repair_region([588, 684, 693, 907], [300, 160, 400, 220]))
        self.assertFalse(same_repair_region(None, [647, 650, 750, 850]))

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
            self.assertFalse(repeated_rejection(folder, target, minimum=3))
            receipt(5, target, 'uncertain')
            self.assertTrue(repeated_rejection(folder, target, minimum=3))

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

    def test_interior_detail_cannot_clear_a_limb_by_treating_its_crop_as_the_frame(self):
        image = Image.new('RGB', (1920, 1080))
        person = [{'name': 'person', 'box': [590, 210, 650, 390]}]
        for state, x, expected in [('hand', 50, False), ('frame_exit', 500, False),
                                    ('occlusion', 500, True)]:
            reading = {'confirmed': True, 'left': {'state': state, 'x': x, 'y': 500},
                       'right': {'state': 'hand', 'x': 650, 'y': 600}}
            with patch('construction_review.verify_arm_endpoints', return_value=reading):
                result = verify_interior_figure(None, image, person, 'local')
            self.assertEqual(result['confirmed'], expected)

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
