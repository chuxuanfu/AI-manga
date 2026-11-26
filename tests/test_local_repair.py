"""Offline regression checks for retaining or escalating a structural repair."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/studio_pipeline'), str(ROOT / 'extensions')]
from construction_review import LARGE, assess_edit_boundary, assess_local_edit, compare_prior_version, verify_interior_figure
from illustration import instruction_edit, reference_draft
from masks import structural_composite
from print_judge import actionable_edit, boxes_agree, crop_brief, disputed_action_resolved, inspect_repaired_region, issue_category, verify_depicted_connection, verify_part_identity, verified_wrist_at
from page_layout import retry_direction
from production import earlier_local_version, previous_scene_candidate, repeated_rejection, same_repair_region
from review import inspection_key
from structure import inspect_small_figure_support


class LocalRepairTests(unittest.TestCase):
    def test_tile_subjects_align_with_the_actual_sampler_in_portrait_and_landscape(self):
        from focal_edit import print_subjects, print_constraints
        from illustration import print_graph
        from manga_materials import MaterialTiles, starts, TILE
        import torch
        restrictions=print_constraints('An ink illustration, no lettering.')
        self.assertIn('no lettering',restrictions)
        def identify(url, images, prompt, schema, tag):
            self.assertIn(restrictions,prompt)
            return json.dumps({n:'surface '+n for n in schema['required']})
        with TemporaryDirectory() as temp, patch('focal_edit.model_identity', return_value='fixture'), \
             patch('focal_edit.backend',return_value=nullcontext('local')),patch('focal_edit.ask',side_effect=identify) as ask:
            for width,height in ((384,256),(256,384)):
                source=Image.new('RGB',(width,height))
                cache=Path(temp)/f'{width}.json'
                subjects=print_subjects(source,1920,cache,constraints=restrictions)
                scaled=(width*5,height*5)
                fields=[(x,y) for y in starts(0,scaled[1],scaled[1],TILE) for x in starts(0,scaled[0],scaled[0],TILE)]
                self.assertEqual(set(subjects),{f'{x},{y}' for x,y in fields})
                _,prompts,plan=MaterialTiles().prepare(torch.zeros((1,scaled[1],scaled[0],3)),None,'整幅通用重绘',
                        'none','Dry ink',json.dumps(subjects))
                self.assertEqual(prompts,[subjects[f'{x},{y}']+'. Dry ink' for x,y,*_ in plan])
                calls=ask.call_count
                self.assertEqual(print_subjects(source,1920,cache,constraints=restrictions),subjects)
                self.assertEqual(calls,ask.call_count)
        g,_=print_graph('source.png','case',23,four_only=True,tile_subjects={3840:subjects})
        sample=next(n for n in g.api.values() if n['class_type']=='KSampler')
        pos=g.api[sample['inputs']['positive'][0]]['inputs']
        split,output=pos['text']
        self.assertEqual((g.api[split]['class_type'],output),('MangaStudioMaterialTiles',1))

    def test_optical_repair_targets_a_verified_site_and_keeps_pixels_outside(self):
        from focal_edit import failed_focus,framing,selection
        rect=(3421,99,3840,530)
        report={'issue':{'category':'unresolved_focus'},'views':[{'view':'comparison','pixels':list(rect)}]}
        self.assertEqual(failed_focus(report),rect)
        self.assertIsNone(failed_focus(dict(report,issue={'category':'anatomy'})))
        self.assertIsNone(failed_focus(dict(report,views=[])))
        ambiguous={'status':'uncertain','issue':{'category':'other','operation':'none','box':None},
                   'views':[{'view':'comparison','pixels':list(rect),'observation':{'operation':'none'}}]}
        self.assertEqual(failed_focus(ambiguous),rect)
        ambiguous['views'][0]['corroboration']={'operation':'redraw'}
        self.assertIsNone(failed_focus(ambiguous))
        field,target=framing((3840,2160),rect)
        self.assertLessEqual(field[2],3840)
        close,inside=framing((3840,2160),rect,magnified=True)
        self.assertLess(close[2]-close[0],field[2]-field[0])
        self.assertEqual(tuple(v+close[i%2] for i,v in enumerate(inside)),rect)
        mask=selection((field[2]-field[0],field[3]-field[1]),target)
        self.assertEqual(mask.getbbox(),target)
        self.assertEqual(mask.getpixel(((target[0]+target[2])//2,(target[1]+target[3])//2)),255)

    def test_interpolated_and_sharpened_prints_require_a_new_render(self):
        from print_judge import inspect, resize_only
        import numpy as np
        rng=np.random.default_rng(2541)
        source=Image.fromarray(rng.integers(20,235,(124,180,3),dtype=np.uint8))
        enlarged=source.resize((1440,992),Image.Resampling.LANCZOS)
        sharp=enlarged.filter(ImageFilter.UnsharpMask(radius=1.4,percent=115,threshold=0))
        self.assertEqual(resize_only((enlarged,sharp))['condition'],'resize_or_sharpen')
        self.assertEqual(resize_only((enlarged,ImageOps.mirror(sharp)))['condition'],'distinct_signal')
        with TemporaryDirectory() as temp, patch('print_judge.model_identity',return_value='test'), \
             patch('print_judge.backend',return_value=nullcontext('offline')), \
             patch('print_judge.ask',side_effect=AssertionError('A pure enlargement is detectable before model inference')):
            folder=Path(temp)
            small=folder/'source.png'; source.save(small)
            for name,image in [('enlarged',enlarged),('sharpened',sharp)]:
                with self.subTest(variant=name):
                    path=folder/(name+'.png');image.save(path)
                    result=inspect(path,'A finely patterned illustrated fabric',folder,{},small=small,source_comparison=True)
                    self.assertEqual(result['status'],'global')
                    self.assertEqual(result['issue']['category'],'unresolved_focus')
                    self.assertGreaterEqual(len(result['resampling_check'][0]['fields']),2)

    def test_latent_rounding_keeps_original_pixels_on_the_requested_print_canvas(self):
        from focus_print import fit_print_aspect
        with TemporaryDirectory() as temp:
            source=Path(temp)/'art.png'
            image=Image.effect_noise((384,214),50).convert('RGB');image.save(source)
            result=fit_print_aspect(source,'16:9',384)
            with Image.open(result) as corrected:
                self.assertEqual(corrected.size,(384,216))
                self.assertIsNone(ImageChops.difference(image,corrected.crop((0,1,384,215))).getbbox())
            self.assertEqual(result,fit_print_aspect(result,'16:9',384))
            self.assertEqual(source,fit_print_aspect(source,'auto',384))

    def test_ctc_word_decoder_preserves_double_letters_separated_by_blanks(self):
        import numpy as np
        from lettering import decode_word
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        sequence = [0] + [alphabet.index(c)+1 if c else 0 for c in ('f','f','e','e','','e','t')]
        logits = np.full((len(sequence),37),-8.,dtype=np.float32)
        logits[np.arange(len(sequence)),sequence] = 8.
        result,score = decode_word(logits)
        self.assertEqual(result,'feet')
        self.assertGreater(score,.99)
        self.assertEqual(decode_word(logits[:1]),('',0.))

    def test_disjoint_word_polygons_do_not_erase_a_neighbouring_contour(self):
        from lettering import glyph_mask
        image=Image.new('RGB',(160,80),'white')
        canvas=ImageDraw.Draw(image)
        support=Image.new('L',image.size)
        for x in (15,105):
            ImageDraw.Draw(support).rectangle((x,30,x+30,52),fill=255)
            for i in range(3):
                canvas.rectangle((x+4+i*8,35,x+7+i*8,47),fill='black')
        canvas.rectangle((68,33,73,50),fill='black')
        mask=glyph_mask(image,support)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(mask.crop((15,30,45,52)).getbbox())
        self.assertIsNone(mask.crop((60,25,86,57)).getbbox())

    def test_optical_recheck_grades_the_actual_print_before_comparing(self):
        from print_judge import compare_local_definition
        source, final = object(), object()
        clear = {'condition': 'well_defined', 'visible_edges': 'separate feather tips'}
        blurred = {'condition': 'optically_degraded', 'visible_edges': 'fused tips'}
        for replies, expected in [([blurred], blurred), ([clear, blurred], blurred),
                                  ([clear, clear], clear)]:
            with self.subTest(replies=replies), patch('print_judge.ask', side_effect=[
                    json.dumps(reading) for reading in replies]) as ask:
                result = compare_local_definition('local', [source, final])
                self.assertEqual(result['condition'], expected['condition'])
                self.assertEqual(ask.call_args_list[0].args[1], [final])
                self.assertEqual(ask.call_count, len(replies))
                if len(replies) == 2:
                    self.assertEqual(ask.call_args_list[1].args[1], [final, source])

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

    def test_qwen_text_generator_draws_on_an_empty_canvas_with_matching_weights(self):
        from illustration import DRAW_LORA, DRAW_UNET, text_composition
        graph = text_composition('Two bookends and a fern', 17, 1792, 1008, 'draft').api
        nodes = {node['class_type']: node['inputs'] for node in graph.values()}
        self.assertEqual(nodes['EmptySD3LatentImage'], {'width':1792, 'height':1008, 'batch_size':1})
        self.assertEqual(nodes['UNETLoader']['unet_name'], DRAW_UNET)
        self.assertEqual(nodes['LoraLoaderModelOnly']['lora_name'], DRAW_LORA)
        self.assertEqual(nodes['KSampler']['steps'], 4)
        self.assertEqual(nodes['KSampler']['seed'], 17)
        self.assertNotIn('LoadImage', nodes)
        self.assertEqual(graph[nodes['KSampler']['latent_image'][0]]['class_type'], 'EmptySD3LatentImage')

    def test_unchanged_native_tile_requires_same_pixels_brief_location_and_model(self):
        from print_judge import native_review_key
        tile = Image.new('RGB', (8, 8), 'white')
        arguments = [(0, 0, 8, 8), (12, 12), 'Ink', 'model-A']
        key = native_review_key(tile, *arguments)
        self.assertEqual(key, native_review_key(tile.copy(), *arguments))
        for index, other in enumerate(((1, 0, 9, 8), (16, 12), 'Pastel', 'model-B')):
            changed = list(arguments)
            changed[index] = other
            self.assertNotEqual(key, native_review_key(tile, *changed))
        tile.putpixel((5, 5), (254, 255, 255))
        self.assertNotEqual(key, native_review_key(tile, *arguments))

    def test_only_a_general_lettering_ban_enables_surface_cleaning(self):
        from lettering import blank_surfaces_requested
        for text in ('A poster. No captions, signatures or lettering.', 'No text or watermarks.',
                     '书店，无文字。', 'A figure without lettering.'):
            self.assertTrue(blank_surfaces_requested(text), text)
        for text in ('No captions. Storefront with a sign.', 'No lettering on her jacket.',
                     'No lettering except the station name.', 'A page with no text bubbles.',
                     '手绘海报，不要文字气泡。', 'A sign with legible text.'):
            self.assertFalse(blank_surfaces_requested(text), text)

    def test_glyph_masks_clean_both_polarities_and_leave_the_surface_edge_intact(self):
        from lettering import glyph_mask
        for paper, ink in (('white','black'),('black','white')):
            art=Image.new('RGB',(200,120),paper)
            draw=ImageDraw.Draw(art)
            draw.line((0,94,199,94),fill=ink,width=3)
            draw.text((60,50),'POST 27',fill=ink,stroke_width=1)
            region=Image.new('L',art.size)
            ImageDraw.Draw(region).rectangle((53,41,119,72),fill=255)
            mask=glyph_mask(art,region)
            self.assertIsNotNone(mask)
            self.assertIsNone(mask.crop((0,90,200,120)).getbbox())
        self.assertIsNone(glyph_mask(Image.new('RGB',art.size,'grey'),region))

    def test_two_hands_on_one_prop_do_not_trigger_an_overlap_redraw(self):
        from construction_review import overlapping_digits
        regions={'R1':{'box':[250,250,750,750], 'label':'human hands and fingers'}}
        first={'relation':'fingers_of_different_hands_overlap','between':'an envelope'}
        other={'relation':'separate_visible_contours','first_hand':'top edge','second_hand':'bottom edge',
               'between':'an uninterrupted strip of paper'}
        with patch('construction_review.ask',side_effect=[json.dumps(first),json.dumps(other)]) as read, \
                patch('construction_review.available',return_value=True):
            rows=overlapping_digits(None,Image.new('RGB',(180,180)),regions,LARGE)
        self.assertEqual(read.call_count,2)
        self.assertEqual(rows[0]['spatial_check'],other)
        self.assertNotIn('mirror',rows[0])

    def test_verified_render_resume_skips_gpu_and_changed_files_require_a_new_save(self):
        import hashlib
        import io
        from run import post_graph, _post_graph
        api={'1':{'class_type':'SaveImage','inputs':{'filename_prefix':'image'}}}
        payload=json.dumps({'prompt':api}).encode()
        digest=hashlib.sha256(payload).hexdigest()
        outputs={'1':{'images':[{'type':'output','filename':'image.png','subfolder':''}]}}
        with TemporaryDirectory() as temp, patch('run.release_idle_renderer'):
            root=Path(temp)
            source=root/'image.png'
            source.write_bytes(b'original')
            (root/'drawing-task.json').write_text(json.dumps({'id':'original','digest':digest,'renderer':'ComfyUI',
                'outputs':outputs,'output_sha256':{'image.png':hashlib.sha256(b'original').hexdigest()}}))
            with patch('run.OUTPUT',root),patch('run.RUNTIME',root/'no-compute-lock'):
                self.assertEqual(post_graph(api,8193,root,'drawing'),outputs)
            source.write_bytes(b'externally changed')
            history={'new':{'status':{'status_str':'success','messages':[]},'outputs':{}}}
            replies=[io.BytesIO(b'{"prompt_id":"new"}'),io.BytesIO(json.dumps(history).encode())]
            with patch('run.OUTPUT',root),patch('run.request.urlopen',side_effect=replies) as service:
                _post_graph(payload,digest,8193,root,'drawing',1.25)
            submitted=service.call_args_list[0].args[0]
            self.assertTrue(submitted.full_url.endswith('/prompt'))
            self.assertTrue(json.loads(submitted.data)['prompt']['1']['inputs']['filename_prefix'].startswith('image-restored-'))
            self.assertEqual(json.loads((root/'drawing-task.json').read_text())['id'],'new')

    def test_crop_omission_requires_a_located_attachment_and_an_independent_native_read(self):
        from print_judge import verify_outside_crop
        claim='The crop shows part of the bird but excludes the feet and support.'
        found={'condition':'satisfied_in_view','trace':'Feet reach the bar.','clause':'Bird perched on rail','left':420,'top':530,'right':550,'bottom':590}
        for outcome,expected in (('uncertain',False),('satisfied_in_view',True),('visible_fault',False)):
            replies=[json.dumps(found),json.dumps({'condition':outcome,'evidence':'Two toes curl over the bar.'})]
            with patch('print_judge.model_identity',return_value='model'),patch('print_judge.ask',side_effect=replies):
                value=verify_outside_crop(None,Image.new('RGB',(400,300)),'Bird perched on rail',claim)
            self.assertEqual(value['confirmed'],expected)
        edge=dict(found,left=848,top=480,right=999,bottom=999)
        with patch('print_judge.model_identity',return_value='model'),patch('print_judge.ask',return_value=json.dumps(edge)) as read:
            self.assertFalse(verify_outside_crop(None,Image.new('RGB',(400,300)),'Bird perched on rail',claim)['confirmed'])
            self.assertEqual(read.call_count,1)
        with patch('print_judge.ask') as reader:
            self.assertIsNone(verify_outside_crop(None,None,'Bird on rail','A talon visibly merges with the wood.'))
            reader.assert_not_called()

    def test_full_scene_repair_sees_the_native_detail_but_uses_the_full_canvas(self):
        graph = instruction_edit('Repair this connection', 77, 'whole.png', 'trial',
                                 masked=True, detail_reference='detail.png').api
        encoder = next(n for n in graph.values() if n['class_type'] == 'TextEncodeQwenImageEditPlus')
        latent = next(n for n in graph.values() if n['class_type'] == 'VAEEncode')
        full = encoder['inputs']['image1'][0]
        detail = encoder['inputs']['image2'][0]
        self.assertEqual(graph[full]['inputs']['image'], 'whole.png')
        self.assertEqual(graph[detail]['inputs']['image'], 'detail.png')
        self.assertEqual(latent['inputs']['pixels'][0], full)
        mask = next(n for n in graph.values() if n['class_type'] == 'SetLatentNoiseMask')
        self.assertEqual(mask['inputs']['mask'], [full, 1])

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

    def test_overturning_a_structural_finding_needs_an_independent_whole_part_reading(self):
        reading = {'visible_shapes':'A branch continues into its trunk.',
                   'depicted_action':'The branch supports the leaf cluster.'}
        for condition in ('insufficient', 'visible_fault', 'physically_plausible'):
            self.assertEqual(disputed_action_resolved(dict(reading, condition=condition)),
                             condition == 'physically_plausible')
        self.assertFalse(disputed_action_resolved({'condition':'physically_plausible'}))

    def test_retry_brief_cannot_copy_objects_from_a_discarded_composition(self):
        for note, unexpected in [('The strands around the unrequested cleat have tangled crossings.', 'cleat'),
                                 ('An extra red suitcase appears.', 'suitcase'),
                                 ('多余的蓝色雨伞遮挡了河岸。', '雨伞')]:
            direction = retry_direction(note)
            self.assertTrue(direction)
            self.assertNotIn(unexpected, direction)
        self.assertEqual(retry_direction(''), '')
        self.assertEqual(retry_direction('An unfamiliar artifact in the picture.'), '')

    def test_a_neighbor_cannot_corroborate_the_reported_part(self):
        actual = [160, 692, 263, 757]
        self.assertFalse(boxes_agree(actual, [46, 563, 129, 659]))
        self.assertTrue(boxes_agree(actual, [215, 695, 242, 737]))
        self.assertFalse(boxes_agree(actual, None))

    def test_localized_uncertainty_is_a_repair_candidate_not_an_approved_draft(self):
        from run import draft_repair_issue
        issue = {'operation':'redraw','box':[100, 200, 220, 330],'fix':'Reconnect the lower branch.'}
        self.assertIs(draft_repair_issue({'status':'uncertain','issues':[issue]}), issue)
        for status, change in [('global',{}),('pass',{}),('uncertain',{'box':None}),
                               ('uncertain',{'operation':'none'}),('uncertain',{'fix':'No repair needed.'})]:
            self.assertIsNone(draft_repair_issue({'status':status,'issues':[dict(issue,**change)]}))

    def test_wrist_attachment_evidence_cannot_clear_malformed_fingers_or_a_different_hand(self):
        point = [720, 900]
        crop = {'visible_hand':True,'joins_own_wrist':True,'visible_unsupported_end':False}
        right = {'state':'hand','source_point':point,'crop_check':crop}
        both = {'confirmed':True,'native_figure':{'confirmed':True,'right':right}}
        pre = {'局部连接复审':{'arm_endpoint_evidence':both}}
        im = Image.new('RGB',(1000,1000))
        box = [680, 720, 750, 950]
        self.assertEqual(verified_wrist_at(pre, im, box, 'The sleeve lacks a visible hand.'), point)
        self.assertIsNone(verified_wrist_at(pre, im, box, 'The fingers merge with the thumb.'))
        self.assertIsNone(verified_wrist_at(pre, im, [100,350,230,450], 'The sleeve lacks a hand.'))
        both['native_figure']['left'] = right
        self.assertIsNone(verified_wrist_at(pre, im, box, 'The sleeve lacks a hand.'))

    def test_an_earlier_version_requires_locally_plausible_form_and_a_specific_boundary_win(self):
        for state, winner, expected in [('ambiguous','A',True),('defective','A',False),
                                        ('ambiguous','both_intact',False),('ambiguous','B',False)]:
            uncertain = {'verdict':'uncertain','blind_comparison':{'better':'equal'},
                         'independent_construction':{'candidate':'A','samples':{'A':{'construction':state}}}}
            boundary = {'status':'intact' if winner!='B' else 'new_damage', 'candidate':'A','transition':winner}
            with patch('construction_review.assess_local_edit',return_value=uncertain), patch(
                    'construction_review.assess_edit_boundary', return_value=boundary):
                result = compare_prior_version(None,None,None,'A connection',None,'Artwork')
            self.assertEqual(bool(result.get('eligible_for_full_review')), expected)
            self.assertEqual(result['verdict'], 'uncertain')

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

    def test_sides_of_a_boundary_comparison_do_not_reveal_which_image_was_edited(self):
        before = Image.new('RGB', (320, 200), 'gray')
        after = before.copy()
        ImageDraw.Draw(after).line((100, 78, 197, 115), fill='black', width=6)
        pairs = []
        def compare(url, images, *args, **kwargs):
            pairs.append([im.tobytes() for im in images])
            return json.dumps({'transition':'A', 'trace':'Stroke continues in sample A.'})
        box = [280, 330, 680, 660]
        with patch('construction_review.ask', side_effect=compare), patch(
                'construction_review.backend', return_value=nullcontext(None)):
            forward = assess_edit_boundary(before, after, box)
            reverse = assess_edit_boundary(after, before, box)
        self.assertEqual(pairs[0], pairs[1])
        self.assertEqual({forward['status'], reverse['status']}, {'intact', 'new_damage'})
        self.assertEqual(len(forward['transition_sites']), 4)

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

    def test_reverting_a_patch_cannot_revert_a_distant_fix_or_an_unverified_image(self):
        from print_judge import fingerprint
        with TemporaryDirectory() as tmp:
            rounds = Path(tmp) / 'rounds'
            first, next_round = rounds/'0001', rounds/'0002'
            reports = first/'print-review'
            reports.mkdir(parents=True)
            next_round.mkdir()
            earlier, current = first/'before.png', first/'after.png'
            old = Image.new('RGB', (512, 512), 'white')
            old.save(earlier)
            new = old.copy()
            ImageDraw.Draw(new).line((220, 230, 250, 235), fill='black', width=3)
            new.save(current)
            box = [400, 400, 510, 510]
            report = dict(sha256=fingerprint(earlier), brief='A drawing', size=[512, 512],
                          comparison_sha256=fingerprint(earlier), source=str(earlier), status='local',
                          issue={'box':box})
            receipt = reports/'read.json'
            receipt.write_text(json.dumps(report))
            def find_version():
                return earlier_local_version(next_round, current, earlier, box, 'A drawing', 'v')
            self.assertEqual(find_version()[0], earlier.resolve())
            receipt.write_text(json.dumps(dict(report, sha256='wrong')))
            self.assertIsNone(find_version())
            receipt.write_text(json.dumps(report))
            new.putpixel((1, 511), (0, 0, 0))
            new.save(current)
            self.assertIsNone(find_version())

    def test_previous_scene_requires_matching_renderer_review_and_file(self):
        from print_judge import fingerprint
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            folder = root/'round'; folder.mkdir()
            image = folder/'scene.png'; Image.new('RGB',(40, 20)).save(image)
            digest = fingerprint(image)
            row = {'stage':'scene','file':image.name,'hash':digest,'review':{'status':'uncertain'}}
            receipt = {'renderer':'ComfyUI','model_files':['local-editor'],'output_sha256':{'round/scene.png':digest}}
            (folder/'scene-task.json').write_text(json.dumps(receipt))
            self.assertEqual(previous_scene_candidate(folder, 'scene', [row], root)[0], image)
            (folder/'render-history').mkdir()
            (folder/'scene-task.json').rename(folder/'render-history/scene-previous.json')
            self.assertEqual(previous_scene_candidate(folder, 'scene', [row], root)[0], image)
            Image.new('RGB',(40, 20),'red').save(image)
            self.assertIsNone(previous_scene_candidate(folder, 'scene', [row], root))

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

    def test_tall_subject_in_a_wide_frame_can_be_inspected_in_subject_coordinates(self):
        image = Image.new('RGB', (1920, 1080))
        person = [{'name': 'person', 'box': [542, 102, 809, 961]}]
        reading = {'confirmed': True, 'left': {'state':'occlusion', 'x':460, 'y':470},
                   'right': {'state':'hand', 'x':710, 'y':820}}
        with patch('construction_review.verify_arm_endpoints', return_value=reading):
            checked = verify_interior_figure(None, image, person, 'local')
        self.assertTrue(checked['confirmed'])
        self.assertLess(checked['source_pixels'][2]-checked['source_pixels'][0], image.width*.5)

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
