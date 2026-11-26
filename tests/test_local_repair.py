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
        focused=dict(report,size=[3840,2160],views=[dict(report['views'][0],unresolved_focal_pixels=[800,400,1248,848])])
        self.assertEqual(failed_focus(focused),(800,400,1248,848))
        focused['views'][0]['unresolved_focal_pixels']=[-100,20,80,200]
        self.assertEqual(failed_focus(focused),rect)
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

    def test_source_restoration_conditions_sampling_and_color_on_same_selected_artwork(self):
        from restoration import restoration_graph
        graph,node=restoration_graph('selected.png','job/pending-review/restored',3840,19)
        ops=graph.api
        saved=ops[node]['inputs']['images'][0]
        post=ops[saved]
        self.assertEqual(post['inputs']['color_correction_method'],'wavelet')
        guide=post['inputs']['original_resized_images'][0]
        self.assertEqual(ops[guide]['inputs']['largest_size'],3840)
        self.assertEqual(ops[ops[guide]['inputs']['image'][0]]['inputs']['image'],'selected.png')
        sampler=next(x['inputs'] for x in ops.values() if x['class_type']=='KSampler')
        condition=ops[sampler['positive'][0]]['inputs']
        self.assertEqual(condition['vae_conditioning'],sampler['latent_image'])
        encoder=ops[sampler['latent_image'][0]]['inputs']
        self.assertEqual(ops[encoder['pixels'][0]]['inputs']['resized_images'][0],guide)
        self.assertEqual(sampler['steps'],1)

    def test_lettering_vote_cannot_erase_a_depicted_contour_when_evidence_rules_out_writing(self):
        from lettering import denies_lettering
        self.assertTrue(denies_lettering('Mustard-yellow brushstrokes rather than tiny writing.'))
        self.assertTrue(denies_lettering('These strokes do not form any letters.'))
        self.assertFalse(denies_lettering('Four small painted letters rather than animal contours.'))
        self.assertFalse(denies_lettering('A line of illegible printed glyphs is visible.'))
        self.assertFalse(denies_lettering('A row of printed glyphs with no readable words.'))

    def test_new_radiating_lines_across_an_empty_ground_are_held(self):
        import math
        from print_artifacts import unsupported_lines
        source=Image.new('RGB',(448,448),'#174874')
        ImageDraw.Draw(source).ellipse((145,100,330,360),fill='#d6e4ee',outline='#163b5b',width=3)
        painted=source.copy();draw=ImageDraw.Draw(painted)
        for deg in range(0,360,7):
            angle=math.radians(deg)
            draw.line((220,224,round(220+180*math.cos(angle)),round(224+180*math.sin(angle))),fill='white',width=2)
        self.assertTrue(unsupported_lines(source,painted)['hold'])
        self.assertFalse(unsupported_lines(painted,painted)['hold'])
        boundary=source.copy()
        ImageDraw.Draw(boundary).line((50,20,385,420),fill='white',width=2)
        self.assertFalse(unsupported_lines(source,boundary)['hold'])

    def test_breeds_share_a_counting_species_but_different_species_and_extra_animals_still_fail(self):
        from counts import inventory_counts
        from evidence import count_conflict
        def observed(*pairs):
            return inventory_counts([dict(category=label,counting_species=name,kind='animal',count=n)
                                     for label,name,n in pairs])
        wanted={'dog':1}
        shiba,species=observed(('Shiba Inu','dog',1))
        self.assertFalse(count_conflict(wanted,shiba,[],{'dog'},species))
        literal,species=observed(('Shiba Inu','Shiba Inu',1))
        self.assertEqual(literal,{'dog':1})
        self.assertFalse(count_conflict(wanted,literal,[],{'dog'},species))
        fox,species=observed(('red fox','fox',1))
        self.assertTrue(count_conflict(wanted,fox,[],{'dog'},species))
        pair,species=observed(('Shiba Inu','dog',1),('basenji','dog',1),('dog','dog',2))
        self.assertEqual(count_conflict(wanted,pair,[],{'dog'},species)[0]['observed'],2)

    def test_self_contradictory_geometry_verdict_requires_an_independent_pixel_trace(self):
        from print_judge import inspect_disputed_action, disputed_action_resolved
        image=Image.new('RGB',(1800,1100))
        contradiction=dict(condition='visible_fault',visible_shapes='paired fins on a tapered body',
                           depicted_action='swimming',problem='No visible deformities or contradictions in structure are present.')
        trace=dict(first_landmark='head',second_landmark='pectoral fins',connection='single unbroken contour')
        for decision,repair,expected in [('continuous_and_plausible','none',True),
                                         ('specific_visible_error','Reconnect the detached fin.',False),
                                         ('cannot_trace','none',False)]:
            with patch('print_judge.model_identity',return_value='local'),patch('print_judge.ask',side_effect=[
                    json.dumps(contradiction),json.dumps(dict(trace,repair=repair,verdict=decision))]):
                result=inspect_disputed_action('localhost',image,[200,400,500,600],'A koi swims in a pond.')
            self.assertEqual(disputed_action_resolved(result),expected)
            if decision=='specific_visible_error':self.assertIn('detached fin',result['problem'])

    def test_scene_plan_cache_survives_rounds_but_changes_with_prompt_and_model(self):
        from contextlib import nullcontext
        from manga_materials import planner
        reply=dict(caption='a red cat',medium='woodcut',palette=[],art_direction='',surfaces=[])
        with TemporaryDirectory() as directory,patch.object(planner,'MODEL_ROOT',Path(directory)/'models'),patch('evidence.available',return_value=True),patch('evidence.backend',return_value=nullcontext('local')),patch('evidence.model_identity',return_value='model-a') as identity,patch('evidence.ask',return_value=json.dumps(reply)) as query:
            first=planner.dedicated_plan('A red cat.')
            self.assertEqual(first,planner.dedicated_plan('A red cat.'))
            self.assertEqual(query.call_count,1)
            planner.dedicated_plan('A blue cat.')
            identity.return_value='model-b'
            planner.dedicated_plan('A red cat.')
            self.assertEqual(query.call_count,3)
            self.assertEqual(len(list(Path(directory).rglob('*.json'))),3)

    def test_named_animal_evidence_distinguishes_related_species_and_preserves_aliases(self):
        from animal_identity import named_animal_conflict
        vocabulary={'barn owl':{'1'},'tyto alba':{'1'},'horned owl':{'2'},'great horned owl':{'3'},
                    'red fox':{'5'},'arctic fox':{'6'}}
        parents={'1':{'4'},'2':{'4'},'3':{'2'},'5':{'7'},'6':{'7'}}
        with TemporaryDirectory() as scratch:
            dictionary=Path(scratch)/'wordnet.zip'
            dictionary.touch()
            with patch('animal_identity.DICTIONARY',dictionary),patch('animal_identity.animal_names',return_value=(vocabulary,parents)):
                self.assertEqual(named_animal_conflict('A barn owl','A great horned owl with ear tufts.')['observed'],'great horned owl')
                self.assertEqual(named_animal_conflict('A red fox','An arctic fox.')['observed'],'arctic fox')
                self.assertIsNone(named_animal_conflict('Barn owl','A Tyto alba.'))
                self.assertIsNone(named_animal_conflict('Horned owl','A great horned owl.'))
                self.assertIsNone(named_animal_conflict('Barn owl','Possibly a great horned owl.'))
                self.assertIsNone(named_animal_conflict('Barn owl','A great horned owl and an arctic fox.'))

    def test_reinspection_uses_the_signed_release_and_its_actual_baseline(self):
        from production import previous_print_pair
        from print_judge import fingerprint
        with TemporaryDirectory() as scratch:
            root=Path(scratch);source=root/'round';source.mkdir();final=root/'final';final.mkdir()
            base=source/'draft.png';printed=final/'4K.png'
            Image.new('L',(1680,944)).save(base);Image.new('L',(3840,2160)).save(printed)
            receipt={'sha256':fingerprint(printed),'对照指纹':fingerprint(base)}
            record={'底图':'draft.png','自动定稿':receipt}
            self.assertEqual(previous_print_pair(root,source,record),[str(base.resolve()),str(printed.resolve())])
            Image.new('L',(1680,944),color=150).save(base)
            self.assertIsNone(previous_print_pair(root,source,record))

    def test_outline_repair_cannot_introduce_an_unrequested_action(self):
        from construction_review import scoped_repair_direction
        request='Separate the visible digits and reconnect the cuff.'
        draft='The hand rests naturally on the wooden bench, its digits separated.'
        self.assertNotIn('bench',scoped_repair_direction(draft,request))
        self.assertIn(request,scoped_repair_direction(draft,request))
        self.assertIn('bench',scoped_repair_direction(draft,'Restore contact between the hand and its bench.'))
        self.assertNotIn('grips',scoped_repair_direction('The foot grips a branch.','Restore the rear toe.'))

    def test_physical_example_matches_only_the_specific_connection_being_repaired(self):
        from illustration import construction_guide
        guide=construction_guide('Repair how the mooring rope turns through the iron ring on the quay.')
        self.assertTrue(guide[0].is_file())
        self.assertIn('rigid ring',guide[1]['construction'])
        self.assertIsNone(construction_guide('Repaint the boat and sharpen its waterline.'))
        self.assertIsNone(construction_guide('Draw a ring on her finger beside the moored boat.'))
        self.assertIsNone(construction_guide('Redraw the fish. Preserve the mooring rope and iron ring.'))
        self.assertIsNone(construction_guide('Remove the mooring rope and its ring.'))

    def test_pigment_corrections_do_not_pin_the_previous_target_color(self):
        from illustration import pigment_edit
        for instruction in ('Repaint the fur pigmentation of the cat.', 'Correct its fabric color.', '重新上色'):
            self.assertTrue(pigment_edit(instruction))
        for instruction in ('Reconstruct a wrist; preserve the colors of the shirt.', 'Correct the finger count.'):
            self.assertFalse(pigment_edit(instruction))

    def test_ambiguous_local_label_requires_two_physical_traces_of_the_same_visible_part(self):
        from construction_review import verify_repair_location
        source=Image.new('RGB',(600,700))
        suspect=dict(verdict='local',target='[400,210,600,480] wrist',observation='Wrist has a broken outline.',
                     box=[300,250,650,575],fix='Reconnect the wrist.')
        label=dict(state='target_absent_or_unclear',target_visible=True,evidence='Fabric wraps over the base of the palm.')
        trace=dict(part_visible=True,landmark_one='sleeve',landmark_two='wrist',
                   trace='A continuous edge joins the sleeve and wrist.',finding='visible_and_coherent')
        with patch('construction_review.available',return_value=True),patch('construction_review.ask',side_effect=[
                json.dumps(label),json.dumps(trace),json.dumps(trace)]) as query:
            verify_repair_location('local',source,suspect,'test-model')
        self.assertEqual(suspect['verdict'],'coherent')
        self.assertEqual(len(suspect['location_verification'][0]['independent_traces']),2)
        for call in query.call_args_list:self.assertNotIn('[400,210,600,480]',call.args[2])

    def test_existing_print_requires_a_smaller_matching_baseline(self):
        from focus_print import validate_print_pair
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            base=Path(directory)/'draft.png';candidate=Path(directory)/'4k.png'
            Image.new('L',(1680,944)).save(base)
            Image.new('L',(3840,2160)).save(candidate)
            validate_print_pair(base,candidate)
            with self.assertRaises(ValueError):validate_print_pair(candidate,candidate)
            Image.new('L',(960,1440)).save(base)
            with self.assertRaises(ValueError):validate_print_pair(base,candidate)
            Image.new('L',(1024,1536)).save(candidate)
            with self.assertRaises(ValueError):validate_print_pair(base,candidate)

    def test_specific_requirements_must_be_quoted_and_observed_before_returning_a_repair(self):
        from brief_check import subject_witness
        image=Image.new('RGB',(1800,1000));overview=image.resize((900,500))
        brief='A child looks through a brass telescope.'
        coordinates={'left':300,'top':200,'right':600,'bottom':530}
        items=[{'quote':'looks through a brass telescope','inspect':'Compare eye and eyepiece.',**coordinates},
               {'quote':'wears a blue coat','inspect':'Observe coat.',**coordinates}]
        prediction={'reading':'The eyepiece touches the nose below the eye.','fix':'Align the eyepiece with the pupil.',
                    'decision':'visible_contradiction'}
        with patch('brief_check.ask',side_effect=[json.dumps({'checks':items}),json.dumps({'crop_1':prediction}),
            json.dumps({'actual_landmarks':'The eyepiece touches the bridge of the nose below the eye.',
                        'finding':'contradiction','repair_instruction':'Align the eyepiece with the pupil.'}),
            'The lens touches the nose below the eye.',
            json.dumps({'relation':'explicit_incompatibility','incompatible_words':'below the eye'})]) as query:
            notes,issue=subject_witness('local',image,overview,brief,'model')
        self.assertEqual(len(notes),1)
        self.assertEqual(issue['operation'],'redraw')
        self.assertEqual(issue['left'],300)
        self.assertIs(query.call_args_list[-3].args[1][0],overview)
        self.assertGreater(query.call_args_list[-3].args[1][1].width,500)
        with patch('brief_check.ask',return_value=json.dumps({'checks':items[1:]})) as query:
            self.assertEqual(subject_witness('local',image,overview,brief,'model'),([],None))
        self.assertEqual(query.call_count,1)

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

    def test_blank_surface_restoration_requires_a_single_verified_material(self):
        from lettering import finish_blank_surface
        mask=Image.new('L',(180,110))
        ImageDraw.Draw(mask).rectangle((48,46,131,67),fill=255)
        clean=Image.new('RGB',mask.size,(24,37,35))
        ghost=clean.copy()
        ImageDraw.Draw(ghost).line((48,51,131,62),fill='#bbbbbb',width=4)
        restored=finish_blank_surface(clean,ghost,mask)
        self.assertLess(max(restored.getpixel((87,56))),45)
        self.assertEqual(restored.getpixel((44,49)),ghost.getpixel((44,49)))
        wood=clean.copy()
        for x in range(0,wood.width,4):ImageDraw.Draw(wood).line((x,0,x,109),fill='white',width=2)
        self.assertIs(finish_blank_surface(wood,ghost,mask),ghost)

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

    def test_planar_material_must_have_plausible_thickness_and_unbroken_occlusion(self):
        image=Image.new('RGB',(620,570))
        obs=dict(feature='envelope',material='paper',attachment='slot',image_evidence='Paper edge enters slot.')
        audit=dict(is_claimed_part='yes',blind_identification_visible=True,intact_contour=True,trace='Edges visible.')
        suspected=dict(operation='redraw',target='envelope',evidence='The item is too flat and its thickness is incorrect.')
        for natural,overlap,other,expected in [('thin_sheet_or_layer',True,False,True),
                 ('thin_sheet_or_layer',False,False,False),('thin_sheet_or_layer',True,True,False),
                 ('volumetric',True,False,False)]:
            evidence=dict(natural_form=natural,first_edge='upper fold',second_edge='lower lip',
                          perspective='Both edges converge at the slot.',thickness_matches=True,
                          overlap_matches_action=overlap,other_geometry_error=other)
            with patch('print_judge.model_identity',return_value='local'),patch('print_judge.ask',side_effect=[
                    json.dumps(obs),json.dumps(audit),json.dumps(evidence)]):
                result=verify_part_identity(None,image,[180,290,410,480],suspected)
            self.assertEqual(result['disproved'],expected)

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
        scene=dict(reading,condition='physically_plausible',depth_verification={'matches_pose':True})
        self.assertFalse(disputed_action_resolved(dict(scene,silhouette_inventory={'condition':'unresolved'})))
        self.assertTrue(disputed_action_resolved(dict(scene,silhouette_inventory={'condition':'fully_traced'})))

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

    def test_native_line_pairs_require_repeated_members_across_a_stroke(self):
        from print_artifacts import resolved_line_pair
        from print_judge import resize_only
        from PIL import ImageFilter
        im=Image.new('RGB',(192,192),'#c8b79b');d=ImageDraw.Draw(im)
        for x in range(45,150,8):d.line((x,24,x,167),fill='#35465e',width=2)
        small=im.resize((32,32),Image.Resampling.LANCZOS).resize(im.size,Image.Resampling.LANCZOS)
        self.assertIsNotNone(resolved_line_pair(small,im))
        self.assertIsNone(resolved_line_pair(small,small))
        self.assertEqual(resize_only((small,small.filter(ImageFilter.UnsharpMask(2,200,0))))['condition'],'resize_or_sharpen')

    def test_flat_color_spill_must_cross_the_target_and_destroy_existing_lines(self):
        from construction_review import overpainted_rectangle
        src=Image.new('RGB',(240,160),'#235742')
        for x in range(0,240,7):
            ImageDraw.Draw(src).line((x,0,x-20,159),fill='white')
        dst=src.copy()
        ImageDraw.Draw(dst).rectangle((60,52,185,105),fill='orange')
        field=(0,0,*src.size)
        self.assertIsNotNone(overpainted_rectangle(src,dst,(64,45,127,123),field))
        self.assertIsNone(overpainted_rectangle(src,dst,(50,42,200,120),field))
        self.assertIsNone(overpainted_rectangle(src,src,(64,45,127,123),field))
        plain=Image.new('RGB',src.size,'navy')
        overlay=plain.copy()
        ImageDraw.Draw(overlay).rectangle((60,52,185,105),fill='yellow')
        self.assertIsNone(overpainted_rectangle(plain,overlay,(64,45,127,123),field))

    def test_disputed_pigmentation_checks_the_exterior_surface_and_shading(self):
        before=Image.new('RGB',(300,180),'gray')
        after=before.copy()
        ImageDraw.Draw(after).ellipse((65,70,110,104),fill='brown')
        for edge,lighting,expected in [('continuous','consistent','intact'),('bleed_or_cut','consistent','new_damage'),
                                       ('continuous','broken','new_damage'),('unclear','consistent','uncertain')]:
            evidence={'inside_landmark':'cloth hem','outside_landmark':'bench slat','trace':'The hem meets the slat.',
                      'edge':edge,'light_and_style':lighting}
            with patch('construction_review.backend',return_value=nullcontext(None)), \
                    patch('construction_review.available',return_value=True), \
                    patch('construction_review.ask',side_effect=[json.dumps({'transition':'ambiguous','trace':'Pattern changed.'}),
                                                                json.dumps(evidence)]) as reader:
                result=assess_edit_boundary(before,after,[200,300,440,640],'Recolor the scarf to show a chequered pattern.')
                self.assertEqual(result['status'],expected)
                self.assertIn('EXTERIOR',reader.call_args_list[0].args[2])
                self.assertEqual(reader.call_args_list[1].kwargs['tag'],LARGE)

    def test_depth_review_uses_the_observed_fault_instead_of_parsing_a_negative_description(self):
        from print_judge import inspect_disputed_action
        first={'visible_shapes':'hand','depicted_action':'contact with platform','problem':'unclear','condition':'insufficient'}
        scene={'parent':'person','location_on_parent':'beside thigh','adjacent_surface':'jacket hem',
               'plane_evidence':'hand occludes hem','relation':'overlap_at_different_depths'}
        for fault,result,expected in [(False,'coherent','physically_plausible'),(True,'malformed','visible_fault'),
                                       (True,'coherent','insufficient')]:
            witness={'part':'hand','endpoints_and_edges':'traced the outline from wrist to fingertip',
                     'specific_fault':'Fused digits' if fault else 'No missing or fused part.',
                     'visible_fault':fault,'matches_pose':True,'result':result}
            with patch('print_judge.model_identity',return_value='local'),patch('print_judge.ask',side_effect=[
                    json.dumps(first),json.dumps(scene),json.dumps(witness)]):
                read=inspect_disputed_action(None,Image.new('RGB',(900,520)),[430,770,500,990],'A person standing.')
            self.assertEqual(read['condition'],expected)

    def test_visual_action_reading_includes_a_located_body_scale_view(self):
        from print_judge import inspect_disputed_action
        canvas=Image.new('RGB',(1440,810),'gray')
        observation={'visible_shapes':'A lowered limb.','depicted_action':'Hangs free.',
                     'problem':'','condition':'physically_plausible'}
        traced={'part':'limb','visible_ends':['free toe'],'base_and_branches':'Ankle continues into foot.',
                'occlusion_evidence':'','contradiction':'','condition':'fully_traced'}
        for expected,outcome in [('physically_plausible',traced),('visible_fault',
                dict(traced,condition='malformed',contradiction='Two long spikes attach where the palm must branch.'))]:
            with patch('print_judge.model_identity',return_value='fixture'), \
                    patch('print_judge.ask',side_effect=map(json.dumps,(observation,outcome))) as reader:
                actual=inspect_disputed_action(None,canvas,[450,850,520,999],'A figure standing.')
                self.assertEqual(actual['condition'],expected)
                self.assertEqual(len(reader.call_args_list[1].args[1]),1)
        full,middle,native=reader.call_args_list[0].args[1]
        self.assertGreater(full.width,middle.width)
        self.assertGreater(middle.width,native.width)
        self.assertIn((190,54,116),[color for _,color in middle.getcolors()])
        self.assertNotIn((190,54,116),[color for _,color in native.getcolors()])

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
