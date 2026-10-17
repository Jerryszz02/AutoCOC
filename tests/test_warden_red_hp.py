"""Same-frame current Warden evidence; these crops are not training labels."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from autococ.hero_state import locate_current_deployed_hero_card, recognize_hero_state
from autococ.post_deployment_skill import _current_warden


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/warden_red_hp_023658_18_647_2'
BOX = [647, 595, 736, 711]


def recognize(path, *, box=BOX, unit='grand_warden', version='18.647.2'):
    return recognize_hero_state(path, box, unit_id=unit, client_version=version)


class WardenRedHPTests(unittest.TestCase):
    def test_four_same_group_real_frames_have_live_fill_and_independent_ready_anchors(self):
        for path in sorted(FIXTURES.glob('*.png')):
            with self.subTest(frame=path.name):
                locator = locate_current_deployed_hero_card(path, 'grand_warden', client_version='18.647.2')
                self.assertEqual(locator['bbox'], BOX)
                state = recognize(path)
                self.assertEqual(state['state'], 'ability_ready')
                self.assertTrue(state['deployed'])
                self.assertFalse(state['defeated'])
                self.assertTrue(state['ability_ready'])
                self.assertFalse(state['ability_used'])
                evidence = state['evidence']
                self.assertIsNone(evidence['health_bar'])
                self.assertIsNone(evidence['yellow_health_bar'])
                hp = evidence['red_health_bar']
                self.assertEqual(hp['fill_bbox'], [655, 574, 670, 581])
                self.assertEqual(hp['frame_support'], [1., 1., 1., 1.])
                self.assertGreaterEqual(hp['fill_density'], .95)
                self.assertIsNotNone(evidence['current_equipment_matches']['ready_book'])
                self.assertIsNotNone(evidence['current_equipment_matches']['ready_card'])
                target = _current_warden(SimpleNamespace(client_version='18.647.2'),
                                         SimpleNamespace(screenshot_path=path))
                self.assertTrue(target['verified'])
                self.assertTrue(target['state']['ability_ready'])

    def test_red_outline_clutter_partial_fill_and_broken_card_cannot_prove_live_hp(self):
        source = cv2.imdecode(np.fromfile(FIXTURES / 'skill_before_31.png', np.uint8), cv2.IMREAD_COLOR)
        cases = {}
        for name, box in {
            'empty_interior': (655, 574, 726, 581),
            'top_edge_missing': (654, 572, 727, 573),
            'bottom_edge_missing': (654, 582, 727, 583),
            'left_edge_missing': (653, 574, 655, 582),
            'right_edge_missing': (726, 574, 728, 582),
            'card_left_missing': (643, 624, 651, 696),
            'card_right_missing': (731, 624, 740, 696),
            'card_bottom_missing': (647, 704, 737, 716),
            'portrait_missing': (676, 625, 735, 680),
        }.items():
            image = source.copy()
            x0, y0, x1, y1 = box
            image[y0:y1, x0:x1] = 0
            cases[name] = image
        image = source.copy()
        image[574:581, 655:726] = 0
        image[574:581:2, 655:670:2] = (0, 0, 255)
        cases['sparse_noise'] = image
        image = source.copy()
        image[574:581, 655:726] = 0
        image[574:581, 680:695] = source[574:581, 655:670]
        cases['detached_fill'] = image
        image = source.copy()
        image[574:581, 655:726] = 0
        image[574:581, 655:659] = (0, 0, 255)
        cases['outline_and_tiny_prefix'] = image
        image = source.copy()
        image[574:581, 655:726] = (0, 0, 255)
        cases['solid_red_backdrop'] = image
        image = source.copy()
        image[574:581, 655:726] = 0
        image[565:570, 655:680] = (0, 0, 255)
        cases['red_outside_frame'] = image
        with TemporaryDirectory() as directory:
            for name, image in cases.items():
                with self.subTest(case=name):
                    path = Path(directory) / f'{name}.png'
                    cv2.imencode('.png', image)[1].tofile(path)
                    state = recognize(path)
                    self.assertIsNone(state['evidence'].get('red_health_bar'))
                    self.assertIsNot(state['deployed'], True)
                    self.assertIsNot(state['ability_ready'], True)
                    self.assertIsNot(state['ability_used'], True)
                    target = _current_warden(SimpleNamespace(client_version='18.647.2'),
                                             SimpleNamespace(screenshot_path=path))
                    self.assertFalse(target['verified'])

    def test_red_health_alone_does_not_prove_ability_phase(self):
        source = cv2.imdecode(np.fromfile(FIXTURES / 'skill_before_31.png', np.uint8), cv2.IMREAD_COLOR)
        # Preserve the real HP/portrait/card. An absent or gray book must not
        # borrow readiness from that HP, or infer use from a gray icon alone.
        cases = {'missing_book': np.zeros((37, 40, 3), np.uint8),
                 'gray_book': cv2.cvtColor(cv2.cvtColor(source[530:567, 650:690],
                    cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)}
        with TemporaryDirectory() as directory:
            for name, replacement in cases.items():
                image = source.copy()
                image[530:567, 650:690] = replacement
                path = Path(directory) / f'{name}.png'
                cv2.imencode('.png', image)[1].tofile(path)
                state = recognize(path)
                self.assertTrue(state['deployed'])
                self.assertIsNotNone(state['evidence']['red_health_bar'])
                self.assertIsNot(state['ability_ready'], True)
                self.assertIsNot(state['ability_used'], True)

    def test_wrong_version_identity_and_shifted_slot_remain_unknown(self):
        path = FIXTURES / 'skill_before_31.png'
        for unit, version, box in (
            ('grand_warden', '18.600.7', BOX), ('grand_warden', None, BOX),
            ('archer_queen', '18.647.2', BOX), ('minion_prince', '18.647.2', BOX),
            ('grand_warden', '18.647.2', [744, 595, 833, 711]),
            ('grand_warden', '18.647.2', [648, 595, 737, 711]),
        ):
            with self.subTest(unit=unit, version=version, box=box):
                state = recognize(path, unit=unit, version=version, box=box)
                self.assertIsNone(state['evidence'].get('red_health_bar'))
                self.assertIsNot(state['ability_ready'], True)

    def test_earlier_red_hp_frame_is_not_an_empty_hp_negative(self):
        path = ROOT / 'tests/fixtures/hero_skill_stabilization_054932_00027_18_647_2.png'
        state = recognize(path)
        self.assertTrue(state['ability_ready'])
        self.assertEqual(state['evidence']['red_health_bar']['fill_bbox'], [655, 574, 668, 581])

    def test_provenance_and_anonymous_pixels(self):
        metadata = json.loads((FIXTURES / 'provenance.json').read_text(encoding='utf8'))
        self.assertFalse(metadata['human_ground_truth'])
        self.assertFalse(metadata['dynamic_event_validation'])
        self.assertFalse(metadata['training_data'])
        self.assertEqual(len(metadata['fixtures']), 4)
        for name, row in metadata['fixtures'].items():
            path = FIXTURES / f'{name}.png'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['fixture_sha256'])
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (720, 1280, 3))
            x0, y0, x1, y1 = row['card_bbox_1280']
            mask = np.ones(image.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = False
            self.assertFalse(np.any(image[mask]))
            self.assertEqual(hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest(), row['card_pixels_sha256'])
            evidence_root = os.environ.get('AUTOCOC_SOURCE_EVIDENCE_ROOT')
            if evidence_root:
                source = Path(evidence_root) / row['source_relative_to_reports']
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), row['source_sha256'])
                original = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
                original = cv2.resize(original, (1280, 720), interpolation=cv2.INTER_AREA)
                self.assertTrue(np.array_equal(original[y0:y1, x0:x1], image[y0:y1, x0:x1]))
