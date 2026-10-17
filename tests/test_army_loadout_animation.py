"""Real pet animation pixels and fail-closed current-frame descriptor wiring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.army_editor import _visual_fingerprint, recognize_hero_loadout, visual_fingerprint_matches
from autococ.ocr import OCRText

FIXTURES = Path(__file__).parent / 'fixtures'


class PetAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provenance = json.loads((FIXTURES / 'pet-animation-005103-provenance.json').read_text(encoding='utf8'))
        cls.rows = {(r['frame_number'], r['hero_id']): r for r in cls.provenance['crops']}

    def crop(self, frame, hero):
        row = self.rows[frame, hero]
        path = FIXTURES / row['crop']
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['crop_sha256'])
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    def fingerprint(self, frame, hero, *, context='archer_queen'):
        return _visual_fingerprint(self.crop(frame, hero),
                                   tuple(self.rows[frame, hero]['bbox_at_baseline']),
                                   animation_hero_id=context)

    def test_original_pixels_reproduce_cached_hash_and_animation_false_rejection(self):
        for (frame, hero), row in self.rows.items():
            with self.subTest(frame=frame, hero=hero):
                actual = self.fingerprint(frame, hero, context=hero)
                for field in ('phash', 'mean_bgr', 'std_bgr', 'bbox'):
                    self.assertEqual(actual[field], row['cached_fingerprint'][field])
        before = self.fingerprint(11, 'archer_queen')
        after = self.fingerprint(15, 'archer_queen')
        self.assertEqual((int(before['phash'], 16) ^ int(after['phash'], 16)).bit_count(), 10)
        self.assertFalse(visual_fingerprint_matches(self.rows[11, 'archer_queen']['cached_fingerprint'],
                                                   self.rows[15, 'archer_queen']['cached_fingerprint']))
        self.assertTrue(visual_fingerprint_matches(before, after))

    def test_other_real_pets_cannot_replace_the_current_pet_under_same_hero_context(self):
        heroes = sorted({key[1] for key in self.rows})
        for first in heroes:
            for second in heroes:
                with self.subTest(first=first, second=second):
                    self.assertEqual(visual_fingerprint_matches(
                        self.fingerprint(11, first), self.fingerprint(15, second)), first == second)

    def test_animation_evidence_missing_malformed_or_from_other_context_is_rejected(self):
        before = self.fingerprint(11, 'archer_queen')
        after = self.fingerprint(15, 'archer_queen')
        cases = []
        item = deepcopy(after); item.pop('animation'); cases.append(item)
        for field, value in (('schema', 'other'), ('client_version', 'unknown'),
                             ('hero_id', 'grand_warden'), ('gray16', '00'),
                             ('gray16', 'gg' * 256), ('gray16', '7f' * 256)):
            item = deepcopy(after); item['animation'][field] = value; cases.append(item)
        item = deepcopy(after)
        item['animation']['gray16'] = bytes(255 - x for x in bytes.fromhex(item['animation']['gray16'])).hex()
        cases.append(item)
        for item in cases:
            with self.subTest(item=item):
                self.assertFalse(visual_fingerprint_matches(before, item))
        bad_context = deepcopy(before); bad_context['animation']['hero_id'] = 'not_a_hero'
        other = deepcopy(after); other['animation']['hero_id'] = 'not_a_hero'
        self.assertFalse(visual_fingerprint_matches(bad_context, other))

    def test_strong_structure_cannot_bypass_color_or_hash_change_limits(self):
        before = self.fingerprint(11, 'archer_queen')
        after = self.fingerprint(15, 'archer_queen')
        cases = []
        for field in ('mean_bgr', 'std_bgr'):
            item = deepcopy(after); item[field][0] = before[field][0] + 3.01; cases.append(item)
        item = deepcopy(after); item['mean_bgr'][0] = float('nan'); cases.append(item)
        item = deepcopy(after); item['std_bgr'][0] = float('inf'); cases.append(item)
        item = deepcopy(after); item['phash'] = f"{int(before['phash'], 16) ^ ((1 << 13) - 1):016x}"; cases.append(item)
        for item in cases:
            with self.subTest(item=item):
                self.assertFalse(visual_fingerprint_matches(before, item))

    def test_descriptor_is_emitted_only_for_version_verified_pet_not_equipment(self):
        # Synthetic layout tests wiring only; identity comes from an independent
        # recognizer in production, while the pet pixels below are real crops.
        for verified, version, identity_id, expected in ((True, '18.647.2', 'archer_queen', True),
                                             (False, '18.647.2', 'archer_queen', False),
                                             (True, 'another', 'archer_queen', False),
                                             (True, '18.647.2', 'dragon_duke', False)):
            with self.subTest(verified=verified, version=version), TemporaryDirectory() as folder:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                canvas[533:599, 421:529] = self.crop(11, 'archer_queen')
                rng = np.random.default_rng(17)
                canvas[602:658, 421:472] = rng.integers(0, 256, (56, 51, 3), dtype=np.uint8)
                canvas[602:658, 478:529] = rng.integers(0, 256, (56, 51, 3), dtype=np.uint8)
                path = Path(folder) / 'synthetic-layout.png'
                cv2.imencode('.png', canvas)[1].tofile(str(path))
                cards = [{'unit_id': 'archer_queen', 'kind': 'hero', 'source': 'army', 'count': 1,
                          'confidence': .99, 'card_bbox': [413, 185, 537, 668],
                          'evidence': {'identity': {'unit_id': identity_id, 'version_verified': verified,
                                                   'template_client_version': version}}}]
                result = recognize_hero_loadout(path, [OCRText('我的军队', .99, (210, 60, 310, 90)),
                                                       OCRText('1/4', .99, (50, 150, 98, 176))], cards)
                self.assertTrue(result['complete'])
                slots = result['hero_loadout']['archer_queen']
                self.assertEqual('animation' in slots['pet_visual'], expected)
                self.assertNotIn('animation', slots['equipment_1_visual'])
                self.assertNotIn('animation', slots['equipment_2_visual'])


if __name__ == '__main__':
    unittest.main()
