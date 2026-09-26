from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.terrain import find_clear_ground_probes, find_west_deployment_points, measure_camera_motion, measure_cloud_cover


class CloudCoverTests(unittest.TestCase):
    def test_thin_cloud_over_controls_is_distinct_from_pale_clear_map_tiles(self):
        root = Path(__file__).resolve().parents[1]
        cases = (("reports/20260924-042448-049906-99d83d3d/frames/00017-battle-candidate.png", True),
                 ("reports/20260924-033932-954707-638055a7/frames/00018-deploy-west-edge.png", False),
                 ("reports/20260924-033932-954707-638055a7/frames/00020-deploy-west-edge.png", False))
        if not all((root / path).is_file() for path, _ in cases):
            self.skipTest("Optional thin-cloud and pale-map fixtures are absent")
        for path, expected in cases:
            with self.subTest(frame=path):
                self.assertIs(measure_cloud_cover(root / path)["obscured"], expected)

    def test_observed_arrival_clouds_are_rejected_until_the_map_is_clear(self):
        root = Path("reports/20260924-041607-629356-a6aed4aa/frames")
        if not root.exists():
            self.skipTest("Optional arrival-cloud fixture is absent")
        for name, obscured in (("00016-battle-candidate.png", True), ("00017-deploy-west-edge.png", True),
                               ("00018-deploy-west-edge.png", False), ("00019-deploy-west-edge.png", False)):
            with self.subTest(frame=name):
                self.assertIs(measure_cloud_cover(root/name)["obscured"], obscured)

    def test_white_hud_outside_map_does_not_hide_clear_map(self):
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        image[150:490, 240:1050] = (50, 115, 77)
        with TemporaryDirectory() as temp:
            path = Path(temp)/"clear-map.png"
            cv2.imwrite(str(path), image)
            self.assertFalse(measure_cloud_cover(path)["obscured"])
            image[:] = 240
            cv2.imwrite(str(path), image)
            self.assertTrue(measure_cloud_cover(path)["obscured"])


class TerrainTests(unittest.TestCase):
    def test_thin_antialiased_boundary_can_use_two_pixel_support(self):
        from autococ.terrain import find_line_deployment_edges

        edges = find_line_deployment_edges(Path(__file__).parent / "fixtures/line-boundary-antialias.png")
        self.assertEqual(len(edges), 4)
        for edge in edges:
            self.assertLess(edge["evidence"]["intercept"], 510 if edge["edge"] == 0 else 215)
            self.assertGreater(edge["evidence"]["intercept"], 460 if edge["edge"] == 0 else 180)

    def test_purple_scenery_uses_wrapped_red_boundary_instead_of_inner_roofs(self):
        from autococ.terrain import find_line_deployment_edges

        edges = find_line_deployment_edges(Path(__file__).parent / "fixtures/line-boundary-purple.png")
        self.assertEqual(len(edges), 4)
        for edge in edges:
            self.assertLess(edge["evidence"]["intercept"], 490 if edge["edge"] == 0 else 260)
            self.assertGreater(edge["evidence"]["intercept"], 455 if edge["edge"] == 0 else 220)

    def test_shifted_live_boundary_moves_the_single_line_outside_red(self):
        from autococ.terrain import find_line_deployment_edges

        path = Path(__file__).parent / "fixtures/line-boundary-shifted.png"
        edges = find_line_deployment_edges(path)
        self.assertEqual(len(edges), 4)
        for item in edges[:2]:
            x, y = item["point"]
            self.assertLess(x, (445 - y) / .75 - 20)
        self.assertLess(edges[0]["point"][0], 450)

    def test_short_sunlit_boundary_is_not_replaced_by_long_interior_roofs(self):
        from autococ.terrain import find_line_deployment_edges

        edges = find_line_deployment_edges(Path(__file__).parent / "fixtures/line-boundary-short.png")
        self.assertEqual(len(edges), 4)
        self.assertLess(edges[0]["evidence"]["intercept"], 505)
        self.assertGreater(edges[0]["evidence"]["intercept"], 470)

    def test_lower_flank_stays_outside_short_western_boundary(self):
        from autococ.terrain import find_line_deployment_edges

        # Redacted from the live battle in which the old lower-flank point
        # landed inside the red deployment boundary and consumed no troop.
        path = Path(__file__).parent / "fixtures/line-boundary-western-outside.png"
        edges = find_line_deployment_edges(path)
        self.assertEqual(len(edges), 4)
        for edge in edges[2:]:
            x, y = edge["point"]
            outer_x = 186 + (y - 390) * 54 / 40
            self.assertLess(x + 13, outer_x)

    def test_short_visible_outer_edge_vetoes_inner_lower_line(self):
        import cv2
        import numpy as np
        from autococ.terrain import find_line_deployment_edges

        image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        cv2.line(image, (300, 275), (500, 125), (25, 40, 180), 2)
        cv2.line(image, (250, 368), (410, 488), (25, 40, 180), 2)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "boundary.png"
            cv2.imwrite(str(path), image)
            self.assertEqual(len(find_line_deployment_edges(path)), 4)
            # Too short to be proposed as a flank itself, but enough to show
            # that the otherwise valid lower line lies inside the perimeter.
            cv2.line(image, (185, 389), (215, 411), (25, 40, 180), 2)
            cv2.imwrite(str(path), image)
            self.assertEqual(find_line_deployment_edges(path), [])

    def test_line_edges_follow_current_boundary_translation(self):
        import cv2
        import numpy as np
        from autococ.terrain import find_line_deployment_edges

        with TemporaryDirectory() as directory:
            results = []
            for shift in (0, 40):
                image = np.zeros((720, 1280, 3), np.uint8)
                for start, end in (((300, 275), (500, 125)), ((250, 368), (410, 488))):
                    cv2.line(image, (start[0] + shift, start[1]), (end[0] + shift, end[1]), (25, 40, 180), 2)
                path = Path(directory) / f"{shift}.png"
                cv2.imencode(".png", image)[1].tofile(path)
                result = find_line_deployment_edges(path)
                self.assertEqual(len(result), 4)
                for point in result:
                    proof = point["evidence"]
                    x, y = point["point"]
                    boundary = (y - proof["intercept"]) / proof["slope"]
                    self.assertGreater(boundary - x, 30)
                results.append(result)
            self.assertAlmostEqual(results[1][0]["point"][0] - results[0][0]["point"][0], 40, delta=3)

    def test_missing_red_edges_do_not_produce_fixed_coordinates(self):
        import cv2
        import numpy as np
        from autococ.terrain import find_line_deployment_edges

        with TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            cv2.imencode(".png", np.zeros((720, 1280, 3), np.uint8))[1].tofile(path)
            self.assertEqual(find_line_deployment_edges(path), [])

    def assert_geometry_only(self, candidates: list[dict]) -> None:
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 3)
        for candidate in candidates:
            x, y = candidate["point"]
            evidence = candidate["evidence"]
            self.assertTrue(225 <= x <= 760 and 165 <= y <= 470)
            self.assertTrue(evidence["ground_unverified"])
            self.assertEqual(evidence["confidence_meaning"], "boundary_geometry_only")
            self.assertEqual(evidence["local_boundary_row_support"], 1)
            self.assertLess(candidate["bbox"][2], evidence["western_edge_at_1280x720"])
            self.assertFalse(evidence["deployment_confirmed"])

    def test_visible_west_corner_produces_points_outside_both_segments(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        cv2.polylines(image, [np.array([[600, 152], [350, 340], [600, 528]])], False, (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.png"
            cv2.imwrite(str(path), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
            candidates = find_west_deployment_points(path)
        self.assertGreaterEqual(len(candidates), 3)
        for candidate in candidates:
            x, y = candidate["point"]
            boundary_x = 350 + abs(y - 340) / (188 / 250)
            self.assertLess(x, boundary_x - 25)
            self.assertFalse(candidate["evidence"]["deployment_confirmed"])

    def test_red_decoration_or_missing_boundary_is_not_a_deployment_region(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        cv2.rectangle(image, (300, 220), (345, 260), (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.png"
            cv2.imwrite(str(path), image)
            self.assertEqual(find_west_deployment_points(path), [])

    def test_disconnected_outer_edge_vetoes_an_inner_red_polyline(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        inner = np.array([[760, 180], [720, 210], [740, 225], [700, 255],
                          [720, 270], [680, 300], [700, 315], [660, 345],
                          [680, 360], [640, 390]])
        cv2.polylines(image, [inner], False, (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inner.png"
            cv2.imwrite(str(path), image)
            self.assertGreater(len(find_west_deployment_points(path)), 0)
            # Separate short segments cannot independently propose deployment,
            # but their location proves these inner candidates are not outside.
            for y in range(150, 421, 42):
                cv2.line(image, (230, y), (190, y + 30), (24, 60, 172), 2)
            cv2.imwrite(str(path), image)
            self.assertEqual(find_west_deployment_points(path), [])

    def test_observed_bobby_roof_is_not_a_western_deployment_boundary(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/20260923-234612-265649-c577d13b/frames/00010-battle-candidate.png"
        if not path.is_file():
            self.skipTest("Optional Bobby deployment-failure fixture is absent")
        # The old candidates (672,189), (642,213), (660,237) were inside the
        # actual red perimeter; every issued probe left the troop count at 10.
        self.assertEqual(find_west_deployment_points(path), [])

    def test_observed_west_grass_is_detected_without_hardcoded_point(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/edge-002701/frames/00008-west-edge.png"
        if not path.is_file():
            self.skipTest("Optional live west-edge fixture is absent")
        candidates = find_west_deployment_points(path)
        self.assertGreaterEqual(len(candidates), 3)
        self.assertTrue(all(item["evidence"]["grass_fraction"] >= .97 for item in candidates))
        self.assertFalse(any(item["evidence"].get("ground_unverified") for item in candidates))

    def test_bright_scenery_grass_is_detected_but_nearby_obstacles_are_excluded(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/edge-refreshed-010415/frames/00001-before.png"
        if not path.is_file():
            self.skipTest("Optional bright-scenery fixture is absent")
        candidates = find_west_deployment_points(path)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(any(abs(item["point"][0] - 326) < 10 and abs(item["point"][1] - 339) < 5 for item in candidates))
        # Cakes, trees and logs occupy these boundary-adjacent strips in this fixture.
        self.assertFalse(any(item["point"][1] in range(295, 310) for item in candidates))
        self.assertFalse(any(item["point"][1] in range(258, 273) for item in candidates))
        self.assertFalse(any(item["point"][1] in range(403, 419) for item in candidates))
        self.assertTrue(all(item["evidence"]["texture_contrast"] <= .15 for item in candidates))

    def test_coarse_ground_only_proposes_unverified_geometry_probes(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (30, 90, 40), dtype=np.uint8)
        for left in range(0, 1280, 16):
            image[:, left:left + 8] = (70, 220, 100)
        cv2.polylines(image, [np.array([[600, 152], [350, 340], [600, 528]])], False, (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.png"
            cv2.imwrite(str(path), image)
            # A map tap on unknown ground is only a probe; deployment still requires
            # an observed troop-count decrease, never this appearance classification.
            self.assert_geometry_only(find_west_deployment_points(path))

    def test_observed_notched_brown_flank_proposes_only_clear_dirt(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/live-20260922/battle-complete-004727/frames/00008-deploy-west-edge.png"
        if not path.is_file():
            self.skipTest("Optional brown-scenery fixture is absent")
        candidates = find_west_deployment_points(path)
        self.assertGreaterEqual(len(candidates), 1)
        for item in candidates:
            x, y = item["point"]
            self.assertTrue(395 <= x <= 470 and 225 <= y <= 280)
            evidence = item["evidence"]
            self.assertEqual(evidence["method"], "two_notched_red_flank_segments_and_clear_dirt")
            self.assertGreaterEqual(evidence["ground_fraction"], .98)
            self.assertGreaterEqual(min(evidence["line_support"]), .9)
            self.assertFalse(evidence["deployment_confirmed"])
            left, top, right, bottom = evidence["segments_at_1280x720"][0]
            boundary_x = left + (y - top) * (right - left) / (bottom - top)
            self.assertLessEqual(item["bbox"][2], boundary_x - 15)

    def test_dirt_requires_two_distinct_adjacent_boundary_segments(self) -> None:
        import cv2
        import numpy as np

        base = np.full((720, 1280, 3), (72, 100, 97), dtype=np.uint8)
        cv2.line(base, (350, 335), (527, 202), (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.png"
            cv2.imwrite(str(path), base)
            self.assertEqual(find_west_deployment_points(path), [])
            cv2.line(base, (486, 206), (627, 100), (24, 60, 172), 2)
            cv2.imwrite(str(path), base)
            self.assertGreaterEqual(len(find_west_deployment_points(path)), 1)
            # Same geometry on rough rock/statue-like bands is not clear dirt.
            for left in range(220, 475, 8):
                base[220:325, left:left + 4] = (30, 45, 43)
            cv2.imwrite(str(path), base)
            self.assertEqual(find_west_deployment_points(path), [])

    def test_observed_short_zigzag_boundary_produces_grass_outside_its_frontier(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/20260923-015217-869637-e2129181/frames/00013-deploy-west-edge.png"
        if not path.is_file():
            self.skipTest("Optional short-zigzag fixture is absent")
        candidates = find_west_deployment_points(path)
        self.assertGreater(len(candidates), 0)
        for item in candidates:
            evidence = item["evidence"]
            self.assertEqual(evidence["method"], "open_red_polyline_western_frontier_and_clear_grass")
            self.assertLess(item["bbox"][2], evidence["western_edge_at_1280x720"])
            self.assertGreaterEqual(evidence["diagonal_coverage"], .8)
            self.assertGreaterEqual(evidence["grass_fraction"], .97)
            self.assertLessEqual(evidence["enclosed_area_fraction"], .15)
            self.assertFalse(evidence["deployment_confirmed"])

    def test_short_polyline_points_follow_translation_and_resolution(self) -> None:
        import cv2
        import numpy as np

        vertices = np.array([[620, 180], [580, 210], [600, 225], [560, 255], [580, 270],
                             [540, 300], [560, 315], [520, 345], [540, 360], [500, 390]])
        with TemporaryDirectory() as tmp:
            results = []
            for index, offset in enumerate(((0, 0), (48, 24))):
                image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
                cv2.polylines(image, [vertices + offset], False, (24, 60, 172), 2)
                path = Path(tmp) / f"terrain-{index}.png"
                cv2.imwrite(str(path), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
                results.append(find_west_deployment_points(path))
        self.assertGreater(len(results[0]), 0)
        self.assertEqual([item["point"] for item in results[1]],
                         [[item["point"][0] + 48, item["point"][1] + 24] for item in results[0]])
        self.assertTrue(all(item["evidence"]["method"].startswith("open_red_polyline") for item in results[0]))

    def test_short_polyline_rejects_invalid_geometry_even_on_unknown_ground(self) -> None:
        import cv2
        import numpy as np

        vertices = np.array([[620, 180], [580, 210], [600, 225], [560, 255], [580, 270],
                             [540, 300], [560, 315], [520, 345], [540, 360], [500, 390]])
        base = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        closed = base.copy()
        cv2.polylines(closed, [np.array([[350, 250], [410, 205], [470, 250], [410, 295]])], True, (24, 60, 172), 2)
        isolated = base.copy()
        cv2.line(isolated, (350, 335), (527, 202), (24, 60, 172), 2)
        occluded = base.copy()
        cv2.polylines(occluded, [vertices], False, (24, 60, 172), 2)
        for x, y in vertices:
            cv2.rectangle(occluded, (int(x) - 9, int(y) - 9), (int(x) + 9, int(y) + 9), (60, 60, 60), -1)
        rough = base.copy()
        for left in range(0, 1280, 16):
            rough[:, left:left + 8] = (70, 220, 100)
        cv2.polylines(rough, [vertices], False, (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            for name, image in (("closed", closed), ("isolated", isolated), ("occluded", occluded), ("rough", rough)):
                with self.subTest(case=name):
                    path = Path(tmp) / f"{name}.png"
                    cv2.imwrite(str(path), image)
                    candidates = find_west_deployment_points(path)
                    if name == "rough":
                        self.assert_geometry_only(candidates)
                    else:
                        self.assertEqual(candidates, [])

    def test_flowered_yellow_ground_uses_geometry_without_changing_grass_thresholds(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/20260923-021120-861689-f14e87d9/frames/00012-deploy-west-edge.png"
        if not path.is_file():
            self.skipTest("Optional flowered-scenery fixture is absent")
        candidates = find_west_deployment_points(path)
        self.assert_geometry_only(candidates)
        self.assertTrue(all(item["evidence"]["grass_fraction"] < .97 for item in candidates))
        self.assertTrue(all(item["evidence"]["diagonal_coverage"] >= .9 for item in candidates))
        self.assertEqual(len(find_west_deployment_points(path, max_points=1)), 1)

    def test_unknown_dirt_requires_intact_local_boundary_before_probing(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (72, 100, 97), dtype=np.uint8)
        for left in range(0, 1280, 8):
            image[:, left:left + 4] = (30, 45, 43)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown-dirt.png"
            cv2.imwrite(str(path), image)
            self.assertEqual(find_west_deployment_points(path), [])
            cv2.line(image, (350, 335), (527, 202), (24, 60, 172), 2)
            cv2.line(image, (486, 206), (627, 100), (24, 60, 172), 2)
            cv2.imwrite(str(path), image)
            candidates = find_west_deployment_points(path)
        self.assert_geometry_only(candidates)
        self.assertTrue(all(item["evidence"]["method"] == "two_notched_red_flank_segments_ground_unverified"
                            for item in candidates))

    def test_clear_ground_is_preferred_over_unknown_ground_on_same_boundary(self) -> None:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8)
        # Leave the lower portion clear while making the upper ground unknowable.
        for left in range(0, 1280, 16):
            image[:335, left:left + 8] = (70, 220, 100)
        cv2.polylines(image, [np.array([[600, 152], [350, 340], [600, 528]])], False, (24, 60, 172), 2)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed-ground.png"
            cv2.imwrite(str(path), image)
            candidates = find_west_deployment_points(path)
        self.assertGreater(len(candidates), 0)
        self.assertFalse(any(item["evidence"].get("ground_unverified") for item in candidates))


class ClearGroundProbeTests(unittest.TestCase):
    def test_observed_map_without_detected_boundary_proposes_only_unverified_grass(self):
        path = Path(__file__).resolve().parents[1] / "reports/boundary-selection-20260924-024915-059899/frames/00004-selected-boundary.png"
        if not path.is_file():
            self.skipTest("Optional Uday grass fixture is absent")
        self.assertEqual(find_west_deployment_points(path), [])
        probes = find_clear_ground_probes(path)
        self.assertEqual(len(probes), 3)
        for probe in probes:
            x, y = probe["point"]
            self.assertTrue(225 <= x <= 760 and 165 <= y <= 470)
            self.assertFalse(probe["evidence"]["boundary_verified"])
            self.assertFalse(probe["evidence"]["deployment_confirmed"])
            self.assertEqual(probe["evidence"]["confidence_meaning"], "ground_appearance_only")
            self.assertGreaterEqual(probe["evidence"]["grass_fraction"], .97)

    def test_observed_red_roofs_do_not_produce_ground_probes(self):
        path = Path(__file__).resolve().parents[1] / "reports/20260923-234612-265649-c577d13b/frames/00010-battle-candidate.png"
        if not path.is_file():
            self.skipTest("Optional Bobby roof fixture is absent")
        self.assertEqual(find_clear_ground_probes(path), [])

    def test_candidates_follow_visible_grass_translation_and_output_resolution(self):
        import cv2
        import numpy as np

        base = np.full((720, 1280, 3), (90, 50, 30), dtype=np.uint8)
        patch = np.clip(np.array([50, 115, 77]) + np.random.default_rng(89).integers(-8, 9, (100, 130, 3)), 0, 255).astype(np.uint8)
        with TemporaryDirectory() as tmp:
            results = []
            for dx, dy in ((0, 0), (80, 30)):
                image = base.copy()
                image[330 + dy:430 + dy, 300 + dx:430 + dx] = patch
                path = Path(tmp) / f"grass-{dx}.png"
                cv2.imwrite(str(path), cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_NEAREST))
                results.append(find_clear_ground_probes(path, max_points=1))
            self.assertEqual(len(results[0]), 1)
            self.assertEqual(results[1][0]["point"], [results[0][0]["point"][0] + 80, results[0][0]["point"][1] + 30])
            scaled = find_clear_ground_probes(path, max_points=1, baseline_resolution=(2560, 1440))
            self.assertEqual(scaled[0]["point"], [v * 2 for v in results[1][0]["point"]])

    def test_flat_blank_and_high_contrast_images_are_not_ground(self):
        import cv2
        import numpy as np

        cases = [np.zeros((720, 1280, 3), dtype=np.uint8),
                 np.full((720, 1280, 3), (50, 115, 77), dtype=np.uint8),
                 np.random.default_rng(90).integers(0, 256, (720, 1280, 3), dtype=np.uint8)]
        with TemporaryDirectory() as tmp:
            for index, image in enumerate(cases):
                with self.subTest(case=index):
                    path = Path(tmp) / f"not-ground-{index}.png"
                    cv2.imwrite(str(path), image)
                    self.assertEqual(find_clear_ground_probes(path), [])

    def test_grass_colored_controls_outside_map_region_never_become_probes(self):
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (90, 50, 30), dtype=np.uint8)
        rng = np.random.default_rng(91)
        for left, top, right, bottom in ((0, 0, 180, 200), (400, 0, 800, 90), (900, 500, 1270, 710), (0, 570, 850, 720)):
            image[top:bottom, left:right] = np.clip(np.array([50, 115, 77]) + rng.integers(-8, 9, (bottom-top, right-left, 3)), 0, 255)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "controls.png"
            cv2.imwrite(str(path), image)
            self.assertEqual(find_clear_ground_probes(path), [])


class CameraMotionTests(unittest.TestCase):
    def test_observed_failed_pan_is_stationary_despite_new_building_labels(self) -> None:
        directory = Path(__file__).resolve().parents[1] / "reports/20260924-022202-413811-d4361b72/frames"
        before = directory / "00011-battle-candidate.png"
        after = directory / "00014-deploy-west-edge.png"
        if not before.is_file() or not after.is_file():
            self.skipTest("Optional stationary-camera fixture is absent")
        motion = measure_camera_motion(before, after)
        self.assertIs(motion["stationary"], True)
        self.assertGreater(motion["inliers"], 400)

    def test_translation_is_distinguished_from_stationary_map_at_two_resolutions(self) -> None:
        import cv2
        import numpy as np

        source = np.random.default_rng(417).integers(0, 256, (720, 1280), dtype=np.uint8)
        with TemporaryDirectory() as tmp:
            before, after = Path(tmp) / "before.png", Path(tmp) / "after.png"
            cv2.imwrite(str(before), source)
            for dx, dy in ((0, 0), (80, 0), (-60, 30)):
                with self.subTest(translation=(dx, dy)):
                    shifted = cv2.warpAffine(source, np.float32([[1, 0, dx], [0, 1, dy]]), (1280, 720))
                    cv2.imwrite(str(after), cv2.resize(shifted, (2560, 1440), interpolation=cv2.INTER_NEAREST))
                    motion = measure_camera_motion(before, after)
                    self.assertIs(motion["stationary"], dx == dy == 0)
                    self.assertAlmostEqual(motion["affine"][0][2], dx, delta=2)
                    self.assertAlmostEqual(motion["affine"][1][2], dy, delta=2)

    def test_missing_unrelated_or_transformed_map_does_not_authorize_a_retry(self) -> None:
        import cv2
        import numpy as np

        rng = np.random.default_rng(418)
        source = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
        cases = {
            "blank": np.zeros_like(source),
            "unrelated": rng.integers(0, 256, source.shape, dtype=np.uint8),
            "zoom": cv2.warpAffine(source, cv2.getRotationMatrix2D((640, 360), 0, 1.12), (1280, 720)),
            "rotation": cv2.warpAffine(source, cv2.getRotationMatrix2D((640, 360), 8, 1), (1280, 720)),
        }
        with TemporaryDirectory() as tmp:
            before, after = Path(tmp) / "before.png", Path(tmp) / "after.png"
            cv2.imwrite(str(before), source)
            for name, transformed in cases.items():
                with self.subTest(case=name):
                    cv2.imwrite(str(after), transformed)
                    self.assertIsNone(measure_camera_motion(before, after)["stationary"])


if __name__ == "__main__":
    unittest.main()
