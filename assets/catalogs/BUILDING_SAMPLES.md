# Building sample review

`building_templates.json` deliberately contains no active samples yet. A screenshot
with rubble does not identify which building was destroyed. To enable a variant:

1. Review two independent battle reports and label one `air_defense` or
   `spell_factory` in both. Record its alive and destroyed screenshots at the
   same minimum zoom and fixed camera. If the exact building level is not
   established, use `"unknown_variant"`; do not guess a level.
2. Record a tight 1280×720 `[left, top, right, bottom]` box around that building
   in each frame. Use the same footprint for its alive and rubble states. Add a
   same-size negative crop showing unrelated terrain/building, and set
   `reviewed_label: true` only after visual inspection.
3. Save an annotation JSON with `client` and `variants`. Each variant needs
   `type`, `level`, `zoom: "minimum"`, `reviewed_label`, and five objects named
   `source_alive`, `source_destroyed`, `heldout_alive`,
   `heldout_destroyed`, `negative`; each object contains `frame` and `bbox`.
4. Run `python scripts/prepare_building_samples.py annotations.json`. The script
   requires both camera pairs to be stationary, positive held-out template
   scores ≥ 0.94, and cross-state/negative scores < 0.80. It writes templates
   and activates the manifest entry only if all variants pass. Source reports
   are read-only. Then run `pytest tests/test_building_vision.py` and inspect
   detections on additional battle frames.

Example variant:

```json
{
  "client": "home_village_client_version",
  "variants": [{
    "type": "air_defense", "level": "unknown_variant", "zoom": "minimum",
    "reviewed_label": true,
    "source_alive": {"frame": "reports/run_a/frames/alive.png", "bbox": [300, 200, 338, 234]},
    "source_destroyed": {"frame": "reports/run_a/frames/destroyed.png", "bbox": [300, 200, 338, 234]},
    "heldout_alive": {"frame": "reports/run_b/frames/alive.png", "bbox": [500, 300, 538, 334]},
    "heldout_destroyed": {"frame": "reports/run_b/frames/destroyed.png", "bbox": [500, 300, 538, 334]},
    "negative": {"frame": "reports/run_b/frames/alive.png", "bbox": [700, 300, 738, 334]}
  }]
}
```

Paths are resolved relative to the JSON file. The coordinates above only show
the format; they are not validated game samples.
