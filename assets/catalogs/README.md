# Recognition samples and coverage

`unit_templates.json` is a versioned list of *observed portraits*, not a list of
unlocked or fully supported units. All current crops came from Clash of Clans
18.600.7 (versionCode 180600008) in MuMu at a 1280 × 720 baseline. Each entry
records its source frame, crop, surface, and card state. The corresponding
sanitized source fixtures are in `tests/fixtures/`; the original runtime frames
are under the ignored `reports/` directory. The unit registry in
`unit_catalog.py` is broader than these samples by design.

| Surface | Sampled IDs | Independent replay evidence | Remaining gaps |
| --- | --- | --- | --- |
| `army` | `meteor_golem`, `bowler`, `wall_breaker`, `archer`; `totem_spell`, `overgrowth_spell`, `revival_spell`, `freeze_spell`; large cards for `grand_warden`, `dragon_duke`, `minion_prince`, `archer_queen` | A second real My Army frame (`tests/fixtures/army_current_second_frame_20260926.png`) checks troop/spell cards. Two separately captured, sanitized left panels (`army_heroes_first_frame_20260926.png` and `army_heroes_second_frame_20260926.png`) verify hero identity independently of column and 12/12 pet/equipment fingerprints. Numeric counts in the separate confirmation frame are covered by `test_army_manifest.py`. | No independent all-unit catalog scan, selected/gray cards, horizontal scroll, siege identity, or editor `+`/`−` controls. Hero samples cover only the observed skins/animation range. |
| `battle` | The same four troops; `grand_warden`, `dragon_duke`, `minion_prince`, `archer_queen`; `totem_spell`, `overgrowth_spell`, `revival_spell` | A later candidate frame in `reports/daily-smoke-2/20260926-155242-563854-422f2415/frames/00013-battle-candidate.png` and a changed camera frame in `tests/fixtures/zoomed_battle_bar_20260926.png` retain the four troop identities/counts. The latter has a dedicated regression in `test_battle_cards.py`. | `freeze_spell`, other units, spent/gray cards, shifted bar pages, and same-portrait clan reinforcements lack independent identity samples. |
| `editor` / `picker` | None complete | Real UI frames are in `reports/live-army-probe/frames/` and `reports/live-army-editor-probe/frames/`. Large hero samples also recognize two visible saved-editor portraits, but the third remains unknown. | No complete editor or picker identity/control coverage. Automatic edits must report unavailable before mutation. |

The source frame matching its own crop is only a sanity check. It is **not**
independent validation. No sample proves a card's level, unlock status, current
activity, or quantity; these require their own current-frame evidence. In
particular, a catalogued event troop is not automatically available. Changed
hero equipment/pets are not part of a hero's identity. The large hero matcher
requires both a face template and an upper-card visual profile. If the
connected client version differs from the manifest, named template identity is
unavailable.

The currently observed `revival_spell` card was confirmed through the game's
information panel as `3级复苏法术` in
`reports/live-army-editor-probe/frames/00002-army-spell-info-after.png`.
`meteor_golem` is a permanent troop in this client, but its card identity has
only visual sample evidence; no direct in-game information panel capture has
yet independently confirmed the name. The production implementation must keep
that evidence distinction visible.

`village_type_provenance.json` records the positive Home Village and Builder
Base anchors, including the Home Village attack-button variant with stars.
Those templates guard navigation; absence of a Builder Base match alone cannot
establish Home Village.
