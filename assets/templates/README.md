# Resource collection templates

`battle_event_super_pekka.png` is the observed extra gray event troop portrait from
`reports/20260925-155528-356263-da52437c/frames/00012-deploy-west-edge.png`.
The 1280x720 `INTER_AREA` baseline crop is `(98, 627, 176, 679)`; it excludes the
quantity. A separate high-confidence `xN` reading is required. The battle-bar
regression fixture is cropped from the same calibration frame, not an independent
live deployment validation. This event card is not present in the My Army manifest.

These templates are cropped from the observed Chinese home village screenshot
`reports/live-20260922/request-open.png` captured on 2026-09-22. Despite that
exploration filename, the frame shows the village with a selected wall; it is
not a reinforcement request dialog.

The source frame is 2560x1440. It was resized to the 1280x720 baseline using
OpenCV `INTER_AREA` before taking these exclusive-right/bottom crops:

| Template | Baseline crop (left, top, right, bottom) |
| --- | --- |
| collect_gold.png | 715, 370, 755, 406 |
| collect_elixir.png | 601, 264, 641, 300 |
| collect_dark_elixir.png | 943, 279, 983, 315 |
| hud_army.png | 18, 488, 93, 558 |
| hud_chat.png | 18, 280, 94, 366 |

The crops include the resource icon and pale bubble border, rather than only
the icon that also appears in the balance HUD. Detection additionally excludes
the HUD corner, screen edges, and bottom building-action menu. Nearby peaks and
cross-scale matches are suppressed so each bubble is returned once.

These templates currently cover the observed UI appearance. Missing detection
does not prove that the village has no collectable resources outside the frame.

The army/chat templates were extracted by the root agent from the same source
and baseline. The army HUD match was used to open the real army panel, recorded
in `reports/live-20260922/army-probe/frames/00002-army-open.png`.

`chat_request.png` comes from the open clan chat screenshot
`reports/live-20260922/chat-probe/frames/00003-chat-open.png`, cropped at
baseline `(380, 647, 477, 709)`. It identifies navigation into the request dialog.
It does not prove request eligibility, receipt of troops, or a full clan castle.

`battle_hero_pet_0.png` through `battle_hero_pet_3.png` are the four visible
hero pet icons in `reports/live-20260922/scout-from-army-235605/frames/00007-scout-from-army.png`.
Their baseline crops are `(604,600,638,636)`, `(700,600,736,637)`,
`(796,600,831,637)`, and `(892,600,928,637)`. They distinguish the observed hero
cards from purple spell cards whose count OCR may be missing. An unmatched card
is `unknown`; a missing count does not establish that a card is a hero.

Pet matching now compares the opaque inner ellipse of each rounded icon instead
of its map-colored corners. Both card location and hero-state recognition use the
same matcher, with observed 1.0, 1.05, and 1.1 scales for selection animation.
This does not replace independent card borders or the hero health-bar check.

`chat_latest.png` is the green down-arrow navigation button from
`reports/20260923-011419-927154-a8f7a5a8/frames/00012-request-verify.png`,
resized to the same baseline and cropped at `(17,585,68,650)`.
It is matched only in confirmed clan chat, inside `(0,530,100,650)`.
It returns to recent messages; it never submits a message or request.

`settlement_gold.png`, `settlement_elixir.png`, and `settlement_dark_elixir.png`
come from the stable three-row defeat result
`reports/live-20260922/settlement-010735/frames/00008-end-confirmation.png`.
At 1280x720 their crops are `(674,310,715,355)`, `(674,358,715,403)`,
and `(674,405,715,450)`. Matching is restricted to the central result rows;
numbers are bound to the matching resource icon, not a fixed vertical row index.
The later gold/elixir-only result in
`reports/live-20260922/battle-current-013325/frames/00001-before.png`
has different row positions. An omitted dark-elixir row is provisionally zero
only for that fully verified layout and must pass return-inventory reconciliation.

The dark-elixir matcher excludes map background using the opaque drop-body ellipse
at template center `(21,26)`, radii `(14,16)`; the original 41x45 bounding box is
preserved for row alignment. The score threshold remains 0.9, non-finite scores
are rejected, and separated duplicate matches remain ambiguous. A new three-row
result at `reports/20260924-003409-344711-0e4a3e7a/frames/00044-battle-settlement.png`
was read and reconciled by the software after this change.

`enemy_gold.png`, `enemy_elixir.png`, and `enemy_dark_elixir.png` come from
`reports/20260923-012829-323923-b0c8cf92/frames/00010-battle-candidate.png`.
Their 1280x720 crops are `(31,99,59,129)`, `(31,137,59,167)`, and
`(34,178,57,203)`, using `INTER_AREA` for the original 2560x1440 frame.
The smaller dark-elixir crop excludes changing map background. The manually
inspected amounts are 1688575 gold, 897982 elixir, and 15439 dark elixir;
the nearby 94 is a defender hero level. Numeric rows are anchored to matched
icons and their font height/alignment; missing icons or ambiguous OCR stay unknown.

`hero_warden_active_book.png` and `hero_warden_ready_card.png` use
`reports/live-20260922/hero-state-023106/frames/00004-hero-placement-after.png`,
with baseline crops `(600,526,642,568)` and `(603,631,685,680)`.
`hero_warden_used_book.png` and `hero_warden_used_card.png` use the same crops in
`reports/live-20260922/warden-activate-023930/frames/00004-after-ability.png`.
Both templates and a separately verified card health bar are required for each
ability state. The activation-flash frame is unknown; gray portrait alone does
not prove ability use or defeat. These states cover the observed Warden equipment.

`hero_warden_ready_card_phase.png` adds the observed gold-book glow phase from
`reports/20260924-004347-675110-2e2b80bd/frames/00027-deploy-hero-verify.png`,
baseline crop `(506,631,588,680)`. It is an alternative lower-card template;
the separate active-book and enclosed health-bar requirements are unchanged.

`settlement_bonus_gold.png`, `settlement_bonus_elixir.png`, and
`settlement_bonus_dark_elixir.png` come from
`reports/20260924-004347-675110-2e2b80bd/frames/00061-battle-settlement-reread.png`.
After `INTER_AREA` resize to 1280x720, their crops are `(1007,349,1036,381)`,
`(1007,385,1036,418)`, and `(1007,421,1036,454)`. They locate the independent
right-side victory reward rows. A reward label and explicit plus-signed number
must also be present; missing rows never imply zero. This observed three-row
layout has one real earned-star sample; two/three-star shape tests are synthetic.

The `hero_barbarian_king_*`, `hero_minion_prince_*`, and `hero_archer_queen_*`
templates use controlled, one-click ability samples from
`reports/hero-ability-samples-20260924-015325-792650/frames/`:

| Hero prefix | Card left at baseline | Ready frame | Used frame |
| --- | ---: | --- | --- |
| `barbarian_king` | 599 | `00014-hero-deployed.png` | `00016-ability-after.png` |
| `minion_prince` | 696 | `00020-hero-deployed.png` | `00022-ability-after.png` |
| `archer_queen` | 792 | `00026-hero-deployed.png` | `00028-ability-after.png` |

Each prefix has `ready_equipment`, `ready_portrait`, `used_portrait`, and
`used_weapon` PNGs. After `INTER_AREA` resize, their `(x0,y0,x1,y1)` crops are
`(left,528,left+42,569)`, `(left+40,594,left+82,646)` for both portraits,
and `(left+35,647,left+82,679)`. Equipment buttons disappear after activation
in these samples. Used-state evidence therefore requires both gray portrait
and gray weapon, plus the independently enclosed green HP bar. Empty or
unrecognized red HP, enlarged transition frames, missing pieces, and reassigned
pets cannot independently establish an ability state. These templates cover
the observed portraits/equipment only; development samples are not completed
automated battles.

`siege_deployed_release.png` comes from
`reports/20260923-021847-763016-3632dc81/frames/00015-deploy-support-selected.png`,
crop `(531,658,576,701)`. It verifies deployment only together with the independent
siege-card health bar. The automation never clicks this release control.

Release matching now uses the eroded green-arrow/white-passenger foreground,
excluding map corners. Its search extends down to baseline y=720 to accommodate
the observed vertical animation; a complete independently framed HP bar is still
required, including the observed narrower 72-pixel layout. The 00:43 live failures
remain unchanged; the later crop and pet-background fixes are offline-verified.

`battle_clan_badge.png` comes from the already saved manual-play observation
`reports/live-20260922/reconnected-precheck-230909/frames/00001-before.png`,
crop `(1153,594,1175,618)`. It marks an available reinforcement quantity card;
the configured clan capacity alone does not establish receipt of reinforcements.
This frame is not an automated battle result. All crops use the 1280x720 baseline
and `INTER_AREA` resizing from their original resolution.
