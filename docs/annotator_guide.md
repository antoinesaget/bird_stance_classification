# Annotator Guide

## Sample batch import (50 images)
1. Run:
   `uv run python /Users/antoine/bird_leg/scripts/prepare_labelstudio_import.py --data-root "${BIRDS_DATA_ROOT}" --site-id scolop2 --count 50 --sample-mode first --image-relative-root raw_images`
2. In Label Studio, import:
   `/Users/antoine/bird_leg/data/birds_project/labelstudio/imports/scolop2_sample50.tasks.json`

## Label config variants to try
- Baseline strict (recommended first):
  `/Users/antoine/bird_leg/labelstudio/label_config.xml`
- Strict nested variant:
  `/Users/antoine/bird_leg/labelstudio/variants/label_config_b1_strict_nested.xml`
- Strict collapse variant:
  `/Users/antoine/bird_leg/labelstudio/variants/label_config_b2_strict_collapse.xml`
- Strict compact variant:
  `/Users/antoine/bird_leg/labelstudio/variants/label_config_b3_strict_compact.xml`

## Refresh labeling UI on an existing project
1. Open project `Settings` -> `Labeling Interface` -> `Code`.
2. Replace with one config file above and save.
3. Hard refresh the browser tab.
4. Confirm image is left and controls are right.

## Required Label Studio project settings
- `Interactive preannotations`: disabled.
- `Use predictions to prelabel tasks`: enabled.
- `Auto-Accept Suggestions`: disabled in the labeling UI footer.

## Per-image workflow (new schema)
1. Review and adjust Bird boxes.
2. For each Bird region, in this order:
   - `isbird`: `yes` / `no`
   - if `isbird = yes`:
     - `readability`: `readable` / `occluded` / `unreadable`
     - `specie`: `correct` / `incorrect` / `unsure`
     - if `readability != unreadable` and `specie != incorrect`:
       - `behavior`: `flying` / `foraging` / `resting` / `backresting` / `preening` / `display` / `unsure` (optional in UI)
       - `substrate`: `ground` / `water` / `air` / `unsure` (optional in UI)
       - if `behavior in {resting, backresting}` and `substrate in {ground, water, unsure}`:
         - `legs`: `one` / `two` / `unsure`

## Rules
- `isbird` is always required per bbox.
- If `isbird=no`, all other per-bird fields are ignored.
- If `isbird=yes`, `readability` and `specie` are required.
- If `readability=unreadable`, other per-bird fields are irrelevant.
- If `specie=incorrect`, other per-bird fields are irrelevant.
- `legs` is only meaningful for resting/backresting on ground/water/unsure.

## Manual validation matrix (for each config variant)
- `prefill_bbox_visible`: PASS/FAIL
- `prefill_attributes_visible`: PASS/FAIL
- `manual_values_persist_after_update`: PASS/FAIL
- `label_all_tasks_consistent`: PASS/FAIL
- `scrolling_needed`: PASS/FAIL
- `subjective_speed_1_to_5`: 1..5

## Speed tips
- Keep keyboard focus in label panel for hotkey-driven labeling.
- Validate one full task before bulk labeling.
- Accept predictions only when visually correct.

## Bulk apply for flock images (custom helper UI)
- Install and use:
  `/Users/antoine/bird_leg/docs/labelstudio_bulk_apply_helper.md`
- Script source:
  `/Users/antoine/bird_leg/labelstudio/tools/bulk_apply_selected_birds.user.js`
