# Annotator Guide

## Sample batch import (50 images)
1. Run:
   `uv run python /Users/antoine/bird_leg/scripts/prepare_labelstudio_import.py --data-root "${BIRDS_DATA_ROOT}" --site-id scolop2 --count 50 --sample-mode first`
2. In Label Studio, import:
   `/Users/antoine/bird_leg/data/birds_project/labelstudio/imports/scolop2_sample50.tasks.json`

## Refresh labeling UI on an existing project
1. Open project `Settings` -> `Labeling Interface` -> `Code`.
2. Replace with `/Users/antoine/bird_leg/labelstudio/label_config.xml` and save.
3. Hard refresh the browser tab.
4. Confirm the layout is 2-column: image on the left, all radio controls on the right.

## Required Label Studio project settings
- `Interactive preannotations`: disabled.
- `Auto-Accept Suggestions`: disabled in the labeling UI footer.
- `Show predictions to annotators`: optional, but disable it if you want to avoid tab confusion between prediction and manual annotation.

## Per-image workflow
1. Set `image_status` first:
   - `has_usable_birds`
   - `no_usable_birds`
2. Review pre-annotated bird boxes from ML backend.
3. Adjust each box if needed.
4. For each Bird region (Bird region selected):
   - set `readability`
   - if `readable`: set `activity` and `support`
   - if `activity=standing`: set `legs` and `resting_back`

## Rules
- `image_status` is required for every image.
- `readability` is required for every Bird region.
- `activity`, `support`, `legs`, and `resting_back` are optional in UI and masked downstream by normalization/training logic.
- Use `unreadable` for birds that cannot be reliably interpreted.

## UI checks for persistence
- Open one task from the table (single-task mode), edit a predicted attribute, click `Update`.
- Reopen the same task and select the same Bird region; edited values must persist.
- Deleting a prediction should not recreate it immediately when interactive preannotations is disabled.

## Speed tips
- Keep keyboard focus in the label panel for rapid choice changes.
- Validate defaults before moving to next task.
- Accept predictions only when they match visible evidence.
