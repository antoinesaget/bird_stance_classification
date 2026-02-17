# Annotator Guide

## Sample batch import (50 images)
1. Run:
   `uv run python /Users/antoine/bird_leg/scripts/prepare_labelstudio_import.py --data-root "${BIRDS_DATA_ROOT}" --site-id scolop2 --count 50 --sample-mode first`
2. In Label Studio, import:
   `/Users/antoine/bird_leg/data/birds_project/labelstudio/imports/scolop2_sample50.tasks.json`

## Per-image workflow
1. Set `image_status` first:
   - `has_usable_birds`
   - `no_usable_birds`
2. Review pre-annotated bird boxes from ML backend.
3. Adjust each box if needed.
4. For each Bird region:
   - set `readability`
   - if `readable`: set `activity` and `support`
   - if `activity=standing`: set `legs` and `resting_back`

## Rules
- `image_status` is required for every image.
- `readability` is required for every Bird region.
- `legs` and `resting_back` are required only when standing.
- Use `unreadable` for birds that cannot be reliably interpreted.

## UI checks for conditional logic
- When `readability=unreadable`, `activity`, `support`, `legs`, `resting_back` must be hidden/disabled.
- When `readability=readable` and `activity!=standing`, `legs` and `resting_back` must be hidden/disabled.
- When `readability=readable` and `activity=standing`, `legs` and `resting_back` must be visible and required.

## Speed tips
- Keep keyboard focus in the label panel for rapid choice changes.
- Validate defaults before moving to next task.
- Accept predictions only when they match visible evidence.
