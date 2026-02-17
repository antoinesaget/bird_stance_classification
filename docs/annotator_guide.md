# Annotator Guide

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

## Speed tips
- Keep keyboard focus in the label panel for rapid choice changes.
- Validate defaults before moving to next task.
- Accept predictions only when they match visible evidence.
