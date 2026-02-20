# LLM Explanation: `label_config_b1_strict_nested.xml`

## Purpose

This Label Studio config collects:
- image-level usability
- region-level bird attributes
- deeper nested attributes only when earlier answers allow them

The structure is intentionally strict and conditional, so annotators only see relevant questions.

## Data Model

### Image-level field

- `image_status` (required, single choice)
  - `has_usable_birds`
  - `no_usable_birds`

### Region anchor

- `bird_bbox` rectangle label on `image`
  - label value: `Bird`

All nested region attributes depend on a selected `bird_bbox` region.

### Region-level fields (first layer)

- `readability` (required, single choice, per-region)
  - `readable` (default)
  - `occluded`
  - `unreadable`
- `specie` (required, single choice, per-region)
  - `correct` (default)
  - `incorrect`
  - `unsure`

### Region-level fields (second layer, conditional)

Shown only if both conditions are true:
- `readability != unreadable`
- `specie != incorrect`

Then show:
- `behavior` (required, single choice, per-region)
  - `resting`
  - `foraging`
  - `flying`
  - `backresting`
  - `preening`
  - `display`
  - `unsure`
- `substrate` (required, single choice, per-region)
  - `ground`
  - `water`
  - `air`
  - `unsure`

### Region-level fields (third layer, conditional)

Shown only if both conditions are true:
- `behavior in {resting, backresting}`
- `substrate in {ground, water}`

Then show:
- `legs` (required, single choice, per-region)
  - `one`
  - `two`
  - `unsure` (default)
  - `sitting`

## Conditional Logic (Normalized)

For each selected `bird_bbox` region:

1. Ask `readability` and `specie`.
2. Ask `behavior` and `substrate` iff:
   - `readability` is not `unreadable`
   - AND `specie` is not `incorrect`
3. Ask `legs` iff:
   - `behavior` is `resting` or `backresting`
   - AND `substrate` is `ground` or `water`

## Minimal Pseudocode

```text
require image_status

for each selected bird_bbox region:
  require readability
  require specie

  if readability != "unreadable" and specie != "incorrect":
    require behavior
    require substrate

    if behavior in {"resting", "backresting"} and substrate in {"ground", "water"}:
      require legs
```

## Implementation Notes

- `visibleWhen="region-selected" whenTagName="bird_bbox"` gates region UI to active region context.
- `choice-unselected` is used as a negative condition:
  - `choice-unselected readability=unreadable` means readability is anything except `unreadable`.
  - `choice-unselected specie=incorrect` means specie is anything except `incorrect`.
- `choice-selected` with comma-separated values acts like set membership.

