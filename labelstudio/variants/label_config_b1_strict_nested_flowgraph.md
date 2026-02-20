# `label_config_b1_strict_nested.xml` Classification Graph

```mermaid
flowchart TD
  A["Image classification"] --> I{"image_status"}
  I -- "no_usable_birds" --> I0["Image class = no_usable_birds"]
  I -- "has_usable_birds" --> I1["Image class = has_usable_birds"]

  I1 --> R0["Per bird region (bird_bbox): classify attributes"]
  R0 --> R{"readability"}
  R -- "unreadable" --> T1["Terminal region class: readability=unreadable"]
  R -- "readable or occluded" --> S{"specie"}

  S -- "incorrect" --> T2["Terminal region class: specie=incorrect"]
  S -- "correct or unsure" --> B{"behavior"}

  B --> U{"substrate"}
  U --> G{"behavior in {resting, backresting} AND substrate in {ground, water}?"}

  G -- "Yes" --> T3["Terminal fine class: behavior + substrate + legs"]
  G -- "No" --> T4["Terminal class: behavior + substrate"]

  T3 --> L["legs in {one, two, unsure, sitting}"]
```

## Interpretation

- The image has a top-level class: `image_status`.
- Region taxonomy is only meaningful when `image_status=has_usable_birds`.
- Region classification has two early-stop terminal classes:
  - `readability=unreadable`
  - `specie=incorrect`
- Otherwise, the region class is defined by:
  - `behavior` + `substrate`
  - plus `legs` only in the resting/backresting on ground/water subspace.
