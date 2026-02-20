# E2E Sign-Off: `ann_v902`

Date: 2026-02-20  
Source export: `/Users/antoine/bird_leg/project-4-at-2026-02-20-10-24-00d4bdd1.json`

## Run summary
- Normalization: `ann_v902`
- Crops: `/Users/antoine/bird_leg/data/birds_project/derived/crops/ann_v902/`
- Dataset build: `/Users/antoine/bird_leg/data/birds_project/derived/datasets/ds_v905/`
- Determinism rebuild: `/Users/antoine/bird_leg/data/birds_project/derived/datasets/ds_v906/` (`same_assignments=True`)
- Model B smoke training: `/Users/antoine/bird_leg/data/birds_project/models/attributes/convnextv2s_v004/`
- Model C smoke training: `/Users/antoine/bird_leg/data/birds_project/models/image_status/status_v003/`
- Backend verify with new checkpoints: PASS (FastAPI TestClient `/health` + `/predict`)

## Key outputs
- Normalized files:
  - `/Users/antoine/bird_leg/data/birds_project/labelstudio/normalized/ann_v902/images_labels.parquet`
  - `/Users/antoine/bird_leg/data/birds_project/labelstudio/normalized/ann_v902/birds.parquet`
- Dataset manifest:
  - `/Users/antoine/bird_leg/data/birds_project/derived/datasets/ds_v905/manifest.json`
- Training artifacts:
  - `/Users/antoine/bird_leg/data/birds_project/models/attributes/convnextv2s_v004/checkpoint.pt`
  - `/Users/antoine/bird_leg/data/birds_project/models/image_status/status_v003/checkpoint.pt`

## Notes
- Metadata index was extended to include site `wetransfer_641831510-jpg_2026-02-17_1739`, bringing total indexed images to `10037`.
- Export includes cancelled annotations (`was_cancelled=true`) for skipped tasks; this is preserved by Label Studio export behavior.
