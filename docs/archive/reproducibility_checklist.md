# Reproducibility Checklist

- [ ] Raw images under `/data/birds_project/raw_images` are immutable.
- [ ] Label Studio exports are append-only in `/data/birds_project/labelstudio/exports`.
- [ ] Normalized exports use versioned folders: `ann_vXXX`.
- [ ] Datasets use versioned folders: `ds_vXXX`.
- [ ] Model outputs use versioned folders under:
  - `/data/birds_project/models/attributes`
  - `/data/birds_project/models/image_status`
- [ ] Every dataset has `manifest.json` with split policy + source annotation version.
- [ ] Every model version has `config.yaml`, `metrics.json`, and `checkpoint.pt`.
- [ ] Re-running normalization on same export produces deterministic rows and values.
- [ ] Train/val/test image IDs are disjoint.
