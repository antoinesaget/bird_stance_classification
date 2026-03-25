# Black Wing Stilts 1: 5k Random Import From `lines_project`

## Summary
- Use the existing TrueNAS Label Studio project `Black Wing Stilts 1` (`project 7`), which is currently empty, and import a JSON task file into it.
- Serve images from a compressed annotation mirror rooted in `/mnt/tank/media/lines_project`, referenced through `/data/local-files/?d=...`.
- Keep the serving format as JPEG, not WebP.
  - Reason: Label Studio officially supports `.jpg`, `.png`, `.webp`, etc., but this corpus is already almost entirely JPEG, browser decode wins from WebP are not guaranteed, and JPEG keeps the lowest rollout risk with the current local-files path and existing tooling.
  - The useful optimization here is smaller bytes at unchanged resolution, not a format migration.
- Use a fixed `q60` JPEG mirror for annotation serving.
  - Decision basis: the local 10-image sweep on March 25, 2026 showed `q60` at mean SSIM `0.9200` with total size ratio `0.1228`, which is a strong reduction for UI responsiveness while staying visually usable for annotation.
- Select `5,000` images with deterministic uniform random sampling over the full eligible corpus, and preserve an explicit manifest mapping each imported task back to the original file.

## Implementation Changes
### Image Format + Compression
- Treat `/mnt/tank/media/lines_project/raw_images` as the canonical source.
- Build a dedicated annotation mirror under `/mnt/tank/media/lines_project/labelstudio/images_compressed/lines_bw_stilts_q60/`.
- Keep output as progressive JPEG with original pixel dimensions unchanged.
- Convert the small non-JPEG tail (`png`, `tif`, `tiff`) into JPEG in the mirror so the imported batch is homogeneous.
- The compression sweep is complete and the implementation target is now fixed:
  - JPEG quality: `60`
  - progressive JPEG: `enabled`
  - resolution: unchanged
  - output format: `.jpg`
- Do not use WebP unless a later explicit experiment shows a material gain with no rendering issues. It is out of scope for this rollout.

### Sampling + Mapping
- Eligible files are image files under `/mnt/tank/media/lines_project/raw_images`.
- Use deterministic uniform random sampling with a fixed seed over the full eligible set.
- Sample size is `5,000`.
- Because filenames are already effectively random and EXIF is widely present across makes/dates, uniform random is sufficient for this first pass and avoids adding stratification complexity.
- Produce a manifest file alongside the import artifacts containing, at minimum:
  - `sample_index`
  - `sample_seed`
  - `original_absolute_path`
  - `original_relative_path`
  - `served_relative_path`
  - `served_url`
  - `original_bytes`
  - `served_bytes`
- Guarantee no duplicates in the sampled set.
- Keep the sampled manifest stable: same seed + same source tree => same 5k selection.

### Import Flow
- Import into project `7` via JSON task import, not via new project creation and not via local storage sync.
- Extend or add repo tooling so it supports a flat image directory, because current helpers assume site-based subdirectories.
- Generate a JSON task file whose `image` field points to the compressed mirror through local-files URLs, for example:
  - `/data/local-files/?d=lines_project/labelstudio/images_compressed/lines_bw_stilts_q60/<filename>.jpg`
- Put the source mapping into `meta` so it remains visible after import, including:
  - original relative path
  - served relative path
  - sampling seed
  - import batch name
- Name the batch deterministically, for example `lines_bw_stilts_5k_seed_<seed>_q<quality>`.
  - For this rollout, use `lines_bw_stilts_5k_seed_<seed>_q60`.

### Repo Surface
- Add or adapt one repo-owned script to:
  - enumerate eligible files from a flat root
  - optionally run the JPEG sweep on a sample
  - build the compressed mirror
  - select the 5k deterministic random sample
  - emit the manifest CSV/JSON
  - emit the Label Studio import JSON
- Keep the import artifact and manifest under a versioned area in the dataset tree or repo-controlled ops output directory so the batch can be reproduced without guessing.

## Public Interfaces / Outputs
- New import artifact for project 7:
  - `tasks.json` for Label Studio import
- New mapping artifact:
  - `manifest.csv` and/or `manifest.json`
- New compressed annotation mirror:
  - JPEG-only, same resolution, project-specific output root
- Script interface should accept:
  - `--source-root`
  - `--served-root`
  - `--sample-size`
  - `--seed`
  - `--jpeg-quality`
  - `--project-name` or `--batch-name`
  - `--url-prefix`

## Test Plan
- Format decision:
  - verify the generated mirror is encoded at `q60`
  - verify byte reduction is in the same range as the recorded sweep (`~12.3%` of original bytes on the 10-image sample)
  - verify resulting files open correctly in a browser and in Label Studio via local-files URLs
- Sampling:
  - rerun with the same seed and confirm identical 5k manifest
  - rerun with a different seed and confirm materially different selection
  - confirm no duplicates
- Mapping:
  - confirm every imported task has a matching manifest row
  - confirm every manifest row points to both a real source file and a real served file
- Import:
  - import a small canary batch of `50` tasks into project `7` first
  - verify images load quickly and render correctly
  - verify `meta` preserves original-path mapping
  - once the canary passes, import the full `5,000`
- Performance sanity:
  - compare median source bytes vs served bytes on the sampled set
  - confirm the mirror meaningfully reduces transfer size without resizing

## Assumptions
- The destination is the existing Label Studio project `Black Wing Stilts 1` (`project 7`).
- The desired batch size is `5,000`, not `2,000`.
- Uniform random sampling is preferred over EXIF-stratified sampling for this rollout.
- The best near-term choice is JPEG recompression, not WebP migration.
- The chosen mirror quality for this rollout is fixed at `q60`.
- Resolution must remain unchanged; only encoding/compression may change.
