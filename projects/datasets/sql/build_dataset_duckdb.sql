-- Purpose: Capture the historical DuckDB query shape for building the dataset tables.
-- BirdSys dataset assembly template.
-- Runtime values are injected by birdsys datasets build-dataset.

WITH birds AS (
  SELECT *
  FROM read_parquet('{{birds_parquet}}')
),
images AS (
  SELECT *
  FROM read_parquet('{{images_labels_parquet}}')
),
joined AS (
  SELECT
    b.annotation_version,
    b.image_id,
    b.bird_id,
    b.bbox_x,
    b.bbox_y,
    b.bbox_w,
    b.bbox_h,
    b.isbird,
    b.readability,
    b.specie,
    b.behavior,
    b.substrate,
    b.stance,
    i.species_slug,
    i.source_filename,
    i.original_relpath,
    i.compressed_relpath,
    i.labelstudio_localfiles_relpath,
    i.compression_profile,
    i.image_usable,
    i.site_id,
    abs(hash(coalesce(i.site_id, '') || ':' || b.image_id)) % 100 AS split_bucket
  FROM birds b
  LEFT JOIN images i
    ON b.annotation_version = i.annotation_version
   AND b.image_id = i.image_id
)
SELECT
  *,
  '{{crop_root}}/' || replace(bird_id, ':', '_') || '.jpg' AS crop_path
FROM joined
ORDER BY image_id, bird_id;
