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
meta AS (
  SELECT *
  FROM read_parquet('{{metadata_parquet}}')
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
    b.legs,
    i.image_usable,
    m.filepath,
    m.site_id,
    abs(hash(coalesce(m.site_id, '') || ':' || b.image_id)) % 100 AS split_bucket
  FROM birds b
  LEFT JOIN images i
    ON b.annotation_version = i.annotation_version
   AND b.image_id = i.image_id
  LEFT JOIN meta m
    ON b.image_id = m.image_id
)
SELECT
  *,
  '{{crop_root}}/' || replace(bird_id, ':', '_') || '.jpg' AS crop_path
FROM joined
ORDER BY image_id, bird_id;
