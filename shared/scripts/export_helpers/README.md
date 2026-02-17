# Export Helpers

## COCO -> YOLO labels

```bash
python3 /Users/antoine/bird_leg/shared/scripts/export_helpers/coco_to_yolo.py \
  --coco-json /path/to/instances_default.json \
  --out-dir /path/to/yolo-dataset \
  --only-category bird
```

Outputs:
- `classes.txt`
- `labels/*.txt`
