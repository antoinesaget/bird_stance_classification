# Legacy Note

This document described the older mixed host-backend workflow where Label Studio and ML could both run from the same mutable stack.

Use the current managed commands instead:

```bash
make iats-deploy-ml
make truenas-deploy-ui
make iats-import-exports PROJECT_ID=<id> ANNOTATION_VERSION=<ann_vXXX>
make smoke-remote
```
