# Label Studio

This subproject contains the surviving Label Studio integration code and the TrueNAS deployment assets.

## Surviving Entry Points

- `src/birdsys/labelstudio/extract_annotations.py`: canonical end-to-end extraction command for versioned raw exports, strict normalization, comparison reports, and plots
- `src/birdsys/labelstudio/export_snapshot.py`: create or download project exports from Label Studio
- `src/birdsys/labelstudio/import_tasks.py`: import prepared task bundles into a project
- `src/birdsys/labelstudio/create_project.py`: create or update a project from the repo label config and clone source project defaults
- `src/birdsys/labelstudio/prefill_predictions.py`: fetch tasks, call the ML backend, and write prediction rows back into Label Studio
- `src/birdsys/labelstudio/build_localfiles_batch.py`: create species-aware compressed local-files batches plus manifests for imports
- `src/birdsys/labelstudio/sync_species_data.py`: sync one species root from TrueNAS onto a local host before extraction or dataset builds
- `src/birdsys/labelstudio/backfill_task_metadata.py`: one-time metadata backfill for older Label Studio tasks using stored import manifests

## Surviving Assets

- `label_config.xml`: current annotation schema config
- `deploy/docker-compose.truenas.yml`: live TrueNAS Label Studio and Postgres deployment
- `deploy/env/truenas.env.example`: example env file for the TrueNAS deploy
- `deploy/overrides/localfiles_views.py`: local-files override mounted into the Label Studio container

## Current Shape

- The old `cli.py` wrapper is gone.
- The project is now a small collection of direct Python entrypoints plus deployment files.
- The API-oriented scripts plus the new extraction command are the main remaining operational surface.

## Current Status

- `export_snapshot.py` still imports cleanly from the remaining package tree.
- `extract_annotations.py` is now the intended operator surface for human-annotation extraction.
- `create_project.py`, `import_tasks.py`, and `prefill_predictions.py` are built on that same remaining Label Studio surface.
- `build_localfiles_batch.py` now writes the strict relative-path metadata required by extraction.
- TrueNAS deployment assets are still present and appear to be the intended production Label Studio surface.
