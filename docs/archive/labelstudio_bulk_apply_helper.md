# Label Studio Bulk-Apply Helper

This helper adds a custom panel in Label Studio with an explicit button:
- `Apply To Selected Birds`

It is intended for flock cases where multiple bird regions share the same attributes.

## What it does

When you click `Apply To Selected Birds`, the helper:
1. Reads currently selected bird regions in the right sidebar.
2. Activates each selected region one-by-one.
3. Sends configured hotkeys for:
   - `behavior`
   - `substrate`
   - optional `legs`

This works around the default Label Studio behavior where per-region controls disappear in multi-select mode.

## Install (Tampermonkey)

1. Install Tampermonkey in your browser.
2. Create a new userscript.
3. Paste the content of:
   `/Users/antoine/bird_leg/labelstudio/tools/bulk_apply_selected_birds.user.js`
4. Save.
5. Open/reload Label Studio task page (for example `http://localhost:8080/projects/<id>/data`).

## Use

1. In the right `Regions` sidebar, select multiple birds with `Ctrl`/`Cmd` + click.
2. In panel `Bird Bulk Apply` (top-left):
   - choose `Behavior` (example: `flying`)
   - choose `Substrate` (example: `air`)
   - optionally choose `Legs` (`skip` by default)
3. Click `Apply To Selected Birds`.
4. Verify a few regions manually, then save task.

## Notes

- `Shift` range selection is not reliable in all Label Studio views.
- The helper only runs on:
  - `http://localhost:8080/*`
  - `http://127.0.0.1:8080/*`
- Hotkeys are aligned to your current config in:
  `/Users/antoine/bird_leg/labelstudio/label_config.xml`
