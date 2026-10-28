# Screenshots

Images used by the root [README](../README.md). The root README references these exact filenames:

| File | What to show |
|------|--------------|
| `demo.gif` | Full flow: upload photo 1 → detection + results → upload photo 2 → CO₂ comparison tab. Under 10 s, under 5 MB. |
| `01_detection.png` | "Detection" column: bounding boxes + ArUco overlay, and the rectified image. |
| `02_results.png` | One expanded container card: crop, GN size, fill, food, mass, CO₂. |
| `03_co2_comparison.png` | "CO₂ comparison" tab: the three metrics and the per-container breakdown. |

## Capture tips

- Run with real models (`STUB_MODE = False`), and use photos with the ArUco marker visible, so the app runs in calibrated mode.
- Set the browser window ~1280 px wide and use the light theme.
- For the GIF, record with ScreenToGif or similar, then compress (e.g. [ezgif](https://ezgif.com/optimize)).

Once the files are here, uncomment the `Demo` block in the root README.
