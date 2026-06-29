# Regression Fixtures

Place the real raster inputs here:

- `class_reklam.png`
- `arcaates.png`

PNG is preferred for stable visual regression. JPG and WebP can also be used, but update `../manifest.json` if the file names or extensions change.

Recommended workflow:

```powershell
cd C:\Users\TAYFUN\Desktop\Projeler\tabela-vector-saas\engine
.\.venv\Scripts\python.exe test_visual_regression.py --update-baseline
.\.venv\Scripts\python.exe test_visual_regression.py
```

The first command writes rendered SVG baselines under `../baselines` when CairoSVG or Inkscape rendering is available. The second command compares future outputs against those baselines and validates analyzer, candidate, quality, and DXF expectations.
