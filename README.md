# pxviewer

A custom molecular viewer built on top of [Mol*](https://molstar.org) with a Python-driven scene API.

## Structure

- `frontend/` — TypeScript/React frontend using the `molstar` React plugin helpers.
- `python/` — Python package using `molviewspec` to build MVS scenes and `ciftools` to write BinaryCIF.

## Quick start

### Frontend

```bash
cd frontend
npm install
npm run build
# open index.html in a browser, or serve the build/ directory
```

### Python

```bash
cd python
pip install -e .
```

```python
import pxviewer.mvs_builder as mvs
mvsj = mvs.create_example_view("https://www.ebi.ac.uk/pdbe/entry-files/1cbs.bcif")
print(mvsj)
```

## Next steps

- Add a Python-to-JS bridge for live coordinate updates (e.g. `ModelWithCoordinates`).
- Generate BCIF in Python and load it directly in the frontend.
- Swap the example `loadPdb` call for `loadMvsData`/`loadStructureFromData`.
