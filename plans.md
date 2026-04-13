# Plans

## PyPI Publishing Prerequisites

Before publishing to PyPI:

- [ ] Add `.gitignore` to exclude `egg-info/`, `__pycache__/`, `.pytest_cache/`, and build artifacts
- [ ] Set up GitHub Actions CI workflow to run the test suite automatically on push/PR
- [ ] Remove or archive `main.py` at the repo root — predates the package and will confuse users arriving from PyPI
- [ ] Verify the package builds cleanly with `python -m build`

---

## Headless / Programmatic Soma Selection

**What:** Add a `center` parameter to `process_single` (and optionally `run`) that bypasses the `ginput` interactive window entirely, allowing the soma position to be passed in directly.

**Why:** The current interactive workflow is completely unusable in scripts, Jupyter notebooks, or HPC/server environments. This single change makes the package composable with other tools and opens up automation and batch-scripting use cases.

**Scope:**
- `process_single(filepath, output_dir, stem=None, center=None)` — if `center` is provided, skip the ginput step
- Consider a matching `--center X,Y` CLI flag for non-interactive batch runs

---

## Sholl Statistics

**What:** Compute standard derived Sholl metrics from the raw intersection counts and include them in the summary CSV.

**Why:** Raw intersection counts per radius are an intermediate result. Biologists report derived statistics — these are what appear in papers and are expected by reviewers and journals.

**Metrics to add:**
- Critical radius — radius at which intersections are maximised
- Maximum intersections — the peak count
- Schoenen ramification index — max intersections / number of primary branches
- Area under the Sholl curve (AUC)
- Mean intersections across all radii

**Scope:**
- New `sholl_stats(radii, counts)` function in `geometry.py` (or a new `stats.py`)
- Output as additional columns in `sholl_summary.csv`

---

## Physical Units (Pixel Size / Scale)

**What:** Add an optional `pixel_size` parameter (µm/px) so that radii and the Sholl curve x-axis are expressed in real-world units rather than pixels.

**Why:** Microscopy images have a known pixel size, often stored in TIFF metadata. Pixel-unit outputs require a manual conversion step before results can be reported; journals and collaborators expect µm.

**Scope:**
- `ShollAnalyzer(pixel_size=1.0)` parameter (default 1.0 = pixels, no change in behaviour)
- Apply scaling to `self.radii` when building the summary and curve plots
- `--pixel-size` CLI flag
- Optionally auto-read pixel size from TIFF metadata (lower priority)

---

## Vectorize Endpoint Coordinate Extraction

**What:** Replace the remaining pixel-by-pixel Python loop in `analyzer.py` (lines 390–396) with `np.argwhere`.

**Why:** Same O(n) Python loop + `np.append` pattern fixed in `geometry.py` during the performance cleanup pass. Small but consistent with the rest of the codebase.

**Scope:** Single function change in `_process_image`:
```python
# current
ep = []
for i in range(endpoint_arr.shape[0]):
    for j in range(endpoint_arr.shape[1]):
        if endpoint_arr[i][j] == 255.0:
            ep = np.append(ep, [i, j])
x_ep, y_ep = x_y_separate(ep)

# replacement
ep = np.argwhere(endpoint_arr == 255.0)
x_ep, y_ep = ep[:, 0], ep[:, 1]
```
