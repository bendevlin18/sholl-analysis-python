# Code Cleaning Notes

Issues grouped by severity. File + line numbers included throughout.

---

## Bugs / Correctness

### 1. `detect_and_normalize` silently wrong on single-value images
`image_processing.py:87-100`

The guard `if len(unique_vals) > 2` raises `ValueError` for multi-value images but lets single-value images through. If an all-black or all-white image is passed, `np.argmin(counts)` picks the only value as signal, and the output becomes entirely 255. No error is raised — the pipeline silently continues with garbage data. Fix: add `if len(unique_vals) < 2` to raise early.

### 2. `_finalise` drops already-processed images from the saved summary
`analyzer.py:472-476`

When a batch run skips some images (already done from a prior run) and processes others, `summary_rows` only contains the newly processed images. `_finalise` writes only those rows to `sholl_summary.csv`, overwriting any previously saved summary. The already-done path in `run()` returns early via `_load_existing_summary`, but the mixed case (some done, some new) silently discards prior results from the CSV. Fix: load any existing summary at the start of `run()` and merge new rows into it before writing.

### 3. `process_single` does not catch `UserCancelledError`
`analyzer.py:296`

`_process_image` raises `UserCancelledError` when the user cancels or closes a window. `run()` catches this correctly, but `process_single` does not — the exception propagates unexpectedly to callers using the public API. Fix: catch `UserCancelledError` in `process_single` and either return `None` or re-raise as a more descriptive exception.

### 4. `calc_intersection` logic works only by accident
`geometry.py:71`

```python
if circ_arr[i][j] == skeleton_arr[i][j]:
```

This finds intersections because `circ_arr` background is `1` and `skeleton_arr` background is `0`, so backgrounds never match. Ring pixels and skeleton pixels are both `255`, so they do match. The intent is clearly `circ == 255 AND skel == 255`, but the current code expresses it as `circ == skel`. If the background convention ever changes (e.g. `make_circles` initialised with `np.zeros`), this would produce massive false positives with no obvious reason. Fix: make the condition explicit:
```python
if circ_arr[i][j] == 255 and skeleton_arr[i][j] == 255:
```

---

## Performance

### 5. `calc_intersection` uses pure Python nested loops
`geometry.py:68-73`

A pixel-by-pixel Python loop over the full image for every Sholl ring is extremely slow. A 1000×1000 image with 30 rings = 30 million iterations. Replace with a vectorised numpy call:
```python
rows, cols = np.where((circ_arr == 255) & (skeleton_arr == 255))
return np.column_stack([rows, cols]).flatten()
```
This would be 100–1000× faster.

### 6. `x_y_separate` uses `np.append` in a loop
`geometry.py:94-97`

Building arrays by repeated `np.append` is O(n²) in memory. The same result is achievable with direct slicing:
```python
return arr[::2].copy(), arr[1::2].copy()
```

### 7. `clean_intersections` is O(n²) and uses `pd.concat` in a loop
`geometry.py:141-149`

The nested `iterrows()` loop compares every point against every other point per ring — quadratic in the number of intersection pixels. For rings with many hits this degrades badly. Also, `pd.concat` is called once per ring inside a loop, which is a known pandas anti-pattern (copies the growing DataFrame each time). Fix: collect all per-ring DataFrames in a list and concat once at the end; replace the O(n²) merge with a spatial approach (e.g. round coordinates to nearest `merge_dist` grid cell).

---

## Inconsistencies

### 8. Version mismatch between `pyproject.toml` and `__init__.py`
`pyproject.toml:6` → `version = "0.1.0"`  
`__init__.py:35` → `__version__ = "0.2.0"`

These should agree. Per CLAUDE.md, `__init__.py` is authoritative, so `pyproject.toml` should be updated to `0.2.0`.

### 9. Placeholder URLs in `pyproject.toml`
`pyproject.toml:37-38`

```toml
Homepage = "https://github.com/yourname/sholl-analysis"
Issues   = "https://github.com/yourname/sholl-analysis/issues"
```

These are template placeholders, not the real repository URLs. Should be updated to the actual GitHub repo (`bendevlin18/sholl-analysis-python`).

### 10. `intersection_counts` uses `/` instead of `//`
`analyzer.py:371`

```python
intersection_counts.append(len(hits) / 2)
```

`len(hits)` is always even (pairs of row/col), so `/` produces float counts (e.g. `3.0`) rather than integers. The Sholl curve and summary CSV then contain floats where integers are expected. Change to `len(hits) // 2`.

### 11. `dilate_skeleton` is not exported from `__init__.py`
`image_processing.py:193` / `__init__.py`

All other `image_processing` functions are exported in `__init__.py` and `__all__`, but `dilate_skeleton` is absent. It is part of the pipeline and useful for programmatic access. Either export it consistently or add a comment explaining why it is intentionally internal.

---

## Test Coverage Gaps

### 12. `test_normalize_raises_on_non_binary` tests the wrong condition
`tests/test_geometry.py:108-111`

```python
arr = np.array([0, 1, 2, 3], dtype=np.uint8)
```

This is a 1D array, so `detect_and_normalize` raises on the `ndim != 2` shape check — not the `> 2 unique values` check the test name implies. The test passes but is not testing what it claims. Fix: use a 2D array with more than 2 values, e.g.:
```python
arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
```

### 13. No test for the single-value image bug (issue #1 above)
There is no test for `detect_and_normalize` receiving an all-zeros or all-255 image. Adding one would catch the silent-wrong-output bug described above.

### 14. No tests for `image_processing` functions beyond `detect_and_normalize`
`skeletonize_image`, `dilate_skeleton`, `find_endpoints`, and `smooth_binary` have no test coverage. At minimum, smoke tests verifying shape, dtype, and basic correctness on synthetic data would be valuable.

---

## Minor / Clarity

### 15. `make_circles` background value `1` is unexplained
`geometry.py:37`

```python
arr = np.ones(image_shape)
```

Using `1` instead of `0` is load-bearing (it's what makes the equality check in `calc_intersection` work), but there is no comment explaining this. A one-line comment would prevent future breakage.

### 16. `find_endpoints` variable name `counts` is misleading
`image_processing.py:237`

```python
counts = np.unique(endpoint_arr, return_counts=True)
n_endpoints = counts[1][1] if len(counts[1]) > 1 else 0
```

`np.unique(..., return_counts=True)` returns a tuple `(unique_values, value_counts)`. Naming this `counts` when it is a two-element tuple containing values *and* counts is confusing. Rename to `unique_result` or unpack directly:
```python
unique_vals, val_counts = np.unique(endpoint_arr, return_counts=True)
n_endpoints = val_counts[1] if len(val_counts) > 1 else 0
```
