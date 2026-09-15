"""
Largest Triangle Three Buckets (LTTB) downsampling algorithm.

Reduces time-series point count while preserving visual shape —
peaks, valleys, and sharp transitions are kept. This is the same
algorithm used internally by plotly-resampler.
"""
import numpy as np


def lttb_core(xv, yv, target_points):
    n_valid = len(xv)
    if n_valid <= target_points or target_points < 3:
        return xv.copy(), yv.copy()

    out_idx = np.empty(target_points, dtype=np.intp)
    out_idx[0] = 0
    out_idx[-1] = n_valid - 1

    bucket_size = (n_valid - 2) / (target_points - 2)

    # Bucket boundaries B[k] = min(floor(1 + k*bucket_size), n_valid).
    # Bucket used for point i is [B[i-1], B[i]); the "next bucket" used
    # for the triangle area is [B[i], B[i+1]).
    B = np.minimum(
        (1 + np.arange(target_points, dtype=np.float64) * bucket_size).astype(np.intp),
        n_valid,
    )

    # Prefix sums let us compute every bucket mean in O(1) instead of
    # slicing + np.mean() on each iteration.
    psx = np.concatenate(([0.0], np.cumsum(xv)))
    psy = np.concatenate(([0.0], np.cumsum(yv)))

    prev = 0
    for i in range(1, target_points - 1):
        b_start = int(B[i - 1])
        b_end = int(B[i])
        nb_start = int(B[i])
        nb_end = int(B[i + 1])

        cnt = nb_end - nb_start
        if cnt > 0:
            avg_x = (psx[nb_end] - psx[nb_start]) / cnt
            avg_y = (psy[nb_end] - psy[nb_start]) / cnt
        else:
            avg_x = xv[-1]
            avg_y = yv[-1]

        bx = xv[b_start:b_end]
        by = yv[b_start:b_end]
        px, py = xv[prev], yv[prev]

        areas = np.abs((bx - px) * (avg_y - py) -
                       (avg_x - px) * (by - py))

        best = b_start + int(np.argmax(areas))
        out_idx[i] = best
        prev = best

    return xv[out_idx], yv[out_idx]


def lttb_downsample(x, y, target_points):
    """
    Downsample (x, y) arrays to *target_points* using LTTB, 
    preserving NaN gaps so Plotly breaks lines appropriately.
    """
    n = len(x)
    if n <= target_points or target_points < 3:
        return x.copy(), y.copy()

    valid = np.isfinite(y)
    total_valid = int(valid.sum())

    if total_valid <= target_points:
        return x.copy(), y.copy()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid_diff = np.diff(valid.astype(np.int8))
    starts = np.flatnonzero(valid_diff == 1) + 1
    if valid[0]:
        starts = np.concatenate(([0], starts))

    ends = np.flatnonzero(valid_diff == -1)
    if valid[-1]:
        ends = np.concatenate((ends, [n - 1]))

    # First pass: compute per-block targets and total output length so we
    # can preallocate instead of building Python lists.
    blocks = list(zip(starts, ends))
    n_blocks = len(blocks)
    targets = [0] * n_blocks
    out_len = 0
    for i, (s, e) in enumerate(blocks):
        block_len = e - s + 1
        block_target = max(3, int(round(target_points * (block_len / total_valid))))
        if block_target > block_len:
            block_target = block_len
        targets[i] = block_target
        out_len += block_target
        if e < n - 1:
            out_len += 1  # NaN separator preserves the gap

    out_x = np.empty(out_len, dtype=np.float64)
    out_y = np.empty(out_len, dtype=np.float64)

    pos = 0
    for (s, e), block_target in zip(blocks, targets):
        bx, by = lttb_core(x[s:e + 1], y[s:e + 1], block_target)
        nb = len(bx)
        out_x[pos:pos + nb] = bx
        out_y[pos:pos + nb] = by
        pos += nb
        if e < n - 1:
            out_x[pos] = x[e + 1]
            out_y[pos] = np.nan
            pos += 1

    return out_x[:pos], out_y[:pos]