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
    prev = 0

    for i in range(1, target_points - 1):
        b_start = int(np.floor(1 + (i - 1) * bucket_size))
        b_end   = min(int(np.floor(1 + i * bucket_size)), n_valid)

        nb_start = int(np.floor(1 + i * bucket_size))
        nb_end   = min(int(np.floor(1 + (i + 1) * bucket_size)), n_valid)
        avg_x = np.mean(xv[nb_start:nb_end]) if nb_start < nb_end else xv[-1]
        avg_y = np.mean(yv[nb_start:nb_end]) if nb_start < nb_end else yv[-1]

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

    valid_diff = np.diff(valid.astype(int))
    starts = np.where(valid_diff == 1)[0] + 1
    if valid[0]:
        starts = np.insert(starts, 0, 0)
        
    ends = np.where(valid_diff == -1)[0]
    if valid[-1]:
        ends = np.append(ends, n - 1)

    out_x = []
    out_y = []
    
    for s, e in zip(starts, ends):
        block_len = e - s + 1
        block_target = max(3, int(round(target_points * (block_len / total_valid))))
        
        bx, by = lttb_core(
            x[s:e+1].astype(np.float64), 
            y[s:e+1].astype(np.float64), 
            block_target
        )
        out_x.append(bx)
        out_y.append(by)
        
        # Insert a NaN after the block to preserve the gap
        if e < n - 1:
            out_x.append([x[e+1]])
            out_y.append([np.nan])
            
    if not out_x:
        return x.copy(), y.copy()
        
    return np.concatenate(out_x), np.concatenate(out_y)
