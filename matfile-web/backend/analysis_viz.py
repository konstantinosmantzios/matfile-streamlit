import pandas as pd
import numpy as np


def _to_ms(t_val):
    import tzlocal
    local_tz = tzlocal.get_localzone_name()
    if pd.isna(t_val):
        return None
    s = pd.to_datetime(pd.Series([t_val]))
    if s.dt.tz is None:
        s = s.dt.tz_localize(local_tz)
    return int(s.astype("int64").values[0] // 1_000_000)


def _interp_t_to_ms(t_s, resampled_x, resampled_t):
    """Interpolate the absolute timestamp (for Plotly ms) at an exact time_s value.
    Uses linear interpolation between the two nearest beat-averaged timestamps."""
    import tzlocal
    local_tz = tzlocal.get_localzone_name()
    if len(resampled_x) == 0:
        return None
    # Convert resampled_t (datetime64) to float64 unix-ns for interpolation
    t_ns = resampled_t.astype("datetime64[ns]").astype(np.float64)
    interp_ns = np.interp(t_s, resampled_x, t_ns)
    # Reconstruct as naive timestamp then localize
    naive_ts = pd.Timestamp(int(interp_ns), unit='ns')
    local_ts = naive_ts.tz_localize(local_tz)
    return int(local_ts.value // 1_000_000)



def _build_exact_segment(t_start_exact, t_end_exact, resampled_x, resampled_y, resampled_t):
    """Build arrays for an analysis segment with interpolated boundary values.
    
    Returns (x_exact, y_exact, t_ms_exact) where:
    - The first point is at t_start_exact (interpolated)
    - Interior points are the beat-averaged data within the range
    - The last point is at t_end_exact (interpolated)
    - Duplicate boundary points are removed
    - NaN values are excluded
    """
    # Interpolate boundary values on the beat-averaged signal
    finite_mask = np.isfinite(resampled_y)
    if not np.any(finite_mask):
        return np.array([]), np.array([]), np.array([])
    
    rx_valid = resampled_x[finite_mask]
    ry_valid = resampled_y[finite_mask]
    
    y_at_start = float(np.interp(t_start_exact, rx_valid, ry_valid))
    y_at_end = float(np.interp(t_end_exact, rx_valid, ry_valid))
    
    # Get interior beat-averaged points strictly between boundaries
    interior_mask = (resampled_x > t_start_exact) & (resampled_x < t_end_exact) & finite_mask
    x_interior = resampled_x[interior_mask]
    y_interior = resampled_y[interior_mask]
    t_interior_idx = np.where(interior_mask)[0]
    
    # Build exact-boundary arrays
    x_exact = np.concatenate([[t_start_exact], x_interior, [t_end_exact]])
    y_exact = np.concatenate([[y_at_start], y_interior, [y_at_end]])
    
    # Build ms timestamps: interpolate for boundaries, use actual for interior
    ms_start = _interp_t_to_ms(t_start_exact, resampled_x, resampled_t)
    ms_end = _interp_t_to_ms(t_end_exact, resampled_x, resampled_t)
    ms_interior = [_to_ms(resampled_t[idx]) for idx in t_interior_idx]
    t_ms_exact = [ms_start] + ms_interior + [ms_end]
    
    # Deduplicate (if a beat point happens to coincide with boundary)
    unique_mask = np.concatenate([[True], np.diff(x_exact) > 1e-9])
    x_exact = x_exact[unique_mask]
    y_exact = y_exact[unique_mask]
    t_ms_exact = [t_ms_exact[i] for i, m in enumerate(unique_mask) if m]
    
    return x_exact, y_exact, t_ms_exact


def _clip_line_below_with_ms(x, y, t_ms, upper_y):
    """
    Returns (x_clipped, y_clipped, t_ms_clipped) where the line is perfectly 
    clipped at upper_y. Exact intersection points are calculated and inserted 
    so both area calculations and visual shading are mathematically perfect.
    """
    if len(x) == 0:
        return np.array([]), np.array([]), []
        
    x_new, y_new, t_ms_new = [], [], []
    
    for i in range(len(x)):
        if i == 0:
            x_new.append(x[i])
            y_new.append(min(y[i], upper_y))
            t_ms_new.append(t_ms[i])
            continue
            
        y1, y2 = y[i-1], y[i]
        x1, x2 = x[i-1], x[i]
        ms1, ms2 = t_ms[i-1], t_ms[i]
        
        # Check if segment crosses the upper_y threshold
        if (y1 > upper_y and y2 < upper_y) or (y1 < upper_y and y2 > upper_y):
            slope = (y2 - y1) / (x2 - x1)
            x_inter = x1 + (upper_y - y1) / slope
            ms_inter = int(ms1 + (ms2 - ms1) * (x_inter - x1) / (x2 - x1))
            
            x_new.append(x_inter)
            y_new.append(upper_y)
            t_ms_new.append(ms_inter)
            
        x_new.append(x[i])
        y_new.append(min(y2, upper_y))
        t_ms_new.append(t_ms[i])
        
    return np.array(x_new), np.array(y_new), t_ms_new


def generate_analysis_viz(main_signal, resampled_x, resampled_y, resampled_t,
                          comment_list, baseline_window, end_marker_window,
                          use_baseline_area, baseline_end_comment="Transition",
                          end_marker_overrides=None):
    """
    Generate Plotly traces, shapes, and annotations for the Supine to Standing Analysis.
    
    Statistics are computed using linear interpolation at exact comment timestamps
    on the beat-averaged signal, ensuring precise temporal boundaries for integrals
    and metric calculations. Visual markers are snapped to the nearest beat point
    for alignment with the plotted resampled line.
    
    Returns: (traces, shapes, annotations, stats)
    """
    traces = []
    shapes = []
    annotations = []
    stats = []
    
    if len(resampled_x) != len(resampled_y) or len(resampled_x) == 0:
        return traces, shapes, annotations, stats

    # Ensure arrays are float64 for interpolation
    resampled_x = np.asarray(resampled_x, dtype=np.float64)
    resampled_y = np.asarray(resampled_y, dtype=np.float64)
        
    i = 0
    while i < len(comment_list):
        c1 = comment_list[i]
        if "transition" in str(c1['comment_text']).lower():
            if i + 1 < len(comment_list) and str(comment_list[i+1]['comment_text']).lower().strip(" .") in ["stand", "standing"]:
                c_stand = comment_list[i+1]
                t_start = float(c1['time_s'])      # exact comment time
                t_stand = float(c_stand['time_s'])  # exact comment time
                t_end_marker = t_stand + end_marker_window
                
                c1_ms = _to_ms(c1['absolute_time'])
                c_stand_ms = _to_ms(c_stand['absolute_time'])
                
                # Highlight Transition Region
                shapes.append({
                    "type": "rect",
                    "x0": c1_ms,
                    "x1": c_stand_ms,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "fillcolor": "rgba(250, 204, 21, 0.25)",
                    "opacity": 1.0,
                    "layer": "below",
                    "line": {"width": 0}
                })
                annotations.append({
                    "x": c1_ms,
                    "y": 1.0,
                    "yref": "paper",
                    "text": "Transition",
                    "showarrow": False,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "font": {"size": 12, "color": "rgba(250, 204, 21, 0.8)"}
                })
                
                # ============================================================
                # Baseline — computed with interpolated exact window boundaries
                # ============================================================
                t_base_end = t_start if baseline_end_comment.lower() == "transition" else t_stand
                t_base_start = max(0, t_base_end - baseline_window)
                
                # Build exact baseline segment with interpolated boundaries
                x_base, y_base, t_ms_base = _build_exact_segment(
                    t_base_start, t_base_end, resampled_x, resampled_y, resampled_t
                )
                
                baseline_mean = None
                if len(y_base) > 0:
                    valid_y = y_base[np.isfinite(y_base)]
                    if valid_y.size > 0:
                        baseline_mean = float(np.mean(valid_y))
                        
                if baseline_mean is not None and len(t_ms_base) >= 2:
                    traces.append({
                        "x": [t_ms_base[0], t_ms_base[-1]],
                        "y": [baseline_mean, baseline_mean],
                        "mode": "lines+text",
                        "name": "Baseline",
                        "line": {"color": "#3b82f6", "width": 3},
                        "text": ["", "Baseline"],
                        "textposition": "top right",
                        "textfont": {"color": "#3b82f6", "size": 14},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # ============================================================
                # Visual Markers — snapped to nearest beat for plot alignment
                # ============================================================
                
                # Transition Start Marker (on the resampled line)
                finite_mask_all = np.isfinite(resampled_y)
                rx_fin = resampled_x[finite_mask_all]
                ry_fin = resampled_y[finite_mask_all]
                
                trans_val_interp = float(np.interp(t_start, rx_fin, ry_fin)) if len(rx_fin) > 0 else np.nan
                stand_val_interp = float(np.interp(t_stand, rx_fin, ry_fin)) if len(rx_fin) > 0 else np.nan
                end_val_interp = float(np.interp(t_end_marker, rx_fin, ry_fin)) if len(rx_fin) > 0 else np.nan
                
                t_start_ms_interp = _interp_t_to_ms(t_start, resampled_x, resampled_t)
                if t_start_ms_interp is not None and np.isfinite(trans_val_interp):
                    traces.append({
                        "x": [t_start_ms_interp],
                        "y": [trans_val_interp],
                        "mode": "markers",
                        "name": "Transition Start",
                        "marker": {"color": "#f97316", "size": 13, "symbol": "square"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # Stand Start Marker
                t_stand_ms_interp = _interp_t_to_ms(t_stand, resampled_x, resampled_t)
                if t_stand_ms_interp is not None and np.isfinite(stand_val_interp):
                    traces.append({
                        "x": [t_stand_ms_interp],
                        "y": [stand_val_interp],
                        "mode": "markers",
                        "name": "Start",
                        "marker": {"color": "#22c55e", "size": 13, "symbol": "circle"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # End Marker
                t_end_marker_ms_interp = _interp_t_to_ms(t_end_marker, resampled_x, resampled_t)
                if t_end_marker_ms_interp is not None and np.isfinite(end_val_interp):
                    traces.append({
                        "x": [t_end_marker_ms_interp],
                        "y": [end_val_interp],
                        "mode": "markers+text",
                        "name": "End",
                        "marker": {"color": "#ef4444", "size": 13, "symbol": "x"},
                        "text": ["End Marker"],
                        "textposition": "top center",
                        "textfont": {"color": "#ef4444", "size": 12},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # ============================================================
                # Trans-to-End Stats (Orange Area) — interpolated boundaries
                # ============================================================
                area_above_or, area_below_or = np.nan, np.nan
                min_val_or, pct_drop_or = np.nan, np.nan
                
                # Build exact segment for orange area
                x_or, y_or, t_ms_or = _build_exact_segment(
                    t_start, t_end_marker, resampled_x, resampled_y, resampled_t
                )
                
                if len(x_or) > 1:
                    upper_y = baseline_mean if (use_baseline_area and baseline_mean is not None) else trans_val_interp
                    
                    x_or_clip, y_or_clip, t_ms_or_clip = _clip_line_below_with_ms(x_or, y_or, t_ms_or, upper_y)
                    
                    # For area above, we do the same logic but flipped if needed, but since it's removed, we just leave it 0 or calculate standard.
                    area_above_or = float(np.trapezoid(np.maximum(y_or - upper_y, 0), x_or))
                    area_below_or = float(np.trapezoid(upper_y - y_or_clip, x_or_clip))
                    min_val_or = float(np.min(y_or))
                    if upper_y != 0:
                        pct_drop_or = float((upper_y - min_val_or) / abs(upper_y) * 100)
                    
                    # Min marker
                    min_idx_or = np.argmin(y_or)
                    traces.append({
                        "x": [t_ms_or[min_idx_or]],
                        "y": [min_val_or],
                        "mode": "markers",
                        "name": "Min (Or)",
                        "marker": {"color": "#ef4444", "size": 10, "symbol": "diamond"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                    # Shaded area traces (visual, using exact interpolated segment)
                    x_vis = resampled_x[(resampled_x > t_start) & (resampled_x < t_end_marker)]
                    
                    if len(x_or) > 0:
                        traces.append({
                            "x": t_ms_or_clip,
                            "y": [float(upper_y)] * len(t_ms_or_clip),
                            "mode": "lines",
                            "line": {"width": 0},
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        traces.append({
                            "x": t_ms_or_clip,
                            "y": [float(y) for y in y_or_clip],
                            "mode": "lines",
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(249, 115, 22, 0.15)",
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        
                # ============================================================
                # Stand-to-End Stats (Green Area) — interpolated boundaries
                # ============================================================
                area_above_gr, area_below_gr = np.nan, np.nan
                min_val_gr, pct_drop_gr = np.nan, np.nan
                
                x_gr, y_gr, t_ms_gr = _build_exact_segment(
                    t_stand, t_end_marker, resampled_x, resampled_y, resampled_t
                )
                
                if len(x_gr) > 1:
                    upper_y_green = baseline_mean if (use_baseline_area and baseline_mean is not None) else stand_val_interp
                    
                    x_gr_clip, y_gr_clip, t_ms_gr_clip = _clip_line_below_with_ms(x_gr, y_gr, t_ms_gr, upper_y_green)
                    
                    area_above_gr = float(np.trapezoid(np.maximum(y_gr - upper_y_green, 0), x_gr))
                    area_below_gr = float(np.trapezoid(upper_y_green - y_gr_clip, x_gr_clip))
                    min_val_gr = float(np.min(y_gr))
                    if upper_y_green != 0:
                        pct_drop_gr = float((upper_y_green - min_val_gr) / abs(upper_y_green) * 100)
                    
                    # Min marker
                    min_idx_gr = np.argmin(y_gr)
                    traces.append({
                        "x": [t_ms_gr[min_idx_gr]],
                        "y": [min_val_gr],
                        "mode": "markers",
                        "name": "Min (Gr)",
                        "marker": {"color": "#ef4444", "size": 10, "symbol": "diamond"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                    # Shaded area traces (visual, using exact interpolated segment)
                    x_vis_gr = resampled_x[(resampled_x > t_stand) & (resampled_x < t_end_marker)]
                    
                    if len(x_gr) > 0:
                        traces.append({
                            "x": t_ms_gr_clip,
                            "y": [float(upper_y_green)] * len(t_ms_gr_clip),
                            "mode": "lines",
                            "line": {"width": 0},
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        traces.append({
                            "x": t_ms_gr_clip,
                            "y": [float(y) for y in y_gr_clip],
                            "mode": "lines",
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(34, 197, 94, 0.25)",
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        
                # ============================================================
                # Baseline Recovery Analysis
                # ============================================================
                rec_duration = np.nan
                rec_area = np.nan
                rec_started_in = ""
                rec_min_val = np.nan
                rec_pct_drop = np.nan
                t_rec_end_ms_interp = None
                rec_end_val_interp = None
                
                if baseline_mean is not None:
                    finite_mask_all = np.isfinite(resampled_y)
                    if np.any(finite_mask_all):
                        t_end_data = float(resampled_x[finite_mask_all][-1])
                        # Bound t_end_data to not go past our array
                        t_end_data = max(t_start, t_end_data)
                        
                        x_post, y_post, t_ms_post = _build_exact_segment(t_start, t_end_data, resampled_x, resampled_y, resampled_t)
                        
                        if len(x_post) > 0:
                            def _interp_cross(x0, y0, x1, y1, y_target):
                                if y1 == y0: return x0
                                return x0 + (y_target - y0) * (x1 - x0) / (y1 - y0)
                                
                            t_search_start = t_start
                            t_rec_start = None
                            t_rec_end = None
                            
                            while True:
                                x_post, y_post, _ = _build_exact_segment(t_search_start, t_end_data, resampled_x, resampled_y, resampled_t)
                                if len(x_post) < 2:
                                    break
                                
                                # Find where curve crosses DOWN the baseline value (y_prev >= baseline_mean and y_curr < baseline_mean)
                                down_cross = np.where((y_post[:-1] >= baseline_mean) & (y_post[1:] < baseline_mean))[0]
                                if len(down_cross) == 0:
                                    break
                                
                                drop_idx = down_cross[0]
                                candidate_start = _interp_cross(x_post[drop_idx], y_post[drop_idx], x_post[drop_idx+1], y_post[drop_idx+1], baseline_mean)
                                    
                                x_rem, y_rem, _ = _build_exact_segment(candidate_start, t_end_data, resampled_x, resampled_y, resampled_t)
                                if len(x_rem) < 2:
                                    break

                                # Find where curve crosses UP the baseline value (y_prev < baseline_mean and y_curr >= baseline_mean)
                                up_cross = np.where((y_rem[:-1] < baseline_mean) & (y_rem[1:] >= baseline_mean))[0]
                                
                                if len(up_cross) > 0:
                                    end_idx = up_cross[0]
                                    candidate_end = _interp_cross(x_rem[end_idx], y_rem[end_idx], x_rem[end_idx+1], y_rem[end_idx+1], baseline_mean)
                                    
                                    # Dip must last at least 0.5 seconds
                                    if candidate_end - candidate_start >= 0.5:
                                        if candidate_start <= t_start + 20:
                                            t_rec_start = candidate_start
                                            if candidate_end <= t_stand + 30:
                                                t_rec_end = candidate_end
                                        break # Stop searching after finding the first valid-length dip
                                    else:
                                        # Too short, advance search past this crossing
                                        t_search_start = max(candidate_end, candidate_start + 0.5)
                                else:
                                    # Never recovers above baseline before end of data
                                    if t_end_data - candidate_start >= 0.5:
                                        if candidate_start <= t_start + 20:
                                            t_rec_start = candidate_start
                                    break
                                    
                            # Check for manual override
                            if end_marker_overrides and str(i) in end_marker_overrides:
                                override_ms = end_marker_overrides[str(i)]
                                # Calculate resampled_t in unix ms once if needed
                                resampled_t_ms = np.array([_to_ms(dt) for dt in resampled_t])
                                t_rec_end_override = float(np.interp(override_ms, resampled_t_ms, resampled_x))
                                
                                # Apply override
                                t_rec_end = t_rec_end_override
                                # If t_rec_start wasn't detected (e.g. baseline crossed late), we still need it.
                                # Usually if there's an end marker, there should be a start marker.
                                # If it wasn't detected by the 20s rule, we can relax it if an override exists.
                                if t_rec_start is None:
                                    down_cross_all = np.where((resampled_x[:-1] >= t_start) & (resampled_y[:-1] >= baseline_mean) & (resampled_y[1:] < baseline_mean))[0]
                                    if len(down_cross_all) > 0:
                                        idx = down_cross_all[0]
                                        t_rec_start = float(_interp_cross(resampled_x[idx], resampled_y[idx], resampled_x[idx+1], resampled_y[idx+1], baseline_mean))

                                    
                            if t_rec_start is not None:
                                # Always plot the start marker if found within 20s
                                t_rec_start_ms_interp = _interp_t_to_ms(t_rec_start, resampled_x, resampled_t)
                                start_val_interp = float(np.interp(t_rec_start, resampled_x, resampled_y))
                                traces.append({
                                    "x": [t_rec_start_ms_interp],
                                    "y": [start_val_interp],
                                    "mode": "markers",
                                    "name": "Rec Start",
                                    "marker": {"color": "#3b82f6", "size": 13, "symbol": "triangle-down"},
                                    "showlegend": False,
                                    "hoverinfo": "skip"
                                })

                            if t_rec_start is not None and t_rec_end is not None:
                                t_rec_end_ms_interp = _interp_t_to_ms(t_rec_end, resampled_x, resampled_t)
                                
                                finite_mask_all = np.isfinite(resampled_y)
                                rx_fin = resampled_x[finite_mask_all]
                                ry_fin = resampled_y[finite_mask_all]
                                rec_end_val_interp = float(np.interp(t_rec_end, rx_fin, ry_fin)) if len(rx_fin) > 0 else np.nan
                                
                                rec_duration = t_rec_end - t_rec_start
                                rec_started_in = "Transition" if t_rec_start < t_stand else "Standing"
                                        
                                x_rec, y_rec, ms_rec = _build_exact_segment(t_rec_start, t_rec_end, resampled_x, resampled_y, resampled_t)
                                
                                if len(x_rec) > 1:
                                    rec_area = float(np.trapezoid(baseline_mean - y_rec, x_rec))
                                    rec_min_val = float(np.min(y_rec))
                                    if baseline_mean != 0:
                                        rec_pct_drop = float((baseline_mean - rec_min_val) / abs(baseline_mean) * 100)
                                    
                                    # Visual traces (end marker and shading)
                                    traces.append({
                                        "x": [ms_rec[-1]],
                                        "y": [float(y_rec[-1])],
                                        "mode": "markers",
                                        "name": "Rec End",
                                        "marker": {"color": "#3b82f6", "size": 13, "symbol": "triangle-up"},
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })
                                    traces.append({
                                        "x": ms_rec,
                                        "y": [float(baseline_mean)] * len(ms_rec),
                                        "mode": "lines",
                                        "line": {"width": 0},
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })
                                    traces.append({
                                        "x": ms_rec,
                                        "y": [float(y) if np.isfinite(y) else None for y in y_rec],
                                        "mode": "lines",
                                        "line": {"width": 0},
                                        "fill": "tonexty",
                                        "fillcolor": "rgba(59, 130, 246, 0.3)", # Blue
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })

                # ============================================================
                # Summary Stats — using interpolated values and exact durations
                # ============================================================
                base_val = baseline_mean if baseline_mean is not None else np.nan
                transition_duration = t_stand - t_start    # exact from comments
                or_duration = t_end_marker - t_start       # exact
                gr_duration = t_end_marker - t_stand       # exact
                
                def _safe_float(v):
                    try:
                        return float(v) if np.isfinite(v) else None
                    except (TypeError, ValueError):
                        return None
                
                stats.append({
                    "id": i,
                    "t_start_ms": c1_ms,
                    "t_trans_ms": c1_ms,
                    "t_stand_ms": c_stand_ms,
                    "t_end_ms": t_end_marker_ms_interp,
                    "baseline": _safe_float(base_val),
                    "transition_time": _safe_float(transition_duration),
                    "or_trans_val": _safe_float(trans_val_interp),
                    "or_duration": _safe_float(or_duration),
                    "or_min_val": _safe_float(min_val_or),
                    "or_pct_drop": _safe_float(pct_drop_or),
                    "or_area_above": _safe_float(area_above_or),
                    "or_area_below": _safe_float(area_below_or),
                    "gr_stand_val": _safe_float(stand_val_interp),
                    "gr_duration": _safe_float(gr_duration),
                    "gr_min_val": _safe_float(min_val_gr),
                    "gr_pct_drop": _safe_float(pct_drop_gr),
                    "gr_area_above": _safe_float(area_above_gr),
                    "gr_area_below": _safe_float(area_below_gr),
                    "rec_duration": _safe_float(rec_duration),
                    "rec_area": _safe_float(rec_area),
                    "rec_min_val": _safe_float(rec_min_val),
                    "rec_pct_drop": _safe_float(rec_pct_drop),
                    "rec_started_in": rec_started_in,
                    "rec_end_ms": _safe_float(t_rec_end_ms_interp) if 't_rec_end_ms_interp' in locals() else None,
                    "rec_end_val": _safe_float(rec_end_val_interp) if 'rec_end_val_interp' in locals() else None,
                    "end_val": _safe_float(end_val_interp)
                })
                
        i += 1
        
    return traces, shapes, annotations, stats
