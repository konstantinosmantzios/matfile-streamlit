import pandas as pd
import numpy as np

def _to_ms(t_val):
    return int(pd.to_datetime(t_val).value // 1_000_000)

def generate_analysis_viz(main_signal, resampled_x, resampled_y, resampled_t, comment_list, baseline_window, end_marker_window, use_baseline_area, baseline_end_comment="Transition"):
    """
    Generate Plotly traces, shapes, and annotations for the Supine to Standing Analysis.
    Returns: (traces, shapes, annotations, stats)
    """
    traces = []
    shapes = []
    annotations = []
    stats = []
    
    i = 0
    while i < len(comment_list):
        c1 = comment_list[i]
        if "transition" in str(c1['comment_text']).lower():
            if i + 1 < len(comment_list) and str(comment_list[i+1]['comment_text']).lower().strip(" .") in ["stand", "standing"]:
                c_stand = comment_list[i+1]
                t_start = c1['time_s']
                t_stand = c_stand['time_s']
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
                
                # Baseline
                t_base_end = t_start if baseline_end_comment.lower() == "transition" else t_stand
                t_base_start = max(0, t_base_end - baseline_window)
                base_mask = (resampled_x >= t_base_start) & (resampled_x <= t_base_end)
                t_base_start_idx = np.abs(resampled_x - t_base_start).argmin()
                t_base_end_idx = np.abs(resampled_x - t_base_end).argmin()
                y_window = resampled_y[base_mask]
                
                baseline_mean = None
                if y_window.size > 0:
                    valid_y = y_window[np.isfinite(y_window)]
                    if valid_y.size > 0:
                        baseline_mean = float(np.mean(valid_y))
                        
                if baseline_mean is not None:
                    traces.append({
                        "x": [_to_ms(resampled_t[t_base_start_idx]), _to_ms(resampled_t[t_base_end_idx])],
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
                    
                # Transition Start Marker
                trans_idx = np.abs(resampled_x - t_start).argmin()
                if trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx]):
                    traces.append({
                        "x": [_to_ms(resampled_t[trans_idx])],
                        "y": [float(resampled_y[trans_idx])],
                        "mode": "markers",
                        "name": "Transition Start",
                        "marker": {"color": "#f97316", "size": 13, "symbol": "square"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # Stand Start Marker
                start_idx = np.abs(resampled_x - t_stand).argmin()
                if start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx]):
                    traces.append({
                        "x": [_to_ms(resampled_t[start_idx])],
                        "y": [float(resampled_y[start_idx])],
                        "mode": "markers",
                        "name": "Start",
                        "marker": {"color": "#22c55e", "size": 13, "symbol": "circle"},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # End Marker
                end_idx = np.abs(resampled_x - t_end_marker).argmin()
                if end_idx < len(resampled_y) and np.isfinite(resampled_y[end_idx]):
                    traces.append({
                        "x": [_to_ms(resampled_t[end_idx])],
                        "y": [float(resampled_y[end_idx])],
                        "mode": "markers+text",
                        "name": "End",
                        "marker": {"color": "#ef4444", "size": 13, "symbol": "x"},
                        "text": ["End Marker"],
                        "textposition": "top center",
                        "textfont": {"color": "#ef4444", "size": 12},
                        "showlegend": False,
                        "hoverinfo": "skip"
                    })
                    
                # Transition Stats (Orange Area)
                area_above_or, area_below_or = np.nan, np.nan
                min_val_or, pct_drop_or = np.nan, np.nan
                if trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx]):
                    area_mask = (resampled_x >= t_start) & (resampled_x <= t_end_marker)
                    x_area = resampled_x[area_mask]
                    t_area = resampled_t[area_mask]
                    y_area = resampled_y[area_mask]
                    if len(x_area) > 0:
                        upper_y = baseline_mean if (use_baseline_area and baseline_mean is not None) else resampled_y[trans_idx]
                        valid_mask = np.isfinite(y_area)
                        x_val_or = x_area[valid_mask]
                        t_val_or = t_area[valid_mask]
                        y_val_or = y_area[valid_mask]
                        
                        if len(x_val_or) > 1:
                            area_above_or = float(np.trapezoid(np.maximum(y_val_or - upper_y, 0), x_val_or))
                            area_below_or = float(np.trapezoid(np.maximum(upper_y - y_val_or, 0), x_val_or))
                            min_val_or = float(np.min(y_val_or))
                            if upper_y != 0:
                                pct_drop_or = float((upper_y - min_val_or) / abs(upper_y) * 100)
                            min_idx_or = np.argmin(y_val_or)
                            
                            traces.append({
                                "x": [_to_ms(t_val_or[min_idx_or])],
                                "y": [min_val_or],
                                "mode": "markers",
                                "name": "Min (Or)",
                                "marker": {"color": "#ef4444", "size": 10, "symbol": "diamond"},
                                "showlegend": False,
                                "hoverinfo": "skip"
                            })
                            
                        traces.append({
                            "x": [_to_ms(t) for t in t_area],
                            "y": [float(upper_y)] * len(t_area),
                            "mode": "lines",
                            "line": {"width": 0},
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        traces.append({
                            "x": [_to_ms(t) for t in t_area],
                            "y": [float(y) if np.isfinite(y) else None for y in y_area],
                            "mode": "lines",
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(249, 115, 22, 0.15)",
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        
                # Standing Stats (Green Area)
                area_above_gr, area_below_gr = np.nan, np.nan
                min_val_gr, pct_drop_gr = np.nan, np.nan
                if start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx]):
                    area_mask_green = (resampled_x >= t_stand) & (resampled_x <= t_end_marker)
                    x_area_green = resampled_x[area_mask_green]
                    t_area_green = resampled_t[area_mask_green]
                    y_area_green = resampled_y[area_mask_green]
                    if len(x_area_green) > 0:
                        upper_y_green = baseline_mean if (use_baseline_area and baseline_mean is not None) else resampled_y[start_idx]
                        valid_mask_gr = np.isfinite(y_area_green)
                        x_val_gr = x_area_green[valid_mask_gr]
                        t_val_gr = t_area_green[valid_mask_gr]
                        y_val_gr = y_area_green[valid_mask_gr]
                        
                        if len(x_val_gr) > 1:
                            area_above_gr = float(np.trapezoid(np.maximum(y_val_gr - upper_y_green, 0), x_val_gr))
                            area_below_gr = float(np.trapezoid(np.maximum(upper_y_green - y_val_gr, 0), x_val_gr))
                            min_val_gr = float(np.min(y_val_gr))
                            if upper_y_green != 0:
                                pct_drop_gr = float((upper_y_green - min_val_gr) / abs(upper_y_green) * 100)
                            min_idx_gr = np.argmin(y_val_gr)
                            
                            traces.append({
                                "x": [_to_ms(t_val_gr[min_idx_gr])],
                                "y": [min_val_gr],
                                "mode": "markers",
                                "name": "Min (Gr)",
                                "marker": {"color": "#ef4444", "size": 10, "symbol": "diamond"},
                                "showlegend": False,
                                "hoverinfo": "skip"
                            })
                            
                        traces.append({
                            "x": [_to_ms(t) for t in t_area_green],
                            "y": [float(upper_y_green)] * len(t_area_green),
                            "mode": "lines",
                            "line": {"width": 0},
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        traces.append({
                            "x": [_to_ms(t) for t in t_area_green],
                            "y": [float(y) if np.isfinite(y) else None for y in y_area_green],
                            "mode": "lines",
                            "line": {"width": 0},
                            "fill": "tonexty",
                            "fillcolor": "rgba(34, 197, 94, 0.25)",
                            "showlegend": False,
                            "hoverinfo": "skip"
                        })
                        
                # Baseline Recovery Analysis
                rec_duration = np.nan
                rec_area = np.nan
                rec_started_in = ""
                rec_min_val = np.nan
                rec_pct_drop = np.nan
                
                if baseline_mean is not None:
                    # Look from t_start onwards
                    post_start_mask = resampled_x >= t_start
                    x_post = resampled_x[post_start_mask]
                    y_post = resampled_y[post_start_mask]
                    t_post = resampled_t[post_start_mask]
                    
                    if len(x_post) > 0:
                        # Find where y drops below baseline
                        below_mask = y_post < baseline_mean
                        
                        t_area_start_idx = -1
                        if below_mask[0]: # already below at t_start
                            # Check if it reaches baseline within 5 seconds
                            within_5s_mask = (x_post <= t_start + 5) & (~below_mask)
                            if np.any(within_5s_mask):
                                # Reaches baseline within 5s, so find the NEXT drop below baseline after this recovery
                                recovery_idx = np.where(within_5s_mask)[0][0]
                                next_drop_mask = below_mask[recovery_idx:]
                                if np.any(next_drop_mask):
                                    t_area_start_idx = recovery_idx + np.where(next_drop_mask)[0][0]
                            else:
                                t_area_start_idx = 0
                        else:
                            # Not below at t_start, find the first drop
                            if np.any(below_mask):
                                t_area_start_idx = np.where(below_mask)[0][0]
                                
                        if t_area_start_idx != -1:
                            # We have the start, now find the end (when it goes back above or equals baseline)
                            x_rem = x_post[t_area_start_idx:]
                            y_rem = y_post[t_area_start_idx:]
                            t_rem = t_post[t_area_start_idx:]
                            
                            above_mask = y_rem >= baseline_mean
                            if np.any(above_mask):
                                t_area_end_idx = np.where(above_mask)[0][0]
                                
                                start_s = x_rem[0]
                                end_s = x_rem[t_area_end_idx]
                                
                                if end_s - start_s <= 30:
                                    rec_duration = end_s - start_s
                                    rec_started_in = "Transition" if start_s < t_stand else "Standing"
                                    
                                    x_rec_segment = x_rem[:t_area_end_idx+1]
                                    y_rec_segment = y_rem[:t_area_end_idx+1]
                                    t_rec_segment = t_rem[:t_area_end_idx+1]
                                    
                                    # Area = \int (baseline - y) dt
                                    rec_area = float(np.trapezoid(baseline_mean - y_rec_segment, x_rec_segment))
                                    rec_min_val = float(np.min(y_rec_segment))
                                    if baseline_mean != 0:
                                        rec_pct_drop = float((baseline_mean - rec_min_val) / abs(baseline_mean) * 100)
                                    
                                    # Visual trace
                                    traces.append({
                                        "x": [_to_ms(t_rec_segment[0])],
                                        "y": [float(y_rec_segment[0])],
                                        "mode": "markers",
                                        "name": "Rec Start",
                                        "marker": {"color": "#3b82f6", "size": 13, "symbol": "triangle-down"},
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })
                                    traces.append({
                                        "x": [_to_ms(t_rec_segment[-1])],
                                        "y": [float(y_rec_segment[-1])],
                                        "mode": "markers",
                                        "name": "Rec End",
                                        "marker": {"color": "#3b82f6", "size": 13, "symbol": "triangle-up"},
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })
                                    traces.append({
                                        "x": [_to_ms(t) for t in t_rec_segment],
                                        "y": [float(baseline_mean)] * len(t_rec_segment),
                                        "mode": "lines",
                                        "line": {"width": 0},
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })
                                    traces.append({
                                        "x": [_to_ms(t) for t in t_rec_segment],
                                        "y": [float(y) if np.isfinite(y) else None for y in y_rec_segment],
                                        "mode": "lines",
                                        "line": {"width": 0},
                                        "fill": "tonexty",
                                        "fillcolor": "rgba(59, 130, 246, 0.3)", # Blue
                                        "showlegend": False,
                                        "hoverinfo": "skip"
                                    })

                # Summary Stats Object
                trans_val = float(resampled_y[trans_idx]) if (trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx])) else np.nan
                stand_val = float(resampled_y[start_idx]) if (start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx])) else np.nan
                end_val = float(resampled_y[end_idx]) if (end_idx < len(resampled_y) and np.isfinite(resampled_y[end_idx])) else np.nan
                base_val = baseline_mean if baseline_mean is not None else np.nan
                transition_duration = t_stand - t_start
                or_duration = t_end_marker - t_start
                gr_duration = t_end_marker - t_stand
                
                def _safe_float(v):
                    return float(v) if np.isfinite(v) else None
                
                stats.append({
                    "id": i,
                    "t_start_ms": c1_ms,
                    "baseline": _safe_float(base_val),
                    "transition_time": _safe_float(transition_duration),
                    "or_trans_val": _safe_float(trans_val),
                    "or_duration": _safe_float(or_duration),
                    "or_min_val": _safe_float(min_val_or),
                    "or_pct_drop": _safe_float(pct_drop_or),
                    "or_area_above": _safe_float(area_above_or),
                    "or_area_below": _safe_float(area_below_or),
                    "gr_stand_val": _safe_float(stand_val),
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
                    "end_val": _safe_float(end_val)
                })
                
        i += 1
        
    return traces, shapes, annotations, stats
