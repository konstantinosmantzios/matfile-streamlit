import pandas as pd
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_stats_for_signal(sig_name, resampled_x, resampled_y, comment_list, baseline_window, end_marker_window, use_baseline_area):
    stats = []
    
    i = 0
    while i < len(comment_list):
        c1 = comment_list[i]
        if "transition" in str(c1['comment']).lower():
            if i + 1 < len(comment_list) and str(comment_list[i+1]['comment']).lower().strip(" .") in ["stand", "standing"]:
                c_stand = comment_list[i+1]
                t_start = c1['time_s']
                t_stand = c_stand['time_s']
                t_end_marker = t_stand + end_marker_window
                
                # Baseline
                t_base_start = max(0, t_start - baseline_window)
                base_mask = (resampled_x >= t_base_start) & (resampled_x < t_start)
                base_y = resampled_y[base_mask]
                baseline_mean = float(np.nanmean(base_y)) if np.any(np.isfinite(base_y)) else np.nan
                
                # Transition point
                trans_idx = np.abs(resampled_x - t_start).argmin()
                trans_val = resampled_y[trans_idx] if (trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx])) else np.nan
                
                # Stand point
                start_idx = np.abs(resampled_x - t_stand).argmin()
                stand_val = resampled_y[start_idx] if (start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx])) else np.nan
                
                # End point
                end_idx = np.abs(resampled_x - t_end_marker).argmin()
                end_val = resampled_y[end_idx] if (end_idx < len(resampled_y) and np.isfinite(resampled_y[end_idx])) else np.nan
                
                # Transition Segment Stats (Orange)
                area_above_or, area_below_or = np.nan, np.nan
                min_val_or, pct_drop_or = np.nan, np.nan
                
                if trans_idx < len(resampled_y) and np.isfinite(resampled_y[trans_idx]):
                    area_mask = (resampled_x >= t_start) & (resampled_x <= t_end_marker)
                    x_area = resampled_x[area_mask]
                    y_area = resampled_y[area_mask]
                    
                    if len(x_area) > 0:
                        upper_y = baseline_mean if (use_baseline_area and not np.isnan(baseline_mean)) else resampled_y[trans_idx]
                        valid_mask = np.isfinite(y_area)
                        x_val_or = x_area[valid_mask]
                        y_val_or = y_area[valid_mask]
                        
                        if len(x_val_or) > 1:
                            area_above_or = np.trapezoid(np.maximum(y_val_or - upper_y, 0), x_val_or)
                            area_below_or = np.trapezoid(np.maximum(upper_y - y_val_or, 0), x_val_or)
                            min_val_or = float(np.min(y_val_or))
                            if upper_y != 0:
                                pct_drop_or = (upper_y - min_val_or) / abs(upper_y) * 100
                                
                # Standing Segment Stats (Green)
                area_above_gr, area_below_gr = np.nan, np.nan
                min_val_gr, pct_drop_gr = np.nan, np.nan
                
                if start_idx < len(resampled_y) and np.isfinite(resampled_y[start_idx]):
                    area_mask_green = (resampled_x >= t_stand) & (resampled_x <= t_end_marker)
                    x_area_green = resampled_x[area_mask_green]
                    y_area_green = resampled_y[area_mask_green]
                    
                    if len(x_area_green) > 0:
                        upper_y_green = baseline_mean if (use_baseline_area and not np.isnan(baseline_mean)) else resampled_y[start_idx]
                        valid_mask_gr = np.isfinite(y_area_green)
                        x_val_gr = x_area_green[valid_mask_gr]
                        y_val_gr = y_area_green[valid_mask_gr]
                        
                        if len(x_val_gr) > 1:
                            area_above_gr = np.trapezoid(np.maximum(y_val_gr - upper_y_green, 0), x_val_gr)
                            area_below_gr = np.trapezoid(np.maximum(upper_y_green - y_val_gr, 0), x_val_gr)
                            min_val_gr = float(np.min(y_val_gr))
                            if upper_y_green != 0:
                                pct_drop_gr = (upper_y_green - min_val_gr) / abs(upper_y_green) * 100
                                
                stats.append({
                    "Test Number": len(stats) + 1,
                    "Signal": sig_name,
                    "Transition Time": c1['absolute_time'].strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(c1['absolute_time']) else "",
                    "Transition Duration (s)": round(t_stand - t_start, 1),
                    "Baseline Mean": round(baseline_mean, 2) if not np.isnan(baseline_mean) else "—",
                    "Value at Transition Start": round(trans_val, 2) if not np.isnan(trans_val) else "—",
                    "Min (Transition to End)": round(min_val_or, 2) if not np.isnan(min_val_or) else "—",
                    "Drop % (Transition to End)": round(pct_drop_or, 2) if not np.isnan(pct_drop_or) else "—",
                    "Area Above (Transition to End)": round(area_above_or, 2) if not np.isnan(area_above_or) else "—",
                    "Area Below (Transition to End)": round(area_below_or, 2) if not np.isnan(area_below_or) else "—",
                    "Value at Stand Start": round(stand_val, 2) if not np.isnan(stand_val) else "—",
                    "Min (Stand to End)": round(min_val_gr, 2) if not np.isnan(min_val_gr) else "—",
                    "Drop % (Stand to End)": round(pct_drop_gr, 2) if not np.isnan(pct_drop_gr) else "—",
                    "Area Above (Stand to End)": round(area_above_gr, 2) if not np.isnan(area_above_gr) else "—",
                    "Area Below (Stand to End)": round(area_below_gr, 2) if not np.isnan(area_below_gr) else "—",
                    "End Value": round(end_val, 2) if not np.isnan(end_val) else "—"
                })
                i += 1 # skip the standing comment
        i += 1
    return stats


def create_test_plot(t_base_start, t_start, t_stand, t_end_marker, x_fp, y_fp, x_cbf, y_cbf, baseline_fp, baseline_cbf, use_baseline_area, test_number):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.2)
    
    def plot_signal(ax, x, y, baseline, color, title):
        mask = (x >= t_base_start) & (x <= t_end_marker) & ~np.isnan(y)
        x_plot = x[mask]
        y_plot = y[mask]
        ax.plot(x_plot, y_plot, color=color, linewidth=1.5)
        
        if not np.isnan(baseline):
            ax.axhline(baseline, color='blue', linestyle='--', linewidth=1.5, label='Baseline')
            
        mask_or = (x_plot >= t_start) & (x_plot <= t_end_marker)
        x_or = x_plot[mask_or]
        y_or = y_plot[mask_or]
        if len(x_or) > 0:
            upper_y_or = baseline if (use_baseline_area and not np.isnan(baseline)) else y_or[0]
            ax.fill_between(x_or, upper_y_or, y_or, color='#f97316', alpha=0.25, label='Transition to End')
            min_idx = np.argmin(y_or)
            ax.plot(x_or[min_idx], y_or[min_idx], marker='D', color='red', markersize=6)
            
        mask_gr = (x_plot >= t_stand) & (x_plot <= t_end_marker)
        x_gr = x_plot[mask_gr]
        y_gr = y_plot[mask_gr]
        if len(x_gr) > 0:
            upper_y_gr = baseline if (use_baseline_area and not np.isnan(baseline)) else y_gr[0]
            ax.fill_between(x_gr, upper_y_gr, y_gr, color='#22c55e', alpha=0.35, label='Stand to End')
            min_idx = np.argmin(y_gr)
            ax.plot(x_gr[min_idx], y_gr[min_idx], marker='D', color='darkred', markersize=6)
            
        ax.axvline(t_start, color='gray', linestyle=':', label='Transition Start')
        ax.axvline(t_stand, color='black', linestyle=':', label='Stand Start')
        ax.axvline(t_end_marker, color='purple', linestyle=':', label='End Marker')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    plot_signal(ax1, x_fp, y_fp, baseline_fp, '#1f77b4', 'Finger Pressure')
    plot_signal(ax2, x_cbf, y_cbf, baseline_cbf, '#d62728', 'CBF')
    ax2.set_xlabel('Time (s)', fontsize=10)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=6, fontsize=8)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_excel_report(df_filtered, df_resampled, metadata_dict, stats_list, resample_label="Resampled Data", 
                          comment_list=None, baseline_window=None, end_marker_window=None, use_baseline_area=None,
                          fp_signal=None, cbf_signal=None, raw_df=None, agg5_map=None, result_df=None, is_beat_mode=False):

    """
    Generates a multi-sheet Excel file containing the analysis data.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        
        def write_styled_table(df_out, sheet_name, style="Table Style Light 9", col_formats=None):
            if df_out.empty:
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)
                return
            
            # Write data without header
            df_out.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=1)
            worksheet = writer.sheets[sheet_name]
            
            # Define table
            max_row, max_col = df_out.shape
            
            columns_config = []
            for c in df_out.columns:
                col_dict = {'header': str(c)}
                if col_formats and c in col_formats:
                    col_dict['format'] = col_formats[c]
                columns_config.append(col_dict)
                
            worksheet.add_table(0, 0, max_row, max_col - 1, {
                'columns': columns_config,
                'style': style
            })
            
            # Auto-adjust column width
            for i, col in enumerate(df_out.columns):
                col_data = df_out.iloc[:, i].dropna()
                if not col_data.empty:
                    max_data_len = col_data.map(lambda x: len(str(x))).max()
                    max_data_len = int(max_data_len) if pd.notna(max_data_len) else 0
                else:
                    max_data_len = 0
                max_len = max(max_data_len, len(str(col))) + 4
                worksheet.set_column(i, i, min(max_len, 70))
        
        # 1. Filtered Data
        cols_to_keep = [c for c in df_filtered.columns if c not in ["block_index", "time_mmss_millis", "absolute_time"]]
        if "absolute_time" in df_filtered.columns:
            cols_to_keep = ["absolute_time"] + cols_to_keep
            
        df_filt_clean = df_filtered[cols_to_keep].copy()
        
        # Format datetimes
        if 'absolute_time' in df_filt_clean.columns:
            df_filt_clean['absolute_time'] = df_filt_clean['absolute_time'].dt.tz_localize(None)
            
        write_styled_table(df_filt_clean, "Filtered Data", "Table Style Medium 2")
        
        # 2. Resampled Data
        sheet_resampled_name = str(resample_label).replace(':', '').replace('/', '')[:31]
        
        if df_resampled is not None and not df_resampled.empty:
            df_res_clean = df_resampled.copy()
            if 'absolute_time' in df_res_clean.columns:
                df_res_clean['absolute_time'] = df_res_clean['absolute_time'].dt.tz_localize(None)
            write_styled_table(df_res_clean, sheet_resampled_name, "Table Style Medium 2")
        else:
            write_styled_table(pd.DataFrame(columns=["No Data"]), sheet_resampled_name)
        
        # 3. Metadata
        df_meta = pd.DataFrame(list(metadata_dict.items()), columns=["Parameter", "Value"])
        write_styled_table(df_meta, "Metadata", "Table Style Light 11")
        
        # 4. Statistics
        if stats_list:
            fmt_orange = writer.book.add_format({'bg_color': '#FFE6CC'})
            fmt_green = writer.book.add_format({'bg_color': '#E2EFDA'})
            
            stats_col_formats = {
                "Min (Transition to End)": fmt_orange,
                "Drop % (Transition to End)": fmt_orange,
                "Area Above (Transition to End)": fmt_orange,
                "Area Below (Transition to End)": fmt_orange,
                "Min (Stand to End)": fmt_green,
                "Drop % (Stand to End)": fmt_green,
                "Area Above (Stand to End)": fmt_green,
                "Area Below (Stand to End)": fmt_green,
            }
            
            df_stats = pd.DataFrame(stats_list)
            write_styled_table(df_stats, "Statistics", "Table Style Light 1", stats_col_formats)
        else:
            write_styled_table(pd.DataFrame(columns=["No Stats Generated"]), "Statistics")
            
        # 5. Definitions
        definitions_data = {
            "Metric": [
                "Test Number",
                "Transition Time",
                "Transition Duration (s)",
                "Baseline Mean",
                "Value at Transition Start",
                "Min (Transition to End)",
                "Drop % (Transition to End)",
                "Area Above (Transition to End)",
                "Area Below (Transition to End)",
                "Value at Stand Start",
                "Min (Stand to End)",
                "Drop % (Stand to End)",
                "Area Above (Stand to End)",
                "Area Below (Stand to End)",
                "End Value"
            ],
            "Definition": [
                "Sequential number of the Supine-to-Standing test (1, 2, 3...) for the given signal.",
                "The absolute time the transition started (at the 'transition' comment).",
                "Time elapsed from 'transition' comment to 'stand' comment.",
                "Mean value of the signal during the user-defined baseline period.",
                "Signal value exactly at the 'transition' comment.",
                "Minimum signal value from the 'transition' comment until the 'End Marker'.",
                "Percentage drop from the Baseline (or Value at Transition Start) down to Min (Transition to End).",
                "Area of the signal above the Baseline (or Transition Value) from 'transition' to 'End Marker'.",
                "Area of the signal below the Baseline (or Transition Value) from 'transition' to 'End Marker'.",
                "Signal value exactly at the 'stand' comment.",
                "Minimum signal value from the 'stand' comment until the 'End Marker'.",
                "Percentage drop from the Baseline (or Stand Value) down to Min (Stand to End).",
                "Area of the signal above the Baseline (or Stand Value) from 'stand' to 'End Marker'.",
                "Area of the signal below the Baseline (or Stand Value) from 'stand' to 'End Marker'.",
                "Signal value exactly at the 'End Marker' time."
            ]
        }
        df_definitions = pd.DataFrame(definitions_data)
        write_styled_table(df_definitions, "Definitions", "Table Style Light 11")
        
        # We can force column width for definitions to be a bit wider for readability
        worksheet_def = writer.sheets["Definitions"]
        worksheet_def.set_column('A:A', 25)
        worksheet_def.set_column('B:B', 90)


        # 6. Plots (Headless generation)
        if comment_list and fp_signal and cbf_signal:
            worksheet_plots = writer.book.add_worksheet("Plots")
            
            if is_beat_mode and agg5_map is not None and raw_df is not None:
                y_fp = agg5_map.get(fp_signal, np.array([]))
                y_cbf = agg5_map.get(cbf_signal, np.array([]))
                x_data = raw_df['time_s'].values
            elif not is_beat_mode and result_df is not None:
                y_fp = result_df.get(fp_signal, pd.Series()).values
                y_cbf = result_df.get(cbf_signal, pd.Series()).values
                x_data = result_df['time_s'].values
            else:
                y_fp, y_cbf, x_data = None, None, None
                
            if x_data is not None and len(y_fp) > 0 and len(y_cbf) > 0:
                plot_row = 1
                test_number = 1
                i = 0
                while i < len(comment_list):
                    c1 = comment_list[i]
                    if "transition" in str(c1.get('comment', '')).lower():
                        if i + 1 < len(comment_list) and str(comment_list[i+1].get('comment', '')).lower().strip(" .") in ["stand", "standing"]:
                            c_stand = comment_list[i+1]
                            t_start = c1['time_s']
                            t_stand = c_stand['time_s']
                            t_end_marker = t_stand + end_marker_window
                            t_base_start = max(0, t_start - baseline_window)
                            
                            base_mask = (x_data >= t_base_start) & (x_data < t_start)
                            y_fp_base = y_fp[base_mask]
                            baseline_fp = float(np.nanmean(y_fp_base)) if len(y_fp_base) > 0 else np.nan
                            
                            y_cbf_base = y_cbf[base_mask]
                            baseline_cbf = float(np.nanmean(y_cbf_base)) if len(y_cbf_base) > 0 else np.nan
                            
                            img_buf = create_test_plot(t_base_start, t_start, t_stand, t_end_marker, 
                                                       x_data, y_fp, x_data, y_cbf, 
                                                       baseline_fp, baseline_cbf, use_baseline_area, test_number)
                                                       
                            worksheet_plots.write(plot_row, 0, f"Test Number {test_number}", writer.book.add_format({'bold': True, 'font_size': 14}))
                            worksheet_plots.insert_image(plot_row + 2, 0, f'test_{test_number}.png', {'image_data': img_buf})
                            
                            plot_row += 35 
                            test_number += 1
                            i += 1 
                    i += 1

    output.seek(0)

    return output
