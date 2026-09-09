from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="BMP Analyzer",
    page_icon="BMP",
    layout="wide",
)


REFERENCE_VALUES = {
    "Cellulose": 360,
    "Cow Manure": 200,
    "Food Waste": 450,
    "Grass Silage": 350,
    "Corn Silage": 320,
    "Pig Manure": 300,
    "Wastewater Sludge": 240,
}


@st.cache_data(show_spinner=False)
def read_workbook(uploaded_file: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = pd.ExcelFile(BytesIO(uploaded_file), engine="openpyxl")
    sheet_lookup = {name.strip().lower(): name for name in workbook.sheet_names}

    raw_name = sheet_lookup.get("raw data")
    doe_name = sheet_lookup.get("doe")
    if raw_name is None or doe_name is None:
        raise ValueError(
            "The workbook must contain sheets named 'Raw data' and 'DOE'. "
            f"Found: {', '.join(workbook.sheet_names)}"
        )

    raw_sheet = pd.read_excel(workbook, sheet_name=raw_name, header=None)
    header_rows = raw_sheet.apply(
        lambda row: row.astype(str).str.strip().str.lower().isin({"run id", "data"}).any(),
        axis=1,
    )
    if not header_rows.any():
        raise ValueError("Could not find the header row in the Raw data sheet.")
    header_index = header_rows[header_rows].index[0]
    raw_data = raw_sheet.iloc[header_index + 1 :].copy()
    raw_data.columns = [
        str(value).strip() if pd.notna(value) else f"Unnamed_{index}"
        for index, value in enumerate(raw_sheet.iloc[header_index])
    ]
    raw_data = raw_data.rename(
        columns={column: "Run ID" for column in raw_data.columns if str(column).lower() == "data"}
    )
    raw_data = raw_data.dropna(how="all").reset_index(drop=True)

    doe_sheet = pd.read_excel(workbook, sheet_name=doe_name, header=None)
    doe_header_rows = doe_sheet.apply(
        lambda row: row.astype(str).str.strip().str.lower().isin({"type", "run id"}).sum() >= 2,
        axis=1,
    )
    if not doe_header_rows.any():
        raise ValueError("Could not find the header row in the DOE sheet.")
    doe_header_index = doe_header_rows[doe_header_rows].index[0]
    doe_data = doe_sheet.iloc[doe_header_index + 1 :].copy()
    doe_data.columns = [
        str(value).strip() if pd.notna(value) else f"Unnamed_{index}"
        for index, value in enumerate(doe_sheet.iloc[doe_header_index])
    ]
    return raw_data.dropna(how="all").reset_index(drop=True), doe_data.dropna(how="all").reset_index(drop=True)


def clean_raw_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Run ID", "Day", "Daily Biogas (ml)", "CH4%"}
    missing = sorted(required - set(raw_data.columns))
    if missing:
        raise ValueError("Missing required Raw data columns: " + ", ".join(missing))

    data = raw_data.copy()
    data["Run ID"] = data["Run ID"].astype(str).str.strip()
    for column in ["Day", "Daily Biogas (ml)", "CH4%", "CO2%"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    summaries = []
    cleaned_runs = []
    for run_id, run_data in data.dropna(subset=["Run ID"]).groupby("Run ID", sort=False):
        run_data = run_data.sort_values("Day").copy()
        duplicate_mask = run_data.duplicated("Day", keep="first")
        duplicate_count = int(duplicate_mask.sum())
        run_data = run_data.loc[~duplicate_mask].copy()
        days = sorted(run_data["Day"].dropna().unique())
        expected_days = list(range(int(min(days)), int(max(days)) + 1)) if days else []
        missing_days = sorted(set(expected_days) - set(days))

        interpolated = 0
        for column in ["Daily Biogas (ml)", "CH4%"]:
            null_count = int(run_data[column].isna().sum())
            if null_count:
                run_data[column] = run_data[column].interpolate(method="linear", limit_area="inside")
                interpolated += null_count

        run_data["Daily Methane (ml)"] = run_data["Daily Biogas (ml)"] * run_data["CH4%"] / 100
        run_data["Cumulative Methane (mL)"] = run_data["Daily Methane (ml)"].cumsum()
        run_data["Cumulative Biogas (mL)"] = run_data["Daily Biogas (ml)"].cumsum()
        cleaned_runs.append(run_data)
        summaries.append(
            {
                "Run ID": run_id,
                "First day": min(days) if days else np.nan,
                "Last day": max(days) if days else np.nan,
                "Recorded timepoints": len(days),
                "Missing days": ", ".join(map(str, missing_days)) or "None",
                "Duplicate rows": duplicate_count,
                "Values interpolated": interpolated,
            }
        )

    if not cleaned_runs:
        raise ValueError("No valid run rows were found.")
    return pd.concat(cleaned_runs, ignore_index=True), pd.DataFrame(summaries)


def plateau_results(processed: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    incomplete = set(checks.loc[checks["Missing days"] != "None", "Run ID"])
    results = []
    for run_id, run_data in processed.groupby("Run ID", sort=False):
        run_data = run_data.sort_values("Day")
        total = run_data["Cumulative Methane (mL)"].iloc[-1]
        last_three = run_data.tail(3)["Daily Methane (ml)"].mean()
        criterion = last_three / total * 100 if total and pd.notna(total) else np.nan
        if run_id in incomplete:
            status = "INCOMPLETE TIME SERIES"
            reached = False
        elif len(run_data) < 3 or pd.isna(criterion):
            status = "NOT ENOUGH VALID DATA"
            reached = False
        else:
            reached = bool(criterion < 1)
            status = "REACHED PLATEAU" if reached else "NOT REACHED PLATEAU"
        results.append(
            {
                "Run ID": run_id,
                "Status": status,
                "Time series complete": run_id not in incomplete,
                "Days recorded": len(run_data),
                "Total Methane (mL)": total,
                "% of Total in last 3 days": criterion,
            }
        )
    return pd.DataFrame(results)


def calculate_bmp(processed: pd.DataFrame, doe: pd.DataFrame, plateau: pd.DataFrame) -> pd.DataFrame:
    required = {"Type", "Run ID"}
    if not required.issubset(doe.columns):
        raise ValueError("DOE must contain Type and Run ID columns.")

    doe = doe.copy()
    doe["Run ID"] = doe["Run ID"].astype(str).str.strip()
    blanks = doe[doe["Type"].astype(str).str.contains("Blank", case=False, na=False)]["Run ID"].unique()
    blank_data = processed[processed["Run ID"].isin(blanks)]
    if blank_data.empty:
        raise ValueError("No blank controls were found in the raw data.")

    blank_average = blank_data.groupby("Day")["Cumulative Methane (mL)"].mean().rename("Blank methane")
    corrected = processed.join(blank_average, on="Day")
    corrected["Net methane"] = corrected["Cumulative Methane (mL)"] - corrected["Blank methane"]

    records = []
    for _, row in doe.iterrows():
        run_id = row["Run ID"]
        if run_id in blanks:
            continue
        mass = pd.to_numeric(row.get("g.1"), errors="coerce")
        vs_percent = pd.to_numeric(row.get("VS_FS"), errors="coerce")
        if pd.isna(mass) or pd.isna(vs_percent) or mass <= 0 or vs_percent <= 0:
            continue
        vs_grams = mass * vs_percent / 100
        run_data = corrected[corrected["Run ID"] == run_id].sort_values("Day")
        if run_data.empty or vs_grams <= 0:
            continue
        records.append(
            {
                "Run ID": run_id,
                "Type": row.get("Type", ""),
                "VS (g)": vs_grams,
                "Final net methane (mL)": run_data["Net methane"].iloc[-1],
                "BMP (mL CH4/g VS)": run_data["Net methane"].iloc[-1] / vs_grams,
            }
        )
    return pd.DataFrame(records)


def table_html(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "<p>No results available.</p>"
    return dataframe.round(2).to_html(index=False, classes="data-table", border=0)


def chart_uri(figure: plt.Figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_report(
    processed: pd.DataFrame,
    checks: pd.DataFrame,
    plateau: pd.DataFrame,
    bmp: pd.DataFrame,
    doe_data: pd.DataFrame,
    source_name: str,
) -> str:
    processed = processed.copy()
    processed["Replicate group"] = (
        processed["Run ID"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).fillna("Other")
    )

    final_by_run = (
        processed.sort_values("Day")
        .groupby(["Replicate group", "Run ID"], as_index=False)
        .agg(
            Final_methane_mL=("Cumulative Methane (mL)", "last"),
            Last_day=("Day", "last"),
            Maximum_daily_methane_mL=("Daily Methane (ml)", "max"),
        )
    )

    variability = (
        final_by_run.groupby("Replicate group")
        .agg(
            Replicates=("Run ID", "nunique"),
            Mean_final_methane_mL=("Final_methane_mL", "mean"),
            SD_final_methane_mL=("Final_methane_mL", lambda values: values.std(ddof=1)),
            Minimum_final_methane_mL=("Final_methane_mL", "min"),
            Maximum_final_methane_mL=("Final_methane_mL", "max"),
        )
        .reset_index()
    )
    variability["CV_percent"] = np.where(
        variability["Mean_final_methane_mL"].abs() > 0,
        variability["SD_final_methane_mL"] / variability["Mean_final_methane_mL"].abs() * 100,
        np.nan,
    )
    variability["Assessment"] = np.select(
        [variability["Replicates"] < 3, variability["CV_percent"] >= 20],
        ["Not a triplicate in the loaded data", "High variability (CV >= 20%)"],
        default="Acceptable variability (CV < 20%)",
    )

    plateau_summary = plateau.copy()
    duplicate_count = int(checks["Duplicate rows"].sum()) if "Duplicate rows" in checks.columns else 0
    null_count = int(checks["Values interpolated"].sum()) if "Values interpolated" in checks.columns else 0
    not_plateaued = [] if plateau_summary.empty else plateau_summary.loc[
        plateau_summary["Status"] != "REACHED PLATEAU", "Run ID"
    ].tolist()

    series_data = processed.copy()
    series_data["Daily methane percent"] = np.where(
        series_data["Daily Biogas (ml)"].abs() > 0,
        series_data["Daily Methane (ml)"] / series_data["Daily Biogas (ml)"] * 100,
        np.nan,
    )
    replicate_series = (
        series_data.groupby(["Replicate group", "Day"], as_index=False)
        .agg(
            Mean_accumulated_biogas_mL=("Cumulative Biogas (mL)", "mean"),
            Mean_methane_content_percent=("Daily methane percent", "mean"),
        )
    )

    replicate_average_chart = ""
    replicate_summary = pd.DataFrame()
    if not replicate_series.empty:
        replicate_groups = sorted(replicate_series["Replicate group"].unique().tolist())
        panel_columns = 2
        panel_rows = int(np.ceil(len(replicate_groups) / panel_columns))
        replicate_figure, axes = plt.subplots(
            panel_rows, panel_columns, figsize=(14, 4.5 * panel_rows), squeeze=False
        )
        axes = axes.flatten()

        for axis, group_name in zip(axes, replicate_groups):
            group_series = replicate_series[replicate_series["Replicate group"] == group_name].sort_values("Day")
            methane_axis = axis.twinx()
            axis.plot(
                group_series["Day"],
                group_series["Mean_accumulated_biogas_mL"],
                color="tab:blue",
                marker="o",
                linewidth=2,
                label="Biogas",
            )
            axis.set_title(str(group_name))
            axis.set_xlabel("Day")
            axis.set_ylabel("Biogas (mL)")

            methane_axis.plot(
                group_series["Day"],
                group_series["Mean_methane_content_percent"],
                color="tab:orange",
                linestyle="None",
                marker="s",
                markersize=5,
                linewidth=0,
                label="CH4 %",
            )
            methane_axis.set_ylabel("CH4 (%)")
            methane_axis.set_ylim(0, 100)

        for axis in axes[len(replicate_groups) :]:
            axis.axis("off")

        replicate_figure.suptitle("Replicate-averaged biogas and methane time series", fontsize=15, y=1.01)
        replicate_figure.tight_layout(rect=(0, 0, 1, 0.98))
        replicate_average_chart = f'<img src="{chart_uri(replicate_figure)}" alt="Replicate average multipanel">'

        replicate_summary = (
            replicate_series.groupby("Replicate group", as_index=False)
            .agg(
                Final_biogas_mL=("Mean_accumulated_biogas_mL", "last"),
                Mean_CH4_percent=("Mean_methane_content_percent", "mean"),
            )
        )

    cumulative_figure, cumulative_axis = plt.subplots(figsize=(10, 5))
    for run_id, run_data in processed.groupby("Run ID"):
        cumulative_axis.plot(run_data["Day"], run_data["Cumulative Methane (mL)"], marker="o", label=run_id)
    cumulative_axis.set_title("Cumulative methane by run")
    cumulative_axis.set_xlabel("Day")
    cumulative_axis.set_ylabel("Cumulative methane (mL)")
    cumulative_axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    cumulative_figure.tight_layout()
    cumulative_chart = chart_uri(cumulative_figure)

    variability_figure, variability_axis = plt.subplots(figsize=(9, 5))
    variability_axis.bar(
        variability["Replicate group"],
        variability["Mean_final_methane_mL"],
        yerr=variability["SD_final_methane_mL"].fillna(0),
        capsize=5,
    )
    variability_axis.set_title("Mean final methane and replicate variability")
    variability_axis.set_xlabel("Replicate group")
    variability_axis.set_ylabel("Final cumulative methane (mL)")
    variability_figure.tight_layout()
    variability_chart = chart_uri(variability_figure)

    bmp_section = "<p>No BMP results were generated.</p>"
    bmp_chart = ""
    if bmp is not None and not bmp.empty:
        bmp_figure, bmp_axis = plt.subplots(figsize=(9, 5))
        bmp_axis.bar(bmp["Run ID"], bmp["BMP (mL CH4/g VS)"])
        bmp_axis.set_title("Blank-corrected BMP by reactor")
        bmp_axis.set_xlabel("Run ID")
        bmp_axis.set_ylabel("BMP (mL CH4/g VS)")
        bmp_axis.tick_params(axis="x", rotation=45)
        bmp_figure.tight_layout()
        bmp_chart = f'<img src="{chart_uri(bmp_figure)}" alt="BMP by reactor">'
        bmp_section = table_html(bmp)

    doe_columns = [column for column in doe_data.columns if not str(column).startswith("Unnamed")]
    doe_priority = [column for column in ["Run ID", "Type", "Feedstock", "g", "g.1"] if column in doe_columns]
    doe_columns = doe_priority + [column for column in doe_columns if column not in doe_priority]
    doe_section = table_html(doe_data[doe_columns]) if not doe_data.empty else "<p>No DOE data available.</p>"

    highlights = [
        f'Runs analysed: {final_by_run["Run ID"].nunique()}',
        f'Processed data rows: {len(processed)}',
        f'Duplicate rows removed: {duplicate_count}',
        f'Values interpolated: {null_count}',
        f'Runs not at plateau: {len(not_plateaued)}',
    ]
    if not replicate_summary.empty:
        top_group = replicate_summary.sort_values("Final_biogas_mL", ascending=False).iloc[0]
        low_group = replicate_summary.sort_values("Final_biogas_mL", ascending=True).iloc[0]
        gas_delta = top_group["Final_biogas_mL"] - low_group["Final_biogas_mL"]
        high_ch4_group = replicate_summary.sort_values("Mean_CH4_percent", ascending=False).iloc[0]
        low_ch4_group = replicate_summary.sort_values("Mean_CH4_percent", ascending=True).iloc[0]
        ch4_delta = high_ch4_group["Mean_CH4_percent"] - low_ch4_group["Mean_CH4_percent"]
        highlights.extend([
            f'Replicate comparison: {top_group["Replicate group"]} finished with the highest mean cumulative biogas ({top_group["Final_biogas_mL"]:.1f} mL), compared with {low_group["Replicate group"]} at {low_group["Final_biogas_mL"]:.1f} mL (Δ {gas_delta:.1f} mL).',
            f'Average methane content also varied materially: {high_ch4_group["Replicate group"]} had the highest mean CH4 ({high_ch4_group["Mean_CH4_percent"]:.1f}%), while {low_ch4_group["Replicate group"]} was lowest ({low_ch4_group["Mean_CH4_percent"]:.1f}%, Δ {ch4_delta:.1f} percentage points).',
        ])

    events = []
    if duplicate_count:
        events.append(f"{duplicate_count} duplicate day records were removed during cleaning.")
    if null_count:
        events.append(f"{null_count} internal missing values were interpolated during cleaning.")
    for group_name, group_data in final_by_run.groupby("Replicate group"):
        highest_run = group_data.loc[group_data["Final_methane_mL"].idxmax()]
        lowest_run = group_data.loc[group_data["Final_methane_mL"].idxmin()]
        events.append(
            f'{group_name}: highest final methane was {highest_run["Run ID"]} '
            f'({highest_run["Final_methane_mL"]:.2f} mL); lowest was '
            f'{lowest_run["Run ID"]} ({lowest_run["Final_methane_mL"]:.2f} mL).'
        )

    remarks = [
        "BMP values are preliminary because the plateau criterion was not reached."
        if not_plateaued
        else "All runs met the plateau criterion before BMP calculation."
    ]
    for _, row in variability.iterrows():
        cv_text = f'{row["CV_percent"]:.1f}%' if pd.notna(row["CV_percent"]) else "not available"
        remarks.append(
            f'{row["Replicate group"]}: {row["Assessment"]}; n={int(row["Replicates"])}; CV={cv_text}.'
        )

    list_html = lambda values: "".join(f"<li>{html.escape(value)}</li>" for value in values)
    source_label = html.escape(Path(source_name).name)
    warning_class = ' class="warning"' if not_plateaued else ""
    report_html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>BMP analysis report</title><style>
body {{ font-family: Arial, sans-serif; color: #202124; max-width: 1150px; margin: 32px auto; padding: 0 24px; line-height: 1.45; }}
h1 {{ color: #174a5b; border-bottom: 3px solid #8fc9bd; padding-bottom: 10px; }} h2 {{ color: #256b6b; margin-top: 32px; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }} .data-table th, .data-table td {{ border: 1px solid #d8e0e3; padding: 8px; text-align: right; }} .data-table th {{ background: #e7f2f0; color: #174a5b; }} .data-table td:first-child, .data-table th:first-child {{ text-align: left; }} img {{ max-width: 100%; height: auto; border: 1px solid #d8e0e3; }} .meta {{ color: #5f6368; }} .warning {{ background: #fff4d6; border-left: 5px solid #d99b21; padding: 12px 16px; }}
</style></head><body><h1>BMP analysis report</h1><p class="meta">Source workbook: {source_label}</p>
<h2>Highlights</h2><ul>{list_html(highlights)}</ul>
<h2>DOE design and reactor details</h2><p class="meta">Values are reproduced from the DOE sheet; column names and units are retained as recorded.</p>{doe_section}
<h2>Noticeable data and events</h2><ul>{list_html(events) or '<li>No events were flagged.</li>'}</ul>
<h2>Main remarks</h2><p class="meta">Note: n is the number of replicate runs in that group, and CV is the coefficient of variation, calculated as SD ÷ mean × 100%, expressed as a percentage to show relative variability between replicates.</p><div{warning_class}><ul>{list_html(remarks)}</ul></div>
<h2>Plateau results</h2>{table_html(plateau_summary)}<img src="{cumulative_chart}" alt="Cumulative methane by run">
<h2>Replicate variability</h2><p>CV is standard deviation divided by the group mean. Fewer than three runs is not a complete triplicate.</p>{table_html(variability)}<img src="{variability_chart}" alt="Mean final methane and replicate variability">
<h2>Replicate average multipanel</h2>{replicate_average_chart or '<p>No replicate-average multipanel was available.</p>'}
<h2>BMP results</h2>{bmp_section}{bmp_chart}</body></html>'''
    return report_html


def main() -> None:
    st.title("BMP Analyzer")
    st.caption("Upload an Excel workbook containing Raw data and DOE sheets.")
    uploaded = st.file_uploader("Excel workbook", type=["xlsx", "xlsm", "xls"])
    if uploaded is None:
        st.info("Upload a workbook to begin.")
        return

    try:
        raw_data, doe_data = read_workbook(uploaded.getvalue())
        processed, checks = clean_raw_data(raw_data)
        plateau = plateau_results(processed, checks)
    except Exception as exc:
        st.error(str(exc))
        return

    bmp = pd.DataFrame()
    with st.sidebar:
        st.header("Analysis")
        calculate = st.checkbox("Calculate blank-corrected BMP", value=True)
        report_name = f"{Path(uploaded.name).stem}_BMP_report.html"

    if calculate:
        try:
            bmp = calculate_bmp(processed, doe_data, plateau)
        except Exception as exc:
            st.warning(f"BMP calculation unavailable: {exc}")

    tab_overview, tab_series, tab_plateau, tab_bmp, tab_report = st.tabs(
        ["Overview", "Time series", "Plateau", "BMP", "Report"]
    )
    with tab_overview:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Runs", processed["Run ID"].nunique())
        metric_columns[1].metric("Raw rows", len(raw_data))
        metric_columns[2].metric("First day", int(processed["Day"].min()))
        metric_columns[3].metric("Last day", int(processed["Day"].max()))
        st.subheader("Data checks")
        st.dataframe(checks, use_container_width=True, hide_index=True)
        st.subheader("Raw data")
        st.dataframe(raw_data, use_container_width=True, hide_index=True)

    with tab_series:
        selected_runs = st.multiselect(
            "Runs to display",
            options=processed["Run ID"].unique().tolist(),
            default=processed["Run ID"].unique().tolist(),
        )
        figure, axis = plt.subplots(figsize=(12, 6))
        for run_id in selected_runs:
            run_data = processed[processed["Run ID"] == run_id]
            axis.plot(run_data["Day"], run_data["Cumulative Biogas (mL)"], marker="o", label=run_id)
        axis.set_xlabel("Day")
        axis.set_ylabel("Cumulative biogas (mL)")
        axis.set_title("Cumulative biogas by run")
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.grid(alpha=0.25)
        st.pyplot(figure)
        plt.close(figure)

    with tab_plateau:
        st.dataframe(plateau, use_container_width=True, hide_index=True)

    with tab_bmp:
        if bmp.empty:
            st.info("No BMP results are available. Check the DOE mass and VS_FS columns.")
        else:
            st.dataframe(bmp, use_container_width=True, hide_index=True)
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.bar(bmp["Run ID"], bmp["BMP (mL CH4/g VS)"])
            axis.set_ylabel("BMP (mL CH4/g VS)")
            axis.set_xlabel("Run ID")
            axis.tick_params(axis="x", rotation=45)
            st.pyplot(figure)
            plt.close(figure)

    with tab_report:
        report = make_report(processed, checks, plateau, bmp, doe_data, uploaded.name)
        st.download_button(
            "Download HTML report",
            data=report,
            file_name=report_name,
            mime="text/html",
        )
        st.components.v1.html(report, height=700, scrolling=True)


if __name__ == "__main__":
    main()
