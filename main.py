import argparse
import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


PALETTE = {
    "steps": "#F08A24",
    "sleep": "#4B9FE1",
    "heart": "#E0454F",
}


# Install and set xkcd Script font if available
if "xkcd Script" not in [f.name for f in font_manager.fontManager.ttflist]:
    print("Loading xkcd Script font...")
    if Path("./xkcd-script.ttf").exists():
        font_manager.fontManager.addfont("./xkcd-script.ttf")


# Fix dash pattern issue
_original_line2d_draw = mlines.Line2D.draw


def _safe_line2d_draw(self, renderer):
    offset, pattern = self._dash_pattern
    if pattern is None or not any(value > 0 for value in pattern):
        self._dash_pattern = (0.0, [1.5])
    return _original_line2d_draw(self, renderer)


mlines.Line2D.draw = _safe_line2d_draw


def parse_arguments() -> argparse.Namespace:
    default_year = 2025
    while datetime(default_year + 1, 6, 30) < datetime.today():
        default_year += 1

    parser = argparse.ArgumentParser(
        description="Generate yearly steps, sleep patterns, and heart-rate visualizations from Mi Fitness data."
    )
    parser.add_argument("input_csv", type=Path,
                        help="Path to the exported hlth_center_fitness_data CSV file.")
    parser.add_argument("--year", type=int, default=default_year,
                        help=f"Target year to analyze (default: {default_year}).")
    parser.add_argument("--tz", dest="tz_offset", type=int, default=8,
                        help="Timezone offset from UTC in hours (default: +8).")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Directory for generated plots (default: ./output).")
    parser.add_argument("--format", choices=["png", "jpg", "svg"], default="png",
                        help="Image format for plots (default: png).")
    parser.add_argument("--font", default="xkcd Script",
                        help="Matplotlib font family for plots (default: xkcd Script).")
    return parser.parse_args()


def safe_json_loads(value: Any) -> dict[str, Any]:
    """Safely parse JSON strings or return an empty dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


@contextmanager
def themed_axes(font_family: str, figsize=(10, 6)):
    """Context manager to set/reset Matplotlib font styles."""
    with plt.xkcd():
        previous_font = plt.rcParams.get("font.family")
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=figsize)
        cozy_axes(ax)
        try:
            yield fig, ax
        finally:
            plt.rcParams["font.family"] = previous_font


def cozy_axes(ax: plt.Axes, facecolor: str | None = None) -> None:
    """Apply gentle colors, spine accents, and ticks."""
    background = facecolor or "#f4f6f8"
    spine_color = "#9ca9bf"
    tick_color = "#4f5969"
    ax.set_facecolor(background)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.1)
    ax.tick_params(colors=tick_color)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def export_plot(fig: plt.Figure, output_path: Path, image_format: str) -> None:
    fig.tight_layout()
    fig.savefig(output_path, format=image_format)
    plt.close(fig)
    # print(f"Saved plot => {output_path}")


class CSVFileFormatError(Exception):
    """Custom exception for unexpected CSV file format."""
    pass


def prepare_dataframe(csv_path: Path, tz_offset: float, year: int) -> pd.DataFrame:
    """Prepare and filter the dataframe for the specified year."""
    df = pd.read_csv(csv_path)
    if df.columns.tolist() != ['Uid', 'Sid', 'Key', 'Time', 'Value', 'UpdateTime']:
        raise CSVFileFormatError("Unexpected CSV format.")

    local_time = pd.to_datetime(df["Time"], unit="s") + pd.Timedelta(hours=tz_offset)
    df["local_time"] = local_time
    df["date"] = df["local_time"].dt.date
    df["month"] = df["local_time"].dt.to_period("M")
    df["weekday"] = df["local_time"].dt.day_name()
    df = df.sort_values("local_time")
    return df[df["local_time"].dt.year == year]


def ensure_output_dir(path: Path) -> Path:
    """Ensure the output directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(minutes: float) -> str:
    """Format duration in minutes to hours and minutes."""
    mins = int(round(minutes))
    hours = mins // 60
    remaining = mins % 60
    return f"{hours}h {remaining}m"


def describe_top_days(df: pd.DataFrame, value_column: str, label: str, formatter: callable = None) -> None:
    """Print the top 3 days with the highest values for a given column."""
    top3 = df.nlargest(3, value_column)
    if top3.empty:
        print(f"No data available for {label}.", file=sys.stderr)
        return

    print(f"Top 3 {label} days:")
    for _, row in top3.iterrows():
        value = row[value_column]
        formatted_value = formatter(value) if formatter else value
        print(f"  {row['date']}: {formatted_value}")


def analyze_steps(
    df: pd.DataFrame,
    year: int,
    output_dir: Path,
    image_format: str,
    font_family: str,
) -> None:
    """Analyze step data and generate plots."""
    steps_raw = df[df["Key"] == "steps"].copy()
    if steps_raw.empty:
        print("No steps data for the selected year.")
        return

    # Extract and aggregate step data
    steps_raw["steps"] = steps_raw["Value"].apply(lambda val: safe_json_loads(val).get("steps", 0))
    steps_raw["half_hour"] = steps_raw["local_time"].dt.floor("30min")
    half_hour_steps = steps_raw.groupby(["Sid", "half_hour"])["steps"].sum().reset_index()
    idx = half_hour_steps.groupby("half_hour")["steps"].idxmax()
    max_steps_per_half_hour = half_hour_steps.loc[idx].reset_index(drop=True)
    max_steps_per_half_hour["date"] = max_steps_per_half_hour["half_hour"].dt.date
    daily_steps = max_steps_per_half_hour.groupby("date")["steps"].sum().reset_index()

    if daily_steps.empty:
        print("Steps data could not be aggregated.")
        return

    # Print summary statistics
    describe_top_days(daily_steps, "steps", "step count")
    print(f"Average daily steps in {year}: {daily_steps['steps'].mean():.0f}")

    # Generate monthly and weekday plots
    daily_steps["month"] = pd.to_datetime(daily_steps["date"]).dt.to_period("M")
    monthly_avg_steps = daily_steps.groupby("month")["steps"].mean().reset_index()
    weekday_avg_steps = (
        daily_steps.assign(weekday=pd.to_datetime(daily_steps["date"]).dt.day_name())
        .groupby("weekday")["steps"].mean()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )

    # Plot average daily steps per month
    with themed_axes(font_family) as (fig, ax):
        month_labels = monthly_avg_steps["month"].dt.strftime("%b")
        bars = ax.bar(month_labels, monthly_avg_steps["steps"], color=PALETTE["steps"])
        for bar, steps in zip(bars, monthly_avg_steps["steps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{round(steps)}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Average Daily Steps Per Month in {year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Steps")
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"steps_monthly.{image_format}", image_format)

    # Plot average daily steps by weekday
    with themed_axes(font_family) as (fig, ax):
        bars = ax.bar(weekday_avg_steps["weekday"],
                      weekday_avg_steps["steps"], color=PALETTE["steps"])
        for bar, steps in zip(bars, weekday_avg_steps["steps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{round(steps)}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Average Daily Steps by Weekday in {year}")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Steps")
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"steps_weekday.{image_format}", image_format)

    # Analyze hourly step data
    steps_raw["hour"] = steps_raw["local_time"].dt.hour  # Extract hour of the day
    hourly_steps = steps_raw.groupby("hour")["steps"].sum()
    days_with_data = steps_raw["date"].nunique()
    hourly_avg_steps = (hourly_steps / days_with_data).reset_index(name="steps")

    # Plot average steps per hour
    with themed_axes(font_family) as (fig, ax):
        bars = ax.bar(hourly_avg_steps["hour"], hourly_avg_steps["steps"], color=PALETTE["steps"])
        for bar, steps in zip(bars, hourly_avg_steps["steps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{steps:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Average Steps Per Hour in {year}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Steps")
        ax.set_xticks(range(0, 24))
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"steps_daily.{image_format}", image_format)


def parse_sleep_value(value: Any) -> dict[str, Any]:
    data = safe_json_loads(value)
    return {
        "duration": data.get("duration", 0),
        "sleep_awake": data.get("sleep_awake_duration", 0),
        "sleep_light": data.get("sleep_light_duration", 0),
        "sleep_deep": data.get("sleep_deep_duration", 0),
        "sleep_rem": data.get("sleep_rem_duration", 0),
        "awake_count": data.get("awake_count", 0),
        "bedtime": data.get("bedtime", 0),
        "wakeup": data.get("wake_up_time", 0),
    }


def normalize_bedtime_minutes(series: pd.Series) -> pd.Series:
    """
    Normalize bedtime timestamps into a continuous minute scale.

    This function maps bedtime timestamps that cross midnight onto a continuous
    minute axis. Hours earlier than 12 (midnight to noon) are retained as positive
    values, while hours from noon onward are adjusted by subtracting 1440 minutes
    (one day) to make evening times negative and early morning/noon times smaller
    positive values.
    """
    minutes = series.dt.hour * 60 + series.dt.minute
    return minutes.where(series.dt.hour < 12, minutes - 1440)


def analyze_sleep(
    df: pd.DataFrame,
    tz_offset: float,
    year: int,
    output_dir: Path,
    image_format: str,
    font_family: str,
) -> None:
    """Parse sleep data and generate plots."""
    sleep_raw = df[
        df["Key"].isin(["sleep", "watch_night_sleep"])
        & df["Value"].str.contains("item", na=False)
    ].copy()
    if sleep_raw.empty:
        print(f"No sleep data for year {year}.")
        return

    # Parse sleep data into structured columns
    sleep_components = sleep_raw["Value"].apply(parse_sleep_value).apply(pd.Series)
    sleep_data = sleep_raw.join(sleep_components)

    # Convert timestamps to local time
    sleep_data["bedtime_local"] = pd.to_datetime(
        sleep_data["bedtime"], unit="s", errors="coerce") + pd.Timedelta(hours=tz_offset)
    sleep_data["wakeup_local"] = pd.to_datetime(
        sleep_data["wakeup"], unit="s", errors="coerce") + pd.Timedelta(hours=tz_offset)
    sleep_data["bedtime_minutes"] = sleep_data["bedtime_local"].dt.hour * \
        60 + sleep_data["bedtime_local"].dt.minute
    sleep_data["wakeup_minutes"] = sleep_data["wakeup_local"].dt.hour * \
        60 + sleep_data["wakeup_local"].dt.minute
    sleep_data["bedtime_minutes_norm"] = normalize_bedtime_minutes(sleep_data["bedtime_local"])
    sleep_data["month"] = pd.to_datetime(sleep_data["date"]).dt.to_period("M")
    sleep_data["weekday"] = pd.to_datetime(sleep_data["date"]).dt.day_name()

    if sleep_data.empty:
        print("Sleep dataset is empty after parsing.")
        return

    # Calculate and print sleep statistics
    avg_sleep_duration = sleep_data["duration"].mean()
    median_sleep_duration = sleep_data["duration"].median()
    max_sleep = sleep_data.loc[sleep_data["duration"].idxmax()]
    min_sleep = sleep_data.loc[sleep_data["duration"].idxmin()]

    print(f"Average sleep duration in {year}: {format_duration(avg_sleep_duration)}")
    print(f"Median sleep duration: {format_duration(median_sleep_duration)}")
    print(
        "Longest sleep: "
        f"{format_duration(max_sleep['duration'])} on {max_sleep['date']} "
        f"({str(max_sleep['bedtime_local'].time())[:5]} - {str(max_sleep['wakeup_local'].time())[:5]})"
    )
    print(
        "Shortest sleep: "
        f"{format_duration(min_sleep['duration'])} on {min_sleep['date']} "
        f"({str(min_sleep['bedtime_local'].time())[:5]} - {str(min_sleep['wakeup_local'].time())[:5]})"
    )

    # Generate daily, monthly, and weekday plots for sleep data
    daily_sleep = sleep_data[["date", "duration"]].groupby("date").mean().reset_index()
    describe_top_days(daily_sleep.rename(
        columns={"duration": "minutes"}), "minutes", "sleep duration (minutes)",
        formatter=format_duration
    )

    monthly_avg_sleep = sleep_data.groupby("month")["duration"].mean().reset_index()

    with themed_axes(font_family) as (fig, ax):
        month_labels = monthly_avg_sleep["month"].dt.strftime("%b")
        duration_hours = monthly_avg_sleep["duration"] / 60
        bars = ax.bar(month_labels, duration_hours, color=PALETTE["sleep"])
        for bar, duration in zip(bars, monthly_avg_sleep["duration"]):
            label = f"{int(duration // 60)}h {int(duration % 60)}m"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Average Sleep Duration per Month in {year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hours")
        ax.set_ylim(bottom=4)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"sleep_duration_monthly.{image_format}", image_format)

    monthly_sleep_window = (
        sleep_data.groupby("month")[["bedtime_minutes_norm", "wakeup_minutes"]]
        .mean()
        .reset_index()
    )

    with themed_axes(font_family) as (fig, ax):
        month_labels = monthly_sleep_window["month"].dt.strftime("%b")
        bedtime_hours = (monthly_sleep_window["bedtime_minutes_norm"] % 1440) / 60
        wakeup_hours = monthly_sleep_window["wakeup_minutes"] / 60
        sleep_durations = wakeup_hours - bedtime_hours

        bars = ax.bar(month_labels, sleep_durations, bottom=bedtime_hours,
                      color=PALETTE["sleep"], label="Sleep Duration")

        for bar, start, duration in zip(bars, bedtime_hours, sleep_durations):
            hh_start, mm_start = divmod(int(start * 60), 60)
            hh_start = hh_start if hh_start >= 0 else hh_start + 24
            hh_end, mm_end = divmod(int((start + duration) * 60), 60)
            hh_end = hh_end if hh_end >= 0 else hh_end + 24

            ax.text(bar.get_x() + bar.get_width() / 2, start,
                    f"{hh_start:02d}:{mm_start:02d}",
                    ha="center", va="bottom", fontsize=8, color="black")

            ax.text(bar.get_x() + bar.get_width() / 2, start + duration,
                    f"{hh_end:02d}:{mm_end:02d}",
                    ha="center", va="bottom", fontsize=8, color="black")

        max_wakeup = max(12, math.ceil((wakeup_hours.max() + 1) / 2) * 2)
        min_bedtime = min(0, math.floor((bedtime_hours.min() - 1) / 2) * 2)

        ax.set_title(f"Average Bedtime & Wake-up per Month in {year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour of Day")
        ax.set_ylim(max_wakeup, min_bedtime)
        yticks = list(range(min_bedtime, max_wakeup + 1, 2))
        ytick_labels = [tick + 24 if tick < 0 else tick for tick in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"sleep_time_monthly.{image_format}", image_format)

    weekday_sleep_window = (
        sleep_data.groupby("weekday")[["bedtime_minutes_norm", "wakeup_minutes"]].mean()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )

    with themed_axes(font_family) as (fig, ax):
        weekday_labels = weekday_sleep_window["weekday"]
        bedtime_hours = (weekday_sleep_window["bedtime_minutes_norm"] % 1440) / 60
        wakeup_hours = weekday_sleep_window["wakeup_minutes"] / 60
        sleep_durations = wakeup_hours - bedtime_hours

        bars = ax.bar(weekday_labels, sleep_durations, bottom=bedtime_hours,
                      color=PALETTE["sleep"], label="Sleep Duration")

        for bar, start, duration in zip(bars, bedtime_hours, sleep_durations):
            hh_start, mm_start = divmod(int(start * 60), 60)
            hh_start = hh_start if hh_start >= 0 else hh_start + 24
            hh_end, mm_end = divmod(int((start + duration) * 60), 60)
            hh_end = hh_end if hh_end >= 0 else hh_end + 24

            ax.text(bar.get_x() + bar.get_width() / 2, start,
                    f"{hh_start:02d}:{mm_start:02d}",
                    ha="center", va="bottom", fontsize=8, color="black")

            ax.text(bar.get_x() + bar.get_width() / 2, start + duration,
                    f"{hh_end:02d}:{mm_end:02d}",
                    ha="center", va="bottom", fontsize=8, color="black")

        max_wakeup = max(12, math.ceil((wakeup_hours.max() + 1) / 2) * 2)
        min_bedtime = min(0, math.floor((bedtime_hours.min() - 1) / 2) * 2)

        ax.set_title(f"Average Bedtime & Wake-up by Weekday in {year}")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Hour of Day")
        ax.set_ylim(max_wakeup, min_bedtime)
        yticks = list(range(min_bedtime, max_wakeup + 1, 2))
        ytick_labels = [tick + 24 if tick < 0 else tick for tick in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"sleep_time_weekday.{image_format}", image_format)

    avg_stages = {
        "Awake": sleep_data["sleep_awake"].mean(),
        "Light": sleep_data["sleep_light"].mean(),
        "Deep": sleep_data["sleep_deep"].mean(),
        "REM": sleep_data["sleep_rem"].mean(),
    }
    with themed_axes(font_family, figsize=(6, 6)) as (fig, ax):
        ax.pie(
            avg_stages.values(),
            labels=avg_stages.keys(),
            autopct="%1.1f%%",
            startangle=140,
            counterclock=False,
            colors=["#aec6cf", "#87ceeb", "#4682b4", "#5f9ea0"],
            wedgeprops=dict(edgecolor="#fffdf8", width=0.4),
        )
        ax.set_title(f"Average Sleep Stages in {year}")
        export_plot(fig, output_dir / f"sleep_stage.{image_format}", image_format)


def analyze_heart_rate(
    df: pd.DataFrame,
    year: int,
    output_dir: Path,
    image_format: str,
    font_family: str,
) -> None:
    """Analyze heart rate data and generate plots."""
    heart_rate_raw = df[df["Key"] == "heart_rate"].copy()
    if heart_rate_raw.empty:
        print("No heart-rate data for the selected year.")
        return

    # Extract and aggregate heart rate data
    heart_rate_raw["bpm"] = heart_rate_raw["Value"].apply(
        lambda val: safe_json_loads(val).get("bpm", 0))
    heart_rate_raw["date"] = heart_rate_raw["local_time"].dt.date
    daily_avg_hr = heart_rate_raw.groupby("date")["bpm"].mean().reset_index()
    daily_avg_hr["month"] = pd.to_datetime(daily_avg_hr["date"]).dt.to_period("M")
    monthly_avg_hr = daily_avg_hr.groupby("month")["bpm"].mean().reset_index()
    weekday_avg_hr = (
        daily_avg_hr.assign(weekday=pd.to_datetime(daily_avg_hr["date"]).dt.day_name())
        .groupby("weekday")["bpm"].mean()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )

    # Set dynamic bottom limit for y-axis
    bottom_limit = 60
    while min(monthly_avg_hr["bpm"].min(), weekday_avg_hr["bpm"].min()) - 3 < bottom_limit:
        bottom_limit -= 10

    # Generate monthly and weekday plots for heart rate
    with themed_axes(font_family) as (fig, ax):
        month_labels = monthly_avg_hr["month"].dt.strftime("%b")
        bars = ax.bar(month_labels, monthly_avg_hr["bpm"], color=PALETTE["heart"])
        for bar, bpm in zip(bars, monthly_avg_hr["bpm"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{round(bpm)}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Average Daily Heart Rate per Month in {year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("bpm")
        ax.set_ylim(bottom=bottom_limit)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"heart_rate_monthly.{image_format}", image_format)

    with themed_axes(font_family) as (fig, ax):
        bars = ax.bar(weekday_avg_hr["weekday"], weekday_avg_hr["bpm"], color=PALETTE["heart"])
        for bar, bpm in zip(bars, weekday_avg_hr["bpm"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{round(bpm)}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Average Daily Heart Rate by Weekday in {year}")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("bpm")
        ax.set_ylim(bottom=bottom_limit)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"heart_rate_weekday.{image_format}", image_format)

    # Analyze hourly heart rate data
    heart_rate_raw["hour"] = heart_rate_raw["local_time"].dt.hour  # Extract hour of the day
    hourly_avg_hr = heart_rate_raw.groupby("hour")["bpm"].mean().reset_index()

    # Set dynamic bottom limit for y-axis
    bottom_limit = 60
    while hourly_avg_hr["bpm"].min() - 3 < bottom_limit:
        bottom_limit -= 5

    # Plot average heart rate per hour
    with themed_axes(font_family) as (fig, ax):
        bars = ax.bar(hourly_avg_hr["hour"], hourly_avg_hr["bpm"], color=PALETTE["heart"])
        for bar, bpm in zip(bars, hourly_avg_hr["bpm"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{round(bpm)}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Average Heart Rate Per Hour in {year}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("BPM")
        ax.set_xticks(range(0, 25, 4))
        ax.set_ylim(bottom=bottom_limit)
        # ax.set_axisbelow(True)
        export_plot(fig, output_dir / f"heart_rate_daily.{image_format}", image_format)


def main() -> None:
    args = parse_arguments()
    output_dir = ensure_output_dir(args.output)

    try:
        df = prepare_dataframe(args.input_csv, args.tz_offset, args.year)
    except FileNotFoundError:
        print(f"Input CSV file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Input CSV file is empty: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    except CSVFileFormatError:
        print("Input CSV file has an unexpected format. Please ensure the file is hlth_center_fitness_data.csv.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} records for {args.year}.")
    analyze_steps(df, args.year, output_dir, args.format, args.font)
    analyze_sleep(df, args.tz_offset, args.year, output_dir, args.format, args.font)
    analyze_heart_rate(df, args.year, output_dir, args.format, args.font)


if __name__ == "__main__":
    main()
