
import io
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import VerticalPitch

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="xT Comparison Pitch Map", layout="wide")

# Google Drive CSV (ENG1_2526.csv)
MATCH_CSV_URL = (
    "https://drive.google.com/file/d/1ZQ9672gGC4P4NFjVJYhrVo9N_-wCA9u8/view?usp=drive_link"
)

# GitHub Excel (ENG1_2526_playerstats_by_position_group.xlsx)
MINUTE_LOG_XLSX_URL = (
    "https://github.com/WTAnalysis/dashboard/raw/main/"
    "ENG1_2526_playerstats_by_position_group.xlsx"
)

PitchColor = "#f5f6fc"
BackgroundColor = "#381d54"
PitchLineColor = "Black"


# -----------------------------------------------------------------------------
# DATA LOADERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_match_data() -> pd.DataFrame:
    """Load ENG1_2526.csv from Google Drive in a robust way."""
    resp = requests.get(MATCH_CSV_URL)
    try:
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching match CSV from Google Drive: {e}")
        return pd.DataFrame()

    # Check if Google Drive is giving us HTML (e.g. a warning page) instead of the CSV
    content_type = resp.headers.get("Content-Type", "")
    text_sample = resp.text[:200].lower()

    if "text/html" in content_type or "<html" in text_sample:
        st.error(
            "The Google Drive link appears to be returning an HTML page "
            "(e.g. a warning or quota page) instead of the raw CSV.\n\n"
            "Please double-check that the file is shared publicly and that "
            "the URL is a direct download link."
        )
        return pd.DataFrame()

    try:
        # Let pandas handle the bytes directly; be tolerant of bad lines
        df = pd.read_csv(
            io.BytesIO(resp.content),
            engine="python",          # more tolerant parser
            on_bad_lines="skip"       # skip malformed rows instead of failing
            # encoding="utf-8"       # you can uncomment + adjust if needed
        )
    except Exception as e:
        st.error(f"Failed to parse match CSV: {e}")
        return pd.DataFrame()

    return df
@st.cache_data
def load_minute_log() -> pd.DataFrame:
    """Load ENG1_2526_playerstats_by_position_group.xlsx from GitHub."""
    resp = requests.get(MINUTE_LOG_XLSX_URL)
    resp.raise_for_status()
    xlsx_bytes = io.BytesIO(resp.content)
    df = pd.read_excel(xlsx_bytes, usecols=["player_name", "minutes_played"])
    return df


# -----------------------------------------------------------------------------
# PLOTTING FUNCTION
# -----------------------------------------------------------------------------
def plot_xt_comparison_for_player(
    matchdata: pd.DataFrame,
    minute_log: pd.DataFrame,
    position: str,
    playername: str
):
    """Create the xT comparison pitch map for a given player and position."""

    # Filter by user-defined position
    positiondata = matchdata.loc[matchdata["playing_position"] == position].copy()

    if positiondata.empty:
        st.error(f"No data found for position '{position}'.")
        return None

    # Ensure coordinates are within pitch bounds
    positiondata["x"] = positiondata["x"].clip(lower=0, upper=100)
    positiondata["y"] = positiondata["y"].clip(lower=0, upper=100)

    # Bin the pitch into a 10 x 7 grid
    x_bins = 10
    y_bins = 7

    x_edges = np.linspace(0, 100, x_bins + 1)
    y_edges = np.linspace(0, 100, y_bins + 1)

    x_bin = np.digitize(positiondata["x"], x_edges, right=False)
    y_bin = np.digitize(positiondata["y"], y_edges, right=False)

    x_bin = np.clip(x_bin, 1, x_bins)
    y_bin = np.clip(y_bin, 1, y_bins)

    # Flip y for vertical pitch orientation
    y_bin = (y_bins + 1) - y_bin

    positiondata["pitch_bin"] = (x_bin - 1) * y_bins + y_bin

    # Remove event types not needed
    drop_types = ["Player off", "Player on", "Corner Awarded", "Card"]
    positiondata = positiondata.loc[~positiondata["typeId"].isin(drop_types)]

    # Sum xT per player per bin
    xT_summary = (
        positiondata
        .groupby(["playerName", "pitch_bin"], as_index=False)["xT_value"]
        .sum()
    )

    # Merge with minute log
    xT_merged = pd.merge(
        xT_summary,
        minute_log,
        how="left",
        left_on="playerName",
        right_on="player_name",
    )
    xT_merged["xT_value_per_90"] = (
        xT_merged["xT_value"] / xT_merged["minutes_played"]
    ) * 90
    xT_merged = xT_merged.drop(columns="player_name")

    # Average xT per 90 by bin (across all players)
    avg_bin_xt = (
        xT_merged
        .groupby("pitch_bin", as_index=False)["xT_value_per_90"]
        .mean()
        .rename(columns={"xT_value_per_90": "avg_xT_value_per_90"})
    )

    xT_compared = pd.merge(
        xT_merged,
        avg_bin_xt,
        on="pitch_bin",
        how="left",
    )
    xT_compared["xT_value_compared"] = (
        xT_compared["xT_value_per_90"]
        - xT_compared["avg_xT_value_per_90"]
    )

    # Player-specific data
    playertest = xT_compared.loc[xT_compared["playerName"] == playername].copy()

    if playertest.empty:
        st.error(f"No data found for player '{playername}' at this position.")
        return None

    # Ensure we have rows for all bins 1..70
    all_bins = pd.DataFrame({"pitch_bin": range(1, 71)})  # 10 x 7 grid = 70
    playertest = pd.merge(all_bins, playertest, on="pitch_bin", how="left")

    # Fill missing playerName and xT_value_compared
    first_name = playername
    if playertest["playerName"].notna().any():
        first_name = playertest["playerName"].dropna().unique()[0]

    playertest["playerName"] = playertest["playerName"].fillna(first_name)
    playertest["xT_value_compared"] = playertest["xT_value_compared"].fillna(0)

    # Colour map: red (worse) -> white (average) -> green (better)
    colors = ["#d7191c", "#ffffff", "#1a9641"]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_red_white_green", colors, N=256
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)

    # Draw vertical pitch
    pitch = VerticalPitch(
        pitch_type="opta",
        pitch_color=PitchColor,
        line_color=PitchLineColor,
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(BackgroundColor)

    # Draw rectangles per bin
    for _, row in playertest.iterrows():
        bin_num = row["pitch_bin"]
        xT_diff = row["xT_value_compared"]

        x_idx = (bin_num - 1) // y_bins
        y_idx = (bin_num - 1) % y_bins
        y_idx = (y_bins - 1) - y_idx  # Flip horizontally

        x_start = x_idx * (100 / x_bins)
        y_start = y_idx * (100 / y_bins)

        color = cmap(norm(xT_diff))

        rect = plt.Rectangle(
            (y_start, x_start),
            (100 / y_bins),
            (100 / x_bins),
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
        )
        ax.add_patch(rect)

    ax.set_title(
        f"{first_name} | Impact by Pitch Area",
        fontsize=12,
        pad=10,
        color="white",
    )

    return fig


# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    st.title("xT Comparison Pitch Map")
    st.subheader("ENG1 25/26 Season")

    with st.spinner("Loading match and minute data..."):
        matchdata = load_match_data()
        minute_log = load_minute_log()

    st.sidebar.header("User Input")

    position = st.sidebar.text_input(
        "Enter position (e.g. LB, RW, CM)",
        value="LB",
    )

    playername = st.sidebar.text_input(
        "Enter player name (exact as in data)",
        value="N. Williams",
    )

    if st.sidebar.button("Generate Pitch Map"):
        fig = plot_xt_comparison_for_player(
            matchdata=matchdata,
            minute_log=minute_log,
            position=position,
            playername=playername,
        )

        if fig is not None:
            st.pyplot(fig)


if __name__ == "__main__":
    main()
