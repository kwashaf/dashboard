import io
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import VerticalPitch

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="xT Comparison Pitch Map", layout="wide")

# -----------------------------------------------------------------------------
# CONSTANTS / CONFIG
# -----------------------------------------------------------------------------
# GitHub Parquet (ENG1_2526.parquet)
MATCH_PARQUET_URL = (
    "https://github.com/WTAnalysis/dashboard/raw/main/ENG1_2526.parquet"
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
    """Load ENG1_2526.parquet from GitHub."""
    try:
        resp = requests.get(MATCH_PARQUET_URL)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching match Parquet from GitHub: {e}")
        return pd.DataFrame()

    try:
        df = pd.read_parquet(io.BytesIO(resp.content))
    except Exception as e:
        st.error(f"Failed to parse match Parquet file: {e}")
        return pd.DataFrame()

    return df


@st.cache_data
def load_minute_log() -> pd.DataFrame:
    """Load ENG1_2526_playerstats_by_position_group.xlsx from GitHub."""
    try:
        resp = requests.get(MINUTE_LOG_XLSX_URL)
        resp.raise_for_status()
        xlsx_bytes = io.BytesIO(resp.content)
        df = pd.read_excel(xlsx_bytes, usecols=["player_name", "minutes_played"])
    except Exception as e:
        st.error(f"Error loading minute log Excel from GitHub: {e}")
        return pd.DataFrame()
    return df


# -----------------------------------------------------------------------------
# PLOTTING FUNCTION
# -----------------------------------------------------------------------------
def plot_xt_comparison_for_player(
    matchdata: pd.DataFrame,
    minute_log: pd.DataFrame,
    position: str,
    playername: str,
):
    """Create the xT comparison pitch map for a given player and position."""

    # -------------------------------------------------------------------------
    # Filter by user-defined position
    # -------------------------------------------------------------------------
    positiondata = matchdata.loc[matchdata["playing_position"] == position].copy()

    if positiondata.empty:
        st.error(f"No data found for position '{position}'.")
        return None

    # --- DEBUG: show a subsection of the position data ---
    with st.expander("Debug: sample of filtered position data", expanded=False):
        debug_cols = [
            col
            for col in [
                "playerName",
                "playing_position",
                "typeId",
                "x",
                "y",
                "xT_value",
            ]
            if col in positiondata.columns
        ]
        st.dataframe(positiondata[debug_cols].head(50))

    # -------------------------------------------------------------------------
    # Bin the pitch and aggregate xT
    # -------------------------------------------------------------------------
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
    if "typeId" in positiondata.columns:
        positiondata = positiondata.loc[~positiondata["typeId"].isin(drop_types)]

    # Sum xT per player per bin
    xT_summary = (
        positiondata.groupby(["playerName", "pitch_bin"], as_index=False)["xT_value"]
        .sum()
    )

    # -------------------------------------------------------------------------
    # Merge with minute log and compute per-90 + comparison vs average
    # -------------------------------------------------------------------------
    xT_merged = pd.merge(
        xT_summary,
        minute_log,
        how="left",
        left_on="playerName",
        right_on="player_name",
    )

    # Safe per-90 calculation
    xT_merged["xT_value_per_90"] = np.where(
        xT_merged["minutes_played"] > 0,
        (xT_merged["xT_value"] / xT_merged["minutes_played"]) * 90,
        np.nan,
    )
    xT_merged = xT_merged.drop(columns="player_name")
    xT_merged = xT_merged.dropna(subset=["xT_value_per_90"])

    # Average xT per 90 by bin (across all players in that position)
    avg_bin_xt = (
        xT_merged.groupby("pitch_bin", as_index=False)["xT_value_per_90"]
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
        xT_compared["xT_value_per_90"] - xT_compared["avg_xT_value_per_90"]
    )

    # -------------------------------------------------------------------------
    # Player-specific data
    # -------------------------------------------------------------------------
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

    # --- DEBUG: distribution of xT_value_compared for this player ---
    with st.expander("Debug: xT_value_compared distribution", expanded=False):
        st.write(playertest["xT_value_compared"].describe())

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------
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
    fig, ax = pitch.draw(figsize=(4.5, 7.5))  # same aspect as your Python version
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
        fontsize=14,
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

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    with st.spinner("Loading match and minute data..."):
        matchdata = load_match_data()
        minute_log = load_minute_log()

    if matchdata.empty or minute_log.empty:
        st.warning("Data could not be loaded. Please check the data sources.")
        return

    # -------------------------------------------------------------------------
    # Normalise dtypes so Streamlit & local Python behave the same
    # -------------------------------------------------------------------------
    # Force numeric on key columns
    for col in ["x", "y", "xT_value"]:
        if col in matchdata.columns:
            matchdata[col] = pd.to_numeric(matchdata[col], errors="coerce")

    if "minutes_played" in minute_log.columns:
        minute_log["minutes_played"] = pd.to_numeric(
            minute_log["minutes_played"], errors="coerce"
        )

    # Aggregate minutes per player in case of duplicates
    if "player_name" in minute_log.columns:
        minute_log = (
            minute_log.groupby("player_name", as_index=False)["minutes_played"]
            .sum()
        )

    # Optional quick debug ranges in sidebar
    with st.sidebar.expander("Debug: global ranges", expanded=False):
        if "xT_value" in matchdata.columns:
            st.write("xT_value:", matchdata["xT_value"].describe())
        if "minutes_played" in minute_log.columns:
            st.write("minutes_played:", minute_log["minutes_played"].describe())

    # -------------------------------------------------------------------------
    # Sidebar controls
    # -------------------------------------------------------------------------
    st.sidebar.header("User Input")

    # Position selector based on data
    positions = (
        matchdata["playing_position"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    positions = sorted(positions)

    if not positions:
        st.error("No positions found in match data.")
        return

    default_position = "LB" if "LB" in positions else positions[0]
    position = st.sidebar.selectbox(
        "Select position",
        positions,
        index=positions.index(default_position),
    )

    # Player selector based on that position
    positiondata = matchdata.loc[matchdata["playing_position"] == position]
    players = (
        positiondata["playerName"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    players = sorted(players)

    if not players:
        st.error(f"No players found for position {position}.")
        return

    default_player = "N. Williams" if "N. Williams" in players else players[0]
    playername = st.sidebar.selectbox(
        "Select player",
        players,
        index=players.index(default_player),
    )

    # -------------------------------------------------------------------------
    # Generate plot
    # -------------------------------------------------------------------------
    if st.sidebar.button("Generate Pitch Map"):
        fig = plot_xt_comparison_for_player(
            matchdata=matchdata,
            minute_log=minute_log,
            position=position,
            playername=playername,
        )
    
        if fig is not None:
            # Center the pitch and scale it nicely inside the page
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()
