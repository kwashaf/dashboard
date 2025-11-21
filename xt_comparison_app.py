import io
from io import BytesIO
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
# REPO CONFIG – tweak these if layout changes
# -----------------------------------------------------------------------------
REPO_OWNER = "WTAnalysis"
REPO_NAME = "dashboard"
BRANCH = "main"
DATA_DIR = ""  # e.g. "data" if your files are in /data

PitchColor = "#f5f6fc"
BackgroundColor = "#381d54"
PitchLineColor = "Black"

SEASON_MAP = {
    "2024/25": "2425",
    "2025":    "2025",
    "2025/26": "2526",
}

# Competition → league prefix
COMP_MAP = {
    "Premier League": "ENG",
    "La Liga":        "SPA",
    "Bundesliga":     "GER",
    "Ligue 1":        "FRA",
    "Serie A":        "ITA",
    "League of Ireland":    "IRE",
}
# -----------------------------------------------------------------------------
# POSITION NORMALISATION
# -----------------------------------------------------------------------------
def normalize_position(pos: str) -> str:
    """Normalize position strings into standard buckets."""
    if pos is None:
        return pos
    pos = str(pos)

    if pos in ['RWB', 'LWB']:
        return pos
    if 'CM(2)' in pos:
        return 'CM(2)'
    elif 'CM(3)' in pos:
        return 'CM(3)'
    elif 'DM(2)' in pos or 'DM(3)' in pos:
        return 'DM(23)'
    elif 'CB(2)' in pos:
        return 'CB(2)'
    elif 'CB(3)' in pos:
        return 'CB(3)'
    elif 'AM(2)' in pos:
        return 'AM(2)'
    elif 'CF(2)' in pos:
        return 'CF(2)'
    elif 'LW' in pos or 'LM' in pos:
        return 'LMW'
    elif 'RW' in pos or 'RM' in pos:
        return 'RMW'
    elif pos in ['DM', 'AM', 'CF', 'LB', 'RB']:
        return pos
    else:
        return pos  # default fallback
# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def build_raw_url(filename: str) -> str:
    """Build raw GitHub URL for a given file in the repo."""
    prefix = "" if DATA_DIR == "" else (DATA_DIR.rstrip("/") + "/")
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/{BRANCH}/{prefix}{filename}"


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# DATA LOADERS (NO API CALLS)
# -----------------------------------------------------------------------------

def fetch_raw_file(url: str) -> bytes:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content


INDEX_FILE = "file_index.txt"


@st.cache_data
def load_index_list():
    url = build_raw_url(INDEX_FILE)
    try:
        content = fetch_raw_file(url).decode("utf-8")
        return [x.strip() for x in content.splitlines() if x.strip()]
    except Exception as e:
        st.error(f"Error loading file index: {e}")
        return []


# -------------------- FIX APPLIED HERE --------------------
# ❌ Removed f.lower().endswith — filenames must preserve case
# -----------------------------------------------------------

@st.cache_data
def list_parquet_files():
    files = load_index_list()
    return sorted([f for f in files if f.endswith(".parquet")])


@st.cache_data
def list_excel_files():
    files = load_index_list()
    return sorted([
        f for f in files
        if (f.endswith(".xlsx") or f.endswith(".xls"))
        and "playerstats_by_position_group" in f.lower()
    ])


@st.cache_data
def load_match_data(parquet_filename: str):
    url = build_raw_url(parquet_filename)
    try:
        df = pd.read_parquet(io.BytesIO(fetch_raw_file(url)))
        return df
    except Exception as e:
        st.error(f"Error reading parquet file '{parquet_filename}': {e}")
        return pd.DataFrame()


@st.cache_data
def load_minute_log(excel_filename: str):
    url = build_raw_url(excel_filename)
    try:
        df = pd.read_excel(io.BytesIO(fetch_raw_file(url)),
                           usecols=["player_name", "minutes_played"])
        return df
    except Exception as e:
        st.error(f"Error reading Excel file '{excel_filename}': {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# PLOTTING FUNCTION
# -----------------------------------------------------------------------------
def plot_xt_comparison_for_player(
    matchdata: pd.DataFrame,
    minute_log: pd.DataFrame,
    position: str,
    playername: str,
):

    positiondata = matchdata.loc[matchdata["playing_position"] == position].copy()
    bad_types = ["position_change", "goal_conceded", "clean_sheet"]
    if "typeId" in positiondata.columns:
        positiondata = positiondata[~positiondata["typeId"].isin(bad_types)]
    if "throwin" in positiondata.columns:
        positiondata = positiondata.loc[positiondata["throwin"] != 1]

    if positiondata.empty:
        st.error(f"No data found for position '{position}' after filtering.")
        return None

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
                "throwin",
            ]
            if col in positiondata.columns
        ]
        st.dataframe(positiondata[debug_cols].head(50))

    positiondata["x"] = positiondata["x"].clip(lower=0, upper=100)
    positiondata["y"] = positiondata["y"].clip(lower=0, upper=100)

    x_bins = 10
    y_bins = 7

    x_edges = np.linspace(0, 100, x_bins + 1)
    y_edges = np.linspace(0, 100, y_bins + 1)

    x_bin = np.digitize(positiondata["x"], x_edges, right=False)
    y_bin = np.digitize(positiondata["y"], y_edges, right=False)

    x_bin = np.clip(x_bin, 1, x_bins)
    y_bin = np.clip(y_bin, 1, y_bins)

    y_bin = (y_bins + 1) - y_bin

    positiondata["pitch_bin"] = (x_bin - 1) * y_bins + y_bin
# --- DEBUG: Show all events for this player in bin 64 ---
    with st.expander("Debug: Events for player in bin 64", expanded=False):
    
        # Filter only this player's events AND the selected position
        player_events = positiondata[
            (positiondata["playerName"] == playername) &
            (positiondata["pitch_bin"] == 64)
        ]
    
        if player_events.empty:
            st.write(f"No events found for {playername} in bin 64.")
        else:
            st.write(f"Total events for {playername} in bin 64: {len(player_events)}")
            st.dataframe(
                player_events[
                    [
                        "playerName",
                        "playing_position",
                        "typeId",
                        "x",
                        "y",
                        "xT_value",
                        "pitch_bin"
                    ]
                ],
                use_container_width=True,
            )
    drop_types = ["Player off", "Player on", "Corner Awarded", "Card"]
    if "typeId" in positiondata.columns:
        positiondata = positiondata.loc[~positiondata["typeId"].isin(drop_types)]

    xT_summary = (
        positiondata.groupby(["playerName", "pitch_bin"], as_index=False)["xT_value"]
        .sum()
    )

    xT_merged = pd.merge(
        xT_summary,
        minute_log,
        how="left",
        left_on="playerName",
        right_on="player_name",
    )

    xT_merged["xT_value_per_90"] = np.where(
        xT_merged["minutes_played"] > 0,
        (xT_merged["xT_value"] / xT_merged["minutes_played"]) * 90,
        np.nan,
    )
    xT_merged = xT_merged.drop(columns="player_name")
    xT_merged = xT_merged.dropna(subset=["xT_value_per_90"])

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

    playertest = xT_compared.loc[xT_compared["playerName"] == playername].copy()

    if playertest.empty:
        st.error(f"No data found for player '{playername}' at this position.")
        return None

    all_bins = pd.DataFrame({"pitch_bin": range(1, 71)})
    playertest = pd.merge(all_bins, playertest, on="pitch_bin", how="left")

    first_name = playername
    if playertest["playerName"].notna().any():
        first_name = playertest["playerName"].dropna().unique()[0]

    playertest["playerName"] = playertest["playerName"].fillna(first_name)
    playertest["xT_value_compared"] = playertest["xT_value_compared"].fillna(0)
    with st.expander("Debug: Full player bin table (all 70 bins)", expanded=True):
        debug_df = playertest[[
            "pitch_bin",
            "playerName",
            "xT_value_per_90",
            "avg_xT_value_per_90",
            "xT_value_compared"
        ]].copy().sort_values("pitch_bin")
        st.dataframe(debug_df, use_container_width=True)
    with st.expander("Debug: xT_value_compared distribution", expanded=False):
        st.write(playertest["xT_value_compared"].describe())

    colors = ["#d7191c", "#ffffff", "#1a9641"]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_red_white_green", colors, N=256
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)

    pitch = VerticalPitch(
        pitch_type="opta",
        pitch_color=PitchColor,
        line_color=PitchLineColor,
    )
    fig, ax = pitch.draw(figsize=(6, 9))
    fig.set_facecolor(BackgroundColor)

    for _, row in playertest.iterrows():
        bin_num = row["pitch_bin"]
        xT_diff = row["xT_value_compared"]

        x_idx = (bin_num - 1) // y_bins
        y_idx = (bin_num - 1) % y_bins
        y_idx = (y_bins - 1) - y_idx

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
    st.subheader("ENG1 25/26 Season (or any selected)")

    st.sidebar.header("Data Sources")
    
    # 1. Select season
    season_choice = st.sidebar.selectbox(
        "Season",
        list(SEASON_MAP.keys()),
        index=0,
    )
    season_fragment = SEASON_MAP[season_choice]
    
    # 2. Select competition
    competition_choice = st.sidebar.selectbox(
        "Competition",
        list(COMP_MAP.keys()),
        index=0,
    )
    league_prefix = COMP_MAP[competition_choice]
    
    # 3. Construct expected filenames
    parquet_choice = f"{league_prefix}1_{season_fragment}.parquet"
    excel_choice   = f"{league_prefix}1_{season_fragment}_playerstats_by_position_group.xlsx"
    
    st.sidebar.markdown(f"""
    **Match data file:** `{parquet_choice}`  
    **Minutes file:** `{excel_choice}`
    """)

    with st.spinner("Loading match and minute data..."):
        matchdata = load_match_data(parquet_choice)
        minute_log = load_minute_log(excel_choice)
    # -----------------------------------------
    # NORMALISE PLAYING POSITION COLUMN
    # -----------------------------------------
    if "playing_position" in matchdata.columns:
        matchdata["playing_position"] = matchdata["playing_position"].apply(normalize_position)
        if matchdata.empty or minute_log.empty:
            st.warning("Data could not be loaded. Please check the data sources.")
            return

    for col in ["x", "y", "xT_value"]:
        if col in matchdata.columns:
            matchdata[col] = pd.to_numeric(matchdata[col], errors="coerce")

    if "minutes_played" in minute_log.columns:
        minute_log["minutes_played"] = pd.to_numeric(
            minute_log["minutes_played"], errors="coerce"
        )

    if "player_name" in minute_log.columns:
        minute_log = (
            minute_log.groupby("player_name", as_index=False)["minutes_played"]
            .sum()
        )

    with st.sidebar.expander("Debug: global ranges", expanded=False):
        if "xT_value" in matchdata.columns:
            st.write("xT_value:", matchdata["xT_value"].describe())
        if "minutes_played" in minute_log.columns:
            st.write("minutes_played:", minute_log["minutes_played"].describe())

    st.sidebar.header("User Input")

    all_players = (
        matchdata["playerName"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    all_players = sorted(all_players)

    if not all_players:
        st.error("No players found in match data.")
        return

    default_player = "N. Williams" if "N. Williams" in all_players else all_players[0]
    playername = st.sidebar.selectbox(
        "Select player",
        all_players,
        index=all_players.index(default_player),
    )

    player_rows = matchdata.loc[matchdata["playerName"] == playername]

    positions = (
        player_rows["playing_position"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    positions = sorted(positions)

    if not positions:
        st.error(f"No positions found for player {playername}.")
        return

    preferred_positions = ["LB", "LCB(2)", "RCB(2)", "RB"]
    default_position = positions[0]
    for p in preferred_positions:
        if p in positions:
            default_position = p
            break

    position = st.sidebar.selectbox(
        "Select position",
        positions,
        index=positions.index(default_position),
    )

    if st.sidebar.button("Generate Pitch Map"):
        fig = plot_xt_comparison_for_player(
            matchdata=matchdata,
            minute_log=minute_log,
            position=position,
            playername=playername,
        )

        if fig is not None:
            left, center, right = st.columns([1, 2, 1])
            with center:
                img_bytes = fig_to_png_bytes(fig)
                st.image(img_bytes, width=450)


if __name__ == "__main__":
    main()
