import io
from io import BytesIO
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PIL import Image
from urllib.request import urlopen

from mplsoccer import VerticalPitch, PyPizza, add_image
# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Player Impact by Position", layout="wide")

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
    "Scottish Premiership": "SCO",
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
        return 'LW'
    elif 'RW' in pos or 'RM' in pos:
        return 'RW'
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

TEAMLOG_FILE = "teamlog.csv"

@st.cache_data
def load_teamlog():
    url = build_raw_url(TEAMLOG_FILE)
    try:
        df = pd.read_csv(io.BytesIO(fetch_raw_file(url)))
        return df
    except Exception as e:
        st.error(f"Error loading team log: {e}")
        return pd.DataFrame()
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
        # Load all columns – we need more than just minutes now
        df = pd.read_excel(io.BytesIO(fetch_raw_file(url)))
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
    season: str,
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

  #  with st.expander("Debug: sample of filtered position data", expanded=False):
  #      debug_cols = [
  #          col
  #          for col in [
  #              "playerName",
  #              "playing_position",
  #              "typeId",
  #              "x",
  #              "y",
  #              "xT_value",
  #              "throwin",
  #          ]
  #          if col in positiondata.columns
  #      ]
  #      st.dataframe(positiondata[debug_cols].head(50))

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
#    with st.expander("Debug: Events for player in bin 64", expanded=False):
#    
#        # Filter only this player's events AND the selected position
#        player_events = positiondata[
#            (positiondata["playerName"] == playername) &
#            (positiondata["pitch_bin"] == 64)
#        ]
#    
#        if player_events.empty:
#            st.write(f"No events found for {playername} in bin 64.")
#        else:
#            st.write(f"Total events for {playername} in bin 64: {len(player_events)}")
#            st.dataframe(
#                player_events[
#                    [
#                        "playerName",
#                        "playing_position",
#                        "typeId",
#                        "x",
#                        "y",
#                        "xT_value",
#                        "pitch_bin"
#                    ]
#                ],
#                use_container_width=True,
#            )
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
    #with st.expander("Debug: Full player bin table (all 70 bins)", expanded=True):
    #    debug_df = playertest[[
    #        "pitch_bin",
    #        "playerName",
    #        "xT_value_per_90",
    #        "avg_xT_value_per_90",
    #        "xT_value_compared"
    #    ]].copy().sort_values("pitch_bin")
    #    st.dataframe(debug_df, use_container_width=True)
    #with st.expander("Debug: xT_value_compared distribution", expanded=False):
    #    st.write(playertest["xT_value_compared"].describe())

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
        f"{first_name} | Impact by Pitch Area as {position} in {season}",
        fontsize=14,
        pad=10,
        color="white",
    )

    return fig

def build_player_pizza(
    player_stats: pd.DataFrame,
    teamlog: pd.DataFrame,
    playername: str,
    position: str,
    competition_name: str,
    season_name: str,
    minute_threshold: int,
):
    """Builds and returns the PyPizza figure for the selected player."""

    # -----------------------------------------------------------------
    # 1. Filter by position and minute threshold
    # -----------------------------------------------------------------
    if "position_group" not in player_stats.columns:
        st.error("Column 'position_group' missing from stats file.")
        return None

    if "minutes_played" not in player_stats.columns:
        st.error("Column 'minutes_played' missing from stats file.")
        return None

    # Filter by selected position
    position_data = player_stats.loc[player_stats["position_group"] == position].copy()

    # Minute filtering
    position_data["minutes_played"] = pd.to_numeric(position_data["minutes_played"], errors="coerce")
    position_data = position_data.loc[position_data["minutes_played"] >= minute_threshold]

    if position_data.empty:
        st.warning("No players meet the minute threshold for this position.")
        return None

    # -----------------------------------------------------------------
    # 2. Percentile computation
    # -----------------------------------------------------------------
    meta_cols = position_data.columns[:10]
    numeric_cols = position_data.columns[10:]

    def pct_0_to_100(s: pd.Series) -> pd.Series:
        n = s.count()
        if n <= 1:
            return pd.Series([0] * len(s), index=s.index, dtype=float)
        return (s.rank(method="min") - 1) / (n - 1) * 100

    percentiles = position_data[numeric_cols].apply(pct_0_to_100)
    percentile_df = pd.concat([position_data[meta_cols], percentiles], axis=1)

    # -----------------------------------------------------------------
    # 3. Metric selection per position
    # -----------------------------------------------------------------
    if position in 'CM(2), CM(3) DM(23)':
        cols = [
            "goals_per_90", "xG_per_90", "assists_per_90",
            "keyPasses_per_90", "touches_in_box_per_90",
            "pass_completion", "pass_completion_final_third", "%_passes_are_progressive",
            "prog_carries_per_90",
            "total_threat_created_per_90",
            "defensive_actions_per_90", "successful_defensive_actions_per_90",
            "interceptions_per_90", "blocked_shots_per_90",
            "total_threat_prevented_per_90",
            "threat_value_per_90",
        ]

    elif position in ['LB', 'LWB', 'RB', 'RWB']:
        cols = [
            "keyPasses_per_90", "xA_per_90", "assists_per_90",
            "xG_per_90", "successful_attacking_actions_per_90",
            "pass_completion", "pass_completion_final_third", "%_passes_are_progressive",
            "prog_carries_per_90",
            "total_threat_created_per_90",
            "successful_defensive_actions_per_90", "defensive_actions_per_90",
            "interceptions_per_90", "blocked_shots_per_90",
            "total_threat_prevented_per_90",
            "threat_value_per_90",
        ]
    elif position in ['CB(3)', 'CB(2)']:
        cols = [
            "goals_per_90", "keyPasses_per_90", "xA_per_90",
            "prog_carries_per_90", "successful_attacking_actions_per_90",
            "pass_completion", "prog_passes_per_90",
            "%_passes_are_progressive", "passing_yards_per_90",
            "passing_threat_per_90",
            "successful_defensive_actions_per_90", "interceptions_per_90",
            "tackle_win_rate", "def_aerial_win_rate",
            "total_threat_prevented_per_90",
            "threat_value_per_90",
        ]
    elif position in ['CF', 'LW', 'RW', 'AM']:
        cols = [
            "shots_per_90", "shot_accuracy", "xG_per_90",
            "goals_per_90", "shot_quality",
            "prog_carries_per_90", "carrying_yards_per_90",
            "carry_threat_per_90", "dribbles_per_90",
            "successful_attacking_actions_per_90",
            "pass_completion_final_third",
            "passing_threat_per_90", "keyPasses_per_90",
            "xA_per_90", "assists_per_90",
            "threat_value_per_90",
        ]
    else:
        st.warning(f"No pizza metrics configured for position '{position}'.")
        return None

    selected_cols = ["player_name", "team_name", "position_group"] + cols

    # Missing columns?
    missing = set(selected_cols) - set(percentile_df.columns)
    if missing:
        st.error(f"Stats file is missing required metrics: {missing}")
        return None

    # -----------------------------------------------------------------
    # 4. Build player-specific percentile row
    # -----------------------------------------------------------------
    pdf = percentile_df[selected_cols].copy()
    pdf[cols] = pdf[cols].round(0).astype(int)

    # add extra aggregation for threat
    metric_block = pdf.iloc[:, 3:-1]
    pdf["role_average"] = metric_block.mean(axis=1)
    pdf["threat_combined"] = (pdf["role_average"] + pdf["threat_value_per_90"]) / 2
    pdf["threat_final"] = (pdf["threat_combined"].rank(pct=True) * 100).round(0).astype(int)

    pdf = pdf.drop(columns=["threat_value_per_90", "role_average", "threat_combined"])
    pdf = pdf.rename(columns={"threat_final": "threat_value_per_90"})

    playerrow = pdf.loc[
        (pdf["player_name"] == playername)
        & (pdf["position_group"] == position)
    ]

    if playerrow.empty:
        st.warning("Player has no percentile data for this position.")
        return None

    playerrow = playerrow.iloc[0]
    values = playerrow[cols].tolist()

    # -----------------------------------------------------------------
    # 5. Load images (WTA + team badge)
    # -----------------------------------------------------------------
    wtaimaged = Image.open(
        requests.get(
            "https://github.com/WTAnalysis/dashboard/raw/main/wtatransnew.png",
            stream=True,
        ).raw
    )

    # TEAM BADGE LOOKUP
    teamname = playerrow["team_name"]
    teamcode = None
    if not teamlog.empty:
        matchrow = teamlog.loc[teamlog["name"] == teamname]
        if not matchrow.empty:
            teamcode = matchrow["id"].iloc[0]

    teamimage = None
    if teamcode:
        badge_url = (
            "https://omo.akamai.opta.net/image.php"
            f"?h=www.scoresway.com&sport=football&entity=team&description=badges"
            f"&dimensions=150&id={teamcode}"
        )
        try:
            teamimage = Image.open(urlopen(badge_url))
        except:
            teamimage = None

    # -----------------------------------------------------------------
    # 6. Parameter display names
    # -----------------------------------------------------------------
    if position in 'CM(2), CM(3) DM(23)':
        params = [
            "Goals", "xG", "Assists/ShotAssist",
            "Creativity", "Box Touches",
            "Pass%", "Final 3rd Pass%", "Prog Pass%",
            "Prog Carries",
            "Threat Created",
            "Def Actions", "Success Def Act",
            "Interceptions", "Blocks", "Threat Prevented",
            "Player Impact",
        ]
    elif position in ['LB', 'LWB', 'RB', 'RWB']:
        params = [
            "Key Pass", "xA", "Assists",
            "xG", "Attacking Success",
            "Pass%", "Final 3rd Pass%", "Prog Pass%",
            "Prog Carries",
            "Threat Created",
            "Def Actions", "Success Def",
            "Interceptions", "Blocks", "Threat Prevented",
            "Player Impact",
        ]
    elif position in ['CB(2)', 'CB(3)']:
        params = [
            "Goals", "Key Pass", "xA",
            "Prog Carries", "Attacking Success",
            "Pass%", "Prog Pass", "Prog Pass%",
            "Pass Yards", "Passing Threat",
            "Success Def", "Interceptions", "Tackle%",
            "Aerial%", "Threat Prevented",
            "Player Impact",
        ]
    else:   # CF / LW / RW / AM
        params = [
            "Shots", "Shot Acc", "xG",
            "Goals", "Shot Quality",
            "Prog Carries", "Carry Yards", "Carry Threat",
            "Dribbles", "Attacking Success",
            "Final 3rd Pass%",
            "Pass Threat", "Key Pass", "xA", "Assists",
            "Player Impact",
        ]

    # Must match metric count
    if len(params) != len(values):
        st.error(f"Mismatch: {len(params)} params vs {len(values)} values.")
        return None

    # -----------------------------------------------------------------
    # 7. Build PyPizza
    # -----------------------------------------------------------------
    slice_colors = ["red"] * 5 + ["#63ace3"] * 5 + ["#2f316a"] * 5 + ["#fcba03"] * 1
    text_colors = ["#000000"] * 10 + ["white"] * 5 + ["black"]

    baker = PyPizza(
        params=params,
        background_color="#F2F2F2",
        straight_line_color="#F2F2F2",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(8, 8.5),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.40,
        kwargs_slices=dict(edgecolor="#F2F2F2", linewidth=1, zorder=2),
        kwargs_params=dict(color="#000000", fontsize=11, va="center"),
        kwargs_values=dict(
            color="#000000",
            fontsize=11,
            bbox=dict(
                edgecolor="#000000",
                facecolor="cornflowerblue",
                boxstyle="round,pad=0.2",
                lw=1,
            ),
        ),
    )

    # Title & subtitle
    fig.text(
        0.52, 0.975,
        f"{playername} – {teamname} – Percentile Rank (0–100)",
        ha="center", size=16, color="#000000"
    )

    fig.text(
        0.52, 0.952,
        f"Compared with other {position} in {competition_name} | {season_name}",
        ha="center", size=13, color="#000000"
    )

    # Credits
    fig.text(
        0.05, 0.02,
        "Data: Opta & Transfermarkt | Per 90 metrics | Minimum 360 mins",
        size=9, color="#000000"
    )

    # Category headings
    fig.patches.extend([
        plt.Rectangle((0.31, 0.9225), 0.025, 0.021, color="red", transform=fig.transFigure),
        plt.Rectangle((0.462, 0.9225), 0.025, 0.021, color="#63ace3", transform=fig.transFigure),
        plt.Rectangle((0.632, 0.9225), 0.025, 0.021, color="#2f316a", transform=fig.transFigure),
    ])

    # WTA Logo
    add_image(wtaimaged, fig, left=0.465, bottom=0.44, width=0.095, height=0.108)

    # Team badge
    if teamimage is not None:
        add_image(teamimage, fig, left=0.05, bottom=0.05, width=0.20, height=0.125)

    return fig

# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    st.title("Player Impact by Position")
    st.subheader("Select Season, Competition, Player & Position on the Left")

    st.sidebar.header("Data Sources")
    
    # --------------------------
    # 1. Season selection
    # --------------------------
    season_choice = st.sidebar.selectbox(
        "Season",
        list(SEASON_MAP.keys()),
        index=0,
    )
    season_fragment = SEASON_MAP[season_choice]
    
    # --------------------------
    # 2. Competition selection
    # --------------------------
    competition_choice = st.sidebar.selectbox(
        "Competition",
        list(COMP_MAP.keys()),
        index=0,
    )
    league_prefix = COMP_MAP[competition_choice]
    
    # --------------------------
    # 3. Build file paths
    # --------------------------
    parquet_choice = f"{league_prefix}1_{season_fragment}.parquet"
    excel_choice   = f"{league_prefix}1_{season_fragment}_playerstats_by_position_group.xlsx"

    # --------------------------
    # LOAD DATA
    # --------------------------
    with st.spinner("Loading match and player stats data..."):
        matchdata = load_match_data(parquet_choice)           # match event data
        player_stats = load_minute_log(excel_choice)          # full stats file (Excel)
        teamlog = load_teamlog()                              # teamcode lookup file

    # --------------------------
    # Normalise positions in matchdata
    # --------------------------
    if "playing_position" in matchdata.columns:
        matchdata["playing_position"] = matchdata["playing_position"].apply(normalize_position)

    if matchdata.empty or player_stats.empty:
        st.warning("Data could not be loaded. Please check the data sources.")
        return

    # Cast match data numeric fields
    for col in ["x", "y", "xT_value"]:
        if col in matchdata.columns:
            matchdata[col] = pd.to_numeric(matchdata[col], errors="coerce")

    # Build minute_log specifically for xT map (aggregated minutes)
    if "minutes_played" in player_stats.columns:
        minutes_only = player_stats[["player_name", "minutes_played"]].copy()
        minutes_only["minutes_played"] = pd.to_numeric(
            minutes_only["minutes_played"], errors="coerce"
        )
        minute_log = (
            minutes_only.groupby("player_name", as_index=False)["minutes_played"]
            .sum()
        )
    else:
        st.error("Column 'minutes_played' not found in player stats file.")
        return

    # --------------------------
    # USER INPUT — Player + Position
    # --------------------------
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
        st.error("No players found.")
        return

    default_player = "" if "N. Williams" in all_players else all_players[0]

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
        st.error(f"No positions found for {playername}.")
        return

    preferred_positions = ["LB", "LCB(2)", "RCB(2)", "RB"]
    default_position = next((p for p in preferred_positions if p in positions), positions[0])

    position = st.sidebar.selectbox(
        "Select position",
        positions,
        index=positions.index(default_position),
    )

    # --------------------------
    # NEW — Minute threshold selector
    # --------------------------
    minuteinput = st.sidebar.number_input(
        "Minute Threshold",
        min_value=0,
        max_value=5000,
        value=360,
        step=30,
        key="minuteinput",
    )

# --------------------------
# TABS — Pitch Map + Player Pizza
# --------------------------
tab1, tab2 = st.tabs(["Pitch Impact Map", "Player Pizza"])

# Initialise session state if not set
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Pitch Impact Map"

# ================================================================
# TAB 1 — PITCH MAP
# ================================================================
with tab1:
    st.session_state.active_tab = "Pitch Impact Map"

    st.header("Pitch Impact Map")

    if st.sidebar.button("Generate Pitch Map"):
        fig = plot_xt_comparison_for_player(
            matchdata=matchdata,
            minute_log=minute_log,
            position=position,
            playername=playername,
            season=season_choice,
        )

        if fig is not None:
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.image(fig_to_png_bytes(fig), width=450)

# ================================================================
# TAB 2 — PLAYER PIZZA
# ================================================================
with tab2:
    st.session_state.active_tab = "Player Pizza"

    st.header("Player Pizza")

    # ⭐ Only run pizza when this tab is active
    if st.session_state.active_tab == "Player Pizza":
        pizza_fig = build_player_pizza(
            player_stats=player_stats,
            teamlog=teamlog,
            playername=playername,
            position=position,
            competition_name=competition_choice,
            season_name=season_choice,
            minute_threshold=minuteinput,
        )

        if pizza_fig is not None:
            left, center, right = st.columns([1, 2, 1])
            with center:
                img_bytes = fig_to_png_bytes(pizza_fig)
                st.image(img_bytes, width=450)

if __name__ == "__main__":
    main()
