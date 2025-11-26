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
st.set_page_config(page_title="WT Analysis - Player Dashboard", layout="wide")

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
TextColor = "White"

SEASON_MAP = {
    "2025":    "2025",
    "2025/26": "2526",
}

# Competition → league prefix
COMP_MAP = {
    "Premier League": "ENG1",
    "La Liga":        "SPA1",
    "Bundesliga":     "GER1",
    "Ligue 1":        "FRA1",
    "Serie A":        "ITA1",
    "League of Ireland":    "IRE1",
    "Scottish Premiership": "SCO1",
    "Allsvenskan":    "SWE1",
    "Austrian Bundesliga":    "AUT1",
    "Pro League":    "BEL1",
    "Superligaen":    "DEN1",
    "Liga Portugal":    "POR1",
    "Brasilerao":    "BRA1",
    "Championship":    "ENG2",
    "League One":    "ENG3",
    "League Two":    "ENG4",
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

# -----------------------------------------------------------------------------
# PLAYER PROFILING CONFIG
# -----------------------------------------------------------------------------
PROFILE_EXCEL_URL = (
    "https://github.com/WTAnalysis/dashboard/raw/main/"
    "player%20profiles%20streamlit.xlsx"
)
FOOTBALL_IMG_URL = (
    "https://github.com/WTAnalysis/dashboard/raw/main/football.png"
)

CLOSE_BAND = 20
LOW_THRESHOLD = 60
# MIN_MINUTES will no longer be used — profiling now uses user input 'minuteinput'

# =====================================================================
#      PROFILING HELPER FUNCTIONS (PASTE ALL OF THESE TOGETHER)
# =====================================================================
def _norm(s):
    return str(s).strip().lower()

def extract_metric_pairs_adjacent(row: pd.Series, columns: list) -> dict:
    pairs = {}
    for i, col in enumerate(columns):
        if str(col).strip().lower().startswith("metric"):
            metric = row.get(col, None)
            wcol = columns[i + 1] if i + 1 < len(columns) else None
            weight = row.get(wcol, None) if wcol is not None else None
            if pd.isna(metric) or metric is None or str(metric).strip() == "":
                continue
            if (weight is None) or pd.isna(weight) or str(weight).strip() == "":
                continue
            pairs[str(metric).strip()] = float(weight)
    return pairs

def build_profiles_by_position(profiles_df: pd.DataFrame,
                               pos_col="Position",
                               name_col="Description") -> dict:
    profs = {}
    cols = list(profiles_df.columns)
    for _, r in profiles_df.iterrows():
        pos = str(r.get(pos_col, "")).strip()
        name = str(r.get(name_col, "")).strip()
        if not pos or not name:
            continue
        weights = extract_metric_pairs_adjacent(r, cols)
        if not weights:
            continue
        if pos not in profs:
            profs[pos] = {}
        profs[pos][name] = weights
    return profs

def candidates_for(metric: str):
    cands = set()
    base = metric.strip()
    cands.update({base, base.replace("_"," "), base.replace(" ","_"),
                  base.replace("_per_90"," per 90"), base.replace(" per 90","_per_90"),
                  base.replace("%","pct"), base.replace("pct","%"),
                  base.replace("-","_"), base.replace("_","-")})
    cands |= {c.lower() for c in cands}
    return list(cands)

def strict_percentile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(index=s.index, dtype=float)
    if len(valid)==0:
        return out
    if valid.nunique()<=1:
        out.loc[valid.index] = 100.0
        return out
    ranks = valid.rank(method="average")
    out.loc[valid.index] = ((ranks-1)/(len(valid)-1))*100
    return out

def weighted_score_from_pcts(row: pd.Series, metric_to_pctcol: dict, weights: dict):
    vals, wts = [], []
    for metric,w in weights.items():
        pctcol = metric_to_pctcol.get(metric)
        if pctcol is None:
            continue
        v = row.get(pctcol, np.nan)
        if pd.notna(v):
            vals.append(v); wts.append(w)
    if not wts:
        return np.nan
    return float(np.clip(np.average(vals,weights=wts),0,100))

def safe_score_col(profile_name: str) -> str:
    import re
    return "score_" + re.sub(r'[^A-Za-z0-9_]+','_',profile_name).strip("_")

def nice(col_name:str)->str:
    return col_name.replace("score_","").replace("_"," ")

POSITION_ALIAS = {}

@st.cache_data
def load_profile_definitions():
    resp = requests.get(PROFILE_EXCEL_URL)
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content))

@st.cache_data
def load_football_image_array():
    resp = requests.get(FOOTBALL_IMG_URL)
    resp.raise_for_status()
    return plt.imread(io.BytesIO(resp.content))

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_profile_polygon_with_ball(
    labels,
    scores,
    player_name="Player",
    position="",
    levels=(20,40,60,80,100),
    centroid_emphasis=1.5,
    ball_img=None,
    football_zoom=0.08,
    fig_bg="#381d54",
    poly_fill="#f5f6fc",
    polygon_color="#381d54",
    polygon_alpha=0.5,
    grid_color="black",
    grid_alpha=0.15,
    label_color="white",
    title_color="white",
    title_pad=28,
):
    """
    Automatically draws a diamond (4 profiles) or pentagon (5 profiles)
    depending on number of labels/scores.
    """

    n = len(labels)
    if n not in (4, 5):
        raise ValueError("This visual supports ONLY 4 or 5 profiles.")

    scores_arr = np.clip(np.array(scores, dtype=float), 0, 100) / 100.0

    # -------- SHAPE DEFINITION --------
    if n == 5:
        # Pentagon (top → clockwise)
        angles = np.deg2rad(np.linspace(90, 90 - 360, 5, endpoint=False))
        base_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    elif n == 4:
        # Diamond (top → right → bottom → left)
        base_pts = np.array([
            [0,  1],   # top
            [1,  0],   # right
            [0, -1],   # bottom
            [-1, 0],   # left
        ], dtype=float)

    # -------- FIGURE --------
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)

    # Fill inside
    ax.fill(base_pts[:,0], base_pts[:,1], color=poly_fill, zorder=0)

    # -------- GRID --------
    for lv in sorted(levels):
        f = lv / 100.0
        ring = base_pts * f
        ring = np.vstack([ring, ring[0]])
        ax.plot(ring[:,0], ring[:,1], color=grid_color, alpha=grid_alpha, zorder=1)

    # Spokes
    for x, y in base_pts:
        ax.plot([0, x], [0, y], color=grid_color, alpha=0.25, zorder=1)

    # Outer boundary
    outer = np.vstack([base_pts, base_pts[0]])
    ax.plot(outer[:,0], outer[:,1], linewidth=2, color=grid_color, alpha=0.9, zorder=2)

    # -------- SCORE POLYGON --------
    poly_pts = base_pts * scores_arr[:, None]
    poly = np.vstack([poly_pts, poly_pts[0]])

    ax.fill(poly[:,0], poly[:,1], color=polygon_color, alpha=polygon_alpha, zorder=3)
    ax.plot(poly[:,0], poly[:,1], linewidth=2, color=polygon_color, alpha=1.0, zorder=3)

    # -------- CENTROID / FOOTBALL (ALWAYS INSIDE POLYGON) --------
    w = np.nan_to_num(scores_arr, nan=0.0)

    # emphasise larger scores, but work on the *scaled* points
    if centroid_emphasis != 1.0:
        w = w ** centroid_emphasis

    if w.sum() > 0:
        w /= w.sum()
        # use poly_pts (scaled by scores), not base_pts
        cx, cy = (w[:, None] * poly_pts).sum(axis=0)
    else:
        cx, cy = 0.0, 0.0
    if ball_img is not None:
        try:
            imagebox = OffsetImage(ball_img, zoom=football_zoom)
            ab = AnnotationBbox(imagebox, (cx, cy), frameon=False, zorder=5)
            ax.add_artist(ab)
        except:
            ax.plot(cx, cy, "o", color="white", markersize=12, zorder=5)
    else:
        ax.plot(cx, cy, "o", color="white", markersize=12, zorder=5)

    # -------- LABELS --------
    label_offset = 1.12
    for (x, y), label, score in zip(base_pts, labels, scores):
        ha = "center"
        va = "center"

        if n == 4:
            # Diamond-specific label positioning (matching your screenshot)
            if (x, y) == (0, 1):      # top
                va = "bottom"
            elif (x, y) == (1, 0):    # right
                ha = "left"
            elif (x, y) == (0, -1):   # bottom
                va = "top"
            elif (x, y) == (-1, 0):   # left
                ha = "right"
        else:
            # Generic nice positioning for pentagon
            if abs(y) < 0.15:
                ha = "left" if x > 0 else "right"
            elif y > 0:
                va = "bottom"
            else:
                va = "top"

        ax.text(
            x * label_offset,
            y * label_offset,
            f"{label}\n({score:.1f})",
            ha=ha,
            va=va,
            color=label_color,
        )

    # Title
    ax.set_title(
        f"{player_name} – Position Profile ({position})",
        pad=title_pad,
        color=title_color,
        size=18,
    )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


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

    # -----------------------------------------------------------------
    # NORMALISE LMW / RMW → LW / RW
    # -----------------------------------------------------------------
    position_replacements = {
        "LMW": "LW",
        "RMW": "RW",
    }
    player_stats["position_group"] = player_stats["position_group"].replace(position_replacements)

    # Filter by selected position
    position_data = player_stats.loc[player_stats["position_group"] == position].copy()

    # Minute filtering
    position_data["minutes_played"] = pd.to_numeric(position_data["minutes_played"], errors="coerce")
    position_data = position_data.loc[position_data["minutes_played"] >= minute_threshold]
    sample_size = len(position_data)

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
            "clearances_per_90", "interceptions_per_90",
            "tackle_win_rate", "def_aerial_win_rate",
            "total_threat_prevented_per_90",
            "threat_value_per_90",
        ]
    elif position in ['CF', 'LW', 'RW', 'AM', 'CF(2)', 'AM(2)']:
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
        st.warning(f"Player hasn't played over {minute_threshold} minutes in selected position")
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
            "Goals", "xG", "Assists",
            "Shot Assists", "Touches in Box",
            "Pass Completion \n%", "Final 3rd \nPass Completion \n%", "% Passes are \nProgressive",
            "Progressive \n Carries",
            "Threat Created",
            "Defensive Actions", "Successful \nDefensive Actions",
            "Interceptions", "Shots Blocked", "Threat Prevented",
            "Player Impact",
        ]
    elif position in ['LB', 'LWB', 'RB', 'RWB']:
        params = [
            "Shot Assists", "xA", "Assists",
            "xG", "Successful \nAttacking Actions",
            "Pass Completion \n%", "Final 3rd \nPass Completion \n%", "% Passes are \nProgressive",
            "Progressive \n Carries",
            "Threat Created",
            "Defensive Actions", "Successful \nDefensive Actions",
            "Interceptions", "Shots Blocked", "Threat Prevented",
            "Player Impact",
        ]
    elif position in ['CB(2)', 'CB(3)']:
        params = [
            "Goals", "Shot Assists", "xA",
            "Progressive \nCarries", "Successful \nAttacking Actions",
            "Pass Completion \n%", "Progressive \nPasses", "% Passes are \nProgressive",
            "Passing Yards", "Passing Threat",
            "Clearances", "Interceptions", "Tackle \n%",
            "Aerial \n%", "Threat Prevented",
            "Player Impact",
        ]
    else:   # CF / LW / RW / AM
        params = [
            "Shots", "Shot Accuracy %", "xG",
            "Goals", "Shot Quality",
            "Progressive \nCarries", "Carrying Yards", "Carrying Threat",
            "Dribbles", "Successful \nAttacking Actions",
            "Final 3rd \nPass Completion \n%",
            "Passing Threat", "Shot Assists", "xA", "Assists",
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
        param_location=112.5,     # <-- ADD THIS LINE
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
    sample_size = len(position_data)
    
    fig.text(
        0.05, 0.02,
        f"Data: Opta | Metrics per 90 unless stated otherwise | "
        f"{sample_size} players have played at least {minute_threshold} mins as a {position}",
        size=9, color="#000000"
    )

    # Category headings
    #fig.patches.extend([
    #    plt.Rectangle((0.31, 0.9225), 0.025, 0.021, color="red", transform=fig.transFigure),
    #    plt.Rectangle((0.462, 0.9225), 0.025, 0.021, color="#63ace3", transform=fig.transFigure),
    #    plt.Rectangle((0.632, 0.9225), 0.025, 0.021, color="#2f316a", transform=fig.transFigure),
    #])

    # WTA Logo
    add_image(wtaimaged, fig, left=0.465, bottom=0.44, width=0.095, height=0.108)

    # Team badge
    if teamimage is not None:
        add_image(teamimage, fig, left=0.05, bottom=0.05, width=0.20, height=0.125)

    return fig
def create_player_actions_figure(
    attackingevents,
    defensiveevents,
    playerrecpass,
    playername,
    teamname,
    competition_name,
    season_name,
    teamimage,
    wtaimaged,
    BackgroundColor,
    PitchColor,
    PitchLineColor,
    TextColor
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from mplsoccer import VerticalPitch
    from scipy.spatial import ConvexHull
    from scipy.stats import gaussian_kde
    import numpy as np

    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.25), facecolor=BackgroundColor)
    plt.subplots_adjust(wspace=.1)

    # Define pitches
    pitch_arrows = VerticalPitch(pitch_type='opta', pitch_color=PitchColor, line_color=PitchLineColor)
    pitch_bins = VerticalPitch(pitch_type='opta', pitch_color=PitchColor, line_color=PitchLineColor)

    # Draw pitches
    pitch_arrows.draw(ax=axes[0], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
    axes[0].set_title(f'{playername} - Attacking Event Locations', fontsize=10, color=TextColor)

    pitch_bins.draw(ax=axes[1], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
    axes[1].set_title(f'{playername} - Defensive Event Locations', fontsize=10, color=TextColor)

    pitch_bins.draw(ax=axes[2], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
    axes[2].set_title(f'{playername} - Pass Reception Locations', fontsize=10, color=TextColor)

    # ---------------------------------------------------------
    # SHARED KDE GRID (IMPORTANT FIX)
    # ---------------------------------------------------------
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 100, 100),
        np.linspace(0, 100, 100)
    )
    # ---------------------------------------------------------
    # ATTACKING EVENTS (PITCH 1)
    # ---------------------------------------------------------
    points_pass = np.array([(row['y'], row['x']) for _, row in attackingevents.iterrows()])

    if len(points_pass) > 3:
        kde_pass = gaussian_kde(points_pass.T)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
        density_pass = kde_pass(np.vstack([x_grid.ravel(), y_grid.ravel()]))

        max_density_index_pass = np.argmax(density_pass)
        max_density_x_pass = x_grid.ravel()[max_density_index_pass]
        max_density_y_pass = y_grid.ravel()[max_density_index_pass]

        radius = 15
        points_within_radius_pass = points_pass[
            ((points_pass[:, 0] - max_density_x_pass) ** 2 +
             (points_pass[:, 1] - max_density_y_pass) ** 2) < radius ** 2
        ]

        if len(points_within_radius_pass) >= 3:
            hull_pass = ConvexHull(points_within_radius_pass)
            x_hull_pass = points_within_radius_pass[hull_pass.vertices, 0]
            y_hull_pass = points_within_radius_pass[hull_pass.vertices, 1]
            hull_patch_pass = MplPolygon(
                np.column_stack([x_hull_pass, y_hull_pass]),
                closed=True,
                edgecolor=BackgroundColor,
                facecolor=BackgroundColor,
                alpha=0.3
            )
            axes[0].add_patch(hull_patch_pass)

    # Plot attacking events
    for _, row in attackingevents.iterrows():
        axes[0].plot(row['y'], row['x'], marker='o', markerfacecolor='none',
                     color=BackgroundColor, markersize=2)

    # ---------------------------------------------------------
    # DEFENSIVE EVENTS (PITCH 2)
    # ---------------------------------------------------------
    points_def = np.array([(row['y'], row['x']) for _, row in defensiveevents.iterrows()])

    if len(points_def) > 3:
        kde_def = gaussian_kde(points_def.T)
        density_def = kde_def(np.vstack([x_grid.ravel(), y_grid.ravel()]))

        max_density_idx_def = np.argmax(density_def)
        max_density_x_def = x_grid.ravel()[max_density_idx_def]
        max_density_y_def = y_grid.ravel()[max_density_idx_def]

        points_within_radius_def = points_def[
            ((points_def[:, 0] - max_density_x_def) ** 2 +
             (points_def[:, 1] - max_density_y_def) ** 2) < radius ** 2
        ]

        if len(points_within_radius_def) >= 3:
            hull_def = ConvexHull(points_within_radius_def)
            x_hull_def = points_within_radius_def[hull_def.vertices, 0]
            y_hull_def = points_within_radius_def[hull_def.vertices, 1]
            hull_patch_def = MplPolygon(
                np.column_stack([x_hull_def, y_hull_def]),
                closed=True,
                edgecolor=BackgroundColor,
                facecolor=BackgroundColor,
                alpha=0.3
            )
            axes[1].add_patch(hull_patch_def)

    # Plot defensive events
    for _, row in defensiveevents.iterrows():
        axes[1].plot(row['y'], row['x'], marker='o', markerfacecolor='none',
                     color=BackgroundColor, markersize=2)

    # ---------------------------------------------------------
    # PASS RECEPTIONS (PITCH 3)
    # ---------------------------------------------------------
    points_rec = np.array([(row['end_y'], row['end_x']) for _, row in playerrecpass.iterrows()])

    if len(points_rec) > 3:
        kde_rec = gaussian_kde(points_rec.T)
        density_rec = kde_rec(np.vstack([x_grid.ravel(), y_grid.ravel()]))

        max_density_idx_rec = np.argmax(density_rec)
        max_density_x_rec = x_grid.ravel()[max_density_idx_rec]
        max_density_y_rec = y_grid.ravel()[max_density_idx_rec]

        points_within_radius_rec = points_rec[
            ((points_rec[:, 0] - max_density_x_rec) ** 2 +
             (points_rec[:, 1] - max_density_y_rec) ** 2) < radius ** 2
        ]

        if len(points_within_radius_rec) >= 3:
            hull_rec = ConvexHull(points_within_radius_rec)
            x_hull_rec = points_within_radius_rec[hull_rec.vertices, 0]
            y_hull_rec = points_within_radius_rec[hull_rec.vertices, 1]
            hull_patch_rec = MplPolygon(
                np.column_stack([x_hull_rec, y_hull_rec]),
                closed=True,
                edgecolor=BackgroundColor,
                facecolor=BackgroundColor,
                alpha=0.3
            )
            axes[2].add_patch(hull_patch_rec)

    # Plot pass receptions
    for _, row in playerrecpass.iterrows():
        axes[2].plot(row['end_y'], row['end_x'], marker='o', markerfacecolor='none',
                     color=BackgroundColor, markersize=2)

    # ---------------------------------------------------------
    # TEXT LABELS
    # ---------------------------------------------------------
    axes[0].text(50, -5, 'Dots show locations of events', ha='center', fontsize=9, color=TextColor)

    axes[0].text(50, -13, f'{playername} - {teamname}', ha='center', fontsize=12, color=TextColor, fontweight='bold')
    axes[0].text(50, -20, f'{competition_name} | {season_name}', ha='center', fontsize=12, color=TextColor, fontweight='bold')

    axes[1].text(50, -5, 'Data from Opta', ha='center', fontsize=9, color=TextColor)

    axes[2].text(50, -5, 'Shaded area shows most frequent area for action', ha='center', fontsize=9, color=TextColor)
    axes[2].text(50, -10, 'Attacking Events are shots, dribbles, shot assists & aerial duels',
                 ha='center', fontsize=9, color=TextColor)
    axes[2].text(50, -15, 'Defensive Events are tackles, challenges, aerials,',
                 ha='center', fontsize=9, color=TextColor)
    axes[2].text(50, -18, 'interceptions, ball recoveries and blocked shots',
                 ha='center', fontsize=9, color=TextColor)
    axes[2].text(50, -23, 'Pass Receptions are where player receives pass',
                 ha='center', fontsize=9, color=TextColor)
    axes[2].text(50, -26, 'from team-mate',
                 ha='center', fontsize=9, color=TextColor)

    # ---------------------------------------------------------
    # LOGOS
    # ---------------------------------------------------------
    add_image(teamimage, fig, left=0.57, bottom=-0.03, width=0.05, alpha=1)
    add_image(wtaimaged, fig, left=0.433, bottom=-0.02175, width=0.06, alpha=1)

    return fig
# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    st.title("WT Analysis - Player Dashboard")
    st.subheader("Select Season, Competition, Player & Position on the Left")

    st.sidebar.header("Data Sources")
    
    # --------------------------
    # 1. Season selection
    # --------------------------
    season_choice = st.sidebar.selectbox(
        "Season",
        list(SEASON_MAP.keys()),
        index=1,   # selects "2025/26"
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
    parquet_choice = f"{league_prefix}_{season_fragment}.parquet"
    excel_choice   = f"{league_prefix}_{season_fragment}_playerstats_by_position_group.xlsx"

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
    if "pass_recipient_position" in matchdata.columns:
        matchdata["pass_recipient_position"] = matchdata["pass_recipient_position"].apply(normalize_position)
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
    # --- NEW: TEAM DROPDOWN for duplicate-name protection ---
    player_teams = (
        matchdata.loc[matchdata["playerName"] == playername, "team_name"]
        .dropna()
        .unique()
        .tolist()
    )
    
    team_choice = st.sidebar.selectbox(
        "Select Team",
        player_teams,
        index=0,
        key="team_choice"
    )
    player_rows = matchdata.loc[(matchdata["playerName"] == playername) & (matchdata["team_name"] == team_choice)]
    # -------------------------------------------------------
    # TEAM NAME (needed for Pizza & Player Actions)
    # -------------------------------------------------------
    if "team_name" in matchdata.columns:
        try:
            teamname = (
                matchdata.loc[matchdata["playerName"] == playername, "team_name"]
                .dropna()
                .iloc[0]
            )
        except:
            teamname = "Unknown"
    else:
        teamname = "Unknown"

    teamlog = pd.read_csv("teamlog.csv")   # or st.cache_data if needed
    
    try:
        teamcode = teamlog.loc[teamlog["name"] == teamname, "id"].iloc[0]
    except:
        teamcode = None
    
    if teamcode:
        teamlogo_url = f"https://omo.akamai.opta.net/image.php?h=www.scoresway.com&sport=football&entity=team&description=badges&dimensions=150&id={teamcode}"
    
        from urllib.request import urlopen
        teamimage = Image.open(urlopen(teamlogo_url))
    else:
        teamimage = None   # fallback
    wtaimaged = Image.open(
        requests.get(
            "https://github.com/WTAnalysis/dashboard/raw/main/wtatransnew.png",
            stream=True,
        ).raw
    )
    # -------------------------------------------------------
    # BUILD POSITIONS SORTED BY MINUTES PLAYED
    # -------------------------------------------------------
    position_minutes = (
        player_stats[player_stats["player_name"] == playername]
        .groupby("position_group")["minutes_played"]
        .sum()
        .reset_index()
    )
    
    # FILTER OUT POSITIONS WITH UNDER 25 MINUTES
    position_minutes = position_minutes[position_minutes["minutes_played"] >= 25]
    
    # Sort by minutes descending
    position_minutes = position_minutes.sort_values("minutes_played", ascending=False)
    
    positions = position_minutes["position_group"].astype(str).tolist()
    
    if not positions:
        st.error(f"No positions with at least 25 minutes played for {playername}.")
        return
    
    default_position = positions[0]

    
    if not positions:
        st.error(f"No positions found for {playername}.")
        return
    
    default_position = positions[0]  # highest minutes first

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
    # -------------------------------------------------------
    # DISPLAY PLAYER POSITION MINUTES + STATS SUMMARY
    # -------------------------------------------------------
    st.markdown("### Player Info")

    # Aggregate extended stats per position
    pos_extended = (
        player_stats[player_stats["player_name"] == playername]
        .groupby("position_group")
        .agg({
            "minutes_played": "sum",
            "xG": "sum",
            "goals": "sum",
            "xA": "sum",
            "assists": "sum",
            "pass_completion": "mean",
            "aerial_win_rate": "mean",
            "tackle_win_rate": "mean",
            "successful_defensive_actions_per_90": "mean",
            "successful_attacking_actions_per_90": "mean",
        })
        .reset_index()
    )
    
    # FILTER OUT POSITIONS WITH UNDER 25 MINUTES
    pos_extended = pos_extended[pos_extended["minutes_played"] >= 25]
    
    # Merge with dropdown ordering to match table order
    pos_extended = position_minutes.merge(
        pos_extended,
        on=["position_group", "minutes_played"],
        how="left",
    )
    
    # Rename columns
    pos_extended = pos_extended.rename(columns={
        "position_group": "Position",
        "minutes_played": "Minutes",
        "xG": "xG",
        "goals": "Goals",
        "xA": "xA",
        "assists": "Assists",
        "pass_completion": "Pass %",
        "aerial_win_rate": "Aerial %",
        "tackle_win_rate": "Tackle %",
        "successful_defensive_actions_per_90": "Successful Def. Actions per 90",
        "successful_attacking_actions_per_90": "Successful Att. Actions per 90",
    })
    
    # Format %
    if "Pass %" in pos_extended.columns:
        pos_extended["Pass %"] = (pos_extended["Pass %"] * 100).round(1)
    if "Aerial %" in pos_extended.columns:
        pos_extended["Aerial %"] = (pos_extended["Aerial %"] * 100).round(1)
    if "Tackle %" in pos_extended.columns:
        pos_extended["Tackle %"] = (pos_extended["Tackle %"] * 100).round(1)
    
    # -------------------------------
    # CENTRE HEADERS + CELL CONTENTS
    # -------------------------------
    styled_table = pos_extended.style.set_properties(**{
        "text-align": "center"
    }).set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]}
    ])
    
    st.dataframe(styled_table, hide_index=True, use_container_width=True)

    # --------------------------
    # TABS — Pitch Map + Player Pizza
    # --------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Pitch Impact Map", "Player Pizza", "Player Actions", "Player Profiling"]
    )
    # Init session state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Pitch Impact Map"

    # ================================================================
    # TAB 1 — Pitch Impact Map
    # ================================================================
    with tab1:
        st.session_state["active_tab"] = "Pitch Impact Map"
        st.header("Pitch Impact Map")
    
        # Button must update session_state instead of triggering immediate render
        if st.sidebar.button("Generate Visuals"):
            st.session_state["generate_pitch_map"] = True
    
        # Only generate + display inside this tab
        if st.session_state.get("generate_pitch_map", False):
    
            # Prevent plot appearing in other tabs
            if st.session_state["active_tab"] != "Pitch Impact Map":
                st.stop()
    
            fig = plot_xt_comparison_for_player(
                matchdata=matchdata,
                minute_log=minute_log,
                position=position,
                playername=playername,
                season=season_choice,
            )
    
            if fig is not None:
                left, center, right = st.columns([1, 4, 1])
                with center:
                    st.image(fig_to_png_bytes(fig), width=500)

    # ================================================================
    # TAB 2 — Player Pizza
    # ================================================================
    with tab2:
        st.session_state["active_tab"] = "Player Pizza"
        st.header("Player Pizza")
    
        # Prevent crash if no player or position selected
        if not playername or not position:
            st.warning("Please select a player and position to generate the Player Pizza.")
        else:
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
                left, center, right = st.columns([1, 4, 1])
                with center:
                    img_bytes = fig_to_png_bytes(pizza_fig)
                    st.image(img_bytes, width=700)

# ================================================================
# TAB 3 — Player Actions
# ================================================================
    with tab3:
        st.session_state["active_tab"] = "Player Actions"
        st.header("Player Actions")
    
        # ❗ PREVENT CRASH WHEN NO PLAYER OR POSITION SELECTED
        if not playername or not position:
            st.warning("Please select a player and position to view Player Actions.")
        else:
            # ---------------------------------------------------------
            # 1. Build event datasets from matchdata
            # ---------------------------------------------------------
            playerevents = matchdata.loc[(matchdata["playerName"] == playername) & (matchdata["team_name"] == team_choice)].copy()
            playerevents = playerevents.loc[playerevents["playing_position"] == position]
    
            playerrecpass = matchdata.loc[(matchdata["playerName"] == playername) & (matchdata["team_name"] == team_choice)].copy()
            playerrecpass = playerrecpass.loc[playerrecpass['pass_recipient_position'] == position]
            playerrecpass = playerrecpass.loc[playerrecpass['outcome'] == "Successful"]
    
            # Filter out erroneous coordinates
            playerrecpass = playerrecpass[
                (playerrecpass['x'].between(1, 99)) &
                (playerrecpass['y'].between(1, 99))
            ]
            playerrecpass = playerrecpass[
                (playerrecpass['end_x'].between(1, 99)) &
                (playerrecpass['end_y'].between(1, 99))
            ]
    
            # Defensive events
            defeven = [
                'Tackle','Aerial','Challenge','Interception',
                'Save','Clearance','Ball recovery'
            ]
            defensiveevents = playerevents[playerevents['typeId'].isin(defeven)].copy()
            defensiveevents = defensiveevents[
                ~((defensiveevents['typeId'] == 'Aerial') & (defensiveevents['x'] >= 50))
            ]
    
            # Attacking events
            atteven = ['Take on', 'Miss', 'Attempt Saved', 'Goal', 'Aerial', 'Post']
            
            attackingevents = playerevents[
                (playerevents['typeId'].isin(atteven)) |
                (playerevents.get('keyPass', 0) == 1) |
                (playerevents.get('assist', 0) == 1) |
                (playerevents.get('progressive_carry', 'No') == "Yes")
            ].copy()
    
            # Remove defensive aerials
            attackingevents = attackingevents[
                ~((attackingevents['typeId'] == 'Aerial') & (attackingevents['x'] < 50))
            ]
    
            # Remove corners
            attackingevents = attackingevents[
                ~(attackingevents.get('corner', 0) == 1)
            ]
    
            # ---------------------------------------------------------
            # 2. Build the full 3-pane Matplotlib figure
            # ---------------------------------------------------------
            fig = create_player_actions_figure(
                attackingevents,
                defensiveevents,
                playerrecpass,
                playername,
                teamname,
                competition_choice,
                season_choice,
                teamimage,
                wtaimaged,
                BackgroundColor,
                PitchColor,
                PitchLineColor,
                TextColor
            )
    
            # ---------------------------------------------------------
            # 3. Display figure in center with width control
            # ---------------------------------------------------------
            left, center, right = st.columns([1, 3, 1])
            with center:
                st.image(fig_to_png_bytes(fig), width=1600)
    # ================================================================
    # TAB 4 — Player Profiling
    # ================================================================
    with tab4:
        st.session_state["active_tab"] = "Player Profiling"
        st.header("Player Profiling")

        if not playername or not position:
            st.warning("Please select a player and position.")
            st.stop()

        # 1) Load files
        profiles_df = load_profile_definitions()
        ball_img = load_football_image_array()

        profiles_by_pos = build_profiles_by_position(profiles_df)

        # 2) Prepare playerlist from player_stats
        playerlist = player_stats.copy()
        playerlist["minutes_played"] = pd.to_numeric(playerlist["minutes_played"], errors="coerce")

        # ❗ USE USER-SELECTED THRESHOLD
        base = playerlist.loc[playerlist["minutes_played"] >= minuteinput].copy()
        if base.empty:
            st.warning("No players meet the minute threshold for profiling.")
            st.stop()

        grp = position

        gdf = base.loc[
            base["position_group"].astype(str).str.lower() == _norm(grp)
        ].copy()

        if gdf.empty:
            st.warning(f"No players found for position {grp} with threshold {minuteinput} mins.")
            st.stop()

        pos_key = POSITION_ALIAS.get(grp, grp)
        matches = [k for k in profiles_by_pos if _norm(k)==_norm(pos_key)]
        if not matches:
            st.warning(f"No profiles defined for position '{grp}' in Excel file.")
            st.stop()

        prof_key = matches[0]
        profiles_for_grp = profiles_by_pos[prof_key]

        player_cols_lut = {c.lower():c for c in gdf.columns}
        metric_to_col = {}

        for prof_name,weights in profiles_for_grp.items():
            for metric in weights:
                if metric in metric_to_col:
                    continue
                for cand in candidates_for(metric):
                    if cand in player_cols_lut:
                        metric_to_col[metric] = player_cols_lut[cand]
                        break

        if not metric_to_col:
            st.warning("No metrics matched column names in stats.")
            st.stop()

        metric_to_pctcol = {}
        for metric, col in metric_to_col.items():
            pctcol = f"{col}__pct"
            metric_to_pctcol[metric]=pctcol
            gdf[pctcol] = strict_percentile(gdf[col])

        score_cols=[]
        for prof_name,weights in profiles_for_grp.items():
            scol = safe_score_col(prof_name)
            gdf[scol] = gdf.apply(lambda r: weighted_score_from_pcts(r, metric_to_pctcol, weights), axis=1)
            gdf[scol] = strict_percentile(gdf[scol]).round(1)
            score_cols.append(scol)

        # 3) Get selected player's row
        player_row = gdf.loc[
            (gdf["player_name"]==playername) &
            (gdf["team_name"]==team_choice)
        ]

        if player_row.empty:
            player_row = gdf.loc[gdf["player_name"]==playername]

        if player_row.empty:
            st.warning("No profiling data for this player.")
            st.stop()

        player_row = player_row.iloc[0]

        # 4) Build list of (profile_name, score)
        # --- PRESERVE EXCEL ORDER ---
        profile_scores = []
        for prof_name in profiles_for_grp.keys():   # order preserved from Excel
            s_col = safe_score_col(prof_name)
            if s_col in gdf.columns:
                val = player_row.get(s_col, np.nan)
                if pd.notna(val):
                    profile_scores.append((prof_name, float(val)))
        
        # filter to 4 or 5 profiles in the ORIGINAL Excel order
        if len(profile_scores) >= 5:
            profile_scores = profile_scores[:5]
        elif len(profile_scores) == 4:
            pass  # diamond OK
        else:
            st.warning("Not enough profile data to plot (need at least 4).")
            st.stop()

        if len(profile_scores) < 4:
            st.warning("Not enough profile data to plot (need at least 4 profiles).")
            st.stop()

        labels = [p[0] for p in profile_scores]
        scores = [p[1] for p in profile_scores]

        fig = plot_profile_polygon_with_ball(
            labels=labels,
            scores=scores,
            player_name=playername,
            position=position,   # <-- NEW
            centroid_emphasis=1.8,
            ball_img=ball_img,
            football_zoom=0.06,
            fig_bg=BackgroundColor,
            poly_fill=PitchColor,
            polygon_color=BackgroundColor,
            polygon_alpha=0.5,
            grid_color="black",
            label_color="white",
            title_color="white",
        )

        left,center,right = st.columns([1,3,1])
        with center:
            st.image(fig_to_png_bytes(fig), width=600)
if __name__ == "__main__":
    main()
