import io
from io import BytesIO
import requests
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from urllib.request import urlopen
import plotly.express as px

from mplsoccer import VerticalPitch, PyPizza, add_image
# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="WT Analysis - Player Data Dashboard", layout="wide")

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
    # -----------------------------------------------------------------
    # 5. Load images (WTA + team badge)
    # -----------------------------------------------------------------
@st.cache_resource
def load_wta_image():
    url = "https://github.com/WTAnalysis/dashboard/raw/main/wtatransnew.png"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

wtaimaged = load_wta_image()
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
    "Superleague":   "GRE1",
    "MLS":            "USA1",
    "J-League":        "JAP1",
    "Eredivisie":        "NED1",
}
@st.cache_resource
def load_team_badge(teamcode: str):
    """Safely fetch and cache a team's badge image."""
    badge_url = (
        "https://omo.akamai.opta.net/image.php"
        f"?h=www.scoresway.com&sport=football&entity=team&description=badges"
        f"&dimensions=150&id={teamcode}"
    )
    try:
        # Download the file in memory safely
        with urlopen(badge_url) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        return None
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
from contextlib import contextmanager

@contextmanager
def safe_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)

@contextmanager
def close_after(fig):
    """Context manager to auto-close an existing Matplotlib figure."""
    try:
        yield fig
    finally:
        plt.close(fig)

def build_raw_url(filename: str) -> str:
    """Build raw GitHub URL for a given file in the repo."""
    prefix = "" if DATA_DIR == "" else (DATA_DIR.rstrip("/") + "/")
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/raw/{BRANCH}/{prefix}{filename}"


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

# ---------------------------------------------------------
# SCATTER METRIC LABEL MAP
# ---------------------------------------------------------
SCATTER_METRIC_MAP = {
    "pass_completion": "Pass Completion %",
    "pass_completion_final_third": "Final 3rd Pass Completion %",
    "%_passes_are_progressive": "% of Passes are Progressive",
    "attempted_passes_per_90": "Passes per 90",
    "prog_passes_per_90": "Progressive Passes per 90",
    "keyPasses_per_90": "Shot Assists per 90",
    "assists_per_90": "Assists per 90",
    "xA_per_90": "xA per 90",
    "switches_per_90": "Switches per 90",
    "passing_yards_per_90": "Passing Yards per 90",
    "passing_threat_per_90": "Passing Threat per 90",
    "dribbles_per_90": "Dribbles per 90",
    "prog_carries_per_90": "Progressive Carries per 90",
    "carries_to_final_third_per_90": "Carries to Final Third per 90",
    "carrying_yards_per_90": "Carrying Yards per 90",
    "ten_yard_carries_per_90": "Ten Yard + Carries per 90",
    "carry_threat_per_90": "Carrying Threat per 90",
    "fouled_per_90": "Fouls Won per 90",
    "total_threat_created_per_90": "Threat Created per 90",
    "touches_per_90": "Touches per 90",
    "touches_in_final_third_per_90": "Final Third Touches per 90",
    "touches_in_own_third_per_90": "Own Third Touches per 90",
    "touches_in_middle_third_per_90": "Middle Third Touches per 90",
    "touches_in_box_per_90": "Touches in Box per 90",
    "received_passes_per_90": "Received Passes per 90",
    "received_passes_final_third_per_90": "Received Passes in Final Third per 90",
    "aerials_per_90": "Aerials per 90",
    "def_aerials_per_90": "Defensive Aerials per 90",
    "off_aerials_per_90": "Attacking Aerials per 90",
    "tackles_per_90": "Tackles per 90",
    "interceptions_per_90": "Interceptions per 90",
    "opp_half_interceptions_per_90": "Opposition Half Interceptions per 90",
    "ball_recoveries_per_90": "Ball Recoveries per 90",
    "opp_half_ball_recoveries_per_90": "Opp Half Ball Recoveries per 90",
    "clearances_per_90": "Clearances per 90",
    "box_def_actions_per_90": "Box Defensive Actions per 90",
    "sixbox_def_actions_per_90": "6-Yard Box Defensive Actions per 90",
    "blocked_shots_per_90": "Blocked Shots per 90",
    "last_man_per_90": "Last Man Actions per 90",
    "errors_per_90": "Errors per 90",
    "errors_leading_to_goal_per_90": "Errors Leading to Goal per 90",
    "fouls_per_90": "Fouls per 90",
    "ground_threat_prevented_per_90": "Ground Threat Prevented per 90",
    "aerial_threat_prevented_per_90": "Aerial Threat Prevented per 90",
    "total_threat_prevented_per_90": "Threat Prevented per 90",
    "shots_per_90": "Shots per 90",
    "shots_on_target_per_90": "Shots on Target per 90",
    "xG_per_90": "xG per 90",
    "xGOT_per_90": "xGOT per 90",
    "goals_per_90": "Goals per 90",
    "threat_value_per_90": "Player Impact per 90",
    "attacking_actions_per_90": "Attacking Actions per 90",
    "successful_attacking_actions_per_90": "Successful Attacking Actions per 90",
    "defensive_actions_per_90": "Defensive Actions per 90",
    "successful_defensive_actions_per_90": "Successful Defensive Actions per 90",
    "%_def_actions_in_box": "% Defensive Actions in Box",
    "%_def_actions_in_6box": "% Defensive Actions in 6-Yard Box",
    "%_touches_in_own_third": "% Touches in Own Third",
    "%_touches_in_middle_third": "% Touches in Middle Third",
    "%_touches_in_final_third": "% Touches in Final Third",
    "aerial_win_rate": "Aerial Win Rate %",
    "def_aerial_win_rate": "Defensive Aerial Win Rate %",
    "off_aerial_win_rate": "Offensive Aerial Win Rate %",
    "tackle_win_rate": "Tackle Win Rate %",
    "dribble_win_rate": "Dribble Success %",
    "shot_accuracy": "Shot Accuracy %",
    "shot_conversion": "Shot Conversion %",
    "shot_quality": "Shot Quality %",
    "box_touches_per_shot": "Touches in Box per Shot",
    "xG_per_shot": "xG Per Shot",
}

# ---------------------------------------------------------
# METRICS USED FOR PLAYER SIMILARITY (RAW → PERCENTILES)
# ---------------------------------------------------------
SIMILARITY_METRICS = [
    "pass_completion",
    "pass_completion_final_third",
    "%_passes_are_progressive",
    "attempted_passes_per_90",
    "prog_passes_per_90",
    "keyPasses_per_90",
    "assists_per_90",
    "xA_per_90",
    "switches_per_90",
    "passing_yards_per_90",
    "passing_threat_per_90",
    "dribbles_per_90",
    "prog_carries_per_90",
    "carries_to_final_third_per_90",
    "carrying_yards_per_90",
    "ten_yard_carries_per_90",
    "carry_threat_per_90",
    "fouled_per_90",
    "total_threat_created_per_90",
    "touches_per_90",
    "touches_in_final_third_per_90",
    "touches_in_own_third_per_90",
    "touches_in_middle_third_per_90",
    "touches_in_box_per_90",
    "received_passes_per_90",
    "received_passes_final_third_per_90",
    "aerials_per_90",
    "def_aerials_per_90",
    "off_aerials_per_90",
    "tackles_per_90",
    "challenges_per_90",
    "interceptions_per_90",
    "opp_half_interceptions_per_90",
    "ball_recoveries_per_90",
    "opp_half_ball_recoveries_per_90",
    "clearances_per_90",
    "box_def_actions_per_90",
    "sixbox_def_actions_per_90",
    "blocked_shots_per_90",
    "last_man_per_90",
    "errors_per_90",
    "errors_leading_to_goal_per_90",
    "own_goals_per_90",
    "fouls_per_90",
    "yellow_cards_per_90",
    "red_cards_per_90",
    "ground_threat_prevented_per_90",
    "aerial_threat_prevented_per_90",
    "total_threat_prevented_per_90",
    "shots_per_90",
    "shots_on_target_per_90",
    "xG_per_90",
    "xGOT_per_90",
    "goals_per_90",
    "threat_value_per_90",
    "attacking_actions_per_90",
    "successful_attacking_actions_per_90",
    "defensive_actions_per_90",
    "successful_defensive_actions_per_90",
    "%_def_actions_in_box",
    "%_def_actions_in_6box",
    "aerial_win_rate",
    "def_aerial_win_rate",
    "off_aerial_win_rate",
    "tackle_win_rate",
    "challenge_win_rate",
    "dribble_win_rate",
    "shot_accuracy",
    "shot_conversion",
    "shot_quality",
    "box_touches_per_shot",
    "xG_per_shot",
]

def resolve_metric(display_label: str) -> str:
    """
    Converts a display label like 'Pass Completion %'
    into its internal dataframe key like 'pass_completion'.
    """
    for key, label in SCATTER_METRIC_MAP.items():
        if label == display_label:
            return key
    return None

def metric_is_percent(display_name: str) -> bool:
    """
    Determines if a metric should be shown as a percentage.
    Any metric whose display name includes '%' is considered a percent metric.
    """
    return "%" in display_name

def create_pass_and_carry_sonar(
    matchdata,
    playername,
    team_choice,
    position,
    BackgroundColor,
    PitchColor,
    PitchLineColor,
    TextColor
):
    from mplsoccer import VerticalPitch
    from matplotlib.patches import Wedge
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    # ---------------------------
    # 0. FILTER BASE DATA
    # ---------------------------
    base = matchdata.loc[
        (matchdata['playerName'] == playername) &
        (matchdata['playing_position'] == position) &
        (matchdata['team_name'] == team_choice) &
        (matchdata['throwin'] != 1) &
        (matchdata['corner'] != 1) &
        (matchdata['freekick'] != 1) &
        (matchdata['goalkick'] != 1)
    ].copy()

    # Remove kick-off ghosts
    mask1 = ~((base['timeMin'] == 0) & (base['timeSec'] == 0))
    mask2 = ~((base['timeMin'] == 45) & (base['timeSec'] == 0))
    base = base[mask1 & mask2]

    passingdata = base.loc[base['typeId'] == 'Pass'].copy()
    carryingdata = base.loc[base['typeId'] == 'Carry'].copy()

    # ---------------------------
    # 1. SETTINGS
    # ---------------------------
    x_bins = np.linspace(0, 100, 6)
    y_bins = np.linspace(0, 100, 6)

    def plot_sonar(ax, cx, cy, angles, bins=12, max_radius=8, color='#381d54'):
        edges = np.linspace(0, 360, bins + 1)
        counts, _ = np.histogram(angles, bins=edges)
        radii = (counts / counts.max() * max_radius) if counts.max() else np.zeros_like(counts)

        for i in range(bins):
            if radii[i] > 0:
                wedge = Wedge(
                    center=(cy, cx),
                    r=radii[i],
                    theta1=edges[i],
                    theta2=edges[i + 1],
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.85
                )
                ax.add_patch(wedge)

    def add_grid(ax):
        for xb in x_bins:
            ax.add_line(Line2D([0, 100], [xb, xb], color='white', ls='--', lw=.7, alpha=.25))
        for yb in y_bins:
            ax.add_line(Line2D([yb, yb], [0, 100], color='white', ls='--', lw=.7, alpha=.25))

    def sonar(ax, data, title):
        if data.empty:
            ax.set_title(f"{title} (No data)", color=TextColor)
            return

        df = data.copy()
        # Correct for rectangular pitch distortion (your original logic)
        df['hx'] = df['end_y'] - df['y']
        df['vy'] = df['end_x'] - df['x']
        df['angle'] = (np.degrees(np.arctan2(df['vy'], df['hx'])) + 360) % 360

        df['row'] = pd.cut(df['x'], bins=x_bins, labels=False)
        df['col'] = pd.cut(df['y'], bins=y_bins, labels=False)

        add_grid(ax)

        for r in range(5):
            for c in range(5):
                cell = df[(df['row'] == r) & (df['col'] == c)]
                if not len(cell):
                    continue

                cx = (x_bins[r] + x_bins[r+1]) / 2
                cy = (y_bins[c] + y_bins[c+1]) / 2

                plot_sonar(ax, cx, cy, cell['angle'], max_radius=7)

        ax.set_title(title, color=TextColor)

    # ---------------------------
    # 2. BUILD FIGURE
    # ---------------------------
    pitch = VerticalPitch(
        pitch_type='opta',
        goal_type='box',
        pitch_color=PitchColor,
        line_color=PitchLineColor
    )

    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(12, 9))
    fig.set_facecolor(BackgroundColor)

    # 🔒 Memory-safe wrapper: does NOT change the visual
    with close_after(fig):
        sonar(axes[0], passingdata, f"{playername} - Passing Sonars as {position}")
        sonar(axes[1], carryingdata, f"{playername} - Carrying Sonars as {position}")

        add_image(wtaimaged, fig, left=0.203, bottom=0.4535, width=0.1, alpha=0.25)
        add_image(wtaimaged, fig, left=0.6975, bottom=0.4535, width=0.1, alpha=0.25)

        return fig

def determine_def_zone(row):
    """
    Zones for defensive half only (x <= 50).
    Zone 1 = Penalty box
    Zone 2–6 = 5 horizontal bands
    """
    x = row['x']
    y = row['y']

    # Penalty box horizontally: x <= 18 AND between the posts
    if x <= 17 and 21.1 <= y <= 78.9:
        return 1

    # 5 horizontal bands across defensive half
    if y < 21.1:
        return 6
    elif 21.1 <= y < 40:
        return 5
    elif 40 <= y < 60:
        return 4
    elif 60 <= y < 78.9:
        return 3
    else:
        return 2

def create_defensive_actions_figure(
    matchdata,
    playername,
    team_choice,
    position,
    teamimage,
    wtaimaged,
    BackgroundColor,
    PitchColor,
    PitchLineColor,
    TextColor
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mplsoccer import VerticalPitch

    defensive_types = ['Tackle', 'Aerial', 'Challenge']

    # ---------------------------
    # 1. FILTER DATA
    # ---------------------------
    allevents = matchdata.loc[
        (matchdata['playing_position'] == position) &
        (matchdata['x'] <= 50) & 
        (matchdata['typeId'].isin(defensive_types))
    ].copy()

    allevents['zone'] = allevents.apply(determine_def_zone, axis=1)
    allevents['is_success'] = allevents['outcome'].eq("Successful")

    playerevents = allevents.loc[
        (allevents['playerName'] == playername) &
        (allevents['team_name'] == team_choice)
    ].copy()

    extra_types = ['Ball recovery', 'Interception', 'Attempt Saved', 'Clearance']
    player_extra = matchdata.loc[
        (matchdata['playerName'] == playername) &
        (matchdata['team_name'] == team_choice) &
        (matchdata['typeId'].isin(extra_types)) &
        (matchdata['x'] <= 50)
    ].copy()

    player_extra["x"] = pd.to_numeric(player_extra["x"], errors="coerce")
    player_extra["y"] = pd.to_numeric(player_extra["y"], errors="coerce")

    # ---------------------------
    # 2. SUMMARY STATS
    # ---------------------------
    zone_summary_all = allevents.groupby("zone").agg(
        total=("zone", "size"),
        success=("is_success", "sum")
    ).reset_index()

    zone_summary_player = playerevents.groupby("zone").agg(
        total=("zone", "size"),
        success=("is_success", "sum")
    ).reset_index()

    all_z = pd.DataFrame({"zone": range(1, 7)})

    zone_summary_all = all_z.merge(zone_summary_all, on="zone", how="left").fillna(0)
    zone_summary_player = all_z.merge(zone_summary_player, on="zone", how="left").fillna(0)

    zone_summary_all["rate"] = (
        zone_summary_all["success"] /
        zone_summary_all["total"].replace(0, np.nan) * 100
    ).fillna(0)

    zone_summary_player["rate"] = (
        zone_summary_player["success"] /
        zone_summary_player["total"].replace(0, np.nan) * 100
    ).fillna(0)

    # ---------------------------
    # 3. DRAW FIGURE — MEMORY SAFE
    # ---------------------------
    with safe_fig(figsize=(15, 10)) as (fig, ax):

        fig.set_facecolor(BackgroundColor)

        pitch = VerticalPitch(
            pitch_type='opta',
            pitch_color=PitchColor,
            line_color=PitchLineColor
        )
        pitch.draw(ax=ax)

        # Draw defensive band lines
        band_lines = [78.9, 60, 40, 21.1]
        for v in band_lines:
            ax.axvline(v, ymin=0.185, ymax=0.50, color='blue', linestyle='--', alpha=0.3)

        # Title
        ax.set_title(
            f"{playername} | Defensive Zone Success as {position}",
            fontsize=14,
            pad=10,
            color=TextColor,
        )

        # ---------------------------
        # 4. ZONE LABELS
        # ---------------------------
        zone_centers = {2: 89, 3: 70, 4: 50, 5: 30, 6: 11}

        for zone in range(2, 7):
            x = zone_centers[zone]

            league_rate = zone_summary_all.loc[zone_summary_all.zone == zone, "rate"].iloc[0]
            player_rate = zone_summary_player.loc[zone_summary_player.zone == zone, "rate"].iloc[0]
            player_total = zone_summary_player.loc[zone_summary_player.zone == zone, "total"].iloc[0]

            ax.text(x, 53.5, f"({league_rate:.1f}%)", fontsize=8, ha='center')

            if player_total == 0:
                ax.text(x, 51, "N/A", fontsize=10, weight='bold', ha='center', color='black')
                continue

            if player_rate > league_rate:
                rate_color = "green"
            elif player_rate < league_rate:
                rate_color = "red"
            else:
                rate_color = "black"

            ax.text(
                x, 51,
                f"{player_rate:.1f}%",
                fontsize=10,
                weight='bold',
                ha='center',
                color=rate_color
            )

        # ---------------------------
        # Penalty box (zone 1)
        # ---------------------------
        league_rate = zone_summary_all.loc[zone_summary_all.zone == 1, "rate"].iloc[0]
        player_rate = zone_summary_player.loc[zone_summary_player.zone == 1, "rate"].iloc[0]
        player_total = zone_summary_player.loc[zone_summary_player.zone == 1, "total"].iloc[0]

        ax.text(50, 15.5, f"({league_rate:.1f}%)", fontsize=8, ha='center')

        if player_total == 0:
            ax.text(50, 13, "N/A", fontsize=10, weight='bold', ha='center')
        else:
            if player_rate > league_rate:
                rate_color = "green"
            elif player_rate < league_rate:
                rate_color = "red"
            else:
                rate_color = "black"

            ax.text(50, 13, f"{player_rate:.1f}%", fontsize=10, weight='bold', ha='center', color=rate_color)

        # Notes
        ax.text(50, 61, "Orange circles = Non-duel defensive actions", fontsize=8, ha='center')
        ax.text(50, 64, "Zones are split into 5 strips + penalty box", fontsize=8, ha='center')
        ax.text(50, 67, "League average shown in brackets", fontsize=8, ha='center')

        # Defensive Events
        markers = {"Tackle": ">", "Challenge": ">", "Aerial": "s"}

        for _, ev in playerevents.iterrows():
            marker = markers.get(ev['typeId'], 'o')
            color = 'green' if ev['is_success'] else 'red'
            ax.scatter(ev['y'], ev['x'], marker=marker, color=color, s=25, alpha=0.45)

        for _, ev in player_extra.iterrows():
            ax.scatter(ev['y'], ev['x'], s=25, color='orange', marker='o', edgecolors='none', alpha=0.3)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='>', color='none', markerfacecolor='green', label='Tackle'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor='green', label='Aerial')
        ]
        ax.legend(
            handles=legend_elements,
            fontsize=6,
            loc='center left',
            bbox_to_anchor=(0.79, 0.58),
            frameon=False
        )

        # Logos
        add_image(teamimage, fig, left=0.3625, bottom=0.75, width=0.05)
        add_image(teamimage, fig, left=0.6125, bottom=0.75, width=0.05)
        add_image(wtaimaged, fig, left=0.4825, bottom=0.645, width=0.06)

        return fig

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

    overlay_img=None,
    overlay_size=0.28,
    overlay_alpha=0.25,
    overlay_center=True,
    overlay_left=None,
    overlay_bottom=None,
):
    """
    Radar/diamond polygon figure with optional overlay image.
    MEMORY SAFE: wrapped in safe_fig so figures close automatically.
    """

    n = len(labels)
    if n not in (4, 5):
        raise ValueError("This visual supports ONLY 4 or 5 profiles.")

    scores_arr = np.clip(np.array(scores, dtype=float), 0, 100) / 100.0

    # -------- SHAPE DEFINITION --------
    if n == 5:
        angles = np.deg2rad(np.linspace(90, 90 - 360, 5, endpoint=False))
        base_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        base_pts = np.array([
            [ 0,  1],  # top
            [ 1,  0],  # right
            [ 0, -1],  # bottom
            [-1,  0],  # left
        ], dtype=float)

    # ---------------------------
    # MEMORY-SAFE FIGURE CREATION
    # ---------------------------
    with safe_fig(figsize=(6, 6)) as (fig, ax):

        fig.patch.set_facecolor(fig_bg)
        ax.set_facecolor(fig_bg)

        # Fill background shape
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
        ax.plot(poly[:,0], poly[:,1], linewidth=2, color=polygon_color, zorder=3)

        # -------- CENTROID / BALL --------
        w = np.nan_to_num(scores_arr, nan=0.0)
        if centroid_emphasis != 1.0:
            w = w ** centroid_emphasis

        if w.sum() > 0:
            w /= w.sum()
            cx, cy = (w[:, None] * poly_pts).sum(axis=0)
        else:
            cx, cy = 0.0, 0.0

        # Add ball or dot
        try:
            if ball_img is not None:
                imagebox = OffsetImage(ball_img, zoom=football_zoom)
                ab = AnnotationBbox(imagebox, (cx, cy), frameon=False, zorder=5)
                ax.add_artist(ab)
            else:
                ax.plot(cx, cy, "o", color="white", markersize=12, zorder=5)
        except:
            ax.plot(cx, cy, "o", color="white", markersize=12, zorder=5)

        # -------- LABELS --------
        label_offset = 1.12
        for (x, y), label, score in zip(base_pts, labels, scores):

            ha = "center"
            va = "center"

            if n == 4:
                if (x, y) == (0, 1): va = "bottom"
                elif (x, y) == (1, 0): ha = "left"
                elif (x, y) == (0, -1): va = "top"
                elif (x, y) == (-1, 0): ha = "right"
            else:
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

        # -------- TITLE --------
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

        # -------- CENTER OVERLAY --------
        if overlay_img is not None:
            try:
                arr = overlay_img.copy()

                if overlay_alpha < 1:
                    arr = arr.astype(float) / 255
                    arr[..., :3] *= overlay_alpha
                    arr = (arr * 255).astype("uint8")

                size = overlay_size
                left = 0.5 - size / 2
                right = 0.5 + size / 2
                bottom = 0.5 - size / 2
                top = 0.5 + size / 2

                ax.imshow(
                    arr,
                    extent=[left, right, bottom, top],
                    zorder=4,
                    transform=ax.transAxes,
                    aspect="equal"
                )

            except Exception as e:
                print("Overlay image failed:", e)

        fig.text(
            0.5, 0.1,
            "Created by @WT_Analysis — Data from Opta",
            ha="center",
            va="center",
            fontsize=8,
            color=title_color,
        )

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
COMPARISON_MAP = {
    "Premier League": "ENG1_2526",
    "La Liga":        "SPA1_2526",
    "Bundesliga":     "GER1_2526",
    "Ligue 1":        "FRA1_2526",
    "Serie A":        "ITA1_2526",
    "League of Ireland": "IRE1_2025",
    "Scottish Premiership": "SCO1_2526",
    "Allsvenskan": "SWE1_2025",
    "Austrian Bundesliga": "AUT1_2526",
    "Pro League": "BEL1_2526",
    "Superligaen": "DEN1_2526",
    "Liga Portugal": "POR1_2526",
    "Brasilerao": "BRA1_2025",
    "Championship": "ENG2_2526",
    "League One": "ENG3_2526",
    "League Two": "ENG4_2526",
    "Superleague": "GRE1_2526",
    "MLS": "USA1_2025",
    "J-League": "JAP1_2025",
    "Eredvisie": "NED1_2526",
}

@st.cache_data
def list_excel_files():
    files = load_index_list()
    return sorted([
        f for f in files
        if (f.endswith(".xlsx") or f.endswith(".xls"))
        and "playerstats_by_position_group" in f.lower()
    ])


#@st.cache_data
#def load_match_data(parquet_filename: str):
#    url = build_raw_url(parquet_filename)
#    try:
#        df = pd.read_parquet(io.BytesIO(fetch_raw_file(url)))
#        return df
#    except Exception as e:
#        st.error(f"Error reading match file")
#        return pd.DataFrame()
@st.cache_data(max_entries=1, ttl=600)
def load_match_data(parquet_filename: str):
    url = build_raw_url(parquet_filename)

    try:
        raw_bytes = fetch_raw_file(url)   # Download file bytes (cached separately if needed)
        df = pd.read_parquet(io.BytesIO(raw_bytes))
        return df

    except Exception as e:
        st.error(f"Error reading match file: {e}")
        return pd.DataFrame()

#@st.cache_data
#def load_minute_log(excel_filename: str):
#    url = build_raw_url(excel_filename)
#    try:
#        # Load all columns – we need more than just minutes now
#        df = pd.read_excel(io.BytesIO(fetch_raw_file(url)))
#        return df
#    except Exception as e:
#        st.error(f"Error reading player file")
#        return pd.DataFrame()

@st.cache_data(max_entries=1, ttl=600)
def load_minute_log(excel_filename: str):
    url = build_raw_url(excel_filename)

    try:
        raw_bytes = fetch_raw_file(url)
        df = pd.read_excel(io.BytesIO(raw_bytes))
        return df

    except Exception as e:
        st.error(f"Error reading player file: {e}")
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

    # ---------------------------
    # PRE-PROCESSING (unchanged)
    # ---------------------------
    positiondata = matchdata.loc[matchdata["playing_position"] == position].copy()
    bad_types = ["position_change", "goal_conceded", "clean_sheet"]
    if "typeId" in positiondata.columns:
        positiondata = positiondata[~positiondata["typeId"].isin(bad_types)]
    if "throwin" in positiondata.columns:
        positiondata = positiondata.loc[positiondata["throwin"] != 1]

    if positiondata.empty:
        st.error(f"No data found for position '{position}' after filtering.")
        return None

    positiondata["x"] = positiondata["x"].clip(0, 100)
    positiondata["y"] = positiondata["y"].clip(0, 100)

    x_bins = 10
    y_bins = 7

    x_edges = np.linspace(0, 100, x_bins + 1)
    y_edges = np.linspace(0, 100, y_bins + 1)

    x_bin = np.clip(np.digitize(positiondata["x"], x_edges), 1, x_bins)
    y_bin = np.clip(np.digitize(positiondata["y"], y_edges), 1, y_bins)

    y_bin = (y_bins + 1) - y_bin
    positiondata["pitch_bin"] = (x_bin - 1) * y_bins + y_bin

    drop_types = ["Player off", "Player on", "Corner Awarded", "Card"]
    if "typeId" in positiondata.columns:
        positiondata = positiondata.loc[~positiondata["typeId"].isin(drop_types)]

    xT_summary = (
        positiondata.groupby(["playerName", "pitch_bin"], as_index=False)["xT_value"].sum()
    )

    xT_merged = pd.merge(
        xT_summary, minute_log,
        how="left",
        left_on="playerName",
        right_on="player_name"
    )

    xT_merged["xT_value_per_90"] = np.where(
        xT_merged["minutes_played"] > 0,
        xT_merged["xT_value"] / xT_merged["minutes_played"] * 90,
        np.nan
    )

    xT_merged = xT_merged.drop(columns="player_name").dropna(subset=["xT_value_per_90"])

    avg_bin_xt = (
        xT_merged.groupby("pitch_bin", as_index=False)["xT_value_per_90"]
        .mean()
        .rename(columns={"xT_value_per_90": "avg_xT_value_per_90"})
    )

    xT_compared = pd.merge(xT_merged, avg_bin_xt, on="pitch_bin", how="left")
    xT_compared["xT_value_compared"] = (
        xT_compared["xT_value_per_90"] - xT_compared["avg_xT_value_per_90"]
    )

    playertest = xT_compared.loc[xT_compared["playerName"] == playername].copy()

    if playertest.empty:
        st.error(f"No data found for player '{playername}' at this position.")
        return None

    # Fill missing bins
    all_bins = pd.DataFrame({"pitch_bin": range(1, 71)})
    playertest = pd.merge(all_bins, playertest, on="pitch_bin", how="left")

    first_name = (
        playertest["playerName"].dropna().unique()[0]
        if playertest["playerName"].notna().any()
        else playername
    )

    playertest["playerName"] = playertest["playerName"].fillna(first_name)
    playertest["xT_value_compared"] = playertest["xT_value_compared"].fillna(0)

    colors = ["#d7191c", "#ffffff", "#1a9641"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red_white_green", colors, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)

    # ---------------------------
    # FIGURE CREATION (SAFE)
    # ---------------------------
    pitch = VerticalPitch(
        pitch_type="opta",
        pitch_color=PitchColor,
        line_color=PitchLineColor,
    )

    fig, ax = pitch.draw(figsize=(6, 9))
    fig.set_facecolor(BackgroundColor)

    # Use the close-after wrapper to guarantee cleanup
    with close_after(fig):
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

        add_image(wtaimaged, fig, left=0.4025, bottom=0.43925, width=0.2, alpha=0.25)

        ax.set_title(
            f"{first_name} | Impact by Pitch Area as {position}",
            fontsize=14,
            pad=10,
            color="white",
        )

        return fig  # Streamlit will render this before close_after() closes it

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



    # TEAM BADGE LOOKUP
#    teamname = playerrow["team_name"]
#    teamcode = None
#    if not teamlog.empty:
#        matchrow = teamlog.loc[teamlog["name"] == teamname]
#        if not matchrow.empty:
#            teamcode = matchrow["id"].iloc[0]

#    teamimage = None
#    if teamcode:
#        badge_url = (
#            "https://omo.akamai.opta.net/image.php"
#            f"?h=www.scoresway.com&sport=football&entity=team&description=badges"
#            f"&dimensions=150&id={teamcode}"
#        )
#        try:
#            teamimage = Image.open(urlopen(badge_url))
#        except:
#            teamimage = None
# TEAM BADGE LOOKUP
    teamname = playerrow["team_name"]
    teamcode = None
    
    if not teamlog.empty:
        matchrow = teamlog.loc[teamlog["name"] == teamname]
        if not matchrow.empty:
            teamcode = matchrow["id"].iloc[0]
    
    # Use cached badge loader
    teamimage = load_team_badge(str(teamcode)) if teamcode else None
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

    # Create the figure
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
        param_location=112.5,
    )

    # -------------------------------
    # Memory-safe wrapper begins HERE
    # -------------------------------
    with close_after(fig):

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

        sample_size = len(position_data)
        fig.text(
            0.05, 0.02,
            f"Data: Opta | Metrics per 90 unless stated otherwise | "
            f"{sample_size} players have played at least {minute_threshold} mins as a {position}",
            size=9, color="#000000"
        )

        add_image(wtaimaged, fig, left=0.465, bottom=0.44, width=0.095, height=0.108)

        if teamimage is not None:
            add_image(teamimage, fig, left=0.05, bottom=0.05, width=0.20, height=0.125)

        return fig  # Still returned exactly the same way
def create_player_actions_figure(
    attackingevents,
    defensiveevents,
    playerrecpass,
    playername,
    teamname,
    position,
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

    # ----------------------------------------------------------------------
    # SAFE, FAST, WARNING-FREE HULL CHECK (fix to stop crash + warnings)
    # ----------------------------------------------------------------------
    def _can_hull(points):
        """Check if we can safely compute a 2-D convex hull."""
        if len(points) < 3:
            return False

        uniq = np.unique(points, axis=0)
        if len(uniq) < 3:
            return False

        # Determinant area of triangle (no np.cross(), no NumPy 2.0 warnings)
        p1, p2, p3 = uniq[:3]
        area = abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p2[1] - p1[1]) * (p3[0] - p1[0])
        )

        return area > 1e-6

    # Create a figure with three subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.25), facecolor=BackgroundColor)
    plt.subplots_adjust(wspace=.1)

    with close_after(fig):

        pitch_arrows = VerticalPitch(pitch_type='opta', pitch_color=PitchColor, line_color=PitchLineColor)
        pitch_bins = VerticalPitch(pitch_type='opta', pitch_color=PitchColor, line_color=PitchLineColor)

        # Draw pitches
        pitch_arrows.draw(ax=axes[0], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
        axes[0].set_title(f'{playername} - Attacking Event Locations as {position}', fontsize=10, color=TextColor)

        pitch_bins.draw(ax=axes[1], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
        axes[1].set_title(f'{playername} - Defensive Event Locations as {position}', fontsize=10, color=TextColor)

        pitch_bins.draw(ax=axes[2], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
        axes[2].set_title(f'{playername} - Pass Reception Locations as {position}', fontsize=10, color=TextColor)

        # ---------------------------------------------------------
        # SHARED KDE GRID
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
            density_pass = kde_pass(np.vstack([x_grid.ravel(), y_grid.ravel()]))

            max_density_index_pass = np.argmax(density_pass)
            max_density_x_pass = x_grid.ravel()[max_density_index_pass]
            max_density_y_pass = y_grid.ravel()[max_density_index_pass]

            radius = 15
            points_within_radius_pass = points_pass[
                ((points_pass[:, 0] - max_density_x_pass)**2 +
                 (points_pass[:, 1] - max_density_y_pass)**2) < radius**2
            ]

            # -------------------- FIX APPLIED HERE --------------------
            if _can_hull(points_within_radius_pass):
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
            axes[0].plot(
                row['y'], row['x'], marker='o', markerfacecolor='none',
                color=BackgroundColor, markersize=2
            )

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

            radius = 15
            points_within_radius_def = points_def[
                ((points_def[:, 0] - max_density_x_def)**2 +
                 (points_def[:, 1] - max_density_y_def)**2) < radius**2
            ]

            # -------------------- FIX APPLIED HERE --------------------
            if _can_hull(points_within_radius_def):
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
            axes[1].plot(
                row['y'], row['x'], marker='o', markerfacecolor='none',
                color=BackgroundColor, markersize=2
            )

        # ---------------------------------------------------------
        # PASS RECEPTIONS (PITCH 3)
        # ---------------------------------------------------------
        points_rec = np.array([(row['end_y'], row['end_x']) for _, row in playerrecpass.iterrows()])
        radius = 15

        if len(points_rec) > 3:
            kde_rec = gaussian_kde(points_rec.T)
            density_rec = kde_rec(np.vstack([x_grid.ravel(), y_grid.ravel()]))

            max_density_idx_rec = np.argmax(density_rec)
            max_density_x_rec = x_grid.ravel()[max_density_idx_rec]
            max_density_y_rec = y_grid.ravel()[max_density_idx_rec]

            points_within_radius_rec = points_rec[
                ((points_rec[:, 0] - max_density_x_rec)**2 +
                 (points_rec[:, 1] - max_density_y_rec)**2) < radius**2
            ]

            # -------------------- FIX APPLIED HERE --------------------
            if _can_hull(points_within_radius_rec):
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

        # Plot receptions
        for _, row in playerrecpass.iterrows():
            axes[2].plot(
                row['end_y'], row['end_x'], marker='o', markerfacecolor='none',
                color=BackgroundColor, markersize=2
            )

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
        add_image(teamimage, fig, left=0.56, bottom=-0.05, width=0.06, alpha=1)
        add_image(wtaimaged, fig, left=0.4, bottom=-0.04075, width=0.08, alpha=1)
        plt.close(fig)
        return fig
def create_creative_actions_figure(
    progdata,
    shotassistdata,
    shotlocdata,
    playername,
    teamname,
    competition_name,
    season_name,
    position,
    teamimage,
    wtaimaged,
    BackgroundColor,
    PitchColor,
    PitchLineColor,
    TextColor
):
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    import numpy as np

    # ---------------------------------------------------------
    # Helper: Comet Line
    # ---------------------------------------------------------
    def add_comet(ax, x0, y0, x1, y1, color,
                  n=20, lw_start=0.6, lw_end=2.0,
                  alpha_start=0.10, alpha_end=1.0,
                  z=3):

        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)

        segments = np.stack(
            (np.column_stack([xs[:-1], ys[:-1]]),
             np.column_stack([xs[1:], ys[1:]])),
            axis=1
        )

        widths = np.linspace(lw_start, lw_end, n - 1)
        alphas = np.linspace(alpha_start, alpha_end, n - 1)

        r, g, b, _ = to_rgba(color, 1.0)
        colors = [(r, g, b, a) for a in alphas]

        lc = LineCollection(
            segments,
            linewidths=widths,
            colors=colors,
            capstyle='round',
            joinstyle='round',
            zorder=z
        )
        ax.add_collection(lc)

    # ---------------------------------------------------------
    # 1) Create figure — SAME LAYOUT AS PLAYER ACTIONS
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.25), facecolor=BackgroundColor)
    plt.subplots_adjust(wspace=.1)

    with close_after(fig):

        pitch = VerticalPitch(
            pitch_type='opta',
            pitch_color=PitchColor,
            line_color=PitchLineColor
        )

        # Draw all pitches
        pitch.draw(ax=axes[0], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
        pitch.draw(ax=axes[1], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)
        pitch.draw(ax=axes[2], figsize=(9, 8.25), constrained_layout=True, tight_layout=False)

        # ---------------------------------------------------------
        # PITCH 1 — PROGRESSIVE ACTIONS
        # ---------------------------------------------------------
        axes[0].set_title(f"{playername} - Progressive Actions as {position}", fontsize=10, color=TextColor)

        for _, row in progdata.iterrows():
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]

            if row.get("progressive_pass") == "Yes":
                add_comet(axes[0], x0, y0, x1, y1, color="green")

            if row.get("progressive_carry") == "Yes":
                add_comet(axes[0], x0, y0, x1, y1, color="purple")

        axes[0].text(50, -5,
                     "Green = Progressive Pass | Purple = Progressive Carry",
                     ha='center', fontsize=9, color=TextColor)

        axes[0].text(50, -13, f"{playername} - {teamname}",
                     ha='center', fontsize=12, color=TextColor, fontweight='bold')

        axes[0].text(50, -20, f"{competition_name} | {season_name}",
                     ha='center', fontsize=12, color=TextColor, fontweight='bold')

        # ---------------------------------------------------------
        # PITCH 2 — SHOT ASSISTS
        # ---------------------------------------------------------
        axes[1].set_title(f"{playername} - Shot Assists as {position}", fontsize=10, color=TextColor)

        for _, row in shotassistdata.iterrows():
            x0, y0, x1, y1 = row["y"], row["x"], row["end_y"], row["end_x"]

            if row.get("keyPass", 0) == 1:
                add_comet(axes[1], x0, y0, x1, y1, color="orange")

            if row.get("assist", 0) == 1:
                add_comet(axes[1], x0, y0, x1, y1, color="blue")

        axes[1].text(50, -5,
                     "Orange = Shot Assist | Blue = Assist",
                     ha='center', fontsize=9, color=TextColor)

        # ---------------------------------------------------------
        # PITCH 3 — SHOT ASSIST LOCATIONS
        # ---------------------------------------------------------
        axes[2].set_title(f"{playername} - Shot Assist Locations as {position}", fontsize=10, color=TextColor)

        for _, row in shotlocdata.iterrows():

            if row.get("keyPass", 0) == 1:
                axes[2].plot(
                    row["y"], row["x"],
                    marker="o", markersize=8,
                    markerfacecolor="orange", markeredgecolor="black", linewidth=1.5
                )

            if row.get("assist", 0) == 1:
                axes[2].plot(
                    row["y"], row["x"],
                    marker="o", markersize=8,
                    markerfacecolor="blue", markeredgecolor="black", linewidth=1.5
                )

        axes[2].text(50, -5,
                     "Orange = Shot Assist | Blue = Assist",
                     ha='center', fontsize=9, color=TextColor)
        axes[2].text(50, -10, 'Events shown are open play only',
                     ha='center', fontsize=9, color=TextColor)
        axes[2].text(50, -15, 'Shot Assists are passes that lead directly to a shot',
                     ha='center', fontsize=9, color=TextColor)
        axes[2].text(50, -20, 'Data from Opta',
                     ha='center', fontsize=9, color=TextColor)

        # ---------------------------------------------------------
        # LOGOS
        # ---------------------------------------------------------
        add_image(teamimage, fig, left=0.56, bottom=-0.05, width=0.06, alpha=1)
        add_image(wtaimaged, fig, left=0.4, bottom=-0.04075, width=0.08, alpha=1)

        return fig
# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    st.title("WT Analysis - Player Data Dashboard")
    st.subheader("Select League, Player & Position on the Left - Contact @WT_Analysis on X for more info")
    st.subheader("Data last updated 12/12/2025")

    st.sidebar.header("League Selection")

    # --------------------------
    # 1. Competition selection (now first)
    # --------------------------
    competition_choice = st.sidebar.selectbox(
        "Competition",
        list(COMP_MAP.keys()),
        index=0,
    )

    # --------------------------
    # 2. Season auto-selection (locked)
    # --------------------------
    # Leagues that must use 2025
    forced_2025 = ["Brasilerao", "League of Ireland", "Allsvenskan", "MLS", "J-League"]

    if competition_choice in forced_2025:
        season_choice = "2025"
        st.sidebar.selectbox("Season", ["2025"], index=0, disabled=True)
    else:
        season_choice = "2025/26"
        st.sidebar.selectbox("Season", ["2025/26"], index=0, disabled=True)

    season_fragment = SEASON_MAP[season_choice]

    # --------------------------
    # 3. Build file paths using auto-selected season
    # --------------------------
    league_prefix = COMP_MAP[competition_choice]

    parquet_choice = f"{league_prefix}_{season_fragment}.parquet"
    excel_choice   = f"{league_prefix}_{season_fragment}_playerstats_by_position_group.xlsx"

    # --------------------------
    # LOAD DATA
    # --------------------------
    with st.spinner("Loading match and player stats data..."):
        matchdata = load_match_data(parquet_choice)           # match event data
        player_stats = load_minute_log(excel_choice)          # full stats file (Excel)
        teamlog = load_teamlog()                              # teamcode lookup file
    if "position_group" in player_stats.columns:
        player_stats["position_group"] = (
            player_stats["position_group"]
            .astype(str)
            .apply(normalize_position)
        )
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
    st.sidebar.header("Player & Position")

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
        "Select Player",
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
        teamcode = teamlog.loc[teamlog["name"] == team_choice, "id"].iloc[0]
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
        player_stats[
            (player_stats["player_name"] == playername) &
            (player_stats["team_name"] == team_choice)
        ]
        .groupby("position_group", as_index=False)["minutes_played"]
        .sum()
    )
    
    # FILTER OUT POSITIONS WITH UNDER 25 MINUTES
    position_minutes = position_minutes[position_minutes["minutes_played"] >= 25]
    
    # Sort by minutes descending
    position_minutes = position_minutes.sort_values("minutes_played", ascending=False)
    
    positions = position_minutes["position_group"].astype(str).tolist()
    
    if not positions:
        st.error(f"Please check your selections on the left - no data available with current filters")
        return
    
    # -------------------------------------------------------
    # Determine default position
    # -------------------------------------------------------
    preferred_positions = ["LB", "LCB(2)", "RCB(2)", "RB"]
    default_position = next((p for p in preferred_positions if p in positions), positions[0])
    
    # Sidebar selector
    position = st.sidebar.selectbox(
        "Select Position",
        positions,
        index=positions.index(default_position) if default_position in positions else 0,
    )
    
    # Minute threshold selector
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
    st.markdown(f"#### {playername} | {team_choice} - Player Info (positions with negligible minutes not shown)")
    
    # Aggregate extended stats per position
    pos_extended = (
        player_stats[
            (player_stats["player_name"] == playername) &
            (player_stats["team_name"] == team_choice)
        ]
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
    
    # Merge with dropdown ordering
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
    
    # ---------- FORMAT AS STRINGS ----------
    pos_extended["Minutes"] = pos_extended["Minutes"].map(lambda x: f"{x:.1f}")
    #pos_extended["xG"] = pos_extended["xG"].map(lambda x: f"{x:.3f}")
    #pos_extended["Goals"] = pos_extended["Goals"].map(lambda x: f"{int(x)}")
    pos_extended["xG"] = pos_extended["xG"].apply(
        lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
    )
    pos_extended["Goals"] = (
        pd.to_numeric(pos_extended["Goals"], errors="coerce")  # turns NaN into NaN
          .fillna(0)                                           # replace NaN with 0 (or "" if you prefer)
          .astype(int)                                         # safe now
          .astype(str)
    )
    #pos_extended["xA"] = pos_extended["xA"].map(lambda x: f"{x:.3f}")
    #pos_extended["Assists"] = pos_extended["Assists"].map(lambda x: f"{int(x)}")
    pos_extended["xA"] = pos_extended["xA"].apply(
        lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
    )    
    pos_extended["Assists"] = (
        pd.to_numeric(pos_extended["Assists"], errors="coerce")  # turns NaN into NaN
          .fillna(0)                                           # replace NaN with 0 (or "" if you prefer)
          .astype(int)                                         # safe now
          .astype(str)
    )    
    pos_extended["Pass %"] = pos_extended["Pass %"].map(lambda x: f"{x*100:.2f}%")
    pos_extended["Aerial %"] = pos_extended["Aerial %"].map(lambda x: f"{x*100:.2f}%")
    pos_extended["Tackle %"] = pos_extended["Tackle %"].map(lambda x: f"{x*100:.2f}%")
    
    pos_extended["Successful Def. Actions per 90"] = (
        pos_extended["Successful Def. Actions per 90"].map(lambda x: f"{x:.2f}")
    )
    pos_extended["Successful Att. Actions per 90"] = (
        pos_extended["Successful Att. Actions per 90"].map(lambda x: f"{x:.2f}")
    )
    
    # ---------- HTML + CSS CENTERED TABLE (DARK MODE SAFE) ----------
    table_html = pos_extended.to_html(index=False, classes="playerinfo-table")
    
    css = """
    <style>
    
    /* ---------- COMMON BASE STYLES ---------- */
    .playerinfo-table {
        margin-left: auto;
        margin-right: auto;
        width: 96%;
        border-collapse: collapse;
        font-family: var(--font, "Inter", sans-serif);
        font-size: 0.95rem;
        text-align: center;
    }
        /* FORCE CENTER ALIGNMENT */
    .playerinfo-table th,
    .playerinfo-table td {
        text-align: center !important;
    }
    /* ---------- LIGHT MODE ---------- */
    @media (prefers-color-scheme: light) {
        .playerinfo-table {
            color: #222 !important;
        }
    
        .playerinfo-table th {
            background-color: #e9ecef !important;
            color: #111 !important;
            border-bottom: 1px solid #c9c9c9 !important;
        }
    
        .playerinfo-table td {
            border-bottom: 1px solid #d5d5d5 !important;
        }
    
        .playerinfo-table tr:hover td {
            background-color: rgba(0,0,0,0.05) !important;
        }
    }
    
    /* ---------- DARK MODE ---------- */
    @media (prefers-color-scheme: dark) {
        .playerinfo-table {
            color: #f8f9fa !important;
        }
    
        .playerinfo-table th {
            background-color: rgba(255,255,255,0.10) !important;
            color: #ffffff !important;
            border-bottom: 1px solid rgba(255,255,255,0.15) !important;
        }
    
        .playerinfo-table td {
            border-bottom: 1px solid rgba(255,255,255,0.12) !important;
        }
    
        .playerinfo-table tr:hover td {
            background-color: rgba(255,255,255,0.08) !important;
        }
    }
    
    </style>
    """
    
    # auto-height instead of fixed height
    st.components.v1.html(css + table_html, scrolling=False)
    # --------------------------
    # TABS — Pitch Map + Player Pizza
    # --------------------------
    #tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    #    ["Pitch Impact Map", "Player Pizza", "Player Actions", "Player Profiling",
    #     "Shot Maps", "Creative Actions", "Metric Comparisons",
    #     "Defensive Actions", "Passes & Carries", "Player Similarity"]
    #)
    tab2, tab4, tab3, tab8, tab6, tab9, tab5, tab1, tab7, tab10 = st.tabs(
        ["Player Pizza", "Player Profiling", "Player Actions", 
         "Defensive Actions", "Creative Actions", "Passes & Carries", "Shot Maps", "Pitch Impact",
         "Metric Comparison", "Player Similarity"]
    )
    # Init session state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Pitch Impact Map"

    # ================================================================
    # TAB 1 — Pitch Impact Map
    # ================================================================
  #  with tab1:
  #      st.session_state["active_tab"] = "Pitch Impact Map"
  #      st.header("Pitch Impact Map")
  #  
        # Define the condition under which all required inputs exist
  #      inputs_ready = (
  #          matchdata is not None
  #       and minute_log is not None
  #          and position not in (None, "")
  #          and playername not in (None, "")
  #          and season_choice not in (None, "")
  #      )
    
        # Only generate + display inside this tab when selections are ready
   #     if inputs_ready and st.session_state["active_tab"] == "Pitch Impact Map":
    
   #         fig = plot_xt_comparison_for_player(
   #             matchdata=matchdata,
   #             minute_log=minute_log,
   #             position=position,
   #             playername=playername,
   #             season=season_choice,
   #         )
    
   #         if fig is not None:
   #             left, center, right = st.columns([1, 4, 1])
   #             with center:
   #                 st.image(fig_to_png_bytes(fig), width=550)
    
    with tab1:
        st.session_state["active_tab"] = "Pitch Impact Map"
        st.header("Pitch Impact Map")
    
        inputs_ready = (
            matchdata is not None
            and minute_log is not None
            and position not in (None, "")
            and playername not in (None, "")
            and season_choice not in (None, "")
        )
    
        if inputs_ready and st.session_state["active_tab"] == "Pitch Impact Map":
    
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
                    st.image(fig_to_png_bytes(fig), width=550)   # keep your width
                plt.close(fig)  # absolutely required to prevent crashes
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
                    st.image(img_bytes, width=700)   # ✔ keep your width
                plt.close(pizza_fig)                 # 🔥 required to prevent crashes

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
                position,
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
                plt.close(fig)
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
            plt.close(fig)
    # ================================================================
    # TAB 5 — Shot Maps
    # ================================================================
    # ================================================================
    # TAB 5 — Shot Maps
    # ================================================================
    with tab5:
        st.session_state["active_tab"] = "Shot Maps"
        st.header("Shot Maps")
    
        # ------------------------------------------
        # BUILD SHOT DATA
        # ------------------------------------------
        playershots = matchdata.loc[
            (matchdata['playerName'] == playername) &
            (matchdata['team_name'] == team_choice) &
            (matchdata['playing_position'] == position)
        ].copy()
    
        playershots = playershots.loc[playershots['shotType'].notna()]
    
        shotmaptar2 = playershots.loc[playershots['typeId']=='Attempt Saved']
        shotmaptar2 = shotmaptar2.loc[shotmaptar2['expectedGoalsOnTarget']>0]
    
        shotmapbk2 = playershots.loc[playershots['typeId']=='Attempt Saved']
        shotmapbk2 = shotmapbk2.loc[shotmapbk2['expectedGoalsOnTarget']== 0]
    
        shotmapoff2 = playershots.loc[playershots['typeId'] == 'Miss']
    
        goalmap3 = playershots.loc[
            (playershots['typeId'] == 'Goal') &
            (playershots['typeId'] != 'Own Goal')
        ]
    
        num_goals = len(goalmap3)
        num_shots = len(shotmaptar2) + len(shotmapbk2) + len(shotmapoff2) + len(goalmap3)
        shots_on_target = len(shotmaptar2) + len(goalmap3)
        shot_conversion_rate = round((shots_on_target / num_shots) * 100, 1) if num_shots > 0 else 0
        goal_conversion_rate = round((num_goals / num_shots) * 100, 1) if num_shots > 0 else 0
        xg_sum = round(playershots['expectedGoals'].sum(), 2)
        xgot_sum = round(playershots['expectedGoalsOnTarget'].sum(), 2)
    
        # ---------------------------------------------------------------
        # LAYOUT → Two columns side-by-side
        # ---------------------------------------------------------------
        col1, col2 = st.columns([1, 1], gap="large")
    
    
        # ===============================================================
        # LEFT COLUMN: ORIGINAL SHOT MAP
        # ===============================================================
        with col1:
    
            fig, ax = plt.subplots(figsize=(10, 7.5))
            fig.set_facecolor(BackgroundColor)
    
            pitch_left = VerticalPitch(
                pitch_type='opta',
                half=True,
                pitch_color=PitchColor,
                line_color=PitchLineColor
            )
            pitch_left.draw(ax=ax)
    
            def get_marker_size(expectedGoals, scale_factor=1000):
                return expectedGoals * scale_factor
    
            pitch_left.scatter(
                shotmaptar2.x, shotmaptar2.y,
                s=get_marker_size(shotmaptar2.expectedGoals),
                ax=ax, edgecolor='blue', facecolor='none', marker='o',
                label='Shot on Target'
            )
    
            pitch_left.scatter(
                shotmapbk2.x, shotmapbk2.y,
                s=get_marker_size(shotmapbk2.expectedGoals),
                ax=ax, edgecolor='orange', facecolor='none', marker='o',
                label='Shot Blocked'
            )
    
            pitch_left.scatter(
                shotmapoff2.x, shotmapoff2.y,
                s=get_marker_size(shotmapoff2.expectedGoals),
                ax=ax, edgecolor='red', facecolor='none', marker='o',
                label='Shot off Target'
            )
    
            pitch_left.scatter(
                goalmap3.x, goalmap3.y,
                s=get_marker_size(goalmap3.expectedGoals),
                ax=ax, edgecolor='green', facecolor='none', marker='o',
                label='Goal'
            )
    
            ax.set_title(
                f"{playername} - xG Shot Map as {position} - {team_choice}",
                fontsize=15, color=TextColor
            )
    
            handles, labels = ax.get_legend_handles_labels()
            legend_markers = [
                plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                           markeredgecolor='blue', markersize=10),
                plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                           markeredgecolor='orange', markersize=10),
                plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                           markeredgecolor='red', markersize=10),
                plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                           markeredgecolor='green', markersize=10)
            ]
    
            ax.legend(
                legend_markers, labels,
                facecolor='none', handlelength=5, edgecolor='None',
                bbox_to_anchor=(.22275, .94), fontsize=8
            )
    
            add_image(teamimage, fig, left=0.7, bottom=0.16, width=0.1, alpha=1)
            add_image(wtaimaged, fig, left=0.472, bottom=0.165, width=0.08, alpha=1)
    
            ax.text(99, 73,   f'Goals Scored: {num_goals}', ha='left', fontsize=9, color='black')
            ax.text(99, 71.5, f'Total xG: {xg_sum}',       ha='left', fontsize=9, color='black')
            ax.text(99, 70,   f'Total xGOT: {xgot_sum}',   ha='left', fontsize=9, color='black')
            ax.text(99, 68.5, f'Shots Taken: {num_shots}', ha='left', fontsize=9, color='black')
            ax.text(99, 67,   f'Shots on Target: {shots_on_target}', ha='left', fontsize=9, color='black')
            ax.text(99, 65.5, f'Shots Accuracy: {shot_conversion_rate}%', ha='left', fontsize=9, color='black')
            ax.text(99, 64,   f'Goal Conversion: {goal_conversion_rate}%', ha='left', fontsize=9, color='black')
    
            st.image(fig_to_png_bytes(fig), width=750)
            plt.close(fig)
    
    
    
        # ===============================================================
        # RIGHT COLUMN: NEW GOAL-MOUTH VISUAL
        # ===============================================================
        with col2:
    
            import json
    
            def extract_goalmouth(df):
                df = df.copy()
                df['onGoalShot'] = df['onGoalShot'].apply(
                    lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
                )
                df['goal_x'] = df['onGoalShot'].apply(lambda x: x.get('x') if isinstance(x, dict) else None)
                df['goal_y'] = df['onGoalShot'].apply(lambda x: x.get('y') if isinstance(x, dict) else None)
                df['zoomratio'] = df['onGoalShot'].apply(lambda x: x.get('zoomRatio') if isinstance(x, dict) else None)
                return df
    
            shotmap_gm = extract_goalmouth(shotmaptar2)
            goalmap_gm = extract_goalmouth(goalmap3)
    
            size_factor = 800
            sizes_tar = shotmap_gm['expectedGoalsOnTarget'].fillna(0).clip(lower=0.01) * size_factor
            sizes_goals = goalmap_gm['expectedGoalsOnTarget'].fillna(0).clip(lower=0.01) * size_factor
    
            # NEW background colour here
            figGM, axGM = plt.subplots(figsize=(10, 7.5))
            figGM.set_facecolor(BackgroundColor)
            axGM.set_facecolor(PitchColor)
    
            # Goal frame
            axGM.plot([0, 0], [0, 0.66], color='black', linewidth=4)
            axGM.plot([2, 2], [0, 0.66], color='black', linewidth=4)
            axGM.plot([0, 2], [0.66, 0.66], color='black', linewidth=4)
    
            axGM.scatter(
                shotmap_gm['goal_x'], shotmap_gm['goal_y'],
                s=sizes_tar, facecolors='none', edgecolors='blue',
                alpha=0.7, label='Shots on Target'
            )
    
            axGM.scatter(
                goalmap_gm['goal_x'], goalmap_gm['goal_y'],
                s=sizes_goals, facecolors='green', edgecolors='#381d54',
                linewidths=1.5, label='Goals'
            )
    
            axGM.set_xlim(-0.5, 2.5)
            axGM.set_ylim(-0.2, 1.5)
            axGM.set_xticks([])
            axGM.set_yticks([])
    
            for side in ['top', 'right', 'left', 'bottom']:
                axGM.spines[side].set_visible(False)
    
            for x in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]:
                axGM.plot([x, x], [0, 0.66], color='grey', linewidth=0.5)
            for y in [0.0825, 0.165, 0.2475, 0.33, 0.4125, 0.495, 0.5775]:
                axGM.plot([0, 2], [y, y], color='grey', linewidth=0.5)
    
            axGM.set_title(
                f"{playername} – xG Shot Placement as {position} - {team_choice}",
                fontsize=14, color=TextColor
            )
            add_image(teamimage, figGM, left=0.32, bottom=0.58, width=0.125, alpha=1)
            add_image(wtaimaged, figGM, left=0.58, bottom=0.60, width=0.125, alpha=1)
            st.pyplot(figGM)
            plt.close(figGM)  

# ================================================================
# TAB6 — Creative Actions
# ================================================================
    with tab6:
        st.session_state["active_tab"] = "Creative Actions"
        st.header("Creative Actions")
    
        # Prevent crash
        if not playername or not position:
            st.warning("Please select a player and position to view Creative Actions.")
            st.stop()
    
        # ---------------------------------------------------------
        # 1. Build Data for Creative Actions
        # ---------------------------------------------------------
    
        # Base player events
        playerevents = matchdata.loc[
            (matchdata["playerName"] == playername) &
            (matchdata["team_name"] == team_choice) &
            (matchdata["playing_position"] == position)
        ].copy()
    
        # ------------- PITCH 1: Progressive Actions -------------
        progdata = playerevents[
            (
                # Progressive Pass
                (
                    (playerevents.get("progressive_pass") == "Yes") &
                    (playerevents.get("typeId") == "Pass")
                )
                |
                # Progressive Carry (but only < 30 yards)
                (
                    (playerevents.get("progressive_carry") == "Yes") &
                    (playerevents.get("typeId") == "Carry") &
                    (playerevents.get("carrying_yards", 0) < 30) &
                    (
                        (playerevents.get("end_y") - playerevents.get("y")).abs() <= 40
                    )
                )
            )
            &
            (playerevents.get("corner", 0) != 1)
            &
            (playerevents.get("freekick", 0) != 1)
            &
            (playerevents.get("throwin", 0) != 1)
            &
            (playerevents.get("goalkick", 0) != 1)
        ].copy()
        # ------------- PITCH 2 + 3: Shot Assists -------------
        shotassistdata = playerevents[
            (
                (
                    (playerevents.get("keyPass", 0) == 1) |
                    (playerevents.get("assist", 0) == 1)
                )
                &
                (playerevents.get("typeId") == "Pass")
            )
            &
            (playerevents.get("corner", 0) != 1)
            &
            (playerevents.get("freekick", 0) != 1)
            &
            (playerevents.get("throwin", 0) != 1)
            &
            (playerevents.get("goalkick", 0) != 1)
        ].copy()
    
        # PITCH 3 just plots locations of the same events
        shotlocdata = shotassistdata.copy()
    
        # ---------------------------------------------------------
        # 2. Create the Creative Actions Visual
        # ---------------------------------------------------------
        fig = create_creative_actions_figure(
            progdata=progdata,
            shotassistdata=shotassistdata,
            shotlocdata=shotlocdata,
            playername=playername,
            teamname=team_choice,
            competition_name=competition_choice,
            season_name=season_choice,
            position=position,               # ← REQUIRED
            teamimage=teamimage,
            wtaimaged=wtaimaged,
            BackgroundColor=BackgroundColor,
            PitchColor=PitchColor,
            PitchLineColor=PitchLineColor,
            TextColor=TextColor
        )
    
        # ---------------------------------------------------------
        # 3. Display in Center Column
        # ---------------------------------------------------------
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.image(fig_to_png_bytes(fig), width=1600)
            plt.close(fig)

    # ================================================================
    # TAB 7 — Metric Scatter (FINAL POLISHED & CORRECTED VERSION)
    # ================================================================
    with tab7:
        st.header(f"Interactive Metric Comparison - {position} in {competition_choice}")
    
        # ------------------------------------------------------------
        # Filter dataset by selected position + minutes
        # ------------------------------------------------------------
        df_filtered = player_stats.copy()
        df_filtered["minutes_played"] = pd.to_numeric(df_filtered["minutes_played"], errors="coerce")
        df_filtered = df_filtered[
            (df_filtered["position_group"] == position) &
            (df_filtered["minutes_played"] >= minuteinput)
        ]
    
        if df_filtered.empty:
            st.warning("No players meet the selected position and minute threshold.")
            st.stop()
    
        # ------------------------------------------------------------
        # Metric selector
        # ------------------------------------------------------------
        available_metrics = [m for m in SCATTER_METRIC_MAP if m in df_filtered.columns]
    
        chosen = st.multiselect(
            "Select TWO metrics to plot:",
            options=[SCATTER_METRIC_MAP[m] for m in available_metrics],
            max_selections=2
        )
    
        if len(chosen) == 2:
    
            # ------------------------------------------------------------
            # Resolve metric_x / metric_y from DISPLAY NAMES → DATAFRAME KEYS
            # ------------------------------------------------------------
            metric_x = resolve_metric(chosen[0])
            metric_y = resolve_metric(chosen[1])
    
            if metric_x is None or metric_y is None:
                st.error("Error resolving metric names. Check SCATTER_METRIC_MAP.")
                st.stop()
    
            # ------------------------------------------------------------
            # Detect if these metrics should be displayed as %
            # ------------------------------------------------------------
            is_x_percent = metric_is_percent(SCATTER_METRIC_MAP[metric_x])
            is_y_percent = metric_is_percent(SCATTER_METRIC_MAP[metric_y])
    
            # Filter dropped rows (no missing values)
            df_plot = df_filtered.dropna(subset=[metric_x, metric_y]).copy()
    
            # ------------------------------------------------------------
            # Identify selected player row
            # ------------------------------------------------------------
            df_plot["highlight"] = (
                (df_plot["player_name"] == playername) &
                (df_plot["team_name"] == team_choice)
            )
    
            import plotly.express as px
    
            fig = px.scatter(
                df_plot,
                x=metric_x,
                y=metric_y,
                hover_name="player_name",            # ONLY player name
                hover_data={},                       # no extra hover info
                color=df_plot["highlight"].map({True: "Selected", False: "Other"}),
                size=df_plot["highlight"].map({True: 26, False: 14}),
                color_discrete_map={
                    "Selected": "red",
                    "Other": "black",
                }
            )
    
            # ------------------------------------------------------------
            # Remove legend
            # ------------------------------------------------------------
            fig.update_layout(showlegend=False)
    
            # ------------------------------------------------------------
            # Title + global layout
            # ------------------------------------------------------------
            fig.update_layout(
                title=f"{SCATTER_METRIC_MAP[metric_x]} vs {SCATTER_METRIC_MAP[metric_y]} — {position} only",
                title_font=dict(color=TextColor, size=20),
                title_x=0.52,                # center over plot area
                title_xanchor="center",
    
                xaxis_title=SCATTER_METRIC_MAP[metric_x],
                yaxis_title=SCATTER_METRIC_MAP[metric_y],
    
                plot_bgcolor=PitchColor,     # inner plot background
                paper_bgcolor=BackgroundColor,  # outer background
    
                font=dict(color=TextColor),
    
                width=1000,
                height=650,
                margin=dict(l=80, r=40, t=80, b=80)
            )
    
            # ------------------------------------------------------------
            # Axis formatting (increase font + remove grid + % formatting)
            # ------------------------------------------------------------
            fig.update_xaxes(
                title_font=dict(color=TextColor, size=14),
                tickfont=dict(color=TextColor, size=14),
                showgrid=False,
                tickformat=".0%" if is_x_percent else None
            )
    
            fig.update_yaxes(
                title_font=dict(color=TextColor, size=14),
                tickfont=dict(color=TextColor, size=14),
                showgrid=False,
                tickformat=".0%" if is_y_percent else None
            )
    
            # ------------------------------------------------------------
            # Highlight selected player + hover only name
            # ------------------------------------------------------------
            fig.update_traces(
                marker=dict(line=dict(width=1.5, color="white")),
                hovertemplate="%{hovertext}<extra></extra>"
            )
    
            # ------------------------------------------------------------
            # Add logo inside the plot (top-left, inside axis)
            # ------------------------------------------------------------
            fig.add_layout_image(
                dict(
                    source=wtaimaged,
                    xref="paper", yref="paper",
                    x=0.01,
                    y=0.99,
                    sizex=0.20,
                    sizey=0.20,
                    xanchor="left",
                    yanchor="top",
                    opacity=0.5,
                    layer="above",
                )
            )
    
            # ------------------------------------------------------------
            # Display chart
            # ------------------------------------------------------------
            st.plotly_chart(fig, use_container_width=False)
    
        else:
            st.info("Please select exactly two metrics.")
    # ==========================================================
    # TAB 8 — DEFENSIVE ACTIONS
    # ==========================================================
    with tab8:
        st.header("Defensive Actions")
    
        # -----------------------------------
        # Defensive data prep (LOCAL TO TAB)
        # -----------------------------------
        defensive_types = ['Tackle', 'Aerial', 'Challenge']
    
        defdata = matchdata.loc[
            (matchdata["playing_position"] == position) &
            (matchdata["typeId"].isin(defensive_types))
        ].copy()
    
        defdata["x"] = pd.to_numeric(defdata["x"], errors="coerce")
        defdata["y"] = pd.to_numeric(defdata["y"], errors="coerce")
        defdata["is_success"] = defdata["outcome"].eq("Successful")
    
        fig = create_defensive_actions_figure(
            matchdata=matchdata,
            playername=playername,
            team_choice=team_choice,
            position=position,
            teamimage=teamimage,
            wtaimaged=wtaimaged,
            BackgroundColor=BackgroundColor,
            PitchColor=PitchColor,
            PitchLineColor=PitchLineColor,
            TextColor=TextColor
        )
    
        st.image(fig_to_png_bytes(fig), width=600)
        plt.close(fig)
        
    with tab9:
        st.subheader("Passing & Carrying Sonars")
    
        fig = create_pass_and_carry_sonar(
            matchdata=matchdata,
            playername=playername,
            team_choice=team_choice,
            position=position,
            BackgroundColor=BackgroundColor,
            PitchColor=PitchColor,
            PitchLineColor=PitchLineColor,
            TextColor=TextColor,
        )

        st.image(fig_to_png_bytes(fig), width=1100)
        plt.close(fig)


    # TAB 10 — PLAYER SIMILARITY ENGINE (TOP/BOTTOM METRICS VERSION)
    # + MULTI-LEAGUE SUPPORT
    # ================================================================
    with tab10:
        st.header("Closest Playing Style Comparables")

        # --------------------------------------------
        # 0. League selector for additional datasets
        # --------------------------------------------
        st.subheader("Add Additional Leagues for Further Comparison")

        available_leagues = [
            lg for lg in COMPARISON_MAP.keys()
            if lg != competition_choice  # exclude currently selected league
        ]
        #st.write("Selected league:", competition_choice)
        #st.write("Available COMPARISON_MAP keys:", list(COMPARISON_MAP.keys()))
        selected_extra_leagues = st.multiselect(
            "Select up to 4 additional leagues:",
            options=available_leagues,
            max_selections=4
        )

        # --------------------------------------------
        # 1. Load current league data
        # --------------------------------------------
        df_main = player_stats.copy()

        # Normalise positions in the *main* league as well
        position_replacements = {
            "LMW": "LW",
            "RMW": "RW",
        }
        if "position_group" in df_main.columns:
            df_main["position_group"] = df_main["position_group"].replace(position_replacements)

        # --------------------------------------------
        # 2. Load extra league data files
        # --------------------------------------------
       # @st.cache_data
      #  def load_league_file(mapped_name: str) -> pd.DataFrame | None:
     #       files = list_excel_files()
    #        for f in files:
   #             if mapped_name.lower() in f.lower():
  #                  # NOTE: these files are already local / accessible in your setup
 #                   df = pd.read_excel(f)
#
              #      position_replacements = {
             #           "LMW": "LW",
            #            "RMW": "RW",
           #         }

                    #pos_col = None
                   # for c in df.columns:
                  #      if c.lower() in ["position_group", "position", "pos"]:
                 #           pos_col = c
                #            break

               #     if pos_col is not None:
              #          df[pos_col] = df[pos_col].replace(position_replacements)

             #       return df

            #return None


        @st.cache_data(max_entries=1, ttl=600)
        def load_league_file(mapped_name: str) -> pd.DataFrame | None:
            files = list_excel_files()
        
            for f in files:
                if mapped_name.lower() in f.lower():
        
                    try:
                        df = pd.read_excel(f)
        
                        # Normalise positions across leagues
                        position_replacements = {
                            "LMW": "LW",
                            "RMW": "RW",
                        }
        
                        # Try to detect the correct position column
                        pos_col = next(
                            (c for c in df.columns if c.lower() in ["position_group", "position", "pos"]),
                            None
                        )
        
                        if pos_col:
                            df[pos_col] = df[pos_col].replace(position_replacements)
        
                        return df
        
                    except Exception as e:
                        st.error(f"Error reading league file: {e}")
                        return None
        
            return None
        extra_frames = []
        for lg in selected_extra_leagues:
            mapped = COMPARISON_MAP[lg]
            df_extra = load_league_file(mapped)
            if df_extra is not None:
                extra_frames.append(df_extra)
        # --------------------------------------------
        # 3. Combine all datasets
        # --------------------------------------------
        if extra_frames:
            combined_df = pd.concat([df_main] + extra_frames, ignore_index=True)
        else:
            combined_df = df_main.copy()

        # --------------------------------------------
        # 4. Percentiles for the requested similarity metrics
        # --------------------------------------------
        def pct_0_to_100(s: pd.Series) -> pd.Series:
            s_num = pd.to_numeric(s, errors="coerce")
            n = s_num.count()
            if n <= 1:
                return pd.Series([np.nan] * len(s_num), index=s_num.index, dtype=float)
            return (s_num.rank(method="min") - 1) / (n - 1) * 100

        # Keep only metrics that actually exist in this combined dataset
        metric_cols_available = [c for c in SIMILARITY_METRICS if c in combined_df.columns]

        if not metric_cols_available:
            st.warning("None of the configured similarity metrics are present in the stats data.")
            st.stop()

        if "position_group" not in combined_df.columns:
            st.error("Column 'position_group' is required for similarity comparison.")
            st.stop()

        # Apply percentiles within position_group for each metric
        for col in metric_cols_available:
            combined_df[col + "__pct"] = (
                combined_df
                .groupby("position_group")[col]
                .transform(pct_0_to_100)
            )

        # --------------------------------------------
        # 5. Filter for selected position + minutes
        # --------------------------------------------
        df = combined_df.copy()
        df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce")

        df = df[
            (df["position_group"] == position) &
            (df["minutes_played"] >= minuteinput)
        ].copy()

        if df.empty:
            st.warning("No players available for similarity comparison for this position and minute threshold.")
            st.stop()

        # --------------------------------------------
        # 6. Work out THIS player's top 15 & bottom 10 percentile metrics
        # --------------------------------------------
        # Percentile columns corresponding to the metric list
        pct_cols = [c + "__pct" for c in metric_cols_available]

        # Find selected player row
        mask_player = (df["player_name"] == playername) & (df["team_name"] == team_choice)
        if mask_player.sum() == 0:
            st.warning("Could not find a matching row for the selected player in the stats table.")
            st.stop()

        player_index = df.index[mask_player][0]
        player_pcts = df.loc[player_index, pct_cols].dropna()

        if player_pcts.empty:
            st.warning("Selected player has no percentile data for the configured metrics.")
            st.stop()

        # Strongest 15 metrics (highest percentiles)
        top_15 = player_pcts.sort_values(ascending=False).head(15)

        # Weakest 10 metrics (lowest percentiles)
        bottom_10 = player_pcts.sort_values(ascending=True).head(10)

        # Final set of percentile features driving similarity
        selected_pct_cols = pd.Index(top_15.index).union(bottom_10.index).tolist()

        # --------------------------------------------
        # 6a. Show which metrics are being used
        # --------------------------------------------
        def nice_metric_name(pct_col: str) -> str:
            base = pct_col.replace("__pct", "")
            # Try to use the scatter map labels if available, else raw name
            return SCATTER_METRIC_MAP.get(base, base)

        # --------------------------------------------
        # 7. Build feature set:
        #    - the 25 (max) selected percentile metrics
        #    - plus % of touches in each third with reduced weight
        # --------------------------------------------
        touch_cols = [
            "%_touches_in_own_third",
            "%_touches_in_middle_third",
            "%_touches_in_final_third",
        ]
        touch_cols = [c for c in touch_cols if c in df.columns]

        feature_cols = selected_pct_cols + touch_cols

        # Ensure numeric
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop players with absolutely no usable metrics
        df = df.dropna(subset=feature_cols, how="all")

        if df.empty:
            st.warning("After filtering for usable metrics, no players remain for comparison.")
            st.stop()

        # All percentile / percentage metrics → 0–1
        X = df[feature_cols].astype(float)
        X01 = X / 100.0

        # Re-compute player index after any row drops
        mask_player = (df["player_name"] == playername) & (df["team_name"] == team_choice)
        if mask_player.sum() == 0:
            st.warning("Selected player was filtered out due to missing data.")
            st.stop()

        player_index = df.index[mask_player][0]
        player_vec = X01.loc[player_index].to_numpy()
        arr = X01.to_numpy()

        # --------------------------------------------
        # WEIGHTED RMS DIFFERENCE
        #  - 1.0 for top/bottom percentile metrics
        #  - 0.3 for touch-distribution metrics
        # --------------------------------------------
        weights = np.ones(len(feature_cols), dtype=float)
        for i, col in enumerate(feature_cols):
            if col in touch_cols:
                weights[i] = 0.3

        W = np.tile(weights, (len(df), 1))

        diff = arr - player_vec
        valid = ~np.isnan(arr)

        diff[~valid] = np.nan
        W[~valid] = np.nan

        weighted_sq = W * (diff ** 2)
        n_used = np.sum(~np.isnan(weighted_sq), axis=1)

        with np.errstate(invalid="ignore"):
            mean_sq = np.nansum(weighted_sq, axis=1) / np.where(n_used == 0, np.nan, n_used)
            distance = np.sqrt(mean_sq)

        similarity = (1.0 - distance) * 100.0
        similarity = np.clip(similarity, 0.0, 100.0)

        df["similarity"] = similarity
        df = df[~df["similarity"].isna()].copy()

        # --------------------------------------------
        # 8. Sort & present closest comparables
        # --------------------------------------------
        df_sorted = df.sort_values("similarity", ascending=False)
        df_comps = df_sorted[df_sorted.index != player_index].head(10)

        # ------------------------------------------------------
        # HORIZONTAL BAR CHART OF THE TOP 10 SIMILAR PLAYERS
        # ------------------------------------------------------
        import plotly.express as px
        
        # Start from your top 10 comparisons
        df_plot = df_comps.copy()
        df_plot = df_plot.sort_values("similarity", ascending=True)  # worst of the top 10 at the bottom
        
        # Label used on the y-axis
        df_plot["label"] = df_plot["player_name"] + " (" + df_plot["team_name"] + ")"
        
        fig = px.bar(
            df_plot,
            x="similarity",
            y="label",
            orientation="h",
            range_x=[0, 100],
            color="similarity",
            # 👇 force colour range to be 0–100 so "red" only happens for truly low scores
            range_color=[0, 100],
            color_continuous_scale=[
                (0.00, "red"),         # 0
                (0.40, "orange"),      # 40
                (0.70, "yellowgreen"), # 70
                (1.00, "green"),       # 100
            ],
            labels={"similarity": "Similarity Score", "label": ""},
            height=500,
        )
        
        # 👇 Custom hover text:
        # "playername - teamname\nminutes_played"
        fig.update_traces(
            customdata=df_plot[["player_name", "team_name", "minutes_played", "similarity"]],
            hovertemplate=(
                "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                "Minutes played: %{customdata[2]:.0f}<br>"
                "Similarity Score: %{customdata[3]:.2f}"
                "<extra></extra>"
            ),
            marker_line_width=0,
        )
        
        fig.update_layout(
            title=f"Top 10 Most Similar Players to {playername} ({position})",
            xaxis_title="Similarity Score (0–100)",
            yaxis_title="",
            coloraxis_showscale=False,
        
            plot_bgcolor=PitchColor,           # inner plot background
            paper_bgcolor=BackgroundColor,     # overall figure background
            font=dict(color=TextColor),        # axis + title font colours
        
            width=1000,
        
            xaxis=dict(
                range=[0, 105],
                tickmode="linear",
                dtick=20,
                ticks="outside",
                fixedrange=True,
            )
        )
        fig.add_layout_image(
            dict(
                source=wtaimaged,
                xref="paper", yref="paper",
                x=0.8,
                y=0.01,
                sizex=0.20,
                sizey=0.20,
                xanchor="left",
                yanchor="bottom",
                opacity=0.5,
                layer="above",
            )
        )        
        st.plotly_chart(fig, use_container_width=False)  

        st.info(
            "Similarity is based only on this player's strongest & weakest metrics, coupled with "
            "their most common pitch areas for touches. This does not compare ability, only styles."
        )

if __name__ == "__main__":
    main()
