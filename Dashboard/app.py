from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Page config (MUSS vor Output sein)
# -----------------------------
st.set_page_config(
    page_title="CAS Dashboard – Medizinische Isolationen",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Pfade / Assets
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
DERIVED_DIR = ASSETS_DIR / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

FLOORPLAN_ORIG = ASSETS_DIR / "grundriss.png"
FLOORPLAN_CLEAN = DERIVED_DIR / "grundriss_clean_light.png"
ROOM_POINTS_PATH = ASSETS_DIR / "room_points.csv"


# -----------------------------
# Custom CSS - Modernes Design
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #3b82f6;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .status-container {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .status-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .status-green { 
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #22c55e;
    }
    .status-yellow { 
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 6px solid #f59e0b;
    }
    .status-red { 
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
    }
    .status-title { 
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .status-msg { 
        font-size: 1rem;
        margin-left: 2rem;
        line-height: 1.6;
        opacity: 0.9;
    }
    .warning-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #ef4444;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    section[data-testid="stSidebar"] h2 {
        color: #f1f5f9;
        margin-top: 0.5rem;
    }
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    ">
      <h1 style="color:#f9fafb; margin:0; font-size:2rem; font-weight:700;">
        🧠 CAS Dashboard – Medizinische Isolationen
      </h1>
      <p style="color:#cbd5e1; margin:0.5rem 0 0; font-size:1rem; line-height:1.5;">
        Interaktive Visualisierung von Isolationsfällen • Heatmap-Analyse • Echtzeit-Übersicht
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

COLOR_SEQ = [
    "#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981",
    "#06b6d4", "#f97316", "#6366f1"
]


# -----------------------------
# Helper: Data Processing
# -----------------------------
def convert_date_columns_smart(df: pd.DataFrame, min_fraction: float = 0.7) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue

        s = df[col].astype(str).str.strip()
        s_nonnull = s.replace({"": pd.NA}).dropna()
        if s_nonnull.empty:
            continue

        sample = s_nonnull.sample(min(len(s_nonnull), 200), random_state=0)
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        if parsed.notna().mean() >= min_fraction:
            df[col] = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return df


def detect_interval_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    start_col = None
    stop_col = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["start", "beginn"]) and start_col is None:
            start_col = c
        if any(k in cl for k in ["stop", "ende"]) and stop_col is None:
            stop_col = c
    return start_col, stop_col


@st.cache_data(show_spinner=True)
def load_and_prep_data(file_content: bytes, suffix: str) -> tuple[pd.DataFrame, str | None, str | None]:
    try:
        if suffix == ".csv":
            df_tmp = pd.read_csv(BytesIO(file_content), sep=";")
            if df_tmp.shape[1] == 1:
                df_tmp = pd.read_csv(BytesIO(file_content), sep=",")
            df = df_tmp
        else:
            df = pd.read_excel(BytesIO(file_content))
    except Exception as e:
        st.error(f"Fehler beim Parsen: {e}")
        return pd.DataFrame(), None, None

    df = convert_date_columns_smart(df)

    if "Raum_ID" in df.columns:
        df["Raum_ID"] = df["Raum_ID"].astype(str).str.strip()

    s_col, e_col = detect_interval_cols(df)
    return df, s_col, e_col


def count_active_on(df: pd.DataFrame, as_of: pd.Timestamp, start_col: str, stop_col: str) -> int:
    s = df[start_col]
    e = df[stop_col]
    mask = (s <= as_of) & (e.isna() | (e >= as_of))
    return int(mask.sum())


def check_warnings(df: pd.DataFrame, stichtag: pd.Timestamp, start_col: str, stop_col: str) -> list[str]:
    warnings: list[str] = []

    dates = pd.date_range(end=stichtag, periods=14, freq="D")
    counts = np.array([count_active_on(df, d, start_col, stop_col) for d in dates], dtype=float)

    current = int(counts[-1])
    prev_week_avg = float(counts[-8:-1].mean()) if len(counts) >= 8 else float(np.mean(counts[:-1])) if len(counts) > 1 else 0.0

    if prev_week_avg > 0:
        pct_change = (current - prev_week_avg) / prev_week_avg
        if pct_change >= 0.5 and current >= 5:
            warnings.append(
                f"⚠️ **Sprunghafter Anstieg:** +{pct_change:.0%} gegenüber Vorwochenschnitt ({current} vs ø{prev_week_avg:.1f})."
            )
    elif current >= 5:
        warnings.append("⚠️ **Neuer Ausbruch:** Plötzlich ≥5 Fälle (Vorwoche nahe 0).")

    if len(counts) >= 7:
        recent_trend = counts[-1] - counts[-3]
        older_trend = counts[-4] - counts[-7]
        if recent_trend > 2 * max(1.0, older_trend) and current > 5:
            warnings.append("📈 **Beschleunigtes Wachstum:** Zuwachsrate steigt deutlich (Verdacht auf exponentielle Phase).")

    s = df[start_col]
    e = df[stop_col]
    mask_active = (s <= stichtag) & (e.isna() | (e > stichtag))
    active_now = df.loc[mask_active]

    if not active_now.empty and "Station" in active_now.columns:
        station_counts = active_now["Station"].astype(str).value_counts()
        total_active = len(active_now)
        top_station = station_counts.index[0]
        top_count = int(station_counts.iloc[0])
        share = top_count / max(1, total_active)
        if share > 0.50 and total_active >= 5:
            warnings.append(f"🏘️ **Cluster-Verdacht:** Station '{top_station}' hält {share:.0%} aller aktuellen Fälle ({top_count}).")

    return warnings


def analyze_risk_status(df: pd.DataFrame, stichtag: pd.Timestamp, start_col: str, stop_col: str) -> dict:
    dates = pd.date_range(end=stichtag, periods=14, freq="D")
    counts = np.array([count_active_on(df, d, start_col, stop_col) for d in dates], dtype=float)

    current = int(counts[-1])
    if current == 0:
        return {"level": "green", "msgs": ["Keine aktiven Isolationen am gewählten Stichtag."]}

    prev_week_avg = float(counts[-8:-1].mean()) if len(counts) >= 8 else float(np.mean(counts[:-1])) if len(counts) > 1 else 0.0

    msgs: list[str] = []
    level = "green"

    if len(counts) >= 7:
        recent_trend = counts[-1] - counts[-3]
        older_trend = counts[-4] - counts[-7]
        if recent_trend > 2 * max(1.0, older_trend) and current >= 5:
            level = "red"
            msgs.append("🚨 **Kritisch:** Die Fallzahlen beschleunigen deutlich (Zuwachsrate steigt).")

    if level != "red":
        if prev_week_avg > 0 and (current / prev_week_avg) > 1.2 and current >= 3:
            level = "yellow"
            msgs.append(f"⚠️ **Beobachtung:** Anstieg um {((current/prev_week_avg)-1):.0%} gegenüber Vorwochenschnitt.")

        if "Station" in df.columns:
            s_col, e_col = df[start_col], df[stop_col]
            mask_active = (s_col <= stichtag) & (e_col.isna() | (e_col > stichtag))
            active_now = df.loc[mask_active]
            if not active_now.empty:
                vc = active_now["Station"].astype(str).value_counts()
                top_station = vc.index[0]
                top_count = int(vc.iloc[0])
                if top_count >= 4 and (top_count / len(active_now)) > 0.6:
                    level = "yellow"
                    msgs.append(f"📍 **Cluster-Verdacht:** {top_count} Fälle konzentrieren sich auf Station '{top_station}'.")

    if not msgs:
        msgs.append("Stabil: Keine auffälligen Trends in den letzten 14 Tagen identifiziert.")
    return {"level": level, "msgs": msgs}


# -----------------------------
# Floorplan cleaning
# -----------------------------
@st.cache_data
def ensure_clean_floorplan() -> str:
    if FLOORPLAN_CLEAN.exists():
        return str(FLOORPLAN_CLEAN)
    if not FLOORPLAN_ORIG.exists():
        return str(FLOORPLAN_ORIG)

    img = Image.open(FLOORPLAN_ORIG).convert("L")
    img = ImageEnhance.Brightness(img).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(0.60)
    img.save(FLOORPLAN_CLEAN)
    return str(FLOORPLAN_CLEAN)


# -----------------------------
# Diffuse heat helpers
# -----------------------------
def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = int(3 * sigma)
    radius = max(1, radius)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k


def gaussian_blur_2d(z: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z
    k = gaussian_kernel1d(sigma)
    z1 = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=z)
    z2 = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=z1)
    return z2


def build_diffuse_heat(
    m_df: pd.DataFrame, W: int, H: int, grid: int = 220, sigma: float = 6.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = int(grid)
    gy = max(40, int(grid * (H / W)))

    z = np.zeros((gy, gx), dtype=float)
    xs = m_df["x"].to_numpy(dtype=float)
    ys = m_df["y"].to_numpy(dtype=float)
    ws = m_df["summe"].to_numpy(dtype=float)

    ix = np.clip((xs / W * (gx - 1)).astype(int), 0, gx - 1)
    iy = np.clip((ys / H * (gy - 1)).astype(int), 0, gy - 1)

    np.add.at(z, (iy, ix), ws)

    if sigma > 0:
        z = gaussian_blur_2d(z, sigma=float(sigma))

    x_axis = np.linspace(0, W, gx)
    y_axis = np.linspace(0, H, gy)
    return x_axis, y_axis, z


def auto_params(m_df: pd.DataFrame, W: int, H: int) -> tuple[int, float, float]:
    pos = int((m_df["summe"] > 0).sum())
    grid = int(np.clip(W / 5, 160, 320))
    sigma = float(np.clip(3.0 + 1.2 * np.log1p(pos), 3.5, 8.0))
    opacity = 0.50
    return grid, sigma, opacity


# -----------------------------
# Sidebar: Datei Upload
# -----------------------------
st.sidebar.header("📁 Datenquelle")

uploaded_file = st.sidebar.file_uploader(
    "Datei hochladen",
    type=["csv", "xlsx", "xls"],
    help="Laden Sie eine CSV- oder Excel-Datei mit medizinischen Isolationsdaten hoch.",
)

if uploaded_file is None:
    st.info("👋 **Willkommen!** Bitte laden Sie eine CSV- oder Excel-Datei hoch, um zu beginnen.")
    st.stop()

file_bytes = uploaded_file.getvalue()
suffix = Path(uploaded_file.name).suffix.lower()
df, start_col, stop_col = load_and_prep_data(file_bytes, suffix)

if df.empty:
    st.error("❌ Datei konnte nicht gelesen werden oder ist leer.")
    st.stop()


# -----------------------------
# Reset Button & Filter State
# -----------------------------
if st.sidebar.button("🔄 Alle Filter zurücksetzen", use_container_width=True):
    for key in ["filter_Station", "filter_Klinik", "filter_Zentrum"]:
        if key in st.session_state:
            st.session_state[key] = "(alle)"
    st.session_state["reset_filters"] = True
    st.rerun()

reset = st.session_state.get("reset_filters", False)


# -----------------------------
# Filter: Station -> Klinik -> Zentrum
# -----------------------------
st.sidebar.header("🔍 Filter")

df_for_opts = df.copy()

station_vals = ["(alle)"]
if "Station" in df_for_opts.columns:
    station_vals += sorted(df_for_opts["Station"].dropna().astype(str).unique().tolist())
station_choice = st.sidebar.selectbox("Station", station_vals, key="filter_Station")

df_tmp = df_for_opts.copy()
if station_choice != "(alle)" and "Station" in df_tmp.columns:
    df_tmp = df_tmp[df_tmp["Station"].astype(str) == str(station_choice)]

klinik_vals = ["(alle)"]
if "Klinik" in df_tmp.columns:
    klinik_vals += sorted(df_tmp["Klinik"].dropna().astype(str).unique().tolist())

klinik_index = 0
if not reset and station_choice != "(alle)" and len(klinik_vals) == 2:
    klinik_index = 1

klinik_choice = st.sidebar.selectbox("Klinik", klinik_vals, index=klinik_index, key="filter_Klinik")

df_tmp2 = df_tmp.copy()
if klinik_choice != "(alle)" and "Klinik" in df_tmp2.columns:
    df_tmp2 = df_tmp2[df_tmp2["Klinik"].astype(str) == str(klinik_choice)]

zentrum_vals = ["(alle)"]
if "Zentrum" in df_tmp2.columns:
    zentrum_vals += sorted(df_tmp2["Zentrum"].dropna().astype(str).unique().tolist())

zentrum_index = 0
if not reset and klinik_choice != "(alle)" and len(zentrum_vals) == 2:
    zentrum_index = 1

zentrum_choice = st.sidebar.selectbox("Zentrum", zentrum_vals, index=zentrum_index, key="filter_Zentrum")

if reset:
    st.session_state["reset_filters"] = False

df_filtered = df.copy()
if station_choice != "(alle)" and "Station" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Station"].astype(str) == str(station_choice)]
if klinik_choice != "(alle)" and "Klinik" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Klinik"].astype(str) == str(klinik_choice)]
if zentrum_choice != "(alle)" and "Zentrum" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Zentrum"].astype(str) == str(zentrum_choice)]

if df_filtered.empty:
    st.warning("⚠️ Keine Daten für die aktuelle Filterkombination.")
    st.stop()


# -----------------------------
# Stichtag
# -----------------------------
st.sidebar.header("📅 Stichtag")
default_date = datetime.today().date()
if start_col and start_col in df.columns and not df[start_col].dropna().empty:
    max_date = df[start_col].max()
    if pd.notna(max_date):
        default_date = max_date.date()

stichtag_val = st.sidebar.date_input(
    "Datum für aktive Fälle",
    value=default_date,
    help="Stichtag, für den aktive Isolationen berechnet werden.",
)
stichtag = pd.to_datetime(stichtag_val)

df_active = pd.DataFrame()
if start_col is not None and stop_col is not None:
    s = df_filtered[start_col]
    e = df_filtered[stop_col]
    mask_active = (s <= stichtag) & (e.isna() | (e > stichtag))
    df_active = df_filtered.loc[mask_active].copy()


# -----------------------------
# Status-Zentrale
# -----------------------------
if start_col and stop_col:
    status = analyze_risk_status(df_filtered, stichtag, start_col, stop_col)
    icon = {"green": "✅", "yellow": "⚠️", "red": "🚨"}[status["level"]]
    title = {"green": "Normalbetrieb", "yellow": "Beobachtung erforderlich", "red": "KRITISCHE DYNAMIK"}[status["level"]]

    st.markdown(
        f"""
        <div class="status-container status-{status['level']}">
            <div class="status-title">{icon} {title}</div>
            <div style="font-size:0.9rem; opacity:0.8; margin-bottom:0.5rem;">Stand: {stichtag_val:%d.%m.%Y}</div>
            <div class="status-msg">{"<br>".join(status["msgs"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_warnings = check_warnings(df_filtered, stichtag, start_col, stop_col)
    if detail_warnings:
        html = '<div class="warning-box"><h4 style="margin:0 0 0.75rem 0; color:#991b1b;">⚠️ Detaillierte Risiko-Analyse</h4>'
        for w in detail_warnings:
            html += f'<div style="margin-bottom:0.5rem; padding-left:0.5rem; border-left:3px solid #fca5a5;">{w}</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
else:
    st.warning("⚠️ Zeitspalten (Start/Ende) konnten nicht automatisch erkannt werden.")


# -----------------------------
# Tabs (Default: Überblick)
# -----------------------------
TAB_LABELS = ["📊 Überblick", "🗺️ Heatmap & Infektionen", "📈 Zeitverlauf", "🔎 Detailansicht"]
DEFAULT_TAB_INDEX = 0

if "main_tab_choice" not in st.session_state:
    st.session_state["main_tab_choice"] = TAB_LABELS[DEFAULT_TAB_INDEX]

st.radio(
    "Navigation",
    TAB_LABELS,
    index=TAB_LABELS.index(st.session_state["main_tab_choice"])
    if st.session_state["main_tab_choice"] in TAB_LABELS
    else DEFAULT_TAB_INDEX,
    horizontal=True,
    label_visibility="collapsed",
    key="main_tab_choice",
)

main_tab = st.session_state["main_tab_choice"]


# -----------------------------
# Tab: Überblick
# -----------------------------
if main_tab == "📊 Überblick":
    st.subheader("📊 Wichtige Kennzahlen")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📁 Fälle (Total im Filter)",
            value=f"{len(df_filtered):,}",
            help="Gesamtanzahl der Fälle nach angewendeten Filtern",
        )

    with col2:
        active_count = len(df_active) if (start_col and stop_col) else 0
        st.metric(
            label=f"🔴 Aktive Fälle ({stichtag_val.day}.{stichtag_val.month}.)",
            value=f"{active_count:,}",
            help="Anzahl der am Stichtag aktiven Isolationen",
        )

    with col3:
        if "Infektion" in df_filtered.columns:
            basis_n = df_active if (start_col and stop_col and not df_active.empty) else df_filtered
            infection_types = int(basis_n["Infektion"].nunique())
            st.metric(
                label="🦠 Infektionsarten",
                value=infection_types,
                help="Anzahl unterschiedlicher Infektionstypen",
            )
        else:
            st.metric(label="🦠 Infektionsarten", value="—")

    with col4:
        if stop_col is not None and stop_col in df_filtered.columns:
            offene = int(df_filtered[df_filtered[stop_col].isna()].shape[0])
            st.metric(
                label="⏳ Offene Fälle",
                value=f"{offene:,}",
                help="Fälle ohne Enddatum",
            )
        else:
            st.metric(label="⏳ Offene Fälle", value="—")


# -----------------------------
# Tab: Heatmap & Infektionen
# -----------------------------
elif main_tab == "🗺️ Heatmap & Infektionen":
    st.subheader("🦠 Verteilung nach Infektionstyp")

    if "Infektion" in df_filtered.columns:
        mode_inf = st.radio(
            "Datenbasis",
            ["📊 Alle gefilterten Fälle", "🔴 Nur aktive Fälle (Stichtag)"],
            index=1,
            horizontal=True,
            key="tab_inf_bar_basis",
        )

        base = df_filtered if mode_inf.startswith("📊") else (df_active if (start_col and stop_col) else pd.DataFrame())

        if base is not None and not base.empty:
            infection_counts = base["Infektion"].astype(str).value_counts().reset_index()
            infection_counts.columns = ["Infektion", "Anzahl"]
            infection_counts = infection_counts.sort_values("Anzahl", ascending=True)

            fig_bar = px.bar(
                infection_counts,
                x="Anzahl",
                y="Infektion",
                orientation="h",
                title="",
                text="Anzahl",
                template="plotly_white",
                color="Anzahl",
                color_continuous_scale="Blues",
            )
            fig_bar.update_traces(textposition="outside", textfont_size=12)
            fig_bar.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=max(300, len(infection_counts) * 40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ℹ️ Keine Daten verfügbar für die gewählte Basis.")
    else:
        st.info("ℹ️ Keine Spalte 'Infektion' vorhanden.")

    st.divider()
    st.subheader("🗺️ Interaktive Grundriss-Heatmap")

    if not ROOM_POINTS_PATH.exists():
        st.warning(f"⚠️ Datei 'room_points.csv' nicht gefunden in {ASSETS_DIR}. Heatmap deaktiviert.")
        pts = pd.DataFrame()
    else:
        pts = pd.read_csv(ROOM_POINTS_PATH)
        if "Raum_ID" in pts.columns:
            pts["Raum_ID"] = pts["Raum_ID"].astype(str).str.strip()

    if not pts.empty:
        col_conf1, col_conf2 = st.columns([1, 2])

        with col_conf1:
            heat_basis = st.radio(
                "Datenbasis",
                ["📊 Alle Fälle", "🔴 Aktive Fälle"],
                key="tab_inf_heat_basis",
            )

            render_mode = st.radio(
                "Darstellung",
                ["🌡️ Diffuse Heatmap", "📍 Punktmarkierungen"],
                key="tab_inf_heat_render_mode",
            )

        with col_conf2:
            if "Infektion" in df_filtered.columns:
                inf_options = ["(alle Infektionen)"] + sorted(df_filtered["Infektion"].dropna().astype(str).unique().tolist())
            else:
                inf_options = ["(alle Infektionen)"]
            inf_choice = st.selectbox(
                "🦠 Infektion filtern:",
                inf_options,
                index=0,
                key="tab_inf_heat_inf",
            )

        auto = st.toggle("✨ Automatische Optimierung", value=True, key="tab_inf_heat_auto")

        base_h = df_filtered if heat_basis.startswith("📊") else (df_active if (start_col and stop_col) else pd.DataFrame())

        valid_heat = True
        if base_h is None or base_h.empty:
            st.info("ℹ️ Keine Datenbasis für Heatmap verfügbar.")
            valid_heat = False
        elif "Raum_ID" not in base_h.columns:
            st.error("❌ Datensatz enthält keine 'Raum_ID'.")
            valid_heat = False

        if valid_heat:
            base_h = base_h.copy()
            if inf_choice != "(alle Infektionen)" and "Infektion" in base_h.columns:
                base_h = base_h[base_h["Infektion"].astype(str) == str(inf_choice)]

            if base_h.empty:
                st.warning("⚠️ Keine Fälle nach Filterung vorhanden.")
            else:
                agg = base_h.groupby("Raum_ID").size().reset_index(name="summe")
                m = pts.merge(agg, on="Raum_ID", how="left").fillna({"summe": 0})

                img_path = ensure_clean_floorplan()
                if not Path(img_path).exists():
                    st.error("❌ Grundriss-Bild konnte nicht geladen werden.")
                else:
                    img = Image.open(img_path)
                    W, H = img.size

                    if auto:
                        grid, sigma, opacity = auto_params(m, W, H)
                    else:
                        col_sl1, col_sl2, col_sl3 = st.columns(3)
                        grid = col_sl1.slider("Auflösung", 100, 400, 220, 20)
                        sigma = col_sl2.slider("Diffusion", 0.0, 15.0, 6.0, 0.5)
                        opacity = col_sl3.slider("Deckkraft", 0.1, 1.0, 0.50, 0.05)

                    fig = go.Figure()

                    fig.add_layout_image(
                        dict(
                            source=img,
                            x=0,
                            y=0,
                            xref="x",
                            yref="y",
                            sizex=W,
                            sizey=H,
                            sizing="stretch",
                            layer="below",
                        )
                    )

                    if render_mode.startswith("🌡️"):
                        x_axis, y_axis, z = build_diffuse_heat(m, W, H, grid=grid, sigma=sigma)
                        z_disp = np.log1p(z)

                        nz = z_disp[z_disp > 0]
                        zmax = float(np.percentile(nz, 99)) if nz.size else 1.0

                        fig.add_trace(
                            go.Heatmap(
                                x=x_axis,
                                y=y_axis,
                                z=z_disp,
                                colorscale="RdYlGn_r",
                                zmin=0,
                                zmax=zmax,
                                opacity=float(opacity),
                                colorbar=dict(title="Dichte (log)"),
                                hovertemplate="x: %{x:.0f}px<br>y: %{y:.0f}px<br>Dichte: %{z:.2f}<extra></extra>",
                                showscale=True,
                            )
                        )
                    else:
                        m_pos = m[m["summe"] > 0].copy()
                        if not m_pos.empty:
                            cmax = int(max(1, m_pos["summe"].max()))
                            fig.add_trace(
                                go.Scatter(
                                    x=m_pos["x"],
                                    y=m_pos["y"],
                                    mode="markers+text",
                                    text=m_pos["summe"].astype(int).astype(str),
                                    textposition="middle center",
                                    textfont=dict(color="white", size=11, family="Arial Black"),
                                    customdata=m_pos["Raum_ID"],
                                    hovertemplate="<b>Raum: %{customdata}</b><br>Fälle: %{text}<extra></extra>",
                                    marker=dict(
                                        size=32,
                                        opacity=0.85,
                                        color=m_pos["summe"],
                                        colorscale="RdYlGn_r",
                                        cmin=0,
                                        cmax=cmax,
                                        showscale=True,
                                        colorbar=dict(title="Anzahl Fälle"),
                                        line=dict(width=2, color="white"),
                                    ),
                                )
                            )

                    fig.update_xaxes(visible=False, range=[0, W])
                    fig.update_yaxes(visible=False, range=[0, H], autorange="reversed", scaleanchor="x", scaleratio=1)

                    fig.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=850,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        dragmode="pan",
                    )

                    st.plotly_chart(fig, use_container_width=True, key="heatmap_plot")


# -----------------------------
# Tab: Zeitverlauf
# -----------------------------
elif main_tab == "📈 Zeitverlauf":
    st.subheader("📈 Entwicklung der aktiven Fälle über Zeit")

    if start_col and stop_col and not df_filtered.empty:
        col_ts1, col_ts2 = st.columns([1, 3])

        with col_ts1:
            ts_mode = st.radio(
                "Ansicht",
                ["📊 Gesamtübersicht", "🦠 Nach Infektion"],
                key="tab_time_mode",
            )

        s = df_filtered[start_col]
        e = df_filtered[stop_col]

        if s.notna().any():
            # FIX: Timeline mindestens bis Stichtag (auch wenn Daten früher enden)
            t_min = s.min().replace(day=1)
            max_s = s.max()
            max_e = e.max() if e.notna().any() else max_s
            # Wichtig: bis zum (gewählten) Stichtag weiterzeichnen
            t_max = max(max_s, max_e, pd.Timestamp(stichtag)).replace(day=1)

            # Sicherheitsbremse
            if t_max.year > datetime.now().year + 2:
                t_max = pd.Timestamp.now().replace(day=1)

            timeline = pd.date_range(t_min, t_max, freq="MS")

            if ts_mode == "📊 Gesamtübersicht":
                vals = [{"Monat": t, "Aktive Fälle": count_active_on(df_filtered, t, start_col, stop_col)} for t in timeline]
                ts_df = pd.DataFrame(vals)

                fig_ts = px.area(
                    ts_df,
                    x="Monat",
                    y="Aktive Fälle",
                    title="Gesamtverlauf aktiver Fälle (monatlich) – bis Stichtag",
                    template="plotly_white",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_ts.update_traces(line_shape="spline")
                fig_ts.update_layout(hovermode="x unified", height=450)
                st.plotly_chart(fig_ts, use_container_width=True)

                st.divider()
                st.markdown("#### 📅 Detailansicht: Letzte 30 Tage")

                hist_dates = pd.date_range(end=stichtag, periods=30, freq="D")
                hist_df = pd.DataFrame(
                    [{"Datum": d, "Aktive Fälle": count_active_on(df_filtered, d, start_col, stop_col)} for d in hist_dates]
                )

                fig_trend = px.line(
                    hist_df,
                    x="Datum",
                    y="Aktive Fälle",
                    markers=True,
                    template="plotly_white",
                    color_discrete_sequence=["#8b5cf6"],
                )
                fig_trend.update_traces(fill="tozeroy", line_shape="spline")
                fig_trend.update_layout(hovermode="x unified", height=400)
                st.plotly_chart(fig_trend, use_container_width=True)

            else:
                if "Infektion" in df_filtered.columns:
                    all_infections = sorted(df_filtered["Infektion"].dropna().astype(str).unique().tolist())

                    with col_ts2:
                        selected_infections = st.multiselect(
                            "Infektionen auswählen",
                            all_infections,
                            default=all_infections[:5] if len(all_infections) >= 5 else all_infections,
                        )

                    if not selected_infections:
                        st.info("ℹ️ Bitte mindestens eine Infektion auswählen.")
                    else:
                        records = []
                        for inf in selected_infections:
                            sub_df = df_filtered[df_filtered["Infektion"].astype(str) == str(inf)]
                            if sub_df.empty:
                                continue
                            for t in timeline:
                                records.append(
                                    {"Monat": t, "Aktive Fälle": count_active_on(sub_df, t, start_col, stop_col), "Infektion": inf}
                                )

                        ts_df = pd.DataFrame(records)
                        if not ts_df.empty:
                            fig_ts = px.line(
                                ts_df,
                                x="Monat",
                                y="Aktive Fälle",
                                color="Infektion",
                                markers=True,
                                template="plotly_white",
                                color_discrete_sequence=COLOR_SEQ,
                                title="Aktive Fälle nach Infektion (monatlich) – bis Stichtag",
                            )
                            fig_ts.update_traces(line_shape="spline")
                            fig_ts.update_layout(
                                hovermode="x unified",
                                height=500,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                ),
                            )
                            st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("ℹ️ Keine Infektions-Spalte vorhanden.")
        else:
            st.warning("⚠️ Keine gültigen Datumsangaben gefunden.")
    else:
        st.info("ℹ️ Warten auf Datei-Upload oder Erkennung der Zeit-Spalten.")


# -----------------------------
# Tab: Detailansicht (NEU)
# -----------------------------
elif main_tab == "🔎 Detailansicht":
    st.subheader("Aktuelle Isolationen im Detail")

    if not (start_col and stop_col):
        st.warning("⚠️ Zeitspalten (Start/Ende) konnten nicht automatisch erkannt werden.")
    else:
        if df_active.empty:
            st.info(f"ℹ️ Keine aktiven Fälle am {stichtag_val:%d.%m.%Y} für die gewählte Filterung.")
        else:
            cols_order = ["Klinik", "Zentrum", "Station", "Raum_ID", "fallnummer", "Infektion", start_col, stop_col]
            cols_present = [c for c in cols_order if c in df_active.columns]
            df_active_preview = df_active[cols_present].copy()

            if start_col in df_active_preview.columns:
                df_active_preview = df_active_preview.sort_values(by=start_col, ascending=False)
                df_active_preview[start_col] = df_active_preview[start_col].dt.strftime("%d.%m.%Y")
            if stop_col in df_active_preview.columns:
                df_active_preview[stop_col] = df_active_preview[stop_col].dt.strftime("%d.%m.%Y")

            st.caption(f"**{len(df_active):,} aktive Fälle** am {stichtag_val:%d.%m.%Y} (erste 20 sichtbar, Rest scrollbar)")
            # FIX: mehr sichtbar, Rest scrollbar (nicht truncaten)
            st.dataframe(df_active_preview, use_container_width=True, height=560)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem;">
        CAS Dashboard v2.0 • Visualisierung & Analyse von Isolationsfällen
    </div>
    """,
    unsafe_allow_html=True,
)
