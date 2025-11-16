from __future__ import annotations
from io import BytesIO
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime


# Custom CSS laden
st.markdown(
    """
    <style>
    @import url('assets/style.css');
    </style>
    """,
    unsafe_allow_html=True,
)



tab_overview, tab_inf, tab_time = st.tabs(
    ["Überblick", "Infektionen", "Zeitverlauf"]
)

# Datumsspalten automatisch identifizieren

def convert_date_columns_smart(df: pd.DataFrame, min_fraction: float = 0.7) -> pd.DataFrame:
   
    df = df.copy()

    for col in df.columns:
        # Spalten, die schon datetime sind, einfach lassen
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Zahlen, bool etc. überspringen – meistens keine Datumsstrings
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue

        # Nur auf Text-Spalten anwenden
        s = df[col].astype(str).str.strip()
        s_nonnull = s.replace({"": pd.NA}).dropna()

        if s_nonnull.empty:
            continue

        # Sample, damit es bei großen Tabellen schnell bleibt
        sample = s_nonnull.sample(min(len(s_nonnull), 200), random_state=0)

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )

        frac = parsed.notna().mean()

        if frac >= min_fraction:
            # Diese Spalte sieht stark nach Datum aus -> ganze Spalte konvertieren
            df[col] = pd.to_datetime(
                s,
                errors="coerce",
                dayfirst=True,
                infer_datetime_format=True,
            )

    return df

# Intelligente Datumseingrenzung:

def detect_interval_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Versucht Start- und Stop-Spalte zu finden.
    Sucht nach 'start'/'beginn' und 'stop'/'ende' im Spaltennamen.
    """
    cols = list(df.columns)
    start_col = None
    stop_col = None

    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["start", "beginn"]) and start_col is None:
            start_col = c
        if any(k in cl for k in ["stop", "ende"]) and stop_col is None:
            stop_col = c

    return start_col, stop_col


def count_active_on(df: pd.DataFrame, as_of: pd.Timestamp, start_col: str, stop_col: str) -> int:
    """
    Zählt aktive Fälle am Stichtag:
    - start <= as_of
    - stop ist NaT ODER stop >= as_of
    """
    if start_col not in df.columns or stop_col not in df.columns:
        return 0

    s = pd.to_datetime(df[start_col], errors="coerce")
    e = pd.to_datetime(df[stop_col], errors="coerce")

    mask = (s <= as_of) & (e.isna() | (e >= as_of))
    return int(mask.sum())


# Streamlit App

import plotly.express as px

st.set_page_config(
    page_title="CAS Dashboard – Medizinische Isolationen",
    page_icon="🧠",
    layout="wide",
)

# Einfaches, seriöses App-Header-Panel
st.markdown(
    """
    <div style="
        background-color:#0f172a;
        padding:1.2rem 1.5rem;
        border-radius:0.75rem;
        margin-bottom:1.0rem;
    ">
      <h1 style="color:#f9fafb; margin:0; font-size:1.8rem;">
        🧠 CAS Dashboard – Medizinische Isolationen
      </h1>
      <p style="color:#9ca3af; margin:0.25rem 0 0; font-size:0.95rem;">
        Prototypische Visualisierung von Isolationsfällen nach Klinik, Zentrum, Station und Infektion.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Leichtes globales Styling
st.markdown(
    """
    <style>
    /* Hintergrund etwas aufhellen */
    .stApp {
        background-color: #f3f4f6;
    }

    /* Dataframe-Header etwas absetzen */
    .stDataFrame table {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    /* Sidebar-Header enger */
    section[data-testid="stSidebar"] h2 {
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar: Datei-Upload ----------

st.sidebar.header("Datenquelle")

uploaded_file = st.sidebar.file_uploader(
    "Datei",
    type=["csv", "xlsx", "xls"],
    help="Laden Sie eine Datei hoch, die die medizinischen Isolationsdaten enthält.",
)

df = None

if uploaded_file is not None:
    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix == ".csv":
            raw = uploaded_file.read()
            df_tmp = pd.read_csv(BytesIO(raw), sep=";")
            if df_tmp.shape[1] == 1:
                df_tmp = pd.read_csv(BytesIO(raw), sep=",")
            df = df_tmp

        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(BytesIO(uploaded_file.read()))
        else:
            st.error(
                f"Unbekanntes Dateiformat: {suffix}. Bitte laden Sie eine CSV- oder Excel-Datei hoch."
            )

    except Exception as e:
        st.error(f"Fehler beim Einlesen der Datei: {e}")
else:
    st.info("Noch keine Datei hochgeladen.")
    st.stop()  # nichts weiter anzeigen, bis eine Datei da ist


# ---------- Daten vorbereiten ----------

df = convert_date_columns_smart(df)
start_col, stop_col = detect_interval_cols(df)




# --------------------------------------------------
# RESET BUTTON
# --------------------------------------------------

if st.sidebar.button("Alle Filter löschen"):
    st.session_state["filter_Station"] = "(alle)"
    st.session_state["filter_Klinik"] = "(alle)"
    st.session_state["filter_Zentrum"] = "(alle)"
    st.session_state["reset_filters"] = True
    st.rerun()


reset = st.session_state.get("reset_filters", False)


# --------------------------------------------------
# FILTER: Station → Klinik → Zentrum
# --------------------------------------------------

st.sidebar.header("Filter")

df_for_opts = df.copy()

# -----------------------
# 1) Station
# -----------------------
station_vals = ["(alle)"]
if "Station" in df_for_opts.columns:
    station_vals += sorted(df_for_opts["Station"].dropna().unique().tolist())

station_choice = st.sidebar.selectbox(
    "Station",
    station_vals,
    key="filter_Station"
)

# Filter für nächste Stufe
df_tmp = df_for_opts.copy()
if station_choice != "(alle)" and "Station" in df_tmp.columns:
    df_tmp = df_tmp[df_tmp["Station"] == station_choice]


# -----------------------
# 2) Klinik
# -----------------------
klinik_vals = ["(alle)"]
if "Klinik" in df_tmp.columns:
    klinik_vals += sorted(df_tmp["Klinik"].dropna().unique().tolist())

if reset:
    klinik_index = 0
else:
    # Automatische Vorauswahl, NUR wenn Station gewählt wurde
    klinik_index = 0
    if station_choice != "(alle)":
        unique_kliniken = df_tmp["Klinik"].dropna().unique().tolist()
        if len(unique_kliniken) == 1:
            try:
                klinik_index = klinik_vals.index(unique_kliniken[0])
            except ValueError:
                klinik_index = 0

klinik_choice = st.sidebar.selectbox(
    "Klinik",
    klinik_vals,
    index=klinik_index,
    key="filter_Klinik"
)

# Filter für nächste Stufe
df_tmp2 = df_tmp.copy()
if klinik_choice != "(alle)" and "Klinik" in df_tmp2.columns:
    df_tmp2 = df_tmp2[df_tmp2["Klinik"] == klinik_choice]


# -----------------------
# 3) Zentrum
# -----------------------
zentrum_vals = ["(alle)"]
if "Zentrum" in df_tmp2.columns:
    zentrum_vals += sorted(df_tmp2["Zentrum"].dropna().unique().tolist())

if reset:
    zentrum_index = 0
else:
    zentrum_index = 0
    if klinik_choice != "(alle)":
        unique_zentren = df_tmp2["Zentrum"].dropna().unique().tolist()
        if len(unique_zentren) == 1:
            try:
                zentrum_index = zentrum_vals.index(unique_zentren[0])
            except ValueError:
                zentrum_index = 0

zentrum_choice = st.sidebar.selectbox(
    "Zentrum",
    zentrum_vals,
    index=zentrum_index,
    key="filter_Zentrum"
)


# Reset-Flag ausschalten nach dem Rendern
if reset:
    st.session_state["reset_filters"] = False




# --- 4. Filter anwenden auf df_filtered ---

df_filtered = df.copy()

if station_choice != "(alle)" and "Station" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Station"] == station_choice]

if klinik_choice != "(alle)" and "Klinik" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Klinik"] == klinik_choice]

if zentrum_choice != "(alle)" and "Zentrum" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Zentrum"] == zentrum_choice]

if df_filtered.empty:
    st.warning("Keine Daten für die aktuelle Filterkombination.")
    st.stop()



st.sidebar.subheader("Stichtag aktive Fälle")
stichtag = None
if start_col is not None:
    col_series = df[start_col].dropna()
    if not col_series.empty:
        
        default_date = datetime.today().date()
        stichtag = st.sidebar.date_input(
            "Datum",
            value=default_date,
            help="Stichtag, für den aktive Isolationen berechnet werden.",
        )





# ---------- Initiale aktive Fälle am Stichtag ----------
df_active = pd.DataFrame()

if stichtag is not None and start_col is not None and stop_col is not None:
    as_of = pd.to_datetime(stichtag)

    s = pd.to_datetime(df_filtered[start_col], errors="coerce")
    e = pd.to_datetime(df_filtered[stop_col], errors="coerce")

    # aktive Fälle am Stichtag: start <= as_of und (stop leer oder stop > as_of)
    mask_active = (s <= as_of) & (e.isna() | (e > as_of))
    df_active = df_filtered.loc[mask_active].copy()

# ---------- Daten-Vorschau ----------

st.subheader("Vorschau der aktuellen Isolationen")
if stichtag is not None and start_col is not None and stop_col is not None:
    as_of = pd.to_datetime(stichtag)

    s = pd.to_datetime(df_filtered[start_col], errors="coerce")
    e = pd.to_datetime(df_filtered[stop_col], errors="coerce")

    # aktive Fälle am Stichtag: start <= as_of und (stop leer oder stop > as_of)
    mask_active = (s <= as_of) & (e.isna() | (e > as_of))
    df_active = df_filtered.loc[mask_active].copy()

    # gewünschte Spalten in der richtigen Reihenfolge
    cols_order = ["Klinik", "Zentrum", "Station", "fallnummer", "Infektion", start_col]
    cols_present = [c for c in cols_order if c in df_active.columns]
    df_active = df_active[cols_present]

    # sortieren nach Beginndatum abwärts (neuestes zuerst)
    if start_col in df_active.columns:
        df_active = df_active.sort_values(by=start_col, ascending=False)

    st.subheader("Aktuell aktive Fälle am Stichtag")
    st.caption(f"{len(df_active):,} aktive Fälle am {as_of.date():%d.%m.%Y}")

    if not df_active.empty:
        st.dataframe(df_active, use_container_width=True)
    else:
        st.info("Keine aktiven Fälle für den gewählten Stichtag.")
else:
    st.warning("Start- und Stopspalten oder Stichtag fehlen – keine Vorschau der aktiven Fälle möglich.")


# ---------- KPIs ----------
with tab_overview:

    st.subheader("Kennzahlen")

    col1, col2, col3, col4 = st.columns(4)

    # 1) Alle gefilterten Fälle (rein informational)
    with col1:
        st.metric("Fälle", f"{len(df_filtered):,}")
    # 2) Aktive Fälle am Stichtag (auf Basis df_active)
    with col2:
        st.metric(
            f"Aktive Fälle am {pd.to_datetime(stichtag).date():%d.%m.%Y}",
            f"{len(df_active):,}",
        )


    # 3) Unterschiedliche Infektionen – aber nur unter den AKTIVEN

    with col3:
        if "Infektion" in df_active.columns:
            unique_inf_active = df_active["Infektion"].nunique()
            st.metric("Anzahl unterschiedlicher Infektionen", unique_inf_active)


    # 4) Offene Fälle (Stopdatum leer) optional
    with col4:
        if stop_col is not None:
            offene = df_filtered[df_filtered[stop_col].isna()].shape[0]
            st.metric("Offene Fälle (Stopdatum leer)", f"{offene:,}")

with tab_inf:
    # ---------- Fälle pro Infektion ----------

    st.subheader("Fälle pro Infektion")

    if "Infektion" in df_filtered.columns:
        mode_inf = st.radio(
            "Basis für Auswertung",
            ["Insgesamt (alle gefilterten Fälle)", "Aktiv am Stichtag"],
            horizontal=True,
        )

        # Basis-DataFrame wählen
        if mode_inf == "Insgesamt (alle gefilterten Fälle)":
            base = df_filtered
        else:
            base = df_active
            if base.empty:
                st.info("Keine aktiven Fälle am gewählten Stichtag – kein Diagramm möglich.")
                base = None

        if base is not None and not base.empty:
            infection_counts = (
                base["Infektion"]
                .value_counts()
                .reset_index()
            )
            infection_counts.columns = ["Infektion", "Anzahl"]

            fig = px.bar(
        infection_counts,
        x="Infektion",
        y="Anzahl",
        title=(
            "Anzahl Fälle pro Infektion (gesamt)"
            if mode_inf == "Insgesamt (alle gefilterten Fälle)"
            else f"Anzahl Fälle pro Infektion (aktiv am {pd.to_datetime(stichtag).date():%d.%m.%Y})"
        ),
        text="Anzahl",
        template="plotly_white",
        color_discrete_sequence=COLOR_SEQ,
    )

            
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Keine Spalte 'Infektion' vorhanden.")




# --- Zeitreihe: aktive Fälle (monatlich) gesamt / nach Infektion ---
with tab_time:
    st.subheader("Aktive Fälle über Zeit")

    if start_col is not None and stop_col is not None and not df_filtered.empty:

        # Umschalter: Gesamt vs. nach Infektion
        ts_mode = st.radio(
            "Darstellung",
            ["Gesamt (eine Linie)", "Nach Infektion (mehrere Linien)"],
            horizontal=True,
        )

        s = pd.to_datetime(df_filtered[start_col], errors="coerce")
        e = pd.to_datetime(df_filtered[stop_col], errors="coerce")

        if s.notna().any():
            t_min = s.min().to_period("M").to_timestamp()
            t_max = max(s.max(), e.max(skipna=True)).to_period("M").to_timestamp()
            timeline = pd.date_range(t_min, t_max, freq="MS")

            # --- Modus 1: Gesamt (eine Linie) ---
            if ts_mode == "Gesamt (eine Linie)":
                vals = []
                for t in timeline:
                    active = count_active_on(df_filtered, t, start_col, stop_col)
                    vals.append({"Monat": t, "Aktive Fälle": active})

                ts = pd.DataFrame(vals)

                if ts.empty:
                    st.info("Keine Daten für die Zeitreihe.")
                else:
                    fig_ts = px.line(
                        ts,
                        x="Monat",
                        y="Aktive Fälle",
                        markers=True,
                        title="Aktive Fälle (monatlich, gesamt)",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

            # --- Modus 2: nach Infektion (mehrere Linien) ---
            else:
                if "Infektion" not in df_filtered.columns:
                    st.info("Keine Spalte 'Infektion' vorhanden – keine Aufschlüsselung möglich.")
                else:
                    all_infections = sorted(df_filtered["Infektion"].dropna().unique().tolist())

                    selected_infections = st.multiselect(
                        "Infektionen anzeigen",
                        options=all_infections,
                        default=all_infections,
                        help="Infektionen für die Zeitreihe auswählen.",
                    )

                    infections_for_ts = selected_infections or all_infections

                    if not infections_for_ts:
                        st.info("Keine Infektionen für die Zeitreihe ausgewählt.")
                    else:
                        records = []
                        for inf in infections_for_ts:
                            df_inf = df_filtered[df_filtered["Infektion"] == inf]
                            if df_inf.empty:
                                continue
                            for t in timeline:
                                active = count_active_on(df_inf, t, start_col, stop_col)
                                records.append(
                                    {"Monat": t, "Aktive Fälle": active, "Infektion": inf}
                                )

                        ts = pd.DataFrame(records)

                        if ts.empty:
                            st.info("Keine Daten für die gewählten Infektionen im Zeitraum.")
                        else:
                            fig_ts = px.line(
                            ts,
                            x="Monat",
                            y="Aktive Fälle",
                            color="Infektion",
                            markers=True,
                            title="Aktive Fälle (monatlich) nach Infektion",
                            template="plotly_white",
                            color_discrete_sequence=COLOR_SEQ,
                        )

                            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Keine gültigen Start-Daten für die Zeitreihe 'Aktive Fälle'.")
