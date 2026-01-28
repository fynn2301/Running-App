import datetime
import os
import time
from datetime import date

import pandas as pd

# Added Plotly imports
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from garminconnect import Garmin
from plotly.subplots import make_subplots

# Removed Matplotlib imports
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt


load_dotenv()

# --- Konfiguration der Seite ---
st.set_page_config(page_title="Garmin Stats", layout="wide")


def get_prediction_df(client):
    predictions = {key: [value] for key, value in client.get_race_predictions().items()}
    df = pd.DataFrame.from_dict(predictions)
    df = df.loc[:, ["time5K", "time10K", "timeHalfMarathon", "timeMarathon"]]
    df = df.rename(
        columns={
            "time5K": "5K",
            "time10K": "10K",
            "timeHalfMarathon": "1/2 Marathon",
            "timeMarathon": "Marathon",
        }
    )
    df = df.map(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    return df


# --- Login & Datenabruf ---
@st.cache_data(ttl=600)
def load_data():
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORT")

    try:
        garmin = Garmin(EMAIL, PASSWORD)
        garmin.login()

        # 1. Activities
        activities = garmin.get_activities(0, 50)

        # 2. VO2 Max Metrics
        today = date.today()
        # Ensure we use string format for the API
        today_str = str(today)
        url = f"/metrics-service/metrics/maxmet/daily/2025-10-01/{today_str}"
        vo2_data = garmin.connectapi(url)

        # 3. NEW: Weight Data
        # Using the same start date as your project
        weight_data = garmin.get_weigh_ins("2025-10-01", today_str)

        race_predictions = get_prediction_df(garmin)

        # Return 4 values now
        return activities, vo2_data, race_predictions, weight_data

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# Ladebalken anzeigen
with st.spinner("Lade Daten von Garmin..."):
    activities, vo2_json, race_predictions, weight_json = load_data()

if activities is None:
    st.error("Fehler beim Login oder Abruf der Daten.")
    st.stop()

# --- Datenverarbeitung ---
df = pd.DataFrame(activities)

# 1. Nur Running-Aktivitäten
if "activityType" in df.columns:
    df["is_running"] = df["activityType"].apply(
        lambda x: x["typeKey"] in ["running", "treadmill_running"]
    )
    df = df[df["is_running"]].copy()
else:
    st.warning("Keine Aktivitätstypen gefunden.")
    st.stop()

# 2. Zeitformat korrigieren & Filtern (Aktivitäten)
df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"])
start_date = pd.to_datetime("2025-10-01")
df = df[df["startTimeLocal"] >= start_date]

# 3. Sortieren
df = df.sort_values("startTimeLocal", ascending=True).reset_index(drop=True)

# --- VO2 MAX DATEN VERARBEITUNG ---
vo2_list = []
if vo2_json:
    for entry in vo2_json:
        if "generic" in entry and "vo2MaxPreciseValue" in entry["generic"]:
            val = entry["generic"]["vo2MaxPreciseValue"]
            datum_str = entry["generic"]["calendarDate"]
            vo2_list.append({"date": pd.to_datetime(datum_str), "vo2_precise": val})

df_vo2 = pd.DataFrame(vo2_list)
# Auch VO2 Daten nach Datum sortieren
if not df_vo2.empty:
    df_vo2 = df_vo2.sort_values("date", ascending=True)

# ----------------------------------

if not df.empty:
    # --- BERECHNUNGEN FÜR PLOTS ---

    # A) Effizienz-Index & Rolling Average
    if "averageSpeed" in df.columns and "averageHR" in df.columns:
        df["efficiency_index"] = df.apply(
            lambda row: (
                (row["averageSpeed"] * 60) / row["averageHR"]
                if (pd.notnull(row["averageHR"]) and row["averageHR"] > 0)
                else None
            ),
            axis=1,
        )
        df["efficiency_rolling"] = df["efficiency_index"].rolling(window=3).mean()
    else:
        df["efficiency_index"] = None
        df["efficiency_rolling"] = None

    # B) Cadence Rolling Average
    if "averageRunningCadenceInStepsPerMinute" in df.columns:
        df["cadence_raw"] = df["averageRunningCadenceInStepsPerMinute"]
        df["cadence_rolling"] = df["cadence_raw"].rolling(window=3).mean()
    else:
        df["cadence_raw"] = None
        df["cadence_rolling"] = None

    # --- TABELLEN VORBEREITUNG ---
    df_display_base = df.sort_values("startTimeLocal", ascending=False).copy()

    df_display_base["datum_formatiert"] = df_display_base["startTimeLocal"].dt.strftime(
        "%d.%m.%Y"
    )

    wochentage_map = {
        0: "Montag",
        1: "Dienstag",
        2: "Mittwoch",
        3: "Donnerstag",
        4: "Freitag",
        5: "Samstag",
        6: "Sonntag",
    }
    df_display_base["wochentag"] = df_display_base["startTimeLocal"].dt.weekday.map(
        wochentage_map
    )

    def calculate_pace(duration_sec, distance_km):
        if distance_km <= 0:
            return "0:00"
        pace_seconds = duration_sec / distance_km
        minutes = int(pace_seconds // 60)
        seconds = int(pace_seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def convert_activity(activity_dict):
        key = activity_dict.get("typeKey", "")
        conversion_dict = {"running": "Laufen", "treadmill_running": "Laufband"}
        return conversion_dict.get(key, "-")

    df_display_base["distance [km]"] = df_display_base["distance"].apply(
        lambda x: round(x / 1000, 2)
    )
    df_display_base["cadence_display"] = df_display_base[
        "averageRunningCadenceInStepsPerMinute"
    ].apply(lambda x: round(x) if pd.notnull(x) else 0)
    df_display_base["duration_formated"] = df_display_base["duration"].apply(
        lambda x: str(datetime.timedelta(seconds=int(x)))
    )
    df_display_base["pace [min/km]"] = df_display_base.apply(
        lambda x: calculate_pace(x["duration"], x["distance [km]"]), axis=1
    )
    df_display_base["activity_name"] = df_display_base["activityType"].apply(
        convert_activity
    )

    final_columns = [
        "datum_formatiert",
        "wochentag",
        "distance [km]",
        "duration_formated",
        "pace [min/km]",
        "averageHR",
        "cadence_display",
        "calories",
        "activity_name",
        "description",
    ]
    display_df = df_display_base[
        [c for c in final_columns if c in df_display_base.columns]
    ].copy()

    display_df.fillna("-", inplace=True)
    display_df = display_df.rename(
        columns={
            "datum_formatiert": "Datum",
            "wochentag": "Wochentag",
            "distance [km]": "Distanz (km)",
            "duration_formated": "Dauer",
            "pace [min/km]": "Pace",
            "averageHR": "Ø Puls",
            "cadence_display": "Schrittfreq.",
            "calories": "Kalorien",
            "activity_name": "Aktivität",
            "description": "Notiz",
        }
    )

    weight_list = []
    if weight_json and "dailyWeightSummaries" in weight_json:
        for weighin in weight_json["dailyWeightSummaries"]:
            # Safety check to ensure keys exist
            if "summaryDate" in weighin and "latestWeight" in weighin:
                val = weighin["latestWeight"]["weight"] / 1000.0
                datum_str = weighin["summaryDate"]
                weight_list.append({"date": pd.to_datetime(datum_str), "weight": val})

    df_weight = pd.DataFrame(weight_list)
    if not df_weight.empty:
        df_weight = df_weight.sort_values("date", ascending=True)
        df_weekly = df_weight.set_index("date").resample("W").mean().reset_index()
        df_weekly["date"] = df_weekly["date"] - pd.Timedelta(days=3)
        df_weekly = df_weekly.dropna(subset=["weight"])

    # --- UI: TABELLE ---
    st.subheader("🏃‍♂️ Lauf-Übersicht")
    anzahl = st.number_input(
        "Anzahl der letzten Läufe anzeigen:", min_value=1, value=3, step=1
    )
    df_limited = display_df.head(int(anzahl))
    st.dataframe(df_limited, hide_index=True)

    # --- UI: DOWNLOAD BUTTON ---
    csv = df_limited.to_csv(index=False, sep=";").encode("utf-8-sig")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="💾 Tabelle als CSV herunterladen",
        data=csv,
        file_name=f"{timestamp}_Stats.csv",
        mime="text/csv",
    )

    # Race prediction DF
    st.dataframe(race_predictions, hide_index=True)

    st.markdown("---")

    # Only outdoor running for plots
    df_plot = df.copy()
    df_plot["is_running"] = df["activityType"].apply(
        lambda x: x["typeKey"] in ["running"]
    )
    df_plot = df_plot[df_plot["is_running"]].copy()

    # --- UI: GRAPHEN (Plotly) ---
    st.subheader("📈 Fitness-Entwicklung")

    # Wir erstellen Subplots mit gemeinsamer X-Achse
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "VO2 Max Entwicklung",
            "Laufeffizienz (Meter/Herzschlag)",
            "Schrittfrequenz (Cadence)",
        ),
    )

    # 1. Plot: VO2 Max (Präzise)
    if not df_vo2.empty:
        fig.add_trace(
            go.Scatter(
                x=df_vo2["date"],
                y=df_vo2["vo2_precise"],
                mode="lines+markers",
                name="VO2max",
                line=dict(color="#d62728", width=2),
                marker=dict(size=6),
                hovertemplate="%{y:.2f} ml/kg/min<extra></extra>",  # Custom Hover
            ),
            row=1,
            col=1,
        )

    # 2. Plot: Effizienz
    # Einzelne Läufe (Punkte)
    fig.add_trace(
        go.Scatter(
            x=df_plot["startTimeLocal"],
            y=df_plot["efficiency_index"],
            mode="markers",
            name="Effizienz (Einzel)",
            marker=dict(color="gray", opacity=0.4, size=6),
            hovertemplate="%{y:.2f} m/Schlag<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Trendlinie (Rolling Avg)
    fig.add_trace(
        go.Scatter(
            x=df_plot["startTimeLocal"],
            y=df_plot["efficiency_rolling"],
            mode="lines",
            name="Effizienz (Trend)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{y:.2f} m/Schlag (Ø)<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 3. Plot: Cadence
    mask_cad = df_plot["cadence_raw"] > 0
    # Einzelne Läufe (Punkte)
    fig.add_trace(
        go.Scatter(
            x=df_plot.loc[mask_cad, "startTimeLocal"],
            y=df_plot.loc[mask_cad, "cadence_raw"],
            mode="markers",
            name="Cadence (Einzel)",
            marker=dict(color="gray", opacity=0.4, size=6),
            hovertemplate="%{y:.0f} spm<extra></extra>",
        ),
        row=3,
        col=1,
    )
    # Trendlinie (Rolling Avg)
    fig.add_trace(
        go.Scatter(
            x=df_plot["startTimeLocal"],
            y=df_plot["cadence_rolling"],
            mode="lines",
            name="Cadence (Trend)",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="%{y:.0f} spm (Ø)<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Layout Anpassungen
    fig.update_layout(
        height=800,  # Gesamthöhe des Plots
        showlegend=False,  # Legende ausblenden, um Platz zu sparen (oder True, wenn gewünscht)
        hovermode="x unified",  # Tooltip zeigt alle Werte an der x-Position
        template="plotly_white",
    )

    xmin = "2025-10-30"
    last_run = (
        df_plot["startTimeLocal"].max() if not df_plot.empty else pd.to_datetime(xmin)
    )
    last_vo2 = df_vo2["date"].max() if not df_vo2.empty else pd.to_datetime(xmin)
    xmax = max(last_run, last_vo2) + pd.Timedelta(days=2)

    # 3. Apply the range to the x-axis
    fig.update_xaxes(range=[xmin, xmax])

    # Y-Achsen Beschriftungen
    fig.update_yaxes(title_text="ml/kg/min", row=1, col=1)
    fig.update_yaxes(title_text="m / Schlag", row=2, col=1)
    fig.update_yaxes(title_text="spm", row=3, col=1)

    # Plot anzeigen
    st.plotly_chart(fig)

    st.markdown("---")  # Visual separator
    st.subheader("⚖️ Gewichtsentwicklung")

    if not df_weight.empty:
        # Create a completely new Figure
        fig_weight = go.Figure()

        fig_weight.add_trace(
            go.Scatter(
                x=df_weight["date"],
                y=df_weight["weight"],
                mode="lines+markers",
                name="Gewicht",
                opacity=0.5,
                line=dict(color="#686868", width=3),  # Purple
                marker=dict(size=8),
                hovertemplate="%{y:.1f} kg<extra></extra>",
            )
        )

        fig_weight.add_trace(
            go.Scatter(
                x=df_weekly["date"],
                y=df_weekly["weight"],
                mode="lines+markers",
                name="Wochenschnitt",
                line=dict(color="#07D7B4", width=2),  # Pink/Thicker for the trend
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="%{y:.1f} kg (Schnitt)<extra></extra>",
            )
        )

        start_date = df_weight["date"].min()
        end_date = df_weight["date"].max()

        # Generate all Mondays between start and end
        # 'W-MON' ensures we only get Mondays
        mondays = pd.date_range(start=start_date, end=end_date, freq="W-MON")

        for monday in mondays:
            fig_weight.add_vline(
                x=monday.timestamp() * 1000,  # Plotly uses ms for timestamps
                line_width=1,
                line_dash="dash",
                line_color="rgba(150, 150, 150, 0.5)",  # Subtle gray
            )

        # Independent Layout
        fig_weight.update_layout(
            height=400,  # Shorter height for single plot
            yaxis_title="Gewicht (kg)",
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
        )

        # Optional: Apply the same Oct 15th start date restriction if you want consistency,
        # OR leave it auto-scaled by removing update_xaxes below.
        first_weight = df_weight["date"].min()
        last_weight = df_weight["date"].max()
        xmax_weight = last_weight + pd.Timedelta(days=3)
        xmin_weight = first_weight - pd.Timedelta(days=1)
        fig_weight.update_xaxes(range=[xmin_weight, xmax_weight])

        st.plotly_chart(fig_weight)

    else:
        st.info("Keine Gewichtsdaten im gewählten Zeitraum gefunden.")

else:
    st.info("Keine Läufe im gewählten Zeitraum gefunden.")
