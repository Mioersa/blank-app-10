# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import altair as alt

# ---------------- CONFIG ----------------
st.set_page_config("Rule‑Based Intraday Option Signals", layout="wide")
st.title("📊 Rule‑Based Intraday Option Signal System")

# -------------- SIDEBAR -----------------
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid‑ask spread %", 0.0, 1.0, 0.2)
num_strikes = st.sidebar.number_input("Top strikes by OI", 1, 30, 6)
st.sidebar.markdown("Upload one or more **Option‑Chain CSVs** below 👇")

uploaded_files = st.file_uploader(
    "Drop CSV files (multiple allowed)",
    type=["csv"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("⬅️ Upload CSVs to start.")
    st.stop()

# -------------- LOAD --------------------
frames = []
for f in uploaded_files:
    try:
        base = f.name.replace(".csv", "")
        ts = datetime.strptime(base.split("_")[-2] + "_" + base.split("_")[-1],
                               "%d%m%Y_%H%M%S")
    except Exception:
        ts = datetime.now()
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)

raw_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
st.success(f"✅ Loaded {len(uploaded_files)} file(s), {len(raw_df)} rows total.")

# -------------- CLEAN -------------------
def clean_data(df, spread_cutoff=0.2):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # zero quote filtering
    req = ["CE_buyPrice1","CE_sellPrice1","PE_buyPrice1","PE_sellPrice1"]
    avail = [c for c in req if c in df.columns]
    df = df[(df[avail] > 0).all(axis=1)]

    df["mid_CE"] = (df["CE_buyPrice1"] + df["CE_sellPrice1"]) / 2
    df["mid_PE"] = (df["PE_buyPrice1"] + df["PE_sellPrice1"]) / 2
    df["mid_CE"].replace(0, np.nan, inplace=True)
    df["spread_pct"] = abs(df["CE_sellPrice1"] - df["CE_buyPrice1"]) / df["mid_CE"]
    df = df[df["spread_pct"] < spread_cutoff]

    if "CE_expiryDate" in df.columns:
        df["CE_expiryDate"] = pd.to_datetime(df["CE_expiryDate"], errors="coerce")
        df["days_to_expiry"] = (df["CE_expiryDate"] - df["timestamp"]).dt.days
    else:
        df["days_to_expiry"] = 1
    df["days_to_expiry"] = df["days_to_expiry"].fillna(1).clip(lower=1)

    df["θ_adj_CE"] = df["CE_lastPrice"] / np.sqrt(df["days_to_expiry"])
    df["θ_adj_PE"] = df["PE_lastPrice"] / np.sqrt(df["days_to_expiry"])
    return df

df = clean_data(raw_df, spread_cutoff)

# -------------- FEATURES ----------------
def compute_features(df, rolling_n=5, top_n=6):
    df = df.copy().sort_values("timestamp")

    # pick top strikes by OI
    if {"CE_openInterest","PE_openInterest","CE_strikePrice"}.issubset(df.columns):
        df["total_OI"] = df["CE_openInterest"] + df["PE_openInterest"]
        top_strikes = (
            df.groupby("CE_strikePrice")["total_OI"]
              .mean().nlargest(top_n).index.tolist()
        )
        df = df[df["CE_strikePrice"].isin(top_strikes)]

    # per‑bar volume (handle cumulative)
    df["CE_vol_delta"] = df.groupby("CE_strikePrice")["CE_totalTradedVolume"].diff()
    df["PE_vol_delta"] = df.groupby("CE_strikePrice")["PE_totalTradedVolume"].diff()
    df["CE_vol_delta"].fillna(0, inplace=True)
    df["PE_vol_delta"].fillna(0, inplace=True)

    agg = df.groupby("timestamp").agg({
        "CE_lastPrice":"mean",
        "PE_lastPrice":"mean",
        "CE_openInterest":"sum",
        "PE_openInterest":"sum",
        "CE_changeinOpenInterest":"sum",
        "PE_changeinOpenInterest":"sum",
        "CE_vol_delta":"sum",
        "PE_vol_delta":"sum",
        "CE_impliedVolatility":"mean",
        "PE_impliedVolatility":"mean",
        "CE_buyPrice1":"mean","PE_buyPrice1":"mean",
        "CE_sellPrice1":"mean","PE_sellPrice1":"mean"
    })

    # derived metrics
    agg["ΔPrice_CE"] = agg["CE_lastPrice"].diff()
    agg["ΔPrice_PE"] = agg["PE_lastPrice"].diff()
    agg["ΔOI_CE"] = agg["CE_changeinOpenInterest"].diff()
    agg["ΔOI_PE"] = agg["PE_changeinOpenInterest"].diff()
    agg["OI_skew"] = (agg["CE_openInterest"] - agg["PE_openInterest"]) / (
        agg["CE_openInterest"] + agg["PE_openInterest"]).replace(0, np.nan)
    agg["IV_skew"] = agg["CE_impliedVolatility"] - agg["PE_impliedVolatility"]
    agg["ΔIV"] = agg["IV_skew"].diff()
    agg["PCR_OI"] = agg["PE_openInterest"] / agg["CE_openInterest"].replace(0, np.nan)
    agg["ΔPCR"] = agg["PCR_OI"].diff()

    total_vol = agg["CE_vol_delta"] + agg["PE_vol_delta"]
    agg["Volume_spike"] = total_vol / total_vol.rolling(rolling_n).mean()

    # θ‑adj + straddle
    agg["Straddle_price"] = agg["CE_lastPrice"] + agg["PE_lastPrice"]
    agg["Range_expansion"] = (agg["CE_lastPrice"] + agg["PE_lastPrice"]).pct_change()

    agg.fillna(0, inplace=True)
    return agg

df_feat = compute_features(df, rolling_n, num_strikes)

# -------------- REGIME LOGIC ------------
def detect_regime(row):
    regime, bias = "quiet","neutral"
    if row["ΔPrice_CE"]*row["ΔOI_CE"]>0 and row["Volume_spike"]>1: regime="trend"
    elif abs(row["ΔPrice_CE"])<0.05 and abs(row["ΔOI_CE"])<1000: regime="range"
    elif abs(row["ΔPrice_CE"])>0.2 and row["Volume_spike"]>1.5 and row["ΔIV"]>0: regime="breakout"
    elif row["ΔPrice_CE"]>0 and row["ΔOI_CE"]<0 and row["ΔIV"]<0: regime="exhaustion"

    if row["PCR_OI"]<0.8: bias="bullish"
    elif row["PCR_OI"]>1.2: bias="bearish"
    return regime,bias

def generate_signal(row):
    if row["regime"]=="trend" and row["bias"]=="bullish": return "BUY_CALL"
    if row["regime"]=="trend" and row["bias"]=="bearish": return "BUY_PUT"
    if row["regime"]=="range": return "SELL_STRANGLE"
    if row["regime"]=="breakout": return "MOMENTUM_TRADE"
    if row["regime"]=="exhaustion": return "EXIT_POSITION"
    return "HOLD"

def conclusion_text(row):
    if row["bias"]=="bullish" and row["ΔOI_CE"]>row["ΔOI_PE"]:
        return "CE build‑up > PE build‑up → bullish skew forming."
    if row["regime"]=="breakout":
        return "Big volume spike + IV rise → breakout likely."
    if row["regime"]=="exhaustion":
        return "Price rising but OI + IV drop → long unwind."
    if row["regime"]=="trend" and row["ΔIV"]>0:
        return "Rising IV + price surge → vol expansion."
    if row.get("ΔPCR",0)>0.2:
        return "PCR climbing → put unwinding / optimism."
    if row["Volume_spike"]<0.8 and abs(row["ΔIV"])<0.2:
        return "Flat prices + low IV → stay out or short prem."
    return ""

df_feat[["regime","bias"]] = df_feat.apply(detect_regime, axis=1, result_type="expand")
df_feat["signal"] = df_feat.apply(generate_signal, axis=1)
df_feat["comment"] = df_feat.apply(conclusion_text, axis=1)

# -------------- METRIC CARDS ------------
latest = df_feat.iloc[-1]
colA,colB,colC,colD = st.columns(4)
colA.metric("Current PCR (OI)", round(float(latest["PCR_OI"]),2))
colB.metric("# Trend Bars", int((df_feat["regime"]=="trend").sum()))
colC.metric("Latest Signal", latest["signal"])
colD.metric("Rows Processed", len(df_feat))

# -------------- DISPLAY -----------------
st.subheader("Recent Signals")
st.dataframe(df_feat.tail(10), use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    chart1 = alt.Chart(df_feat.reset_index()).mark_line().encode(
        x="timestamp:T", y="PCR_OI:Q", color="regime:N")
    st.altair_chart(chart1, use_container_width=True)
with col2:
    chart2 = alt.Chart(df_feat.reset_index()).mark_line().encode(
        x="timestamp:T", y="IV_skew:Q", color="bias:N")
    st.altair_chart(chart2, use_container_width=True)

st.subheader("Full Dataset")
st.dataframe(df_feat, use_container_width=True)

st.download_button(
    "⬇️ Download Processed Results",
    data=df_feat.to_csv(index=False).encode("utf-8"),
    file_name="signals_output.csv",
    mime="text/csv"
)
