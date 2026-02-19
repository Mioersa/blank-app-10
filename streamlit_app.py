import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

# ============ CONFIG ============
st.set_page_config("Rule‑Based Intraday Option Signals", layout="wide")
st.title("📊 Rule‑Based Intraday Option Signal System")

# ---- SIDEBAR ----
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid‑ask spread %", 0.0, 1.0, 0.2)
basis = st.sidebar.radio("Top‑strike ranking basis", ["Open Interest", "Volume"])
num_strikes = st.sidebar.number_input("Top strikes by basis", 1, 30, 6)
st.sidebar.markdown("Upload one or more **Option‑Chain CSV files** below 👇")

uploaded_files = st.file_uploader("Drop CSV files (multiple allowed)",
                                  type=["csv"], accept_multiple_files=True)
if not uploaded_files:
    st.info("⬅️ Upload CSVs to start.")
    st.stop()

# ---- LOAD ----
frames=[]
for f in uploaded_files:
    try:
        base=f.name.replace(".csv","")
        ts=datetime.strptime(base.split("_")[-2]+"_"+base.split("_")[-1],"%d%m%Y_%H%M%S")
    except Exception:
        ts=datetime.now()
    df=pd.read_csv(f)
    df["timestamp"]=ts
    frames.append(df)

raw_df=pd.concat(frames,ignore_index=True).sort_values("timestamp")
st.success(f"✅ Loaded {len(uploaded_files)} file(s), {len(raw_df)} rows.")

# ---- CLEAN ----
def clean_data(df,spread_cutoff=0.2):
    df=df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    req=["CE_buyPrice1","CE_sellPrice1","PE_buyPrice1","PE_sellPrice1"]
    avail=[c for c in req if c in df.columns]
    df=df[(df[avail]>0).all(axis=1)]
    df["mid_CE"]=(df["CE_buyPrice1"]+df["CE_sellPrice1"])/2
    df["mid_PE"]=(df["PE_buyPrice1"]+df["PE_sellPrice1"])/2
    df["mid_CE"].replace(0,np.nan,inplace=True)
    df["spread_pct"]=abs(df["CE_sellPrice1"]-df["CE_buyPrice1"])/df["mid_CE"]
    df=df[df["spread_pct"]<spread_cutoff]
    if "CE_expiryDate" in df.columns:
        df["CE_expiryDate"]=pd.to_datetime(df["CE_expiryDate"],errors="coerce")
        df["days_to_expiry"]=(df["CE_expiryDate"]-df["timestamp"]).dt.days
    else:
        df["days_to_expiry"]=1
    df["days_to_expiry"].fillna(1,inplace=True)
    df["θ_adj_CE"]=df["CE_lastPrice"]/np.sqrt(df["days_to_expiry"].clip(lower=1))
    df["θ_adj_PE"]=df["PE_lastPrice"]/np.sqrt(df["days_to_expiry"].clip(lower=1))
    return df

df=clean_data(raw_df,spread_cutoff)

# ---- FEATURE BUILD ----
def compute_features(df,rolling_n=5,top_n=6,basis="Open Interest"):
    df=df.copy().sort_values("timestamp")
    df["CE_vol_delta"]=df.groupby("CE_strikePrice")["CE_totalTradedVolume"].diff().fillna(0)
    df["PE_vol_delta"]=df.groupby("CE_strikePrice")["PE_totalTradedVolume"].diff().fillna(0)
    df["total_vol"]=df["CE_vol_delta"]+df["PE_vol_delta"]
    df["total_OI"]=df["CE_openInterest"]+df["PE_openInterest"]

    metric="total_OI" if basis.startswith("Open") else "total_vol"
    mean_strike=df.groupby("CE_strikePrice")[metric].mean()
    top_strikes=mean_strike.nlargest(top_n)
    covered=round(100*top_strikes.sum()/mean_strike.sum(),2)
    df=df[df["CE_strikePrice"].isin(top_strikes.index)]

    agg=df.groupby("timestamp").agg({
        "CE_lastPrice":"mean","PE_lastPrice":"mean",
        "CE_openInterest":"sum","PE_openInterest":"sum",
        "CE_changeinOpenInterest":"sum","PE_changeinOpenInterest":"sum",
        "CE_vol_delta":"sum","PE_vol_delta":"sum",
        "CE_impliedVolatility":"mean","PE_impliedVolatility":"mean"
    })
    agg["ΔPrice_CE"]=agg["CE_lastPrice"].diff()
    agg["ΔOI_CE"]=agg["CE_changeinOpenInterest"].diff()
    agg["ΔPrice_PE"]=agg["PE_lastPrice"].diff()
    agg["ΔOI_PE"]=agg["PE_changeinOpenInterest"].diff()
    agg["IV_skew"]=agg["CE_impliedVolatility"]-agg["PE_impliedVolatility"]
    agg["ΔIV"]=agg["IV_skew"].diff()
    agg["PCR_OI"]=agg["PE_openInterest"]/agg["CE_openInterest"].replace(0,np.nan)
    agg["ΔPCR"]=agg["PCR_OI"].diff()
    total_vol=agg["CE_vol_delta"]+agg["PE_vol_delta"]
    agg["Volume_spike"]=total_vol/total_vol.rolling(rolling_n).mean()
    agg.fillna(0,inplace=True)
    return agg,covered

df_feat,covered_pct=compute_features(df,rolling_n,num_strikes,basis)
st.caption(f"**Top {num_strikes} strikes** cover ≈ {covered_pct}% of total {basis.lower()}.")

# ---- LOGIC ----
def detect_regime(row):
    regime,bias="quiet","neutral"
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

df_feat[["regime","bias"]]=df_feat.apply(detect_regime,axis=1,result_type="expand")
df_feat["signal"]=df_feat.apply(generate_signal,axis=1)
df_feat["signal_numeric"]=df_feat["signal"].map({
    "BUY_CALL":1,"BUY_PUT":1,"MOMENTUM_TRADE":1,
    "SELL_STRANGLE":0,"HOLD":0,"EXIT_POSITION":-1
}).fillna(0)

# ---- STYLERS ----
def _sig_color(v):
    colors={"BUY_CALL":"#99ff99","BUY_PUT":"#33cc33","SELL_STRANGLE":"#ffcc80",
            "MOMENTUM_TRADE":"#00b300","EXIT_POSITION":"#ff4d4d","HOLD":"#ffd280"}
    return f"background-color:{colors.get(v,'')}"
def _bias_color(v):
    if v=="bullish": return "background-color:#b3ffb3"
    if v=="bearish": return "background-color:#ff9999"
    return "background-color:#ffd480"

# ---- METRICS ----
lat=df_feat.iloc[-1]
c1,c2,c3,c4=st.columns(4)
c1.metric("Current PCR (OI)",round(float(lat["PCR_OI"]),2))
c2.metric("# Trend Bars",int((df_feat["regime"]=="trend").sum()))
c3.metric("Latest Signal",lat["signal"])
c4.metric("Rows Processed",len(df_feat))
def pcr_meaning(p):
    if p<0.7:return"🐂 Bullish – calls lead"
    if 0.7<=p<=1.2:return"🟧 Neutral structure"
    return"🐻 Bearish – puts build"
st.caption(f"**PCR Interpretation:** {pcr_meaning(lat['PCR_OI'])}")

# ---- TABLES ----
st.subheader("🧾 Recent Signals")
st.dataframe(df_feat.tail(10).style.applymap(_sig_color,subset=["signal"])
                              .applymap(_bias_color,subset=["bias"]),
             use_container_width=True)
st.subheader("📄 Full Dataset")
st.dataframe(df_feat.style.applymap(_sig_color,subset=["signal"])
                          .applymap(_bias_color,subset=["bias"]),
             use_container_width=True)

# ---- TIMELINE ----
st.subheader("🌀 Signal / Bias Timeline")
sig_chart=alt.Chart(df_feat.reset_index()).mark_circle(size=80).encode(
    x="timestamp:T",
    y=alt.Y("signal_numeric:Q",scale=alt.Scale(domain=[-1.2,1.2]),
            title="Signal (‑1 = Sell, 0 = Hold, +1 = Buy)"),
    color="bias:N",
    tooltip=["timestamp","signal","bias","regime"])
st.altair_chart(sig_chart,use_container_width=True)

# ---- DEEP PRICE–VOLUME CORRELATION ----
st.subheader("📊 Deep Price–Volume Correlation Analysis")
top_vol = st.slider("Select Top Strikes by Avg Volume", 1, 20, 5)
df["total_vol"]=df.groupby("CE_strikePrice")[["CE_totalTradedVolume","PE_totalTradedVolume"]].diff().sum(axis=1).fillna(0)
avg_vol=df.groupby("CE_strikePrice")["total_vol"].mean().nlargest(top_vol)
sel=avg_vol.index
subset=df[df["CE_strikePrice"].isin(sel)]
corrs=[]
for k,g in subset.groupby("CE_strikePrice"):
    if g["total_vol"].std()>0 and g["CE_lastPrice"].diff().std()>0:
        c=np.corrcoef(g["total_vol"],g["CE_lastPrice"].diff())[0,1]
        corrs.append({"strike":k,"corr":round(c,2)})
corr_df=pd.DataFrame(corrs)
if corrs:
    st.dataframe(corr_df,use_container_width=True)
    corr_chart=alt.Chart(corr_df).mark_bar().encode(
        x="strike:O",y="corr:Q",
        color=alt.condition("datum.corr>0",
                            alt.value("#2ECC71"),alt.value("#E74C3C"))
    ).properties(height=300)
    st.altair_chart(corr_chart,use_container_width=True)
else:
    st.info("No valid correlations computed (check uploaded data time series).")

# ---- DOWNLOAD ----
st.download_button("⬇️ Download Processed CSV",
                   df_feat.to_csv(index=False).encode("utf‑8"),
                   "signals_output.csv","text/csv")


