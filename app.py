# ============================
# app.py — GrowthOracle AI — Next Gen (Full)
# PART 1/5: Imports, Config, Logger, Validation Core
# ============================
import os, io, re, sys, math, json, time, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import to_html
except Exception:
    px = None
    go = None
    to_html = None

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STM = True
except Exception:
    _HAS_STM = False

# ---- Page ----
st.set_page_config(page_title="GrowthOracle AI — Next Gen", layout="wide", initial_sidebar_state="expanded")
st.title("GrowthOracle AI — Next Gen")
st.caption("Time-aware insights • English SEO-title cannibalization • Interactive analytics • Explainable recommendations")

# ---- Logger ----
@st.cache_resource
def get_logger(level=logging.INFO):
    logger = logging.getLogger("growthoracle")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
logger = get_logger()

# ---- Defaults / Config ----
_DEFAULT_CONFIG = {
    "thresholds": {
        "striking_distance_min": 11, "striking_distance_max": 20,
        "ctr_deficit_pct": 1.0, "similarity_threshold": 0.60,
        "min_impressions": 100, "min_clicks": 10
    },
    "presets": {
        "Conservative": {"striking_distance_min": 15, "striking_distance_max": 25, "ctr_deficit_pct": 2.0, "similarity_threshold": 0.7, "min_impressions": 300, "min_clicks": 30},
        "Standard":     {"striking_distance_min": 11, "striking_distance_max": 20, "ctr_deficit_pct": 1.0, "similarity_threshold": 0.6, "min_impressions": 100, "min_clicks": 10},
        "Aggressive":   {"striking_distance_min": 8,  "striking_distance_max": 18, "ctr_deficit_pct": 0.5,"similarity_threshold": 0.5, "min_impressions": 50,  "min_clicks": 5}
    },
    "expected_ctr_by_rank": {1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07, 6: 0.05, 7: 0.045, 8: 0.038, 9: 0.032},
    "performance": {"sample_row_limit": 350_000, "seed": 42},
    "defaults": {"date_lookback_days": 60}
}
@st.cache_resource
def load_config():
    cfg = _DEFAULT_CONFIG.copy()
    if yaml is not None:
        for candidate in ["config.yaml", "growthoracle.yaml", "settings.yaml"]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        user_cfg = yaml.safe_load(f) or {}
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in cfg: cfg[k].update(v)
                        else: cfg[k] = v
                    logger.info(f"Loaded configuration from {candidate}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {candidate}: {e}")
    return cfg
CONFIG = load_config()

# ---- Validation Core ----
@dataclass
class ValidationMessage:
    category: str  # "Critical" | "Warning" | "Info"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    def __init__(self):
        self.messages: List[ValidationMessage] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.lineage_notes: List[str] = []

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def add_exc(self, where: str, exc: Exception):
        self.exceptions.append({
            "where": where, "type": type(exc).__name__, "message": str(exc),
            "traceback": "\n".join(str(exc).splitlines()[-5:])
        })

    def checkpoint(self, name: str, **data):
        self.checkpoints.append({"name": name, **data})

    def quality_score(self) -> float:
        crit = sum(1 for m in self.messages if m.category == "Critical")
        warn = sum(1 for m in self.messages if m.category == "Warning")
        info = sum(1 for m in self.messages if m.category == "Info")
        score = 100 - (25*crit + 8*warn + 1*info)
        return float(max(0, min(100, score)))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages:
            return pd.DataFrame(columns=["category","code","message","context"])
        return pd.DataFrame([{
            "category": m.category, "code": m.code, "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])

PROCESS_LOG: List[Dict[str, Any]] = []
def log_event(event: str, **kws):
    entry = {"ts": pd.Timestamp.utcnow().isoformat(), "event": event, **kws}
    PROCESS_LOG.append(entry); logger.debug(f"[LOG] {event} | {kws}")

def coerce_numeric(series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float,float]]=None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad: vc.add("Warning", "NUM_COERCE", f"Non-numeric values coerced to NaN in {name}", bad_rows=int(bad))
    if clamp:
        lo, hi = clamp
        before = ((s < lo) | (s > hi)).sum()
        if before:
            s = s.clip(lo, hi)
            vc.add("Info", "NUM_CLAMP", f"{name} clipped to bounds", lo=lo, hi=hi, affected=int(before))
    return s

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    parsed = pd.to_datetime(col, errors="coerce", utc=True, infer_datetime_format=True)
    bad = parsed.isna().sum()
    if bad: vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in c.lower() for k in ["date","dt","time","timestamp","publish"])]

DEFAULT_VALIDATION_STRICTNESS = "Standard"
MERGE_STRATEGY = {"gsc_x_prod": "left", "ga4_align": "left"}
def add_lineage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy(); df["_source"] = source; return df

# ============================
# PART 2/5: Templates, Strong Readers, Date Standardizer, Mapping, UI State
# ============================
def _make_template_production():
    return pd.DataFrame({
        "Msid":[101,102,103],
        "Title":["Budget 2025 highlights explained","IPL 2025 schedule & squads","Monsoon updates: city-by-city guide"],
        "Path":["/business/budget-2025/highlights","/sports/cricket/ipl-2025/schedule","/news/monsoon/guide"],
        "Publish Time":["2025-08-01 09:15:00","2025-08-10 18:30:00","2025-09-01 07:00:00"]
    })
def _make_template_ga4():
    return pd.DataFrame({
        "customEvent:msid":[101,101,102,102,103],
        "date":["2025-08-01","2025-08-02","2025-08-10","2025-08-11","2025-09-01"],
        "screenPageViews":[5000,6000,15000,12000,7000],
        "totalUsers":[4000,4500,10000,8000,5200],
        "userEngagementDuration":[52.3,48.2,41.0,44.7,63.1],
        "bounceRate":[0.42,0.45,0.51,0.49,0.38]
    })
def _make_template_gsc():
    return pd.DataFrame({
        "Date":["2025-08-01","2025-08-02","2025-08-10","2025-08-11","2025-09-01"],
        "Page":[
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/news/monsoon/guide/103.cms"
        ],
        "Query":["budget 2025","budget highlights","ipl 2025 schedule","ipl squads","monsoon city guide"],
        "Clicks":[200,240,1200,1100,300],
        "Impressions":[5000,5500,40000,38000,7000],
        "CTR":[0.04,0.0436,0.03,0.0289,0.04286],
        "Position":[8.2,8.0,12.3,11.7,9.1]
    })

def download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def read_csv_safely(upload, name: str, vc: ValidationCollector, sample_rows: int = 1000) -> Optional[pd.DataFrame]:
    if upload is None:
        vc.add("Critical","NO_FILE",f"{name} file not provided"); return None
    try_encodings = [None, "utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc) if enc else pd.read_csv(upload)
            if df.empty or df.shape[1]==0:
                vc.add("Critical","EMPTY_CSV",f"{name} appears empty"); return None
            nullish_headers = sum(1 for c in df.columns if str(c).strip().lower() in ("unnamed: 0","", "nan"))
            if nullish_headers: vc.add("Warning","HEADER_SUSPECT",f"{name} had unnamed/blank headers",count=int(nullish_headers))
            vc.checkpoint(f"{name}_read", rows=int(min(len(df), sample_rows)), cols=int(df.shape[1]), encoding=enc or "auto")
            return add_lineage(df, name)
        except Exception as e:
            last_err = e
            continue
    vc.add("Critical","CSV_ENCODING",f"Failed to read {name} with common encodings", last_error=str(last_err))
    vc.add_exc(f"read_csv:{name}", last_err or Exception("Unknown"))
    return None

def _guess_colmap(prod_df, ga4_df, gsc_df):
    prod_map = {
        "msid":"Msid" if "Msid" in prod_df.columns else next((c for c in prod_df.columns if c.lower()=="msid"), None),
        "title":"Title" if "Title" in prod_df.columns else next((c for c in prod_df.columns if "title" in c.lower()), None),
        "path":"Path" if "Path" in prod_df.columns else next((c for c in prod_df.columns if "path" in c.lower()), None),
        "publish":"Publish Time" if "Publish Time" in prod_df.columns else next((c for c in prod_df.columns if "publish" in c.lower()), None),
    }
    ga4_map = {
        "msid":"customEvent:msid" if "customEvent:msid" in ga4_df.columns else next((c for c in ga4_df.columns if "msid" in c.lower()), None),
        "date":"date" if "date" in ga4_df.columns else next((c for c in ga4_df.columns if c.lower()=="date"), None),
        "pageviews":"screenPageViews" if "screenPageViews" in ga4_df.columns else next((c for c in ga4_df.columns if "pageview" in c.lower()), None),
        "users":"totalUsers" if "totalUsers" in ga4_df.columns else next((c for c in ga4_df.columns if "users" in c.lower()), None),
        "engagement":"userEngagementDuration" if "userEngagementDuration" in ga4_df.columns else next((c for c in ga4_df.columns if "engagement" in c.lower()), None),
        "bounce":"bounceRate" if "bounceRate" in ga4_df.columns else next((c for c in ga4_df.columns if "bounce" in c.lower()), None),
    }
    gsc_map = {
        "date":"Date" if "Date" in gsc_df.columns else next((c for c in gsc_df.columns if c.lower()=="date"), None),
        "page":"Page" if "Page" in gsc_df.columns else next((c for c in gsc_df.columns if "page" in c.lower()), None),
        "query":"Query" if "Query" in gsc_df.columns else next((c for c in gsc_df.columns if "query" in c.lower()), None),
        "impr":"Impressions" if "Impressions" in gsc_df.columns else next((c for c in gsc_df.columns if "impr" in c.lower()), None),
        "ctr":"CTR" if "CTR" in gsc_df.columns else next((c for c in gsc_df.columns if "ctr" in c.lower()), None),
        "pos":"Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
        "clicks": "Clicks" if "Clicks" in gsc_df.columns else next((c for c in gsc_df.columns if "click" in c.lower()), None), # ADD THIS LINE
    "pos":"Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
    }
    return prod_map, ga4_map, gsc_map

def guess_colmap_enhanced(prod_df, ga4_df, gsc_df):
    prod_map, ga4_map, gsc_map = _guess_colmap(prod_df, ga4_df, gsc_df)
    prod_dates = [c for c in detect_date_cols(prod_df) if "publish" in c.lower() or "time" in c.lower()]
    if prod_dates: prod_map["publish"] = prod_dates[0]
    if not ga4_map.get("date"):
        for c in detect_date_cols(ga4_df):
            if c.lower()=="date": ga4_map["date"]=c; break
    if not gsc_map.get("date"):
        for c in detect_date_cols(gsc_df):
            if c.lower()=="date": gsc_map["date"]=c; break
    return prod_map, ga4_map, gsc_map
    

def validate_columns_presence(prod_map, ga4_map, gsc_map, vc: ValidationCollector):
    req_prod = ["msid"]; req_ga4 = ["msid"]; req_gsc = ["date","page","query","impr","pos"]
    missing=[]
    for k in req_prod:
        if not prod_map.get(k): missing.append(f"Production: {k}")
    for k in req_ga4:
        if not ga4_map.get(k): missing.append(f"GA4: {k}")
    for k in req_gsc:
        if not gsc_map.get(k): missing.append(f"GSC: {k}")
    if missing: vc.add("Critical","MISSING_COLMAP","Missing/ambiguous mappings", items=missing)
    return missing

def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(ts):
            return ts
        # If tz-aware, convert; if tz-naive, localize.
        try:
            return ts.tz_convert("UTC")
        except Exception:
            try:
                return ts.tz_localize("UTC")
            except Exception:
                return ts  # last resort

    def normalize_date_only(df, col_name, out_name):
        if col_name in df.columns:
            dt = safe_dt_parse(df[col_name], col_name, vc)  # tz-aware UTC from safe_dt_parse
            df[out_name] = dt.dt.date
            if dt.notna().any():
                maxd, mind = dt.max(), dt.min()
                maxd_utc, mind_utc = _ensure_utc(maxd), _ensure_utc(mind)
                now_utc = pd.Timestamp.now(tz="UTC")  # ✅ tz-aware current time

                if (pd.notna(maxd_utc)) and (maxd_utc > now_utc + pd.Timedelta(days=1)):
                    vc.add("Warning", "FUTURE_DATE", f"{out_name} has future dates", sample=str(maxd_utc))

                if (pd.notna(mind_utc)) and (mind_utc < pd.Timestamp(2020, 1, 1, tz="UTC")):
                    vc.add("Info", "OLD_DATE", f"{out_name} includes <2020 dates", earliest=str(mind_utc))

    p = prod_df.copy()
    if mappings["prod"].get("publish") and mappings["prod"]["publish"] in p.columns:
        p["Publish Time"] = safe_dt_parse(p[mappings["prod"]["publish"]], "Publish Time", vc)
    else:
        for cand in detect_date_cols(p):
            if "publish" in cand.lower():
                p["Publish Time"] = safe_dt_parse(p[cand], cand, vc)
                vc.add("Info","DATE_DETECT","Detected publish column in Production", column=cand)
                break

    g4 = ga4_df.copy()
    g4_date_col = mappings["ga4"].get("date") or "date"
    if g4_date_col in g4.columns:
        normalize_date_only(g4, g4_date_col, "date")

    gs = gsc_df.copy()
    gs_date_col = mappings["gsc"].get("date") or "Date"
    if gs_date_col in gs.columns:
        normalize_date_only(gs, gs_date_col, "date")

    return p, g4, gs



# ---- Session defaults + sidebar ----
def init_state_defaults():
    if "config" not in st.session_state: st.session_state.config = CONFIG
    if "thresholds" not in st.session_state: st.session_state.thresholds = CONFIG["thresholds"].copy()
    if "date_range" not in st.session_state:
        end = date.today(); start = end - timedelta(days=CONFIG["defaults"]["date_lookback_days"])
        st.session_state.date_range = (start, end)
    if "log_level" not in st.session_state: st.session_state.log_level = "INFO"
init_state_defaults()

with st.sidebar:
    st.subheader("Configuration")
    preset = st.selectbox("Presets", ["Standard","Conservative","Aggressive"], index=0)
    if st.button("Apply Preset"):
        st.session_state.thresholds.update(st.session_state.config["presets"][preset])
    t = st.session_state.thresholds
    c1, c2 = st.columns(2)
    with c1:
        t["striking_distance_min"] = st.slider("Striking Distance — Min Position", 5, 50, t["striking_distance_min"])
        t["ctr_deficit_pct"] = st.slider("CTR Deficit Threshold (%)", 0.5, 10.0, float(t["ctr_deficit_pct"]), step=0.1)
        t["min_impressions"] = st.number_input("Min Impressions", min_value=0, value=int(t["min_impressions"]), step=50)
    with c2:
        t["striking_distance_max"] = st.slider("Striking Distance — Max Position", 5, 50, t["striking_distance_max"])
        t["similarity_threshold"] = st.slider("Content Similarity Threshold", 0.30, 0.90, float(t["similarity_threshold"]), step=0.05)
        t["min_clicks"] = st.number_input("Min Clicks", min_value=0, value=int(t["min_clicks"]), step=5)

    st.markdown("---")
    st.subheader("Analysis Period")
    start_def, end_def = st.session_state.date_range
    start_date = st.date_input("Start Date", value=start_def)
    end_date = st.date_input("End Date", value=end_def)
    if start_date > end_date:
        st.warning("Start date is after end date. Swapping."); start_date, end_date = end_date, start_date
    st.session_state.date_range = (start_date, end_date)

    st.markdown("---")
    st.subheader("Logging & Merge Options")
    st.session_state.log_level = st.selectbox("Log Level", ["DEBUG","INFO","WARNING","ERROR"], index=1)
    get_logger(getattr(logging, st.session_state.log_level))
    strictness = st.selectbox("Validation Strictness", ["Strict","Standard","Lenient"], index=1)
    st.session_state.strictness = strictness
    ms1 = st.selectbox("GSC × Production Join", ["left","inner"], index=0, help="Left keeps all GSC rows; inner keeps only MSIDs present in Production")
    ms2 = st.selectbox("Attach GA4 on (msid,date)", ["left","inner"], index=0)
    MERGE_STRATEGY["gsc_x_prod"] = ms1
    MERGE_STRATEGY["ga4_align"] = ms2

# ============================
# PART 3/5: Onboarding Steps, Mapping UI, Validation UI, Robust Processing
# ============================
st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", ["1) Get CSV Templates", "2) Upload & Map Columns", "3) Validate & Process", "4) Configure & Analyze"], horizontal=True)

if step == "1) Get CSV Templates":
    st.info("Download sample CSV templates to understand required structure.")
    colA, colB, colC = st.columns(3)
    with colA:
        df = _make_template_production()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_production.csv", "Download Production Template")
    with colB:
        df = _make_template_ga4()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_ga4.csv", "Download GA4 Template")
    with colC:
        df = _make_template_gsc()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_gsc.csv", "Download GSC Template")
    st.stop()

# Step 2: Uploads
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1: prod_file = st.file_uploader("Production Data (CSV)", type=["csv"], key="prod_csv")
    with col2: ga4_file  = st.file_uploader("GA4 Data (CSV)", type=["csv"], key="ga4_csv")
    with col3: gsc_file  = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")
if not all([prod_file, ga4_file, gsc_file]): st.stop()

# Read raw files with strong reader for preview/mapping stage
vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw  = read_csv_safely(ga4_file,  "GA4",        vc_read)
gsc_df_raw  = read_csv_safely(gsc_file,  "GSC",        vc_read)
if any(df is None or df.empty for df in [prod_df_raw, ga4_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable. See Validation section below.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

if step == "2) Upload & Map Columns":
    prod_map_guess, ga4_map_guess, gsc_map_guess = guess_colmap_enhanced(prod_df_raw, ga4_df_raw, gsc_df_raw)
    st.subheader("Column Mapping")
    st.caption("We guessed likely columns. Adjust if needed.")

    with st.expander("Production Mapping", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        prod_map = {}
        prod_map["msid"]   = c1.selectbox("MSID", prod_df_raw.columns, index=max(0, prod_df_raw.columns.get_loc(prod_map_guess.get("msid","")) ) if prod_map_guess.get("msid") in prod_df_raw.columns else 0)
        prod_map["title"]  = c2.selectbox("Title", prod_df_raw.columns, index=max(0, prod_df_raw.columns.get_loc(prod_map_guess.get("title","")) ) if prod_map_guess.get("title") in prod_df_raw.columns else 0)
        prod_map["path"]   = c3.selectbox("Path", prod_df_raw.columns, index=max(0, prod_df_raw.columns.get_loc(prod_map_guess.get("path","")) ) if prod_map_guess.get("path") in prod_df_raw.columns else 0)
        prod_map["publish"]= c4.selectbox("Publish Time", prod_df_raw.columns, index=max(0, prod_df_raw.columns.get_loc(prod_map_guess.get("publish","")) ) if prod_map_guess.get("publish") in prod_df_raw.columns else 0)

    with st.expander("GA4 Mapping", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        ga4_map = {}
        ga4_map["msid"]      = c1.selectbox("MSID", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("msid","")) ) if ga4_map_guess.get("msid") in ga4_df_raw.columns else 0)
        ga4_map["date"]      = c2.selectbox("Date", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("date","")) ) if ga4_map_guess.get("date") in ga4_df_raw.columns else 0)
        ga4_map["pageviews"] = c3.selectbox("Pageviews", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("pageviews","")) ) if ga4_map_guess.get("pageviews") in ga4_df_raw.columns else 0)
        ga4_map["users"]     = c4.selectbox("Users", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("users","")) ) if ga4_map_guess.get("users") in ga4_df_raw.columns else 0)
        ga4_map["engagement"]= c5.selectbox("Engagement Duration", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("engagement","")) ) if ga4_map_guess.get("engagement") in ga4_df_raw.columns else 0)
        ga4_map["bounce"]    = st.selectbox("Bounce Rate", ga4_df_raw.columns, index=max(0, ga4_df_raw.columns.get_loc(ga4_map_guess.get("bounce","")) ) if ga4_map_guess.get("bounce") in ga4_df_raw.columns else 0)

    with st.expander("GSC Mapping", expanded=True):
        c1, c2, c3 = st.columns(3)
        c4, c5, c6, c7 = st.columns(4)
        gsc_map = {}
        gsc_map["date"]   = c1.selectbox("Date", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("date","")) ) if gsc_map_guess.get("date") in gsc_df_raw.columns else 0)
        gsc_map["page"]   = c2.selectbox("Page URL", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("page","")) ) if gsc_map_guess.get("page") in gsc_df_raw.columns else 0)
        gsc_map["query"]  = c3.selectbox("Query", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("query","")) ) if gsc_map_guess.get("query") in gsc_df_raw.columns else 0)
        gsc_map["clicks"] = c4.selectbox("Clicks", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("clicks","")) ) if gsc_map_guess.get("clicks") in gsc_df_raw.columns else 0)
        gsc_map["impr"]   = c5.selectbox("Impressions", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("impr","")) ) if gsc_map_guess.get("impr") in gsc_df_raw.columns else 0)
        gsc_map["ctr"]    = c6.selectbox("CTR", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("ctr","")) ) if gsc_map_guess.get("ctr") in gsc_df_raw.columns else 0)
        gsc_map["pos"]    = c7.selectbox("Position", gsc_df_raw.columns, index=max(0, gsc_df_raw.columns.get_loc(gsc_map_guess.get("pos","")) ) if gsc_map_guess.get("pos") in gsc_df_raw.columns else 0)

    # quick validation
    _ = validate_columns_presence(prod_map, ga4_map, gsc_map, vc_read)
    rep_df = vc_read.to_dataframe()
    if not rep_df.empty:
        st.markdown("**Preliminary Reader/Mapping Warnings**")
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

    st.session_state.mapping = {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}
    st.success("Column mapping saved. Proceed to **Step 3**.")
    st.stop()

# Step 3: Validate & Process
if "mapping" not in st.session_state:
    st.warning("Please complete **Step 2** (column mapping) first."); st.stop()
prod_map = st.session_state.mapping["prod"]
ga4_map  = st.session_state.mapping["ga4"]
gsc_map  = st.session_state.mapping["gsc"]

def run_validation_pipeline(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    missing_map = validate_columns_presence(prod_map, ga4_map, gsc_map, vc)
    for name, df in [("Production", prod_df_raw), ("GA4", ga4_df_raw), ("GSC", gsc_df_raw)]:
        if df is None or df.empty: vc.add("Critical","EMPTY_FILE",f"{name} is empty or unreadable"); continue
        if df.shape[1] < 2: vc.add("Critical","TOO_FEW_COLS",f"{name} has too few cols", cols=int(df.shape[1]))
        if df.duplicated().any(): vc.add("Info","DUP_ROWS",f"{name} contained fully duplicated rows", rows=int(df.duplicated().sum()))
        cand = detect_date_cols(df)
        if cand: vc.add("Info","DATE_CANDIDATES", f"Possible date columns in {name}", columns=cand[:6])
    try:
        p_m = set(pd.to_numeric(prod_df_raw[prod_map["msid"]], errors="coerce").dropna().astype("int64"))
        def msid_from_url(u):
            if isinstance(u,str):
                m = re.search(r"(\d+)\.cms", u)
                return int(m.group(1)) if m else None
            return None
        g_m = set(pd.to_numeric(gsc_df_raw[gsc_map["page"]].apply(msid_from_url), errors="coerce").dropna().astype("int64"))
        only_p, only_g = len(p_m - g_m), len(g_m - p_m)
        if only_p: vc.add("Info","MSID_ONLY_PROD","MSIDs appear only in Production", count=int(only_p))
        if only_g: vc.add("Warning","MSID_ONLY_GSC","MSIDs appear only in GSC", count=int(only_g))
    except Exception as e:
        vc.add_exc("preview_msid_consistency", e)
    return vc

st.subheader("Data Validation Report")
vc0 = run_validation_pipeline(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)
rep_df = vc0.to_dataframe()
if rep_df.empty:
    st.success("No issues detected in preliminary checks ✅")
else:
    tabs = st.tabs(["Critical","Warning","Info"])
    for i, cat in enumerate(["Critical","Warning","Info"]):
        with tabs[i]:
            sub = rep_df[rep_df["category"]==cat]
            st.dataframe(sub if not sub.empty else pd.DataFrame({"message":["None"]}), use_container_width=True, hide_index=True)
    st.caption(f"Data Quality Score (pre-processing): **{vc0.quality_score():.0f} / 100**")
    st.download_button("Download Validation Report (CSV)", data=rep_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ---- Robust processing function (drop-in replacement) ----
@st.cache_data(show_spinner=False)
def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                           vc_serialized: Optional[str]=None,
                           merge_strategy: Optional[Dict[str,str]]=None):
    vc = ValidationCollector()
    if vc_serialized:
        try:
            for item in json.loads(vc_serialized):
                ctx = item.get("context")
                ctx = json.loads(ctx) if isinstance(ctx, str) else (ctx or {})
                vc.add(item["category"], item["code"], item["message"], **ctx)
        except Exception:
            pass
    ms = merge_strategy or MERGE_STRATEGY

    prod_df = prod_df_raw.copy(); ga4_df = ga4_df_raw.copy(); gsc_df = gsc_df_raw.copy()
    if prod_map.get("msid") not in prod_df.columns:
        vc.add("Critical","MISSING_KEY","Production missing MSID column", want=prod_map.get("msid")); return None, vc
    prod_df.rename(columns={prod_map["msid"]: "msid"}, inplace=True)
    if prod_map.get("title") in prod_df.columns:   prod_df.rename(columns={prod_map["title"]: "Title"}, inplace=True)
    if prod_map.get("path") in prod_df.columns:    prod_df.rename(columns={prod_map["path"]: "Path"}, inplace=True)
    if prod_map.get("publish") in prod_df.columns: prod_df.rename(columns={prod_map["publish"]: "Publish Time"}, inplace=True)

    if ga4_map.get("msid") not in ga4_df.columns:
        vc.add("Critical","MISSING_KEY","GA4 missing MSID column", want=ga4_map.get("msid")); return None, vc
    ga4_df.rename(columns={ga4_map["msid"]: "msid"}, inplace=True)
    if ga4_map.get("date") in ga4_df.columns:      ga4_df.rename(columns={ga4_map["date"]: "date"}, inplace=True)
    if ga4_map.get("pageviews") in ga4_df.columns: ga4_df.rename(columns={ga4_map["pageviews"]: "screenPageViews"}, inplace=True)
    if ga4_map.get("users") in ga4_df.columns:     ga4_df.rename(columns={ga4_map["users"]: "totalUsers"}, inplace=True)
    if ga4_map.get("engagement") in ga4_df.columns:ga4_df.rename(columns={ga4_map["engagement"]: "userEngagementDuration"}, inplace=True)
    if ga4_map.get("bounce") in ga4_df.columns:    ga4_df.rename(columns={ga4_map["bounce"]: "bounceRate"}, inplace=True)

    # Inside process_uploaded_files
gsc_ren = {
    gsc_map["date"]: "date", gsc_map["page"]: "page_url", gsc_map["query"]: "Query",
    gsc_map["clicks"]: "Clicks", # ADD THIS LINE
    gsc_map["impr"]: "Impressions", gsc_map.get("ctr","CTR"): "CTR", gsc_map["pos"]: "Position"
}
    for k in list(gsc_ren.keys()):
        if k not in gsc_df.columns:
            vc.add("Critical","MISSING_COL",f"GSC missing required column '{k}'"); return None, vc
    gsc_df.rename(columns=gsc_ren, inplace=True)

    # Early date standardization
    prod_df, ga4_df, gsc_df = standardize_dates_early(prod_df, ga4_df, gsc_df,
                                                       {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}, vc)

    # Types
    for df, col in [(prod_df,"msid"),(ga4_df,"msid")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].isna().sum()
        if bad: vc.add("Warning","MSID_BAD","Non-numeric MSIDs dropped", rows=int(bad))
        df.dropna(subset=[col], inplace=True); df[col] = df[col].astype("int64")

    # msid from url
    def _extract_msid_from_url(url):
        if pd.isna(url): return None
        m = re.search(r"(\d+)\.cms", str(url)); return int(m.group(1)) if m else None
    if "msid" not in gsc_df.columns or gsc_df["msid"].isna().all():
        gsc_df["msid"] = gsc_df["page_url"].apply(_extract_msid_from_url)
        missing = gsc_df["msid"].isna().sum()
        if missing: vc.add("Warning","MSID_FROM_URL","Some GSC rows lacked MSID in URL", unresolved=int(missing))
        gsc_df.dropna(subset=["msid"], inplace=True); gsc_df["msid"] = gsc_df["msid"].astype("int64")

    # Numeric coercions
    gsc_df["Impressions"] = coerce_numeric(gsc_df["Impressions"], "GSC.Impressions", vc, clamp=(0, float("inf")))
    if "CTR" in gsc_df.columns:
        if gsc_df["CTR"].dtype == object:
            tmp = gsc_df["CTR"].astype(str).str.replace("%","", regex=False)
            ctr_val = pd.to_numeric(tmp, errors="coerce")
            if (ctr_val > 1.0).any():
                ctr_val = ctr_val / 100.0
                vc.add("Info","CTR_SCALE","CTR parsed as percentage (÷100)")
            gsc_df["CTR"] = ctr_val
        else:
            gsc_df["CTR"] = pd.to_numeric(gsc_df["CTR"], errors="coerce")
        out = ((gsc_df["CTR"] < 0) | (gsc_df["CTR"] > 1)).sum()
        if out: vc.add("Warning","CTR_CLAMP","CTR values clamped to [0,1]", rows=int(out))
        gsc_df["CTR"] = gsc_df["CTR"].clip(0,1)
    if "Position" in gsc_df.columns:
        gsc_df["Position"] = coerce_numeric(gsc_df["Position"], "GSC.Position", vc, clamp=(1, 100))

    # Categories from Path
    def parse_path(path_str):
        if not isinstance(path_str, str): return ("Uncategorized","Uncategorized")
        s = path_str.strip().strip("/")
        if not s: return ("Uncategorized","Uncategorized")
        parts = [p for p in s.split("/") if p]
        if len(parts)==1: return (parts[0],"General")
        return (parts[0], parts[1])
    if "Path" in prod_df.columns:
        cat_tuples = prod_df["Path"].apply(parse_path)
        prod_df[["L1_Category","L2_Category"]] = pd.DataFrame(cat_tuples.tolist(), index=prod_df.index)
    else:
        prod_df["L1_Category"]="Uncategorized"; prod_df["L2_Category"]="Uncategorized"

    # Cross-file sanity — overlap
    d_range = {}
    for tag, df in [("GA4", ga4_df), ("GSC", gsc_df)]:
        if "date" in df.columns:
            dd = pd.to_datetime(df["date"], errors="coerce")
            if dd.notna().any(): d_range[tag]=(dd.min(), dd.max())
    if len(d_range)==2:
        ga4_min, ga4_max = d_range["GA4"]; gsc_min, gsc_max = d_range["GSC"]
        latest_start = max(ga4_min, gsc_min); earliest_end = min(ga4_max, gsc_max)
        overlap_days = (earliest_end - latest_start).days if earliest_end>=latest_start else -1
        if overlap_days < 0: vc.add("Warning","DATE_NO_OVERLAP","GA4 and GSC dates do not overlap", ga4=str(d_range["GA4"]), gsc=str(d_range["GSC"]))
        elif overlap_days < 7: vc.add("Info","DATE_SMALL_OVERLAP",f"Small GA4↔GSC overlap (~{overlap_days} days)", overlap_days=int(overlap_days))

    # Merge GSC x PROD
    before_counts = {"prod": len(prod_df), "ga4": len(ga4_df), "gsc": len(gsc_df)}
    merged_1 = pd.merge(
        gsc_df,
        prod_df[["msid","Title","Path","Publish Time","L1_Category","L2_Category"]],
        on="msid", how=ms.get("gsc_x_prod","left"), validate="many_to_one"
    )
    vc.checkpoint("merge_gsc_prod", before=before_counts, after_m1=len(merged_1))

    # GA4 daily
    if "date" in ga4_df.columns:
        numeric_cols = ga4_df.select_dtypes(include=[np.number]).columns.tolist()
        ga4_daily = ga4_df.groupby(["msid","date"], as_index=False, dropna=False)[numeric_cols].sum(min_count=1)
    else:
        ga4_daily = ga4_df.copy(); ga4_daily["date"] = pd.NaT; vc.add("Info","GA4_NO_DATE","GA4 had no date; set NaT")

    master_df = pd.merge(merged_1, ga4_daily, on=["msid","date"], how=ms.get("ga4_align","left"))
    vc.checkpoint("merge_ga4", after_master=len(master_df))

    # Dedup
    subset_cols = [c for c in ["date","msid","Query"] if c in master_df.columns]
    if len(subset_cols)==3:
        dup_before = master_df.duplicated(subset=subset_cols).sum()
        if dup_before: vc.add("Info","DEDUP","Duplicate rows removed on (date, msid, Query)", count=int(dup_before))
        master_df = master_df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)

    # Impute
    if "CTR" in master_df.columns: master_df["CTR"] = master_df["CTR"].fillna(0.0)
    if "Position" in master_df.columns:
        miss = master_df["Position"].isna().sum()
        if miss: master_df["Position"] = master_df["Position"].fillna(50.0); vc.add("Info","POSITION_IMPUTE","Missing Position imputed to 50.0", rows=int(miss))
    if "Title" in master_df.columns:
        drop_title_n = master_df["Title"].isna().sum()
        if drop_title_n: vc.add("Warning","TITLE_MISSING","Rows lacking Title dropped", rows=int(drop_title_n))
        master_df = master_df.dropna(subset=["Title"])

    master_df["_lineage"] = "GSC→PROD→GA4"
    return master_df, vc

# Run processing
vc_serialized = rep_df.to_json(orient="records")
with st.spinner("Processing & merging with robust validation..."):
    master_df, vc_after = process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                                                 vc_serialized=vc_serialized, merge_strategy=MERGE_STRATEGY)
if master_df is None or master_df.empty:
    st.error("Processing aborted due to critical issues. See reports above.")
    if vc_after: st.dataframe(vc_after.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Post-merge report
st.subheader("Post-merge Data Quality")
post_df = vc_after.to_dataframe()
if not post_df.empty:
    st.dataframe(post_df, use_container_width=True, hide_index=True)
    st.caption(f"Data Quality Score (post-merge): **{vc_after.quality_score():.0f} / 100**")
    st.download_button("Download Post-merge Report (CSV)", data=post_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"postmerge_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# Master preview + sampling
st.success(f"Master Data: {master_df.shape[0]:,} rows × {master_df.shape[1]} cols")
if "date" in master_df.columns:
    try:
        fresh = pd.to_datetime(master_df["date"], errors="coerce").max()
        if pd.notna(fresh): st.caption(f"Latest date in data: **{pd.to_datetime(fresh).date()}**")
    except Exception:
        pass
if master_df.shape[0] > CONFIG["performance"]["sample_row_limit"]:
    st.info(f"Large dataset. Sampling {CONFIG['performance']['sample_row_limit']:,} rows for interactive analysis.")
    master_df = master_df.sample(CONFIG["performance"]["sample_row_limit"], random_state=CONFIG["performance"]["seed"])
st.dataframe(master_df.head(10), use_container_width=True, hide_index=True)

st.markdown("### Processing Logs & Debug")
if PROCESS_LOG:
    logs_df = pd.DataFrame(PROCESS_LOG)
    st.dataframe(logs_df.tail(200), use_container_width=True, hide_index=True)
    st.download_button("Download Logs (CSV)", data=logs_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"processing_logs_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
else:
    st.caption("No detailed logs captured yet.")

if step == "3) Validate & Process":
    st.success("Data looks good. Move to **Step 4) Configure & Analyze** to run insights.")
    st.stop()

# ============================
# PART 4/5: Filters, Helpers, Core Insight Modules & Charts
# ============================
# Date filter
def filter_by_date(df, start_date, end_date):
    if "date" not in df.columns: return df
    m = df.copy(); m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.date
    mask = (m["date"] >= start_date) & (m["date"] <= end_date)
    return m[mask].copy()
filtered_df = filter_by_date(master_df, *st.session_state.date_range)

TH = st.session_state.thresholds
EXPECTED_CTR = CONFIG["expected_ctr_by_rank"]

# --- Module 1: Striking Distance ---
def find_striking_distance_keywords(df):
    d = df.copy()
    pos_col = next((c for c in ["Position","gsc_avg_position","position","avg_position"] if c in d.columns), None)
    impr_col = next((c for c in ["Impressions","gsc_impressions","impressions"] if c in d.columns), None)
    title_col = next((c for c in ["Title","title","headline"] if c in d.columns), None)
    if not pos_col or not impr_col: return ["**No Striking Distance Insights** — required columns not found."], pd.DataFrame()
    d[pos_col] = pd.to_numeric(d[pos_col], errors="coerce")
    d[impr_col] = pd.to_numeric(d[impr_col], errors="coerce").fillna(0)
    sd = d[(d[pos_col] >= TH["striking_distance_min"]) & (d[pos_col] <= TH["striking_distance_max"]) & (d[impr_col] >= TH["min_impressions"])].copy()
    if sd.empty: return ["No items currently inside configured striking-distance range."], pd.DataFrame()
    if "msid" not in sd.columns: return ["**Consolidation unavailable** — `msid` column not found."], pd.DataFrame()
    if not title_col: title_col="Title_fallback"; sd[title_col]="Untitled"
    article_agg = (sd.groupby(["msid", title_col], as_index=False).agg(total_impr=(impr_col, "sum"), best_pos=(pos_col, "min")))
    article_agg.sort_values(by=["total_impr"], ascending=False, inplace=True)
    out = article_agg.rename(columns={"msid": "MSID", title_col: "Title", "total_impr": "Total Impressions", "best_pos": "Best Avg Position"})
    out = out[["MSID","Title","Best Avg Position","Total Impressions"]]
    return [], out

# --- Module 2: Low CTR ---
def find_low_ctr_opportunities(df):
    d = df.copy()
    pos_col = next((c for c in ["Position","gsc_avg_position"] if c in d.columns), None)
    ctr_col = next((c for c in ["CTR","gsc_avg_ctr"] if c in d.columns), None)
    impr_col = next((c for c in ["Impressions","gsc_impressions"] if c in d.columns), None)
    title_col = next((c for c in ["Title","title","headline"] if c in d.columns), None)
    if not all([pos_col, ctr_col, impr_col]): return ["**Low CTR** needs position, CTR, impressions."], pd.DataFrame()
    d[pos_col] = pd.to_numeric(d[pos_col], errors="coerce")
    d[impr_col] = pd.to_numeric(d[impr_col], errors="coerce").fillna(0)
    if d[ctr_col].dtype == object:
        d[ctr_col] = pd.to_numeric(d[ctr_col].astype(str).str.replace("%","", regex=False), errors="coerce")/100.0
    else:
        d[ctr_col] = pd.to_numeric(d[ctr_col], errors="coerce")
    page1 = d[(d[pos_col] > 0) & (d[pos_col] < 10) & (d[impr_col] >= TH["min_impressions"])].copy()
    if page1.empty: return ["No page-1 items with sufficient impressions."], pd.DataFrame()
    page1['expected_ctr'] = page1[pos_col].round().clip(1, 9).map(EXPECTED_CTR)
    page1['ctr_deficit'] = (page1['expected_ctr'] - page1[ctr_col]).clip(lower=0)
    page1 = page1[page1['ctr_deficit'] >= (TH["ctr_deficit_pct"]/100.0)]
    if page1.empty: return ["No items below the configured CTR expectation."], pd.DataFrame()
    page1['opp_score'] = page1['ctr_deficit'] * page1[impr_col]
    res = page1.sort_values(by='opp_score', ascending=False)
    result_df = res[['msid', title_col, pos_col, ctr_col, impr_col, 'expected_ctr', 'ctr_deficit']].copy()
    result_df.rename(columns={'msid':'MSID', title_col:'Title', pos_col:'Best Avg Position', ctr_col:'CTR', impr_col:'Total Impressions', 'expected_ctr':'Expected CTR', 'ctr_deficit':'CTR Deficit'}, inplace=True)
    return [], result_df

# --- Module 3: Cannibalization (English Titles) ---
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    if not _HAS_ST: return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"SentenceTransformer load failed: {e}"); return None

def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms
def _cosine_sim_matrix(X):
    Xn = _normalize(X); return np.clip(Xn @ Xn.T, -1, 1)
def _tfidf_embed(texts, max_features=5000, ngram_range=(1,2)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    M = vec.fit_transform(texts); return M.toarray(), vec
def _cluster_by_threshold(sim, threshold):
    n = sim.shape[0]; parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] >= threshold: union(i,j)
    root_to_id, cur = {}, 0; labels=[-1]*n
    for i in range(n):
        r=find(i)
        if r not in root_to_id: root_to_id[r]=cur; cur+=1
        labels[i]=root_to_id[r]
    counts = pd.Series(labels).value_counts()
    valid = set(counts[counts>1].index.tolist())
    labels = [lbl if lbl in valid else -1 for lbl in labels]
    return labels
def _extract_keywords_topk(tfidf_vec, vocab, k=5):
    idxs = np.argsort(-tfidf_vec)[:k]; return [vocab[i] for i in idxs if tfidf_vec[i] > 0]
def _keyword_overlap(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a), set(b); inter=len(sa&sb); uni=len(sa|sb)
    return inter/uni if uni else 0.0

def english_cannibalization(df):
    d = df.copy()
    if "Title" not in d.columns or "msid" not in d.columns:
        return ["Analysis unavailable: missing Title or msid."], pd.DataFrame(), pd.DataFrame()
    d["SEO_Title"] = d["Title"].astype(str).fillna("")
    articles = d.groupby("msid").agg(SEO_Title=("SEO_Title","first"), Title=("Title","first"), L2_Category=("L2_Category","first"),
                                     Impressions=("Impressions","sum"), CTR=("CTR","mean"), Position=("Position","mean")).reset_index()
    st_model = load_sentence_transformer()
    if st_model is not None:
        try:
            emb = st_model.encode(articles["SEO_Title"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            sim = (emb @ emb.T)
        except Exception as e:
            logger.warning(f"ST encoding failed, falling back TF-IDF: {e}")
            emb=None
    else:
        emb=None
    Xtfidf, vec = _tfidf_embed(articles["SEO_Title"].tolist())
    vocab = np.array(vec.get_feature_names_out())
    tfidf_mat = Xtfidf
    if emb is None:
        sim = _cosine_sim_matrix(tfidf_mat)
    labels = _cluster_by_threshold(sim, threshold=TH["similarity_threshold"])
    # ... inside english_cannibalization after labels are calculated
articles["similarity_group"] = labels
cann = articles[articles["similarity_group"] != -1].copy()
if cann.empty: return ["No competing content clusters found at current threshold."], pd.DataFrame(), pd.DataFrame()
topk_list = [_extract_keywords_topk(tfidf_mat[i], vocab, k=6) for i in range(tfidf_mat.shape[0])]
articles["top_keywords"] = topk_list

# OLD problematic merge
# detailed = cann.merge(d[["msid","Title","L2_Category","Impressions","CTR","Position","Publish Time"]],
#                      on=["msid","Title","L2_Category"], how="left")

# NEW safer merge
detailed = articles[articles["similarity_group"] != -1].copy()
detailed.rename(columns={"msid":"MSID"}, inplace=True)
    theme_rows=[]
    for gid, sub in articles[articles["similarity_group"]!=-1].groupby("similarity_group"):
        tokens = " ".join(sub["SEO_Title"].tolist()).lower().split()
        tok_series = pd.Series([t for t in tokens if len(t)>3])
        theme = ", ".join(tok_series.value_counts().head(5).index.tolist())
        theme_rows.append({"similarity_group": gid, "Theme": theme})
    theme_df = pd.DataFrame(theme_rows)
    detailed = detailed.merge(theme_df, on="similarity_group", how="left")
    summary = (detailed.groupby("L2_Category").agg(Cannibalization_Count=("MSID","nunique")).reset_index().sort_values("Cannibalization_Count", ascending=False))
    group_conf=[]
    for gid, sub in articles[articles["similarity_group"]!=-1].groupby("similarity_group"):
        idxs = sub.index.tolist()
        if len(idxs) < 2: continue
        pairs=[]
        for i in range(len(idxs)):
            for j in range(i+1,len(idxs)):
                si, sj = idxs[i], idxs[j]
                sim_ij = sim[si,sj]
                ko = _keyword_overlap(articles.loc[si,"top_keywords"], articles.loc[sj,"top_keywords"])
                pairs.append((sim_ij, ko))
        sim_mean = float(np.mean([p[0] for p in pairs])) if pairs else 0.0
        ko_mean  = float(np.mean([p[1] for p in pairs])) if pairs else 0.0
        conf = 0.7*sim_mean + 0.3*ko_mean
        group_conf.append({"similarity_group": gid, "Cluster Confidence": round(conf,3)})
    group_conf_df = pd.DataFrame(group_conf)
    detailed = detailed.merge(group_conf_df, on="similarity_group", how="left")
    keep = ["MSID","Title","L2_Category","Theme","Position","CTR","Impressions","Publish Time","similarity_group","Cluster Confidence"]
    detailed = detailed[keep].drop_duplicates()
    return [], summary, detailed

# --- Module 4: Engagement vs Search Mismatch ---
def engagement_mismatches(df):
    d = df.copy()
    engagement_duration_col = next((c for c in ["userEngagementDuration","engagement_duration"] if c in d.columns), None)
    bounce_rate_col = next((c for c in ["bounceRate","bounce_rate"] if c in d.columns), None)
    clicks_col = next((c for c in ["Clicks","gsc_clicks"] if c in d.columns), None)
    position_col = next((c for c in ["Position","gsc_avg_position"] if c in d.columns), None)
    title_col = next((c for c in ["Title","title","headline"] if c in d.columns), "Title")
    if not all([engagement_duration_col, bounce_rate_col, clicks_col, position_col]):
        return ["**Engagement Mismatch analysis unavailable** — missing required columns."]
    for c in [engagement_duration_col, bounce_rate_col, clicks_col, position_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.dropna(subset=[engagement_duration_col, bounce_rate_col, position_col], inplace=True)
    d = d[d[position_col] <= 50].copy()
    if d.empty: return ["No articles with complete data for mismatch analysis."]
    d["engagement_score"] = (d[engagement_duration_col].rank(pct=True) + (1 - d[bounce_rate_col].rank(pct=True))) / 2
    d["search_score"] = (d[clicks_col].rank(pct=True) + (1 - d[position_col].rank(pct=True))) / 2
    d["mismatch_score"] = (d["engagement_score"] - d["search_score"])
    d["mismatch_type"] = np.where((d["engagement_score"] > 0.8) & (d["search_score"] < 0.2), "Hidden Gem",
                           np.where((d["search_score"] > 0.8) & (d["engagement_score"] < 0.2), "Clickbait Risk", None))
    mismatches = d.dropna(subset=["mismatch_type"]).sort_values(by=abs(d["mismatch_score"]), ascending=False).head(8)
    cards=[]
    for _, row in mismatches.iterrows():
        emoji = "💎" if row["mismatch_type"]=="Hidden Gem" else "⚠️"
        recs = (["- **SEO Optimization:** Expand title/H1 & match intent.","- **Internal Linking:** Add links from high-authority pages.","- **Content Expansion:** Add related sections."]
                if row["mismatch_type"]=="Hidden Gem" else
                ["- **Content Depth:** Address thin content.","- **UX:** Improve speed & mobile.","- **Title Alignment:** Ensure promise matches content."])
        goal = (f"_Goal: Leverage high engagement to improve search visibility from position {row[position_col]:.1f}._"
                if row["mismatch_type"]=="Hidden Gem" else f"_Goal: Improve user experience to match search performance (clicks: {int(row[clicks_col]):,})._")
        with st.expander(f"Why this recommendation? — {emoji} {row.get(title_col,'Untitled')[:90]}", expanded=False):
            st.write(f"- **Engagement Score**: {row['engagement_score']:.2f}, **Search Score**: {row['search_score']:.2f}")
            st.write(f"- **Duration**: {row[engagement_duration_col]:.1f}s, **Bounce**: {row[bounce_rate_col]:.1%}, **Pos**: {row[position_col]:.1f}")
            st.write(f"- **Confidence**: {min(0.95, 0.5+abs(row['mismatch_score'])*0.8):.2f}")
        cards.append(
            f"### {emoji} {row['mismatch_type']}\n"
            f"**MSID:** `{row.get('msid', '—')}`\n"
            f"**Title:** {row.get(title_col,'Untitled')}\n"
            f"**Engagement Score:** **{row['engagement_score']:.2f}** | **Search Score:** **{row['search_score']:.2f}**\n"
            f"**Metrics:** Duration: **{row[engagement_duration_col]:.1f}s** | Bounce: **{row[bounce_rate_col]:.1%}** | Pos: **{row[position_col]:.1f}**\n\n"
            f"**Recommendations:**\n" + "\n".join(recs) + f"\n\n{goal}"
        )
    return cards if cards else ["No significant mismatches detected."]

# --- Charts & helpers ---
def export_plot_html(fig, name):
    if to_html is None:
        st.info("Plotly HTML export not available in this environment."); return
    html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
    st.download_button("Export Chart (HTML)", data=html_str.encode("utf-8"), file_name=f"{name}.html", mime="text/html")

def scatter_engagement_vs_search(df):
    if px is None:
        st.info("Plotly not available for interactive charts."); return
    dur_col = next((c for c in ["userEngagementDuration","engagement_duration"] if c in df.columns), None)
    br_col  = next((c for c in ["bounceRate","bounce_rate"] if c in df.columns), None)
    clicks_col = next((c for c in ["Clicks","gsc_clicks"] if c in df.columns), None)
    pos_col = next((c for c in ["Position","gsc_avg_position"] if c in df.columns), None)
    if not all([dur_col, br_col, clicks_col, pos_col]):
        st.info("Insufficient columns for engagement vs search scatter."); return
    t = df.copy()
    for c in [dur_col, br_col, clicks_col, pos_col]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["engagement_score"] = (t[dur_col].rank(pct=True) + (1 - t[br_col].rank(pct=True))) / 2
    t["search_score"] = (t[clicks_col].rank(pct=True) + (1 - t[pos_col].rank(pct=True))) / 2
    t["L2_Category"] = t.get("L2_Category","Uncategorized")
    agg = t.groupby(["msid","Title","L2_Category"], as_index=False).agg(engagement_score=("engagement_score","mean"),
                                                                       search_score=("search_score","mean"),
                                                                       Position=(pos_col,"mean"),
                                                                       Clicks=(clicks_col,"sum"))
    fig = px.scatter(agg, x="engagement_score", y="search_score", size="Clicks", color="L2_Category",
                     hover_data=["msid","Title","Position","Clicks"], title="Engagement vs Search Performance (Interactive)")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit"); export_plot_html(fig, "engagement_vs_search")

# ============================
# PART 5/5: Category Performance, Trends/Forecast, ROI, Run Modules, Summary
# ============================
def category_heatmap(df, value_col, title):
    if px is None:
        st.info("Plotly not available for heatmaps."); return
    t = df.copy()
    if "L1_Category" not in t.columns: t["L1_Category"] = "Uncategorized"
    if "L2_Category" not in t.columns: t["L2_Category"] = "Uncategorized"
    agg = t.groupby(["L1_Category","L2_Category"]).agg(val=(value_col,"sum")).reset_index()
    fig = px.density_heatmap(agg, x="L1_Category", y="L2_Category", z="val", color_continuous_scale="Viridis",
                             title=title, histfunc="sum")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit"); export_plot_html(fig, f"heatmap_{value_col}")

def analyze_category_performance(df):
    d = df.copy()
    if "L1_Category" not in d.columns: d["L1_Category"]="Uncategorized"
    if "L2_Category" not in d.columns: d["L2_Category"]="Uncategorized"
    if "msid" not in d.columns: d["msid"]=range(len(d))
    engagement_col = next((c for c in ["userEngagementDuration","engagement_duration"] if c in d.columns), None)
    pageviews_col  = next((c for c in ["screenPageViews","pageviews"] if c in d.columns), None)
    users_col      = next((c for c in ["totalUsers","users"] if c in d.columns), None)
    clicks_col     = next((c for c in ["Clicks","gsc_clicks"] if c in d.columns), None)
    traffic_metric_col = pageviews_col or users_col
    if not any([engagement_col, traffic_metric_col, clicks_col]):
        return pd.DataFrame({"error":["Category analysis requires traffic, engagement, or clicks."]})
    for c in [engagement_col, pageviews_col, users_col, clicks_col]:
        if c: d[c] = pd.to_numeric(d[c], errors="coerce")
    agg_dict = {"msid":"nunique"}
    if traffic_metric_col: agg_dict[traffic_metric_col] = "sum"
    if engagement_col: agg_dict[engagement_col] = "mean"
    if clicks_col: agg_dict[clicks_col] = "sum"
    grouped = (d.groupby(["L1_Category","L2_Category"]).agg(agg_dict)
               .rename(columns={"msid":"total_articles",
                                traffic_metric_col:"total_traffic" if traffic_metric_col else None,
                                engagement_col:"avg_engagement_duration" if engagement_col else None,
                                clicks_col:"total_gsc_clicks" if clicks_col else None})
               .reset_index())
    site_avg_traffic = grouped["total_traffic"].mean() if "total_traffic" in grouped.columns else 0
    site_avg_eng = grouped["avg_engagement_duration"].mean() if "avg_engagement_duration" in grouped.columns else 0
    grouped["traffic_index"] = (grouped.get("total_traffic", 0) / site_avg_traffic).fillna(0) if site_avg_traffic>0 else 0
    grouped["engagement_index"] = (grouped.get("avg_engagement_duration", 0) / site_avg_eng).fillna(0) if site_avg_eng>0 else 0
    def quadrant(row):
        ht = row["traffic_index"] >= 1.0; he = row["engagement_index"] >= 1.0
        if ht and he: return "Stars"
        if (not ht) and he: return "Hidden Gems"
        if ht and (not he): return "Workhorses"
        return "Underperformers"
    if "traffic_index" in grouped.columns and "engagement_index" in grouped.columns:
        grouped["quadrant"] = grouped.apply(quadrant, axis=1)
    else:
        grouped["quadrant"] = "N/A"
    return grouped

def time_series_trends(df, metric_col, title):
    if px is None:
        st.info("Plotly not available for time-series charts."); return
    t = df.copy()
    if "date" not in t.columns:
        st.info("No date column for time series."); return
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    agg = t.groupby("date")[metric_col].sum().reset_index()
    fig = px.line(agg, x="date", y=metric_col, title=title)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit"); export_plot_html(fig, f"time_series_{metric_col}")

def forecast_series(daily_series, periods=14):
    daily_series = daily_series.asfreq("D").fillna(method="ffill")
    if _HAS_STM and len(daily_series) >= 14:
        try:
            model = ExponentialSmoothing(daily_series, trend="add", seasonal="add", seasonal_periods=7)
            fit = model.fit(optimized=True)
            fc = fit.forecast(periods)
            resid = daily_series - fit.fittedvalues.reindex(daily_series.index, fill_value=np.nan)
            s = float(resid.std())
            lower = fc - 1.96*s; upper = fc + 1.96*s
            return pd.DataFrame({"date": fc.index, "forecast": fc.values, "low": lower.values, "high": upper.values})
        except Exception as e:
            logger.warning(f"Forecasting failed, fallback: {e}")
    roll = daily_series.rolling(7, min_periods=1).mean()
    last = float(roll.iloc[-1]) if not roll.empty else 0.0
    idx = pd.date_range(daily_series.index.max()+pd.Timedelta(days=1), periods=periods, freq="D")
    fc = pd.Series([last]*periods, index=idx)
    return pd.DataFrame({"date": fc.index, "forecast": fc.values, "low": fc.values*0.9, "high": fc.values*1.1})

def what_if_ctr_gain(impr, ctr, ctr_increase_pct):
    ctr_new = ctr * (1 + ctr_increase_pct/100.0)
    return impr * (ctr_new - ctr)
def expected_ctr_from_rank(rank_rounded):
    return EXPECTED_CTR.get(int(np.clip(int(round(rank_rounded)), 1, 9)), 0.03)
def what_if_position_improvement(impr, current_pos, new_pos):
    base_ctr = expected_ctr_from_rank(round(current_pos))
    new_ctr  = expected_ctr_from_rank(round(new_pos))
    return impr * (new_ctr - base_ctr)

# ---- Fail-safe wrapper ----
def run_module_safely(label: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.warning(f"Module '{label}' encountered an issue and was skipped. Details: {type(e).__name__}: {e}")
        logger.exception(f"[ModuleFail] {label}: {e}")
        return None

# ================= UI: Insights =================
st.header("Insights")

# 1) Striking Distance
st.subheader("Module 1: Striking Distance Keywords")
out = run_module_safely("Striking Distance", find_striking_distance_keywords, filtered_df)
if out is not None:
    err, sd_df = out
    if not sd_df.empty:
        top = sd_df.head(10).copy()
        top["Best Avg Position"] = top["Best Avg Position"].round(1)
        top["Total Impressions"] = top["Total Impressions"].astype(int).map(lambda x: f"{x:,}")
        st.dataframe(top, use_container_width=True, hide_index=True)
        st.download_button(label=f"Download All ({len(sd_df)})",
                           data=sd_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"striking_distance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        st.markdown("**Recommendations & Explanations (Top 5)**")
        show5 = sd_df.head(5)
        for _, r in show5.iterrows():
            msid = r["MSID"]
                with st.expander(f"Why this recommendation? — [{msid}] {r['Title'][:90]}"):
                st.write(f"- **Best Avg Position**: {r['Best Avg Position']:.1f}")
                st.write(f"- **Impressions**: {int(float(r['Total Impressions'])):,}") # Corrected line
                st.write(f"- **Site Avg CTR by rank**: {EXPECTED_CTR.get(int(round(r['Best Avg Position'])), 0.03):.2%}")
                imp_num = int(float(r['Total Impressions'])) # Corrected line
                ctr_gain = st.slider(f"CTR improvement (%) for MSID {msid}", 1, 20, 5, key=f"sd_ctr_{msid}")
                delta_clicks = what_if_ctr_gain(imp_num, expected_ctr_from_rank(r["Best Avg Position"]), ctr_gain)
                st.write(f"- Estimated extra clicks: **{int(delta_clicks):,}**")
    else:
        for e in (err or []): st.info(e)
st.divider()

# 2) Low CTR
st.subheader("Module 2: Low CTR Opportunities")
out2 = run_module_safely("Low CTR", find_low_ctr_opportunities, filtered_df)
if out2 is not None:
    err, low_df = out2
    if not low_df.empty:
        t10 = low_df.head(10).copy()
        t10['Best Avg Position'] = t10['Best Avg Position'].map('{:.1f}'.format)
        t10['CTR'] = t10['CTR'].map('{:.2%}'.format)
        t10['Expected CTR'] = t10['Expected CTR'].map('{:.2%}'.format)
        t10['CTR Deficit'] = t10['CTR Deficit'].map('{:.2%}'.format)
        t10['Total Impressions'] = t10['Total Impressions'].map('{:,.0f}'.format)
        st.dataframe(t10, use_container_width=True, hide_index=True)
        st.download_button(label=f"Download All ({len(low_df)})", data=low_df.to_csv(index=False),
                           file_name=f"low_ctr_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
        st.markdown("**Why these items? (Explainable AI)**")
        for _, row in low_df.head(5).iterrows():
            with st.expander(f"Why this? — [{int(row['MSID'])}] {row['Title'][:90]}"):
                st.write(f"- **Position**: {row['Best Avg Position']:.1f}, **CTR**: {row['CTR']:.2%}, **Expected**: {row['Expected CTR']:.2%}")
                st.write(f"- **Deficit**: {row['CTR Deficit']:.2%}, **Impressions**: {int(row['Total Impressions']):,}")
                conf = min(0.98, 0.4 + (float(row['CTR Deficit'])*100)/10.0)
                st.write(f"- **Confidence**: {conf:.2f}")
                ctr_add = st.slider(f"CTR lift (%) for MSID {int(row['MSID'])}", 1, 20, 5, key=f"low_ctr_{int(row['MSID'])}")
                delta_clicks = what_if_ctr_gain(float(row['Total Impressions']), float(str(row['CTR']).replace('%','')) if isinstance(row['CTR'], str) else float(row['CTR']), ctr_add)
                st.write(f"- Extra clicks if CTR +{ctr_add}%: **{int(what_if_ctr_gain(float(str(row['Total Impressions']).replace(',','')), float(str(row['CTR']).replace('%','')) if isinstance(row['CTR'], str) else float(row['CTR']), ctr_add)):,}**")
    else:
        for e in (err or []): st.info(e)
st.divider()

# 3) Cannibalization
st.subheader("Module 3: Content Cannibalization — English SEO Titles")
with st.spinner("Clustering SEO titles & computing semantic similarity..."):
    errs, summary_df, detailed_df = run_module_safely("Cannibalization", english_cannibalization, filtered_df) or (["Cannibalization failed."], pd.DataFrame(), pd.DataFrame())
if errs:
    for e in errs: st.info(e)
else:
    st.subheader("Categories with Competing Articles")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button("Download Detailed Cannibalization CSV", detailed_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"cannibalization_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    st.divider()
    cats_with_issues = summary_df[summary_df['Cannibalization_Count'] > 1]['L2_Category'].unique().tolist()
    if cats_with_issues:
        tabs = st.tabs(cats_with_issues)
        for i, cat in enumerate(cats_with_issues):
            with tabs[i]:
                subset = detailed_df[detailed_df['L2_Category']==cat]
                for gid, group in subset.groupby("similarity_group"):
                    if len(group) < 2: continue
                    st.write(f"**Competition Group {gid}** — Theme: _{group['Theme'].iloc[0]}_ | Confidence: **{group['Cluster Confidence'].iloc[0]:.2f}**")
                    disp_cols = ["MSID","Title","Position","CTR","Impressions","Publish Time"]
                    disp = group[disp_cols].copy()
                    if "Position" in disp.columns: disp["Position"] = disp["Position"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                    if "CTR" in disp.columns:      disp["CTR"]      = disp["CTR"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                    if "Impressions" in disp.columns: disp["Impressions"] = disp["Impressions"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                    with st.expander("Why this recommendation?"):
                        st.write("- High title similarity and keyword overlap suggests internal competition for similar queries.")
                        st.write("- Consider consolidation/redirects, canonicalization, or differentiating angles & intents.")
                    st.markdown("---")
    else:
        st.info("No categories with multiple competing articles at current thresholds.")
st.divider()

# 4) Engagement vs Search
st.subheader("Module 4: Engagement vs Search Mismatch")
cards = run_module_safely("Engagement vs Search", engagement_mismatches, filtered_df) or []
for c in cards: st.markdown(c)
scatter_engagement_vs_search(filtered_df)
st.divider()

# 5) Category Performance
st.subheader("Module 5: Category Performance")
cat_df = run_module_safely("Category Performance", analyze_category_performance, filtered_df) or pd.DataFrame()
if cat_df.empty or ("error" in cat_df.columns):
    st.info("Category Performance could not be computed.")
else:
    st.dataframe(cat_df, use_container_width=True, hide_index=True)
    if px is not None and "total_traffic" in cat_df.columns and cat_df["total_traffic"].sum() > 0:
        st.subheader("Interactive Heatmap — Category Traffic")
        category_heatmap(filtered_df, "screenPageViews" if "screenPageViews" in filtered_df.columns else "totalUsers", "Category Performance Matrix (Traffic)")
st.divider()

# 6) Trends & Forecasts (simplified)
st.subheader("Module 6: Trends & Forecasts")
if "date" in filtered_df.columns:
    filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")
    perf_col = "totalUsers" if "totalUsers" in filtered_df.columns else ("screenPageViews" if "screenPageViews" in filtered_df.columns else ("Clicks" if "Clicks" in filtered_df.columns else None))
    if perf_col:
        daily = filtered_df.groupby("date")[perf_col].sum().asfreq("D").fillna(0)
        fc = forecast_series(daily, periods=14)
        if px is not None and go is not None:
            base = daily.reset_index().rename(columns={"index":"date", perf_col:"value"})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base["date"], y=base["value"], name="History", mode="lines"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", mode="lines"))
            fig.add_trace(go.Scatter(x=pd.concat([fc["date"], fc["date"][::-1]]),
                                     y=pd.concat([fc["high"], fc["low"][::-1]]),
                                     fill="toself", fillcolor="rgba(0,0,0,0.1)", line=dict(width=0), name="Confidence"))
            fig.update_layout(title="Overall Traffic Forecast (next 14 days)")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit"); export_plot_html(fig, "forecast_overall")
        time_series_trends(filtered_df, {"Position":"Position","CTR":"CTR","Impressions":"Impressions"}.get("Impressions" if "Impressions" in filtered_df.columns else "CTR", "Position"),
                           "Key Metric Over Time")
    else:
        st.info("Need Users/Pageviews/Clicks for forecasting.")
else:
    st.info("No date column for forecasting.")
st.divider()

# 7) ROI & What-ifs
st.subheader("Module 7: What-if & ROI Estimators")
c_col1, c_col2 = st.columns(2)
with c_col1:
    st.write("**CTR Improvement Impact**")
    impr = st.number_input("Impressions", min_value=0, value=10000, step=1000)
    ctr  = st.number_input("Current CTR (%)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
    lift = st.slider("CTR Increase (%)", 0, 50, 5)
    gain = what_if_ctr_gain(impr, ctr/100.0, lift)
    st.write(f"Estimated extra clicks: **{int(gain):,}**")
with c_col2:
    st.write("**Position Improvement Impact**")
    impr2  = st.number_input("Impressions (pos what-if)", min_value=0, value=20000, step=1000)
    pos_now= st.number_input("Current Avg Position", min_value=1.0, max_value=50.0, value=12.0, step=0.1)
    pos_new= st.number_input("New Avg Position (target)", min_value=1.0, max_value=50.0, value=8.0, step=0.1)
    pgain  = what_if_position_improvement(impr2, pos_now, pos_new)
    st.write(f"Estimated extra clicks: **{int(pgain):,}**")
st.divider()

# 8) Executive Summary Export
st.subheader("Executive Summary Export")
summary_sections = {
    "period": f"{st.session_state.date_range[0]} → {st.session_state.date_range[1]}",
    "striking_distance_top5": (out[1].head(5).to_dict(orient="records") if out and isinstance(out, tuple) and not out[1].empty else []),
    "low_ctr_top5": (out2[1].head(5).to_dict(orient="records") if out2 and isinstance(out2, tuple) and not out2[1].empty else []),
}
summary_json = json.dumps(summary_sections, indent=2, default=str)
st.download_button("Download Executive Summary (JSON)", data=summary_json.encode("utf-8"),
                   file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.json", mime="application/json")

st.caption("Robust validation, standardized dates, merge stats, fail-safe modules, and explainable recommendations are enabled.")