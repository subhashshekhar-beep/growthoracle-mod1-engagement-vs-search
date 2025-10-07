# app.py — GrowthOracle (Module 1 only)
# Works on Streamlit Cloud. Use Secrets UI for keys.
# Requires (add to requirements.txt): streamlit pandas numpy pyyaml plotly openai google-genai (or google-generativeai)

import os, re, sys, json, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional YAML
try:
    import yaml
except Exception:
    yaml = None

# Try AI SDKs (feature-detect)
_HAS_OPENAI_CLIENT = False
_HAS_OPENAI = False
try:
    from openai import OpenAI  # modern SDK
    _HAS_OPENAI_CLIENT = True
    _HAS_OPENAI = True
except Exception:
    try:
        import openai  # legacy if present
        _HAS_OPENAI = True
    except Exception:
        pass

_HAS_GOOGLE_GENAI_NEW = False  # google-genai (new)
_HAS_GOOGLE_GENAI_OLD = False  # google-generativeai (legacy)
try:
    from google import genai as _genai_new
    from google.genai import types as _genai_types
    _HAS_GOOGLE_GENAI_NEW = True
except Exception:
    pass
if not _HAS_GOOGLE_GENAI_NEW:
    try:
        import google.generativeai as _genai_old
        _HAS_GOOGLE_GENAI_OLD = True
    except Exception:
        pass

# ---- Page ----
st.set_page_config(
    page_title="GrowthOracle — Module 1: Engagement vs Search",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)
st.title("GrowthOracle — Module 1: Engagement vs Search")
st.caption("Identify hidden gems, low CTR at good ranks, and high-bounce pages — with CSV exports")

# ---- Logger ----
@st.cache_resource
def get_logger(level=logging.INFO):
    logger = logging.getLogger("growthoracle_mod1")
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
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
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

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def quality_score(self) -> float:
        crit = sum(1 for m in self.messages if m.category == "Critical")
        warn = sum(1 for m in self.messages if m.category == "Warning")
        info = sum(1 for m in self.messages if m.category == "Info")
        score = 100 - (25 * crit + 8 * warn + 1 * info)
        return float(max(0, min(100, score)))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages:
            return pd.DataFrame(columns=["category", "code", "message", "context"])
        return pd.DataFrame([{
            "category": m.category,
            "code": m.code,
            "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])

# ---- UI helpers ----
def download_df_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.warning(f"No data to download for {label}")
        return
    try:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad = parsed.isna().sum()
    if bad > 0:
        vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def coerce_numeric(series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float, float]] = None) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad > 0:
        vc.add("Warning", "NUM_COERCE", f"Non-numeric values coerced to NaN in {name}", bad_rows=int(bad))
    if clamp and len(s) > 0:
        lo, hi = clamp
        s = s.clip(lower=lo, upper=hi) if hi is not None else s.clip(lower=lo)
    return s

# ---- CSV Readers ----
def read_csv_safely(upload, name: str, vc: ValidationCollector) -> Optional[pd.DataFrame]:
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided"); return None
    try_encodings = [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None
    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc) if enc else pd.read_csv(upload)
            if df.empty or df.shape[1] == 0:
                vc.add("Critical", "EMPTY_CSV", f"{name} appears empty"); return None
            return df
        except Exception as e:
            last_err = e
            continue
    vc.add("Critical", "CSV_ENCODING", f"Failed to read {name}", last_error=str(last_err))
    return None

# ---- Mapping ----
def _guess_colmap(prod_df, ga4_df, gsc_df):
    if prod_df is None or gsc_df is None:
        return {}, {}, {}
    prod_map = {
        "msid": "Msid" if "Msid" in prod_df.columns else next((c for c in prod_df.columns if c.lower()=="msid"), None),
        "title": "Title" if "Title" in prod_df.columns else next((c for c in prod_df.columns if "title" in c.lower()), None),
        "path": "Path" if "Path" in prod_df.columns else next((c for c in prod_df.columns if "path" in c.lower()), None),
        "publish": "Publish Time" if "Publish Time" in prod_df.columns else next((c for c in prod_df.columns if "publish" in c.lower()), None),
    }
    ga4_map = {}
    if ga4_df is not None and not ga4_df.empty:
        ga4_map = {
            "msid": "customEvent:msid" if "customEvent:msid" in ga4_df.columns else next((c for c in ga4_df.columns if "msid" in c.lower()), None),
            "date": "date" if "date" in ga4_df.columns else next((c for c in ga4_df.columns if c.lower()=="date"), None),
            "users": "totalUsers" if "totalUsers" in ga4_df.columns else next((c for c in ga4_df.columns if "users" in c.lower()), None),
            "engagement": "userEngagementDuration" if "userEngagementDuration" in ga4_df.columns else next((c for c in ga4_df.columns if "engagement" in c.lower()), None),
            "bounce": "bounceRate" if "bounceRate" in ga4_df.columns else next((c for c in ga4_df.columns if "bounce" in c.lower()), None),
        }
    gsc_map = {
        "date": "Date" if "Date" in gsc_df.columns else next((c for c in gsc_df.columns if c.lower()=="date"), None),
        "page": "Page" if "Page" in gsc_df.columns else next((c for c in gsc_df.columns if "page" in c.lower()), None),
        "query": "Query" if "Query" in gsc_df.columns else next((c for c in gsc_df.columns if "query" in c.lower()), None),
        "clicks": "Clicks" if "Clicks" in gsc_df.columns else next((c for c in gsc_df.columns if "clicks" in c.lower()), None),
        "impr": "Impressions" if "Impressions" in gsc_df.columns else next((c for c in gsc_df.columns if "impr" in c.lower()), None),
        "ctr": "CTR" if "CTR" in gsc_df.columns else next((c for c in gsc_df.columns if "ctr" in c.lower()), None),
        "pos": "Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
    }
    # Better publish-time guess
    if prod_df is not None and not prod_map.get("publish"):
        prod_dates = [c for c in detect_date_cols(prod_df) if "publish" in c.lower() or "time" in c.lower()]
        if prod_dates:
            prod_map["publish"] = prod_dates[0]
    return prod_map, ga4_map, gsc_map

# ---- Title fallback from URL path ----
def _fallback_title_from_path(path: str) -> str:
    if not path or not isinstance(path, str):
        return ""
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return ""
    last = parts[-1]
    last = re.sub(r"\d+\.cms$", "", last)
    if not last and len(parts) >= 2:
        last = parts[-2]
    last = last.replace("-", " ").strip()
    return last.title() if last else ""

# ---- Standardization & Merge ----
def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    p = prod_df.copy() if prod_df is not None else None
    if p is not None and mappings["prod"].get("publish") and mappings["prod"]["publish"] in p.columns:
        p["Publish Time"] = safe_dt_parse(p[mappings["prod"]["publish"]], "Publish Time", vc)

    g4 = ga4_df.copy() if ga4_df is not None else None
    if g4 is not None and mappings["ga4"].get("date") and mappings["ga4"]["date"] in g4.columns:
        g4["date"] = pd.to_datetime(g4[mappings["ga4"]["date"]], errors="coerce").dt.date

    gs = gsc_df.copy() if gsc_df is not None else None
    if gs is not None and mappings["gsc"].get("date") and mappings["gsc"]["date"] in gs.columns:
        gs["date"] = pd.to_datetime(gs[mappings["gsc"]["date"]], errors="coerce").dt.date

    return p, g4, gs

def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # Rename to standard names
    std_names = {
        "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "Publish Time"},
        "ga4": {"msid": "msid", "date": "date", "users": "totalUsers", "engagement": "userEngagementDuration", "bounce": "bounceRate"},
        "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks", "impr": "Impressions", "ctr": "CTR", "pos": "Position"}
    }
    try:
        if prod_df is not None: prod_df.rename(columns={prod_map[k]: v for k, v in std_names["prod"].items() if prod_map.get(k)}, inplace=True)
        if ga4_df is not None: ga4_df.rename(columns={ga4_map[k]: v for k, v in std_names["ga4"].items() if ga4_map.get(k)}, inplace=True)
        if gsc_df is not None: gsc_df.rename(columns={gsc_map[k]: v for k, v in std_names["gsc"].items() if gsc_map.get(k)}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Column renaming failed: {e}")
        return None, vc

    # Dates
    prod_df, ga4_df, gsc_df = standardize_dates_early(prod_df, ga4_df, gsc_df, {"prod": std_names["prod"], "ga4": std_names["ga4"], "gsc": std_names["gsc"]}, vc)

    # MSID cleanup
    for df, name in [(prod_df, "Production"), (ga4_df, "GA4")]:
        if df is not None and "msid" in df.columns:
            df["msid"] = pd.to_numeric(df["msid"], errors="coerce")
            df.dropna(subset=["msid"], inplace=True)
            if not df.empty: df["msid"] = df["msid"].astype("int64")

    if gsc_df is not None and "page_url" in gsc_df.columns:
        gsc_df["msid"] = gsc_df["page_url"].astype(str).str.extract(r'(\d+)\.cms').iloc[:, 0]
        gsc_df["msid"] = pd.to_numeric(gsc_df["msid"], errors="coerce")
        gsc_df.dropna(subset=["msid"], inplace=True)
        if not gsc_df.empty: gsc_df["msid"] = gsc_df["msid"].astype("int64")

        # Numeric conversions & clamps
        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 100))]:
            if col in gsc_df.columns:
                gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc, clamp=clamp)

        # CTR cleanup
        if "CTR" in gsc_df.columns:
            if gsc_df["CTR"].dtype == "object":
                tmp = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False).str.replace(",", "").str.strip()
                gsc_df["CTR"] = pd.to_numeric(tmp, errors="coerce") / 100.0
            gsc_df["CTR"] = coerce_numeric(gsc_df["CTR"], "GSC.CTR", vc, clamp=(0, 1))
        elif {"Clicks","Impressions"}.issubset(gsc_df.columns):
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)

    # Merge GSC × Prod
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_PREP_FAIL", "Missing GSC or Production data"); return None, vc

    prod_cols = [c for c in ["msid","Title","Path","Publish Time"] if c in prod_df.columns]
    merged = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how="left")

    # Enrich with categories
    if "Path" in merged.columns:
        cats = merged["Path"].astype(str).str.strip('/').str.split('/', n=2, expand=True)
        merged["L1_Category"] = cats[0].fillna("Uncategorized")
        merged["L2_Category"] = cats[1].fillna("General")
    else:
        merged["L1_Category"] = "Uncategorized"
        merged["L2_Category"] = "General"

    # Title fallback if missing
    if "Title" in merged.columns:
        merged["Title"] = merged["Title"].fillna("").astype(str)
        need = merged["Title"].str.strip().eq("")
        if "Path" in merged.columns:
            merged.loc[need, "Title"] = merged.loc[need, "Path"].apply(_fallback_title_from_path)
        if "Query" in merged.columns:
            still = merged["Title"].str.strip().eq("")
            merged.loc[still, "Title"] = merged.loc[still, "Query"].fillna("").astype(str).str.title()

    return merged, vc

# ---- Module 1 logic ----
EXPECTED_CTR = CONFIG["expected_ctr_by_rank"]

def _expected_ctr_for_pos(pos: float) -> float:
    if pd.isna(pos): return np.nan
    p = max(1, min(50, float(pos)))
    base = EXPECTED_CTR.get(int(min(9, round(p))), EXPECTED_CTR[9])
    if p <= 9:
        return base
    return base * (9.0 / p) ** 0.5  # gentle decay beyond rank 9

# Editor-simple fallback (short)
def _simple_actionlines(tag: str, row: Dict[str, Any]) -> List[str]:
    templates = {
        "Low CTR @ Good Position": [
            "Rewrite the headline. Put the main keyword at the start.",
            "Add a short page description (1–2 lines) that matches the headline.",
            "Add 2–3 internal links from strong related pages.",
            "Include a small FAQ with 2–3 common questions."
        ],
        "Hidden Gem: High CTR @ Poor Position": [
            "Link this page from 3–5 high-traffic related pages.",
            "Expand the intro. Explain the topic in simple words.",
            "Add subheadings for common search questions.",
            "Update old facts, numbers, and examples."
        ],
        "High Bounce @ Good Position": [
            "Move the key answer to the first 3–4 lines.",
            "Shorten long paragraphs. Use bullets for lists.",
            "Compress heavy images; remove autoplay videos.",
            "Add a short summary box at the top."
        ]
    }
    return templates.get(tag, ["Make small, clear edits to title, intro, links, and summary."])

# SEO + Business fallback (what/where/why — structured)
def _biz_card(tag: str, row: Dict[str, Any]) -> str:
    base = {
        "Low CTR @ Good Position": {
            "goal": "Lift clicks without changing rank.",
            "target": "Improve click rate toward 15–25% at positions 1–10.",
            "what": [
                "**SEO Title (meta title):** Lead with main keyword + clear benefit.",
                "**H1 (headline):** Match title promise; remove fluff/brand-first leads.",
                "**Meta description:** 1–2 lines answering 'What will I get?' + soft CTA.",
                "**Intro/dek:** State the core answer/value in first 2–3 lines.",
                "**FAQ (2–4):** Cover top 'People also ask' topics."
            ],
            "where": [
                "CMS → Meta Title, H1, Meta Description, Dek/Intro, FAQ module.",
                "Listing/cards that link here → card title/teaser (if owned)."
            ],
            "why": [
                "Users pick the clearest promise on page-1; copy must signal value fast.",
                "Matching intent + quick answers drives higher click-through."
            ],
            "serp": [
                "Add **FAQPage** or **NewsArticle/Article** schema.",
                "Place concise bullets near the top (often pulled into snippets)."
            ],
            "internal": [
                "Add 3–5 links from high-traffic, closely related pages.",
                "Use exact/topic-match anchors (not 'read more')."
            ],
            "measure": "Track CTR & impressions per query for 7–14 days."
        },
        "Hidden Gem: High CTR @ Poor Position": {
            "goal": "Move a proven hit up the ranks.",
            "target": "Improve position from 16–40 → top 10 while keeping strong click rate.",
            "what": [
                "**Topical depth:** Add H2/H3 sections for related questions/entities.",
                "**Intro & summary:** Add a 'What you’ll learn/see' box.",
                "**E-E-A-T:** Bylines, sources, dates, expert quotes where relevant."
            ],
            "where": [
                "CMS → H2/H3 structure, intro/summary box, citations/sources.",
                "Other pages → add internal links pointing here."
            ],
            "why": [
                "High CTR proves fit; ranking lags due to coverage & internal authority gaps."
            ],
            "serp": [
                "Add/verify **Article/NewsArticle** schema (headline, date, author, image).",
                "Check canonical and avoid index bloat."
            ],
            "internal": [
                "Link from 5–10 related, strong pages **above the fold** or near first mention.",
                "Anchors reflect the exact subtopic (e.g., 'IPL 2025 squads')."
            ],
            "measure": "Track average position & impressions weekly for 2–4 weeks."
        },
        "High Bounce @ Good Position": {
            "goal": "Keep qualified traffic; protect revenue/rankings.",
            "target": "Reduce bounce from >70% → <55%.",
            "what": [
                "**Above-the-fold:** Put the key answer/table/summary in first screen.",
                "**Structure:** Short paragraphs; bullets; early TL;DR box.",
                "**Media:** Compress images; remove autoplay video above the fold."
            ],
            "where": [
                "CMS → Dek/intro, first H2, TL;DR box, image settings.",
                "Template (if available) → move heavy embeds below the fold."
            ],
            "why": [
                "Search visitors leave when the first screen doesn’t deliver or loads slow."
            ],
            "serp": [
                "Keep headings scannable; align early copy with query intent."
            ],
            "internal": [
                "Link onward to deeper pages after the answer (reduce pogo-sticking)."
            ],
            "measure": "Watch bounce, engagement time, and scroll depth for 7–14 days."
        }
    }
    b = base.get(tag, None)
    if not b:
        return "- Make clear, measurable changes to title, intro, structure, and internal links."
    lines = []
    lines.append(f"**Goal (business):** {b['goal']}")
    lines.append(f"**Target:** {b['target']}")
    lines.append("\n**What to change (exact fields)**")
    lines += [f"- {w}" for w in b["what"]]
    lines.append("\n**Where to change**")
    lines += [f"- {w}" for w in b["where"]]
    lines.append("\n**Why**")
    lines += [f"- {w}" for w in b["why"]]
    if b.get("serp"):
        lines.append("\n**SERP enhancements**")
        lines += [f"- {w}" for w in b["serp"]]
    if b.get("internal"):
        lines.append("\n**Internal links (quick wins)**")
        lines += [f"- {w}" for w in b["internal"]]
    lines.append(f"\n**How to measure**\n- {b['measure']}")
    return "\n".join(lines)

def engagement_mismatches(df: pd.DataFrame, thresholds: Dict[str, Any], style: str = "biz") -> List[str]:
    """
    style: 'simple' | 'biz' | 'technical'
    """
    if df is None or df.empty:
        return ["No data available for analysis"]
    d = df.copy()
    for c in ["Clicks","Impressions","CTR","Position","bounceRate"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    insights: List[str] = []

    # Low CTR @ good position
    if {"Position","CTR"}.issubset(d.columns):
        d["expected_ctr"] = d["Position"].apply(_expected_ctr_for_pos)
        d["deficit_pct"] = np.where(d["expected_ctr"]>0, (d["expected_ctr"]-d["CTR"]) / d["expected_ctr"], np.nan)
        mask = (d["Position"] <= 10) & (d["deficit_pct"] >= (thresholds["ctr_deficit_pct"]/100.0)) & (d.get("Impressions", 0) >= thresholds.get("min_impressions", 0))
        for _, row in d[mask].nlargest(2, "deficit_pct").iterrows():
            msid_txt = f"**MSID:** `{int(row.get('msid')) if not pd.isna(row.get('msid')) else 'N/A'}`"
            title_txt = str(row.get('Title') or '').strip() or "(title missing)"
            show_title = f"**Title:** {title_txt[:90]}{'...' if len(title_txt)>90 else ''}"
            if style == "simple":
                steps = _simple_actionlines("Low CTR @ Good Position", row)
                bullet = "\n".join([f"- {s}" for s in steps])
                insights.append(
                    f"### ⚠️ Low clicks even though top on Google\n"
                    f"{msid_txt} | **Position:** {row['Position']:.1f} | **Click rate:** {row['CTR']:.2%}\n"
                    f"{show_title}\n{bullet}"
                )
            elif style == "biz":
                body = _biz_card("Low CTR @ Good Position", row)
                insights.append(
                    f"### ⚠️ Low clicks at good rank\n{msid_txt} | **Position:** {row['Position']:.1f} | **Click rate:** {row['CTR']:.2%}\n{show_title}\n\n{body}"
                )
            else:  # technical
                insights.append(
                    f"""### ⚠️ Low CTR at Good Position
{msid_txt} | **Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}
{show_title}
**Recommendation:** Refresh title/meta, add rich snippets, and tighten intro to match dominant intent."""
                )

    # Hidden gems
    if {"Position","CTR"}.issubset(d.columns):
        mask = (d["Position"] > 15) & (d["CTR"] > 0.05) & (d.get("Impressions", 0) >= thresholds.get("min_impressions", 0))
        for _, row in d[mask].nlargest(2, "CTR").iterrows():
            msid_txt = f"**MSID:** `{int(row.get('msid')) if not pd.isna(row.get('msid')) else 'N/A'}`"
            title_txt = str(row.get('Title') or '').strip() or "(title missing)"
            show_title = f"**Title:** {title_txt[:90]}{'...' if len(title_txt)>90 else ''}"
            if style == "simple":
                steps = _simple_actionlines("Hidden Gem: High CTR @ Poor Position", row)
                bullet = "\n".join([f"- {s}" for s in steps])
                insights.append(
                    f"### 💎 Hidden gem (people like it, but rank is low)\n"
                    f"{msid_txt} | **Position:** {row['Position']:.1f} | **Click rate:** {row['CTR']:.2%}\n"
                    f"{show_title}\n{bullet}"
                )
            elif style == "biz":
                body = _biz_card("Hidden Gem: High CTR @ Poor Position", row)
                insights.append(
                    f"### 💎 Hidden gem (high clicks, low rank)\n{msid_txt} | **Position:** {row['Position']:.1f} | **Click rate:** {row['CTR']:.2%}\n{show_title}\n\n{body}"
                )
            else:
                insights.append(
                    f"""### 💎 Hidden Gem (High CTR @ Poor Position)
{msid_txt} | **Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}
{show_title}
**Recommendation:** Strengthen internal links + on-page SEO to lift ranking; content already resonates."""
                )

    # High bounce
    if {"bounceRate","Position"}.issubset(d.columns):
        mask = (d["bounceRate"] > 0.70) & (d["Position"] <= 15) & (d.get("Impressions", 0) >= thresholds.get("min_impressions", 0))
        for _, row in d[mask].nlargest(2, "bounceRate").iterrows():
            msid_txt = f"**MSID:** `{int(row.get('msid')) if not pd.isna(row.get('msid')) else 'N/A'}`"
            title_txt = str(row.get('Title') or '').strip() or "(title missing)"
            show_title = f"**Title:** {title_txt[:90]}{'...' if len(title_txt)>90 else ''}"
            if style == "simple":
                steps = _simple_actionlines("High Bounce @ Good Position", row)
                bullet = "\n".join([f"- {s}" for s in steps])
                insights.append(
                    f"### 🚨 People leave quickly even though rank is good\n"
                    f"{msid_txt} | **Position:** {row['Position']:.1f} | **Bounce:** {row['bounceRate']:.1%}\n"
                    f"{show_title}\n{bullet}"
                )
            elif style == "biz":
                body = _biz_card("High Bounce @ Good Position", row)
                insights.append(
                    f"### 🚨 High bounce at good rank\n{msid_txt} | **Position:** {row['Position']:.1f} | **Bounce:** {row['bounceRate']:.1%}\n{show_title}\n\n{body}"
                )
            else:
                insights.append(
                    f"""### 🚨 High Bounce at Good Position
{msid_txt} | **Position:** {row['Position']:.1f} | **Bounce:** {row['bounceRate']:.1%}
{show_title}
**Recommendation:** Re-check search intent, improve above-the-fold clarity, compress media, and add TOC anchors."""
                )

    if not insights:
        insights.append("No specific mismatches detected at current thresholds.")
    return insights

def build_mismatch_table(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    for c in ["Clicks","Impressions","CTR","Position","userEngagementDuration","bounceRate"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["expected_ctr"] = d["Position"].apply(_expected_ctr_for_pos) if "Position" in d.columns else np.nan
    d["ctr_deficit"] = (d["expected_ctr"] - d["CTR"]) if {"expected_ctr","CTR"}.issubset(d.columns) else np.nan

    tags = []
    for _, row in d.iterrows():
        tag = None
        pos = row.get("Position", np.nan)
        ctr = row.get("CTR", np.nan)
        impr = row.get("Impressions", 0)
        br = row.get("bounceRate", np.nan)
        exp = row.get("expected_ctr", np.nan)
        if not pd.isna(impr) and impr < thresholds.get("min_impressions", 0):
            tags.append(None); continue
        if not pd.isna(pos) and not pd.isna(ctr) and not pd.isna(exp) and pos <= 10 and exp > 0:
            deficit_pct = (exp - ctr) / exp
            if deficit_pct >= (thresholds["ctr_deficit_pct"]/100.0):
                tag = "Low CTR @ Good Position"
        if tag is None and not pd.isna(pos) and not pd.isna(ctr) and pos > 15 and ctr > 0.05:
            tag = "Hidden Gem: High CTR @ Poor Position"
        if tag is None and not pd.isna(br) and not pd.isna(pos) and br > 0.70 and pos <= 15:
            tag = "High Bounce @ Good Position"
        tags.append(tag)

    d["Mismatch_Tag"] = tags
    d = d[~d["Mismatch_Tag"].isna()].copy()

    keep_cols = [c for c in [
        "date","msid","Title","Path","L1_Category","L2_Category","Query",
        "Position","CTR","expected_ctr","ctr_deficit","Clicks","Impressions",
        "userEngagementDuration","bounceRate"
    ] if c in d.columns]
    return d[["Mismatch_Tag"] + keep_cols].sort_values(["Mismatch_Tag","msid"]) if not d.empty else pd.DataFrame()

# ---------- AI helpers ----------
def _get_secret(key: str, sections=("google","gemini","general")) -> Optional[str]:
    try:
        if key in st.secrets and st.secrets[key]:
            return st.secrets[key]
    except Exception:
        pass
    try:
        for sect in sections:
            if sect in st.secrets:
                sec = st.secrets[sect]
                for k in (key, "API_KEY", "api_key", "key"):
                    if isinstance(sec, dict) and sec.get(k):
                        return sec[k]
    except Exception:
        pass
    return os.getenv(key)

def _format_rows_for_prompt(df: pd.DataFrame, n: int = 8) -> str:
    cols = [c for c in ["msid","Title","Query","L1_Category","L2_Category",
                        "Position","CTR","expected_ctr","Impressions","Clicks","bounceRate","Path"]
            if c in df.columns]
    small = df[cols].head(n).copy()
    if "CTR" in small.columns:
        small["CTR"] = (small["CTR"]*100).round(2)
    if "expected_ctr" in small.columns:
        small["expected_ctr"] = (small["expected_ctr"]*100).round(2)
    return small.to_csv(index=False)

def _make_prompt(tag: str, csv_snippet: str, style: str) -> str:
    """
    style: 'simple' | 'biz' | 'technical'
    """
    if style == "simple":
        tone = (
            "Audience: newsroom content editors (non-technical).\n"
            "Style: short, plain words. Avoid jargon (say 'click rate' not 'CTR').\n"
            "Give 3–5 steps. Start lines with a dash (-). Each step max 14 words.\n"
            "Optionally add one 'Example:' line.\n"
        )
        sections = "Only output the bullet points (and one optional Example line)."
    elif style == "biz":
        tone = (
            "Audience: SEO lead + Business head + Content editor.\n"
            "Style: plain English, on-point. Be specific about WHAT to change, WHERE to change, and WHY.\n"
            "Keep bullets crisp; avoid fluff. No long paragraphs.\n"
        )
        sections = (
            "Output these sections in order:\n"
            "1) Goal (business) — one line.\n"
            "2) Target — one line.\n"
            "3) What to change (exact fields) — 3–6 bullets.\n"
            "4) Where to change — 2–4 bullets.\n"
            "5) Why — 1–3 bullets.\n"
            "6) SERP enhancements — 1–3 bullets (if useful).\n"
            "7) Internal links (quick wins) — 2–4 bullets.\n"
            "8) How to measure — one line.\n"
        )
    else:
        tone = (
            "Audience: SEO specialists.\n"
            "Style: concise, technical. 3–5 bullets. Max 20 words each.\n"
        )
        sections = "Only output the bullets."

    return f"""
Write in English.
{tone}
Context rows (CSV):
{csv_snippet}

Task: For items flagged '{tag}', give the action list.
{sections}
"""

def _gemini_generate(prompt: str) -> Optional[str]:
    api_key = _get_secret("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Google key missing. Set GOOGLE_API_KEY in Streamlit Cloud → Settings → Secrets.")
        return None

    if _HAS_GOOGLE_GENAI_NEW:
        try:
            client = _genai_new.Client(
                api_key=api_key,
                http_options=_genai_types.HttpOptions(api_version="v1"),
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates"):
                text = "\n".join([getattr(c, "text", "") for c in resp.candidates if getattr(c, "text", "")])
            return (text or "").strip() or None
        except Exception as e:
            st.warning(f"Gemini (google-genai) failed: {e}")

    if _HAS_GOOGLE_GENAI_OLD:
        try:
            _genai_old.configure(api_key=api_key)
            model = _genai_old.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(prompt)
            return (getattr(resp, "text", "") or "").strip() or None
        except Exception as e:
            st.warning(f"Gemini (google-generativeai) failed: {e}")

    st.warning("Gemini SDK not installed. Add 'google-genai' or 'google-generativeai' to requirements.txt.")
    return None

def _openai_generate(prompt: str) -> Optional[str]:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI key missing. Set OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")
        return None

    if _HAS_OPENAI_CLIENT:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Be a concise SEO + growth operator."},
                    {"role":"user","content": prompt}
                ],
                temperature=0.3
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            st.warning(f"OpenAI (new SDK) failed: {e}")

    if _HAS_OPENAI and not _HAS_OPENAI_CLIENT:
        try:
            import openai
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"Be a concise SEO + growth operator."},
                    {"role":"user","content": prompt}
                ],
                temperature=0.3
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            st.warning(f"OpenAI (legacy) failed: {e}")

    st.warning("OpenAI SDK not installed. Add 'openai' to requirements.txt.")
    return None

def ai_recommendations(tag: str, df: pd.DataFrame, provider: str, n: int, style: str) -> Optional[str]:
    take = df[df["Mismatch_Tag"] == tag]
    if take.empty:
        return None
    csv_snippet = _format_rows_for_prompt(take, n=n)
    prompt = _make_prompt(tag, csv_snippet, style)
    if provider == "Gemini":
        return _gemini_generate(prompt)
    else:
        return _openai_generate(prompt)

# ---------- UI stepper ----------
st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", [
    "1) Get CSV Templates",
    "2) Upload & Map Columns",
    "3) Validate & Process",
    "4) Analyze (Module 1)"
], horizontal=True)

# Templates
def _make_template_production():
    return pd.DataFrame({
        "Msid": [101, 102, 103],
        "Title": ["Budget 2025 highlights explained", "IPL 2025 schedule & squads", "Monsoon updates: city-by-city guide"],
        "Path": ["/business/budget-2025/highlights", "/sports/cricket/ipl-2025/schedule", "/news/monsoon/guide"],
        "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00", "2025-09-01 07:00:00"]
    })

def _make_template_ga4():
    return pd.DataFrame({
        "customEvent:msid": [101, 101, 102, 102, 103],
        "date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "totalUsers": [4000, 4500, 10000, 8000, 5200],
        "userEngagementDuration": [52.3, 48.2, 41.0, 44.7, 63.1],
        "bounceRate": [0.42, 0.45, 0.51, 0.49, 0.38]
    })

def _make_template_gsc():
    return pd.DataFrame({
        "Date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "Page": [
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/news/monsoon/guide/103.cms"
        ],
        "Query": ["budget 2025", "budget highlights", "ipl 2025 schedule", "ipl squads", "monsoon city guide"],
        "Clicks": [200, 240, 1200, 1100, 300],
        "Impressions": [5000, 5500, 40000, 38000, 7000],
        "CTR": [0.04, 0.0436, 0.03, 0.0289, 0.04286],
        "Position": [8.2, 8.0, 12.3, 11.7, 9.1]
    })

if step == "1) Get CSV Templates":
    st.info("Download sample CSV templates to understand required structure.")
    colA, colB, colC = st.columns(3)
    with colA:
        df = _make_template_production(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_production.csv", "Download Production Template")
    with colB:
        df = _make_template_ga4(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_ga4.csv", "Download GA4 Template")
    with colC:
        df = _make_template_gsc(); st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_gsc.csv", "Download GSC Template")
    st.stop()

# Step 2: uploads + mapping
st.subheader("Upload Your Data Files")
col1, col2, col3 = st.columns(3)
with col1:
    prod_file = st.file_uploader("Production Data (CSV)", type=["csv"], key="prod_csv")
    if prod_file: st.success(f"✓ Production: {prod_file.name}")
with col2:
    ga4_file = st.file_uploader("GA4 Data (CSV) — optional", type=["csv"], key="ga4_csv")
    if ga4_file: st.success(f"✓ GA4: {ga4_file.name}")
with col3:
    gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")
    if gsc_file: st.success(f"✓ GSC: {gsc_file.name}")

if not all([prod_file, gsc_file]):
    st.warning("Please upload Production & GSC files to proceed"); st.stop()

vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read) if ga4_file else None
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)

if any(df is None or df.empty for df in [prod_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Column mapping UI
st.subheader("Column Mapping")
prod_guess, ga4_guess, gsc_guess = _guess_colmap(prod_df_raw, ga4_df_raw if ga4_df_raw is not None else pd.DataFrame(), gsc_df_raw)

with st.expander("Production Mapping", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    prod_map = {}
    prod_map["msid"] = c1.selectbox("MSID", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("msid")) if prod_guess.get("msid") in prod_df_raw.columns else 0)
    prod_map["title"] = c2.selectbox("Title", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("title")) if prod_guess.get("title") in prod_df_raw.columns else 0)
    prod_map["path"] = c3.selectbox("Path", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("path")) if prod_guess.get("path") in prod_df_raw.columns else 0)
    prod_map["publish"] = c4.selectbox("Publish Time", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(prod_guess.get("publish")) if prod_guess.get("publish") in prod_df_raw.columns else 0)

with st.expander("GA4 Mapping (optional)", expanded=False):
    if ga4_df_raw is not None:
        c1, c2, c3 = st.columns(3)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox("MSID (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("msid")) if ga4_guess.get("msid") in ga4_df_raw.columns else 0)
        ga4_map["date"] = c2.selectbox("Date (GA4)", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("date")) if ga4_guess.get("date") in ga4_df_raw.columns else 0)
        ga4_map["bounce"] = c3.selectbox("Bounce Rate", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(ga4_guess.get("bounce")) if ga4_guess.get("bounce") in ga4_df_raw.columns else 0)
    else:
        ga4_map = {}
        st.info("GA4 optional — bounceRate not required for core mismatches.")

with st.expander("GSC Mapping", expanded=True):
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    gsc_map = {}
    gsc_map["date"] = c1.selectbox("Date (GSC)", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("date")) if gsc_guess.get("date") in gsc_df_raw.columns else 0)
    gsc_map["page"] = c2.selectbox("Page URL", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("page")) if gsc_guess.get("page") in gsc_df_raw.columns else 0)
    gsc_map["query"] = c3.selectbox("Query", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("query")) if gsc_guess.get("query") in gsc_df_raw.columns else 0)
    gsc_map["clicks"] = c4.selectbox("Clicks", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("clicks")) if gsc_guess.get("clicks") in gsc_df_raw.columns else 0)
    gsc_map["impr"] = c5.selectbox("Impressions", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("impr")) if gsc_guess.get("impr") in gsc_df_raw.columns else 0)
    gsc_map["ctr"] = c6.selectbox("CTR", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("ctr")) if gsc_guess.get("ctr") in gsc_df_raw.columns else 0)
    gsc_map["pos"] = c7.selectbox("Position", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(gsc_guess.get("pos")) if gsc_guess.get("pos") in gsc_df_raw.columns else 0)

# Process & merge
with st.spinner("Processing & merging datasets..."):
    master_df, vc_after = process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)

if master_df is None or master_df.empty:
    st.error("Data processing failed critically. Please check mappings and file contents.")
    st.dataframe(vc_after.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# ---- Sidebar: thresholds, AI toggles, recommendation style, date range ----
with st.sidebar:
    st.subheader("Thresholds")
    TH = CONFIG["thresholds"].copy()
    TH["ctr_deficit_pct"] = st.slider("CTR Deficit Threshold (%)", 0.5, 10.0, float(TH["ctr_deficit_pct"]), step=0.1, key="ctr_def_pct")
    TH["min_impressions"] = st.number_input("Min Impressions (to consider)", min_value=0, value=int(TH["min_impressions"]), step=50, key="min_impr")

    st.markdown("---")
    st.subheader("AI Recommendations")
    ai_note = []
    if not (_HAS_GOOGLE_GENAI_NEW or _HAS_GOOGLE_GENAI_OLD):
        ai_note.append("Gemini SDK missing")
    if not _HAS_OPENAI:
        ai_note.append("OpenAI SDK missing")
    if ai_note:
        st.caption("• " + " • ".join(ai_note) + " — check requirements.txt")
    use_ai = st.checkbox("Use AI-generated recommendations", value=False)
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"], index=0, disabled=not use_ai)
    max_rows_for_ai = st.slider("Rows per bucket (AI prompt)", 3, 20, 8, disabled=not use_ai)

    st.markdown("---")
    st.subheader("Recommendation style")
    style = st.selectbox("Tone & structure", ["SEO + Business (on-point)", "Simple (Editors)", "Technical (SEO)"], index=0)
    style_key = {"SEO + Business (on-point)": "biz", "Simple (Editors)": "simple", "Technical (SEO)":"technical"}[style]

    st.markdown("---")
    st.subheader("Analysis Period")
    if "date" in master_df.columns:
        _dates = pd.to_datetime(master_df["date"], errors="coerce").dt.date.dropna()
        _min_d, _max_d = _dates.min(), _dates.max()
    else:
        _max_d = date.today()
        _min_d = _max_d - timedelta(days=CONFIG["defaults"]["date_lookback_days"])

    _default_end = _max_d
    _default_start = max(_min_d, _default_end - timedelta(days=CONFIG["defaults"]["date_lookback_days"]))
    start_date = st.date_input("Start Date", value=_default_start, min_value=_min_d, max_value=_max_d, key="start_date_picker")
    end_date   = st.date_input("End Date",   value=_default_end,   min_value=_min_d, max_value=_max_d, key="end_date_picker")
    if start_date > end_date:
        st.warning("Start date is after end date. Swapping.")
        start_date, end_date = end_date, start_date

# Date filter
if "date" in master_df.columns:
    m = master_df.copy()
    try:
        m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.date
        mask = (m["date"] >= start_date) & (m["date"] <= end_date)
        filtered_df = m[mask].copy()
        st.info(f"Date filter applied: {len(filtered_df):,} rows from {start_date} to {end_date}")
    except Exception:
        filtered_df = master_df
else:
    filtered_df = master_df

st.success(f"✅ Master dataset ready: {filtered_df.shape[0]:,} rows × {filtered_df.shape[1]} columns")

# If not on Analyze step, stop here
if step != "4) Analyze (Module 1)":
    st.info("Move to **Step 4** to run the Engagement vs Search analysis.")
    st.stop()

# -----------------------------
# ANALYSIS — Module 1 outputs
# -----------------------------
st.header("📊 Module 1: Engagement vs Search — Insights & Exports")

# 1) Build mismatch table first
mismatch_df = build_mismatch_table(filtered_df, TH)

# 2) Build human-readable insight cards per selected style
cards = engagement_mismatches(filtered_df, TH, style=style_key)

# 3) AI recommendations per tag (if enabled), else fallback to canned cards
if use_ai and mismatch_df is not None and not mismatch_df.empty and "Mismatch_Tag" in mismatch_df.columns:
    for tag in ["Low CTR @ Good Position",
                "Hidden Gem: High CTR @ Poor Position",
                "High Bounce @ Good Position"]:
        if (mismatch_df["Mismatch_Tag"] == tag).any():
            st.markdown(f"### {tag}")
            txt = ai_recommendations(tag, mismatch_df, provider, n=int(max_rows_for_ai), style=style_key)
            if txt:
                st.markdown(txt)
            else:
                for c in cards:
                    if tag in c:
                        st.markdown(c)
                        break
else:
    for card in cards:
        st.markdown(card)

# 4) Show table + export
if mismatch_df is not None and not mismatch_df.empty:
    st.info(f"Found **{len(mismatch_df):,}** mismatch rows.")
    with st.expander("Preview mismatch rows (first 200)", expanded=False):
        st.dataframe(mismatch_df.head(200), use_container_width=True, hide_index=True)
    download_df_button(mismatch_df, f"module1_mismatch_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download ALL mismatch rows (CSV)")
else:
    st.info("No mismatch rows matched your thresholds and filters.")

# 5) Visual: CTR vs Position bubble chart + expected CTR curve (NaN-safe marker sizes)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    st.error(f"Plotly is not available. Add 'plotly' to requirements.txt. Details: {e}")
else:
    try:
        vis = filtered_df.copy()

        # Ensure numeric
        for c in ["Position", "CTR", "Impressions"]:
            if c in vis.columns:
                vis[c] = pd.to_numeric(vis[c], errors="coerce")

        # Attach tags for coloring (if we have keys)
        if mismatch_df is not None and not mismatch_df.empty:
            key_cols = [c for c in ["msid","Query"] if c in vis.columns and c in mismatch_df.columns]
            if key_cols:
                vis = vis.merge(
                    mismatch_df[key_cols + ["Mismatch_Tag"]].drop_duplicates(),
                    on=key_cols,
                    how="left"
                )
        if "Mismatch_Tag" not in vis.columns:
            vis["Mismatch_Tag"] = None

        # Drop rows missing x/y
        vis = vis.dropna(subset=["Position", "CTR"])

        # Decide whether to size by Impressions
        size_col = None
        if "Impressions" in vis.columns and vis["Impressions"].notna().any():
            vis["Impressions"] = vis["Impressions"].fillna(0).clip(lower=0)
            size_col = "Impressions"

        fig = px.scatter(
            vis,
            x="Position",
            y="CTR",
            size=size_col,  # None if no valid sizes
            color="Mismatch_Tag",
            hover_data=[c for c in ["msid","Title","Query","L1_Category","L2_Category","Impressions","Clicks"] if c in vis.columns],
            title="CTR vs Position (bubble = Impressions)" if size_col else "CTR vs Position"
        )

        # Expected CTR curve
        pos_grid = np.linspace(1, 50, 200)
        curve = [_expected_ctr_for_pos(p) for p in pos_grid]
        fig.add_trace(go.Scatter(x=pos_grid, y=curve, mode="lines", name="Expected CTR", hoverinfo="skip"))

        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Average Position", yaxis_title="CTR")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Chart failed. See error below:")
        st.exception(e)

# 6) Summary actions
st.divider()
st.subheader("🎯 Quick Recommendations")
if isinstance(cards, list) and len(cards) > 1:
    st.success("**Priority Actions:**")
    for card in cards[:3]:
        # If technical card, pick the rec; if structured, grab last line as a quick action
        if "**Recommendation:**" in card:
            st.markdown(f"- {card.split('**Recommendation:**')[-1].strip()}")
        else:
            parts = [ln for ln in card.splitlines() if ln.strip().startswith("- ")]
            st.markdown(f"- {parts[0][2:].strip()}" if parts else "- Improve title/intro and add internal links.")
else:
    st.info("Upload more data or tune thresholds to surface actions.")

st.markdown("---")
st.caption("GrowthOracle — Module 1 (Standalone)")
