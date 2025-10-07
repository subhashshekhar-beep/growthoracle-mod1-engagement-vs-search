# app.py — GrowthOracle (Module 1 - Improved)
# Works on Streamlit Cloud. Use Secrets UI for keys.
# Requires (add to requirements.txt): streamlit pandas numpy pyyaml plotly openai google-genai

import os
import re
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Optional YAML for config
try:
    import yaml
except ImportError:
    yaml = None

# Try loading AI SDKs (feature-detection)
_HAS_OPENAI = False
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    pass

_HAS_GEMINI = False
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    pass


# ---- Page Configuration ----
st.set_page_config(
    page_title="GrowthOracle — Module 1: Engagement vs Search",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)

# ---- Logger Setup ----
@st.cache_resource
def get_logger(level=logging.INFO):
    """Initializes and retrieves a singleton logger instance."""
    logger = logging.getLogger("growthoracle_mod1")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# ---- Application Configuration ----
_DEFAULT_CONFIG = {
    "thresholds": {
        "striking_distance_min": 11, "striking_distance_max": 20,
        "ctr_deficit_pct": 25.0, # Increased default for more significant findings
        "min_impressions": 250,  # Increased default to focus on impactful pages
        "min_clicks": 20,
        "high_bounce_rate": 0.75,
        "good_position_max": 10,
        "poor_position_min": 15,
        "high_ctr_min_for_gem": 0.05,
    },
    "expected_ctr_by_rank": {
        1: 0.30, 2: 0.18, 3: 0.12, 4: 0.09, 5: 0.07,
        6: 0.06, 7: 0.05, 8: 0.04, 9: 0.035, 10: 0.03
    },
    "performance": {"sample_row_limit": 500_000, "seed": 42},
    "defaults": {"date_lookback_days": 90}
}

@st.cache_resource
def load_config():
    """Loads config from YAML if available, otherwise uses defaults."""
    cfg = _DEFAULT_CONFIG.copy()
    if yaml is not None and os.path.exists("config.yaml"):
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            for k, v in user_cfg.items():
                if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            logger.info("Loaded configuration from config.yaml")
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return cfg

CONFIG = load_config()

# ---- Validation Core ----
@dataclass
class ValidationMessage:
    """A structured message for data validation issues."""
    category: str  # "Critical" | "Warning" | "Info"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    """Collects and manages validation messages."""
    def __init__(self):
        self.messages: List[ValidationMessage] = []

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def to_dataframe(self) -> pd.DataFrame:
        """Converts collected messages to a pandas DataFrame."""
        if not self.messages:
            return pd.DataFrame(columns=["category", "code", "message", "context"])
        return pd.DataFrame([{
            "category": m.category, "code": m.code, "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])
        # ---- UI & Data Helpers ----
def download_df_button(df: pd.DataFrame, filename: str, label: str):
    """Renders a download button for a DataFrame if it's not empty."""
    if df is None or df.empty:
        st.warning(f"No data available to download for {label}")
        return
    try:
        csv = df.to_csv(index=False).encode("utf-8-sig") # Use utf-8-sig for Excel compatibility
        st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Failed to create download link: {e}")

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    """Safely parses a Series to datetime, logging errors."""
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad_count = parsed.isna().sum()
    if bad_count > 0:
        vc.add("Warning", "DATE_PARSE", f"Could not parse {bad_count} values in '{name}' column.", bad_rows=int(bad_count))
    return parsed

def coerce_numeric(series: pd.Series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float, float]] = None) -> pd.Series:
    """Safely coerces a Series to numeric, logging errors and optionally clamping values."""
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    bad_count = s.isna().sum()
    if bad_count > 0:
        vc.add("Warning", "NUM_COERCE", f"Coerced {bad_count} non-numeric values to NaN in '{name}'.", bad_rows=int(bad_count))
    if clamp and not s.empty:
        lo, hi = clamp
        s = s.clip(lower=lo, upper=hi)
    return s

def read_csv_safely(upload, name: str, vc: ValidationCollector) -> Optional[pd.DataFrame]:
    """Reads a CSV from an upload, trying multiple encodings."""
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided.")
        return None
    try_encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc)
            if df.empty or df.shape[1] == 0:
                vc.add("Critical", "EMPTY_CSV", f"{name} file is empty or malformed.")
                return None
            return df
        except Exception:
            continue
    vc.add("Critical", "CSV_ENCODING", f"Failed to read {name} with any common encoding.")
    return None

def _guess_colmap(prod_df, ga4_df, gsc_df):
    """Guesses column mappings based on common naming conventions."""
    prod_map, ga4_map, gsc_map = {}, {}, {}
    if prod_df is not None:
        cols = prod_df.columns
        prod_map = {
            "msid": next((c for c in cols if "msid" in c.lower()), None),
            "title": next((c for c in cols if "title" in c.lower()), None),
            "path": next((c for c in cols if "path" in c.lower() or "url" in c.lower()), None),
            "publish": next((c for c in cols if "publish" in c.lower() or "date" in c.lower()), None),
        }
    if ga4_df is not None:
        cols = ga4_df.columns
        ga4_map = {
            "msid": next((c for c in cols if "msid" in c.lower()), None),
            "date": next((c for c in cols if c.lower() == "date"), None),
            "bounce": next((c for c in cols if "bounce" in c.lower()), None),
        }
    if gsc_df is not None:
        cols = gsc_df.columns
        gsc_map = {
            "date": next((c for c in cols if c.lower() == "date"), None),
            "page": next((c for c in cols if "page" in c.lower() or "url" in c.lower()), None),
            "query": next((c for c in cols if "query" in c.lower()), None),
            "clicks": next((c for c in cols if c.lower() == "clicks"), None),
            "impr": next((c for c in cols if "impr" in c.lower()), None),
            "ctr": next((c for c in cols if c.lower() == "ctr"), None),
            "pos": next((c for c in cols if "pos" in c.lower()), None),
        }
    return {k: v for k, v in prod_map.items() if v}, \
           {k: v for k, v in ga4_map.items() if v}, \
           {k: v for k, v in gsc_map.items() if v}
    # ---- Data Processing & Merging ----
def _fallback_title_from_path(path: str) -> str:
    """Creates a human-readable title from a URL path as a fallback."""
    if not isinstance(path, str) or not path:
        return ""
    try:
        # Take the last part of the path, remove extension, replace hyphens, and title case it
        file_name = path.strip("/").split("/")[-1]
        cleaned = re.sub(r'\.\w+$', '', file_name) # Remove extension
        return cleaned.replace("-", " ").replace("_", " ").strip().title()
    except Exception:
        return ""

def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    """
    Standardizes and merges the uploaded data files into a single master DataFrame.
    """
    vc = ValidationCollector()
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # 1. Rename columns to standard names
    std_names = {
        "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "Publish Time"},
        "ga4": {"msid": "msid", "date": "date", "bounce": "bounceRate"},
        "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks",
                "impr": "Impressions", "ctr": "CTR", "pos": "Position"}
    }
    try:
        if prod_df is not None: prod_df.rename(columns={v: k for k, v in std_names["prod"].items()}, inplace=True)
        if ga4_df is not None: ga4_df.rename(columns={v: k for k, v in std_names["ga4"].items()}, inplace=True)
        if gsc_df is not None: gsc_df.rename(columns={v: k for k, v in std_names["gsc"].items()}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Column renaming failed: {e}")
        return None, vc

    # 2. Standardize data types and clean values
    if prod_df is not None:
        if "Publish Time" in prod_df.columns:
            prod_df["Publish Time"] = safe_dt_parse(prod_df["Publish Time"], "Publish Time", vc)

    for df, name in [(prod_df, "Production"), (ga4_df, "GA4")]:
        if df is not None and "msid" in df.columns:
            df["msid"] = pd.to_numeric(df["msid"], errors="coerce").dropna().astype("int64")

    if gsc_df is not None:
        if "page_url" in gsc_df.columns:
            gsc_df["msid"] = gsc_df["page_url"].astype(str).str.extract(r'(\d+)\.cms').iloc[:, 0]
            gsc_df["msid"] = pd.to_numeric(gsc_df["msid"], errors="coerce")
            gsc_df = gsc_df.dropna(subset=["msid"])
            if not gsc_df.empty: gsc_df["msid"] = gsc_df["msid"].astype("int64")

        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 200))]:
            if col in gsc_df.columns:
                gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc, clamp=clamp)

        if "CTR" in gsc_df.columns and gsc_df["CTR"].dtype == "object":
            gsc_df["CTR"] = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False).str.strip()
            gsc_df["CTR"] = pd.to_numeric(gsc_df["CTR"], errors="coerce") / 100.0
        elif {"Clicks", "Impressions"}.issubset(gsc_df.columns):
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)
        if "CTR" in gsc_df.columns:
            gsc_df["CTR"] = coerce_numeric(gsc_df["CTR"], "GSC.CTR", vc, clamp=(0, 1))

    # 3. Merge GSC and Production data (core requirement)
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_PREP_FAIL", "GSC or Production data is missing or empty after cleaning.")
        return None, vc
    
    prod_cols = ["msid", "Title", "Path", "Publish Time"]
    merged = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how="left")

    # 4. Merge GA4 data (optional)
    if ga4_df is not None and not ga4_df.empty and "msid" in ga4_df.columns:
        ga4_agg = ga4_df.groupby("msid").agg(bounceRate=("bounceRate", "mean")).reset_index()
        merged = pd.merge(merged, ga4_agg, on="msid", how="left")

    # 5. Enrich with categories and fallbacks
    if "Path" in merged.columns:
        cats = merged["Path"].astype(str).str.strip('/').str.split('/', expand=True)
        merged["L1_Category"] = cats[0].str.capitalize().fillna("Uncategorized")
    else:
        merged["L1_Category"] = "Uncategorized"

    if "Title" in merged.columns:
        needs_title = merged["Title"].isna() | (merged["Title"].str.strip() == "")
        if "Path" in merged.columns:
            merged.loc[needs_title, "Title"] = merged.loc[needs_title, "Path"].apply(_fallback_title_from_path)

    return merged, vc
    # ---- Module 1: Core Analysis & AI Logic ----

def _expected_ctr_for_pos(pos: float) -> float:
    """Calculates the expected CTR for a given search position with a decay curve."""
    if pd.isna(pos): return np.nan
    p = max(1.0, float(pos))
    # Use direct lookup for top 10, then apply a decay function
    if p <= 10:
        return CONFIG["expected_ctr_by_rank"].get(round(p), CONFIG["expected_ctr_by_rank"][10])
    else:
        # Smoother decay for positions > 10
        return CONFIG["expected_ctr_by_rank"][10] * (10 / p)**0.75

# --- IMPROVED PREDEFINED RECOMMENDATIONS ---
def _biz_card(tag: str, row: Dict[str, Any]) -> str:
    """Generates a detailed, structured recommendation card in SEO & Business language."""
    templates = {
        "CTR Leak": {
            "title": "Leak: Low CTR at Good Rank",
            "goal": "Increase organic clicks and traffic value from existing high-ranking keywords.",
            "what": [
                "**SEO Title:** Rewrite to be more compelling. Lead with the primary keyword, add a benefit or number (e.g., '10 Best...'), and keep it under 60 characters.",
                "**Meta Description:** Treat it like ad copy. Answer the user's core question, include a call-to-action (e.g., 'Discover the steps...'), and keep it under 155 characters.",
                "**Headline (H1):** Ensure it aligns perfectly with the SEO Title and the promise made in the search results.",
                "**Rich Snippets:** Implement relevant schema (e.g., FAQ, HowTo, Article) to make your search result stand out and occupy more space."
            ],
            "where": [
                "CMS: Update the 'SEO Title' and 'Meta Description' fields for this article.",
                "CMS: Edit the main H1 headline in the article body.",
                "Developer/SEO Tool: Add or validate structured data (JSON-LD) for the page."
            ],
            "why": "The page is visible (high rank) but isn't compelling enough for users to click. Improving the 'shop window'—the title and description in the SERP—is the highest-leverage action to capture existing traffic.",
            "measure": "Monitor CTR and Clicks for the specific page/query in GSC for the next 14-28 days. Look for a significant uplift in CTR."
        },
        "Hidden Gem": {
            "title": "Gem: High CTR at Poor Rank",
            "goal": "Improve search rankings for content that has already proven to be highly relevant and engaging to users.",
            "what": [
                "**Internal Linking:** Add 3-5 prominent internal links from your high-authority, topically relevant pages *to* this 'Hidden Gem' page. Use keyword-rich anchor text.",
                "**Content Depth:** Expand the content. Add sections that answer related questions (check 'People Also Ask' on Google). Add more detail, examples, or data.",
                "**On-Page Optimization:** Review the page for keyword density, LSI keywords, and ensure the primary keyword is in the URL, H1, and first paragraph.",
                "**External Authority:** Consider this page a prime candidate for link-building campaigns or digital PR efforts."
            ],
            "where": [
                "CMS: Edit your top-performing, related articles to add links pointing to this page.",
                "CMS: Update the body content of this article to be more comprehensive.",
                "Analytics: Identify referring domains for top competitors on this topic and target them."
            ],
            "why": "Users love this content when they see it (high CTR), but Google doesn't yet see it as authoritative enough (low rank). Building its authority through internal and external links is the key to unlocking its potential.",
            "measure": "Track Average Position and Impressions in GSC weekly. As rankings improve, clicks should grow exponentially."
        },
        "High Bounce": {
            "title": "Warning: High Bounce at Good Rank",
            "goal": "Improve user experience and content relevance to retain visitors, protect rankings, and increase conversions.",
            "what": [
                "**Above-the-Fold Content:** Ensure the answer to the primary query is immediately visible without scrolling. Use a summary box, a key takeaway, or a clear introductory paragraph.",
                "**Page Speed:** Optimize images, defer non-critical JavaScript, and use browser caching. A slow-loading page is a primary cause of bounces.",
                "**Content Readability:** Break up long paragraphs. Use subheadings (H2s, H3s), bullet points, and bold text to make the content scannable.",
                "**Clear Next Step:** Provide a clear, relevant internal link or call-to-action to guide the user to their next step, preventing a 'pogo-stick' back to Google."
            ],
            "where": [
                "CMS: Edit the first 200 words of the article to deliver immediate value.",
                "Developer Tool: Use Google PageSpeed Insights to identify and fix performance issues.",
                "CMS: Restructure the article body for better readability."
            ],
            "why": "The page successfully attracts clicks (good rank) but fails to meet user expectations upon arrival. This signals a content or UX mismatch that can harm long-term rankings and business goals.",
            "measure": "Monitor Bounce Rate and Average Engagement Time in GA4. A decrease in bounce rate and an increase in engagement time indicate success."
        }
    }
    b = templates.get(tag)
    if not b: return "No recommendation template found for this tag."

    lines = [f"### {b['title']}", f"**Goal:** {b['goal']}", "\n**What to Change:**"]
    lines.extend([f"- {item}" for item in b["what"]])
    lines.append("\n**Where to Change:**")
    lines.extend([f"- {item}" for item in b["where"]])
    lines.append(f"\n**Why it Matters:**\n{b['why']}")
    lines.append(f"\n**How to Measure Success:**\n{b['measure']}")
    return "\n".join(lines)


def build_mismatch_table(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """Analyzes the DataFrame to tag rows with performance mismatches."""
    if df is None or df.empty or not all(c in df.columns for c in ["Position", "CTR", "Impressions"]):
        return pd.DataFrame()
    
    d = df.copy()
    d["expected_ctr"] = d["Position"].apply(_expected_ctr_for_pos)
    d["ctr_deficit"] = d["expected_ctr"] - d["CTR"]

    conditions = [
        ( # CTR Leak
            (d["Position"] <= thresholds["good_position_max"]) &
            (d["Impressions"] >= thresholds["min_impressions"]) &
            (d["ctr_deficit"] * 100 / d["expected_ctr"] >= thresholds["ctr_deficit_pct"])
        ),
        ( # Hidden Gem
            (d["Position"] >= thresholds["poor_position_min"]) &
            (d["Impressions"] >= thresholds["min_impressions"]) &
            (d["CTR"] >= thresholds["high_ctr_min_for_gem"])
        ),
        ( # High Bounce
            ("bounceRate" in d.columns) and
            (d["Position"] <= thresholds["good_position_max"]) &
            (d["bounceRate"] >= thresholds["high_bounce_rate"])
        )
    ]
    choices = ["CTR Leak", "Hidden Gem", "High Bounce"]
    d["Mismatch_Tag"] = np.select(conditions, choices, default=None)

    result = d[d["Mismatch_Tag"].notna()].copy()
    
    keep_cols = [
        "Mismatch_Tag", "msid", "Title", "L1_Category", "Query", "Position", "CTR",
        "expected_ctr", "Clicks", "Impressions", "bounceRate", "Path"
    ]
    # Filter for columns that actually exist in the dataframe
    final_cols = [c for c in keep_cols if c in result.columns]

    return result[final_cols].sort_values(by=["Mismatch_Tag", "Impressions"], ascending=[True, False])

# ---- AI Recommendation Logic ----
def _get_secret(key: str) -> Optional[str]:
    """Retrieves API keys safely from Streamlit secrets or environment variables."""
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

def _format_rows_for_prompt(df: pd.DataFrame, n: int = 5) -> str:
    """Formats top N rows of a DataFrame into a CSV string for the AI prompt."""
    cols = ["Title", "Query", "L1_Category", "Position", "CTR", "Impressions", "bounceRate"]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].head(n).to_csv(index=False)

def _make_ai_prompt(tag: str, csv_snippet: str) -> str:
    """Creates a detailed, structured prompt for the AI based on the mismatch tag."""
    prompts = {
        "CTR Leak": "These pages rank well but have a low click-through rate (CTR).",
        "Hidden Gem": "These pages have a high CTR, showing user interest, but their rank is low.",
        "High Bounce": "These pages rank well and get clicks, but users leave quickly (high bounce rate)."
    }
    return f"""
You are an expert SEO and Digital Growth Strategist.
Your task is to provide a single, actionable recommendation for a group of web pages experiencing a specific performance issue.

**Issue Context:** {prompts.get(tag, "A performance issue has been identified.")}

**Sample Data (CSV):**
{csv_snippet}

**Your Response Format (Strictly follow this structure):**
### Diagnosis: [A short, one-sentence summary of the core problem]
**Goal:** [A one-sentence business goal for the recommendation.]

**What to Change (Primary Actions):**
- **Action 1:** [Be specific. E.g., "Rewrite SEO Titles to include numbers and power words."]
- **Action 2:** [E.g., "Implement FAQPage schema by adding 3-4 relevant questions and answers."]

**Where to Change:**
- [Location 1: E.g., "In the CMS, locate the 'SEO Title' and 'Meta Description' fields."]
- [Location 2: E.g., "Add JSON-LD schema to the <head> section of the page template."]

**Why it Matters:**
[Explain the strategic reason. E.g., "Improving the SERP snippet is the highest leverage activity to capture traffic you've already earned through ranking."]

**How to Measure Success:**
[Define the metric and timeframe. E.g., "Track CTR for these pages in Google Search Console over the next 28 days."
"""

def _call_ai_provider(prompt: str, provider: str) -> Optional[str]:
    """Calls the selected AI provider (Gemini or OpenAI) with the given prompt."""
    if provider == "Gemini" and _HAS_GEMINI:
        api_key = _get_secret("GOOGLE_API_KEY")
        if not api_key: return "Error: GOOGLE_API_KEY not found in secrets."
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: Gemini API call failed: {e}"
    elif provider == "OpenAI" and _HAS_OPENAI:
        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key: return "Error: OPENAI_API_KEY not found in secrets."
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEO and Digital Growth Strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: OpenAI API call failed: {e}"
    return f"Error: Provider '{provider}' is not configured or its SDK is not installed."

def ai_recommendations(tag: str, df: pd.DataFrame, provider: str, n: int) -> Optional[str]:
    """Orchestrates the AI recommendation generation process."""
    tagged_df = df[df["Mismatch_Tag"] == tag]
    if tagged_df.empty: return None
    
    csv_snippet = _format_rows_for_prompt(tagged_df, n=n)
    prompt = _make_ai_prompt(tag, csv_snippet)
    
    with st.spinner(f"🤖 Calling {provider} to generate insights for '{tag}'..."):
        response = _call_ai_provider(prompt, provider)
    return response
    # ---- Streamlit UI & Application Flow ----

# --- NEW VISUALIZATION ---
def create_l1_aggregated_chart(df: pd.DataFrame):
    """Creates a quadrant chart showing L1 Category performance."""
    if df is None or df.empty or not all(c in df.columns for c in ["L1_Category", "Impressions", "Position", "CTR"]):
        st.warning("Cannot generate category chart. Required columns (L1_Category, Impressions, Position, CTR) are missing.")
        return

    st.subheader("L1 Category Performance Quadrant")

    # Aggregate data by L1 Category
    agg_df = df.groupby('L1_Category').agg(
        Total_Impressions=('Impressions', 'sum'),
        Total_Clicks=('Clicks', 'sum'),
        # Weighted average position and CTR by impressions
        Avg_Position=('Position', lambda x: np.average(x, weights=df.loc[x.index, 'Impressions'])),
        Avg_CTR=('CTR', lambda x: np.average(x, weights=df.loc[x.index, 'Impressions']))
    ).reset_index()

    agg_df = agg_df[agg_df['Total_Impressions'] > 0]
    
    # Define quadrant boundaries
    median_pos = agg_df['Avg_Position'].median()
    median_ctr = agg_df['Avg_CTR'].median()

    fig = px.scatter(
        agg_df,
        x='Avg_Position',
        y='Avg_CTR',
        size='Total_Impressions',
        color='L1_Category',
        text='L1_Category',
        hover_name='L1_Category',
        hover_data={
            'L1_Category': False,
            'Avg_Position': ':.1f',
            'Avg_CTR': ':.2%',
            'Total_Impressions': ':,',
            'Total_Clicks': ':,'
        },
        title='Category Performance: CTR vs. Position (Bubble size = Impressions)'
    )

    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title='Weighted Average Position (Lower is Better)',
        yaxis_title='Weighted Average CTR (Higher is Better)',
        yaxis_tickformat='.2%',
        xaxis_autorange='reversed', # Important: Lower position is better
        showlegend=False
    )

    # Add quadrant lines and labels
    fig.add_vline(x=median_pos, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=median_ctr, line_width=1, line_dash="dash", line_color="grey")

    # Annotations for quadrants
    fig.add_annotation(x=agg_df['Avg_Position'].min(), y=agg_df['Avg_CTR'].max(), text="🚀 Leaders", showarrow=False, xanchor='left', yanchor='top', font=dict(color="green"))
    fig.add_annotation(x=agg_df['Avg_Position'].max(), y=agg_df['Avg_CTR'].max(), text="💎 Hidden Gems", showarrow=False, xanchor='right', yanchor='top', font=dict(color="orange"))
    fig.add_annotation(x=agg_df['Avg_Position'].min(), y=agg_df['Avg_CTR'].min(), text="💧 Leaks", showarrow=False, xanchor='left', yanchor='bottom', font=dict(color="red"))
    fig.add_annotation(x=agg_df['Avg_Position'].max(), y=agg_df['Avg_CTR'].min(), text="🤔 Underperformers", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="grey"))

    st.plotly_chart(fig, use_container_width=True)


# ---- Main App ----
st.title("📈 GrowthOracle — Module 1: Engagement vs. Search")
st.caption("Identify CTR Leaks, Hidden Gems, and High-Bounce pages from your GSC and CMS data.")

# --- Step 1: Upload Files ---
st.subheader("1. Upload Your Data")
col1, col2, col3 = st.columns(3)
prod_file = col1.file_uploader("Production Data (CSV)", type="csv")
gsc_file = col2.file_uploader("GSC Data (CSV)", type="csv")
ga4_file = col3.file_uploader("GA4 Data (CSV, optional)", type="csv")

if not all([prod_file, gsc_file]):
    st.info("Please upload Production & GSC files to begin analysis.")
    st.stop()

# --- Step 2: Read and Map Columns ---
vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read) if ga4_file else None

if any(df is None for df in [prod_df_raw, gsc_df_raw]):
    st.error("Critical error reading files. Please check the files and try again.")
    st.dataframe(vc_read.to_dataframe())
    st.stop()

prod_map_guess, ga4_map_guess, gsc_map_guess = _guess_colmap(prod_df_raw, ga4_df_raw, gsc_df_raw)

with st.expander("2. Verify Column Mappings", expanded=False):
    st.markdown("**Production Data**")
    c1, c2, c3, c4 = st.columns(4)
    prod_map = {
        "msid": c1.selectbox("MSID", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess.get('msid')) if prod_map_guess.get('msid') else 0),
        "title": c2.selectbox("Title", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess.get('title')) if prod_map_guess.get('title') else 1),
        "path": c3.selectbox("Path/URL", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess.get('path')) if prod_map_guess.get('path') else 2),
        "publish": c4.selectbox("Publish Time", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess.get('publish')) if prod_map_guess.get('publish') else 3),
    }

    st.markdown("**GSC Data**")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    gsc_map = {
        "date": c1.selectbox("Date", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('date')) if gsc_map_guess.get('date') else 0),
        "page": c2.selectbox("Page URL", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('page')) if gsc_map_guess.get('page') else 1),
        "query": c3.selectbox("Query", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('query')) if gsc_map_guess.get('query') else 2),
        "clicks": c4.selectbox("Clicks", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('clicks')) if gsc_map_guess.get('clicks') else 3),
        "impr": c5.selectbox("Impressions", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('impr')) if gsc_map_guess.get('impr') else 4),
        "ctr": c6.selectbox("CTR", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('ctr')) if gsc_map_guess.get('ctr') else 5),
        "pos": c7.selectbox("Position", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess.get('pos')) if gsc_map_guess.get('pos') else 6),
    }
    ga4_map = {}
    if ga4_df_raw is not None:
        st.markdown("**GA4 Data**")
        c1, c2 = st.columns(2)
        ga4_map['msid'] = c1.selectbox("MSID (GA4)", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess.get('msid')) if ga4_map_guess.get('msid') else 0)
        ga4_map['bounce'] = c2.selectbox("Bounce Rate", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess.get('bounce')) if ga4_map_guess.get('bounce') else 1)


# --- Step 3: Process Data and Configure Analysis ---
with st.spinner("Processing and merging datasets..."):
    master_df, vc_process = process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)

if master_df is None or master_df.empty:
    st.error("Data processing failed. Please check your mappings and file contents.")
    st.dataframe(vc_process.to_dataframe())
    st.stop()

st.success(f"✅ Data processed successfully! Master dataset has {master_df.shape[0]:,} rows.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    
    # Date Range Filter
    if "date" in master_df.columns:
        min_date, max_date = master_df["date"].min(), master_df["date"].max()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.warning("Start date cannot be after end date. Swapping.")
            start_date, end_date = end_date, start_date
        
        # Apply date filter
        master_df['date_dt'] = pd.to_datetime(master_df['date']).dt.date
        filtered_df = master_df[(master_df['date_dt'] >= start_date) & (master_df['date_dt'] <= end_date)].copy()
    else:
        filtered_df = master_df.copy()

    st.subheader("Thresholds")
    TH = CONFIG["thresholds"]
    TH["ctr_deficit_pct"] = st.slider("CTR Deficit % (for Leaks)", 5.0, 75.0, TH["ctr_deficit_pct"], 5.0)
    TH["min_impressions"] = st.number_input("Minimum Impressions", 0, 10000, TH["min_impressions"], 50)
    TH["good_position_max"] = st.slider("Max 'Good' Position", 1, 20, TH["good_position_max"])
    TH["poor_position_min"] = st.slider("Min 'Poor' Position", 10, 50, TH["poor_position_min"])
    
    st.subheader("AI Recommendations")
    use_ai = st.toggle("Enable AI-Generated Advice", value=False)
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"], disabled=not use_ai)
    max_rows_for_ai = st.slider("Context Rows for AI Prompt", 2, 10, 5, disabled=not use_ai)

# --- Step 4: Run Analysis and Display Results ---
st.header("📊 Analysis & Recommendations")
mismatch_df = build_mismatch_table(filtered_df, TH)

create_l1_aggregated_chart(filtered_df)

if mismatch_df.empty:
    st.info("No significant performance mismatches found with the current thresholds. Try adjusting the sliders in the sidebar.")
    st.stop()

st.info(f"Found **{len(mismatch_df)}** opportunities across {mismatch_df['Mismatch_Tag'].nunique()} categories.")

# Display recommendations
tags_found = mismatch_df["Mismatch_Tag"].unique()
for tag in ["CTR Leak", "Hidden Gem", "High Bounce"]:
    if tag in tags_found:
        with st.container(border=True):
            if use_ai:
                st.markdown(ai_recommendations(tag, mismatch_df, provider, max_rows_for_ai))
            else:
                st.markdown(_biz_card(tag, {})) # Pass empty dict as row is not used in this template version

# --- Step 5: Data Export ---
st.header("💾 Data Export")
download_df_button(mismatch_df, "growth_opportunities.csv", "Download All Identified Opportunities (CSV)")

with st.expander("Preview Opportunity Data"):
    st.dataframe(mismatch_df.head(100))

with st.expander("View Original Article-Level Bubble Chart"):
    st.markdown("This chart shows individual articles. It can be cluttered but is useful for deep dives.")
    fig_bubble = px.scatter(
        filtered_df.dropna(subset=['Position', 'CTR']),
        x="Position", y="CTR", size="Impressions", color="L1_Category",
        hover_name="Title", hover_data=["Query", "Clicks"],
        title="Article-Level Performance: CTR vs. Position"
    )
    fig_bubble.update_layout(xaxis_autorange='reversed')
    st.plotly_chart(fig_bubble, use_container_width=True)
