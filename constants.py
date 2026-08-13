#!/usr/bin/env python3
"""
constants.py — Configuration, Constants, Taxonomies & Utility Functions
========================================================================
Foundation module with zero external dependencies.
All other modules import from here.
"""

import os
import re
import json
import datetime
import logging
import sys
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NewsIntel")

# ---------------------------------------------------------------------------
# Application Constants
# ---------------------------------------------------------------------------
APP_PORT = 8080
APP_VERSION = "2.0.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "refresh_interval_minutes": 2,
    "enable_notifications": True,
    "enable_sound_alerts": True,
    "sound_volume": 0.8,
    "quiet_hours_enabled": False,
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "07:00",
    "min_relevance_notification": 85,
    "min_relevance_ingest": 40.0,
    "volume_spike_threshold_percent": 200,
    "recency_window": "7d",
    "custom_start_date": "",
    "custom_end_date": "",
    "db_path": os.path.join(APP_DIR, "news_intelligence.db"),
    "theme": "Dark",
    "domain_filter_enabled": False,
    "stagger_delay_seconds": 1.5,
    "recency_window": "7d",
    # Search engine settings
    "google_delay_range": [3, 8],
    "google_max_requests_per_cycle": 20,
    "max_workers": 4,
    "max_retries": 3,
    "page_size": 50,
    # Provider toggles
    "google_enabled": True,
    "bing_enabled": True,
    "duckduckgo_enabled": True,
    "gdelt_enabled": True,
    "rss_enabled": True,
    # User agents
    "user_agents": [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 OPR/107.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    ],
}

# ---------------------------------------------------------------------------
# Source Reliability Scores (0-100)
# ---------------------------------------------------------------------------
SOURCE_RELIABILITY: Dict[str, int] = {
    "reuters": 100, "bloomberg": 98, "wsj": 96, "wall street journal": 96,
    "financial times": 95, "ft.com": 95, "associated press": 95, "ap news": 95,
    "cnbc": 92, "yahoo finance": 90, "marketwatch": 88, "barron's": 90,
    "seeking alpha": 80, "business insider": 75, "forbes": 75,
    "the economist": 94, "nikkei": 88, "bbc": 90, "guardian": 85,
    "new york times": 93, "washington post": 91, "investopedia": 78,
    "morningstar": 88, "s&p global": 95, "moody's": 95, "fitch": 95,
}

# ---------------------------------------------------------------------------
# Default Portfolio (examples — user can delete/modify)
# ---------------------------------------------------------------------------
DEFAULT_PORTFOLIO = [
    {"company": "Bank of America", "ticker": "BAC", "industry": "Commercial Banking", "country": "USA",
     "aliases": ["Bank of America", "BofA", "BAC", "BankAmerica"]},
    {"company": "JPMorgan Chase", "ticker": "JPM", "industry": "US Banks", "country": "USA",
     "aliases": ["JPMorgan", "JP Morgan", "JPM", "JPMorgan Chase"]},
    {"company": "Citigroup", "ticker": "C", "industry": "Commercial Banking", "country": "USA",
     "aliases": ["Citigroup", "Citi", "Citibank"]},
    {"company": "Wells Fargo", "ticker": "WFC", "industry": "Regional Banks", "country": "USA",
     "aliases": ["Wells Fargo", "WFC"]},
    {"company": "Goldman Sachs", "ticker": "GS", "industry": "Asset Management", "country": "USA",
     "aliases": ["Goldman Sachs", "Goldman", "GS"]},
    {"company": "Morgan Stanley", "ticker": "MS", "industry": "Asset Management", "country": "USA",
     "aliases": ["Morgan Stanley", "MS"]},
]

DEFAULT_INDUSTRIES = [
    "US Banks", "Commercial Banking", "Regional Banks", "Insurance",
    "Asset Management", "Financial Services", "Real Estate", "Airlines",
    "Technology", "Healthcare", "Energy", "Retail", "Telecommunications",
]

DEFAULT_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com",
    "marketwatch.com", "finance.yahoo.com",
]

DEFAULT_QUERY_CATEGORIES = [
    {"name": "Earnings & Financials", "keywords": ["earnings", "quarterly results", "profit", "net income", "revenue", "guidance", "NAV", "book value", "EPS", "EBITDA", "margin", "NIM"]},
    {"name": "Asset Quality & Credit Risk", "keywords": ["loan loss reserves", "provisioning", "provision", "provisions", "credit cost", "NPA", "defaults", "write-off", "impairment", "NPL", "charge-off"]},
    {"name": "Credit & Liquidity", "keywords": ["liquidity", "redemption", "credit line", "credit facility", "refinancing", "bond issuance", "debt issue"]},
    {"name": "M&A & Restructuring", "keywords": ["merger", "acquisition", "acquisition talks", "buyback", "share repurchase", "restructuring", "insolvency", "bankruptcy", "divestment", "takeover"]},
    {"name": "Funding & Capital", "keywords": ["funding", "capital raise", "CET1", "capital adequacy", "Basel III", "note offering", "IPO", "secondary offering"]},
    {"name": "Governance & Legal", "keywords": ["CEO resignation", "management change", "fraud", "investigation", "lawsuit", "settlement", "subpoena", "SEC", "board change"]},
    {"name": "Regulatory & Compliance", "keywords": ["regulatory action", "regulatory fine", "penalty", "stress test", "OCC", "FDIC", "Federal Reserve", "sanction", "compliance"]},
    {"name": "Rating Actions", "keywords": ["downgrade", "upgrade", "rating action", "Fitch", "Moody's", "S&P", "rating cut", "rating raised", "outlook negative", "outlook stable"]},
]

# ---------------------------------------------------------------------------
# Taxonomy Map for Event Classification
# ---------------------------------------------------------------------------
TAXONOMY_MAP: Dict[str, List[str]] = {
    "Earnings & Financials": [
        "earnings", "quarterly results", "profit", "net income", "revenue",
        "guidance", "nav", "book value", "eps", "ebitda", "margin", "cash flow", "nim",
    ],
    "Mergers & M&A": [
        "merger", "acquisition", "acquisition talks", "buyback", "share repurchase",
        "restructuring", "insolvency", "divestment", "takeover", "stake sale",
    ],
    "Funding & Capital": [
        "funding", "capital raise", "debt issuance", "bond offer", "liquidity",
        "cet1", "capital adequacy", "credit facility", "note offering", "refinancing", "ipo",
    ],
    "Asset Quality & Credit Risk": [
        "loan loss reserves", "provisioning", "provision", "provisions", "credit cost",
        "npa", "non-performing assets", "defaults", "default", "write-off", "impairment", "charge-off", "npl",
    ],
    "Governance & Regulatory": [
        "ceo resignation", "ceo exit", "management change", "lawsuit", "investigation",
        "regulatory fine", "penalty", "downgrade", "upgrade", "rating action", "subpoena", "sec",
        "board change", "compliance", "sanction",
    ],
}

DEFAULT_KEYWORDS = [
    "earnings", "quarterly results", "profit", "net income", "revenue",
    "guidance", "merger", "acquisition", "acquisition talks", "capital raise",
    "provisioning", "provision", "reserves", "loan loss reserves", "credit cost",
    "NPA", "defaults", "bankruptcy", "insolvency", "restructuring", "redemption",
    "write-off", "downgrade", "upgrade", "bond issuance", "liquidity", "CET1",
    "capital adequacy", "Basel III", "dividend", "buyback", "CEO resignation",
    "management change", "fraud", "investigation", "lawsuit", "settlement",
    "regulatory action", "stress test", "dividend cut", "rating action",
    "Fitch", "Moody's", "S&P", "OCC", "FDIC", "Federal Reserve", "NAV", "book value",
    "NIM", "asset quality", "net interest margin",
]

NOISE_WORDS = [
    "movie", "sports", "celebrity", "box office", "hollywood", "nfl", "nba",
    "game recap", "horoscope", "crypto casino", "airfryer", "recipe",
    "entertainment", "film review", "actor", "actress", "concert", "stadium",
    "gang", "gangs", "killer", "killers", "murder", "murderer", "kidnap",
    "podcast", "tv show", "trailer", "streaming", "fashion", "lottery",
]

# ---------------------------------------------------------------------------
# Financial Sentiment Lexicon (~200 terms)
# ---------------------------------------------------------------------------
FINANCIAL_LEXICON: Dict[str, float] = {
    # Very Negative (-0.7 to -1.0)
    "bankruptcy": -1.0, "fraud": -1.0, "default": -0.9, "insolvency": -0.95,
    "liquidation": -0.95, "ponzi": -1.0, "embezzlement": -1.0, "scandal": -0.85,
    "collapse": -0.9, "crash": -0.85, "plunge": -0.8, "crisis": -0.85,
    "downgrade": -0.8, "rating cut": -0.8, "junk": -0.75, "toxic": -0.8,
    "write-off": -0.75, "impairment": -0.7, "charge-off": -0.7,
    "subpoena": -0.8, "indictment": -0.9, "investigation": -0.7,
    "penalty": -0.7, "fine": -0.6, "lawsuit": -0.65, "litigation": -0.6,
    "provision increase": -0.7, "credit loss": -0.7, "loan loss": -0.65,
    "net loss": -0.75, "profit drop": -0.7, "revenue fall": -0.7,
    "earnings miss": -0.75, "missed estimates": -0.7, "guidance cut": -0.7,
    "dividend cut": -0.7, "dividend suspension": -0.8, "layoffs": -0.65,
    "restructuring": -0.5, "delisting": -0.85, "margin squeeze": -0.6,
    "debt downgrade": -0.8, "outlook negative": -0.65, "sell-off": -0.6,
    "plummet": -0.8, "tumble": -0.7, "tank": -0.7, "slump": -0.65,
    "recession": -0.7, "contraction": -0.6, "decline": -0.5,
    "ceo exit": -0.55, "ceo resignation": -0.55, "ceo fired": -0.7,
    "ceo ousted": -0.7, "management shake-up": -0.5,
    "non-performing": -0.65, "delinquency": -0.6, "forbearance": -0.5,

    # Negative (-0.3 to -0.6)
    "concern": -0.35, "risk": -0.3, "volatile": -0.4, "uncertainty": -0.4,
    "headwind": -0.4, "pressure": -0.35, "weak": -0.45, "slower": -0.35,
    "miss": -0.5, "below expectations": -0.5, "disappointing": -0.5,
    "caution": -0.3, "warning": -0.4, "alert": -0.35,

    # Positive (0.3 to 0.6)
    "growth": 0.5, "expansion": 0.5, "recovery": 0.5, "rebound": 0.55,
    "improvement": 0.45, "stable": 0.3, "resilient": 0.45,
    "outperform": 0.5, "above expectations": 0.55, "beat estimates": 0.6,
    "strong demand": 0.5, "market share gain": 0.5, "innovation": 0.4,
    "efficiency": 0.4, "cost reduction": 0.4, "synergy": 0.45,
    "pipeline": 0.35, "backlog": 0.35, "momentum": 0.45,
    "positive outlook": 0.5, "outlook stable": 0.35,

    # Very Positive (0.7 to 1.0)
    "record profit": 0.9, "record revenue": 0.9, "earnings beat": 0.85,
    "upgrade": 0.8, "rating raised": 0.8, "rating upgrade": 0.8,
    "strong revenue": 0.8, "revenue growth": 0.75, "profit surge": 0.85,
    "earnings surge": 0.85, "dividend increase": 0.7, "dividend hike": 0.7,
    "buyback": 0.6, "share repurchase": 0.6, "special dividend": 0.75,
    "ipo": 0.5, "acquisition": 0.4, "merger": 0.35,
    "breakthrough": 0.8, "landmark deal": 0.75, "all-time high": 0.85,
    "soar": 0.75, "surge": 0.7, "rally": 0.65, "boom": 0.7,
    "outperformance": 0.7, "exceeded expectations": 0.8,
    "guidance raise": 0.75, "guidance increase": 0.75, "strong outlook": 0.7,
    "capital return": 0.6, "margin expansion": 0.65, "market rally": 0.6,
}

# ---------------------------------------------------------------------------
# Known Source Domain Map
# ---------------------------------------------------------------------------
KNOWN_SOURCE_DOMAINS: Dict[str, str] = {
    "reuters": "reuters.com", "bloomberg": "bloomberg.com", "cnbc": "cnbc.com",
    "wsj": "wsj.com", "wall street journal": "wsj.com", "financial times": "ft.com",
    "marketwatch": "marketwatch.com", "yahoo finance": "finance.yahoo.com",
    "yahoo": "yahoo.com", "seeking alpha": "seekingalpha.com", "seekingalpha": "seekingalpha.com",
    "business wire": "businesswire.com", "businesswire": "businesswire.com",
    "pr newswire": "prnewswire.com", "prnewswire": "prnewswire.com",
    "globenewswire": "globenewswire.com", "globe newswire": "globenewswire.com",
    "investing.com": "investing.com", "tradingview": "tradingview.com",
    "stock titan": "stocktitan.net", "stocktitan": "stocktitan.net",
    "marketbeat": "marketbeat.com", "marketscreener": "marketscreener.com",
    "barron's": "barrons.com", "barrons": "barrons.com",
    "business insider": "businessinsider.com", "forbes": "forbes.com",
}


# ---------------------------------------------------------------------------
# Config Loader / Saver
# ---------------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """Loads config from config.json, merging with defaults."""
    cfg_path = os.path.join(APP_DIR, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: Dict[str, Any]):
    """Persists config to config.json."""
    cfg_path = os.path.join(APP_DIR, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def parse_pub_date(raw_date: str) -> str:
    """Parses RFC-822, ISO, or string date formats into standardized YYYY-MM-DD HH:MM."""
    if not raw_date or not raw_date.strip():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    raw_date = raw_date.strip()
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    try:
        clean_date = raw_date.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return raw_date[:16] if len(raw_date) >= 16 else raw_date


def auto_generate_aliases(company: str, user_aliases: List[str] = None) -> List[str]:
    """Generates alias list. Includes full name, brand phrases, primary word, and user aliases."""
    if user_aliases and len(user_aliases) > 0:
        result = []
        for a in user_aliases:
            a_clean = a.strip()
            if a_clean and a_clean not in result:
                result.append(a_clean)
        return result

    clean = company.strip()
    aliases = [clean]
    tokens = clean.split()
    suffixes = {"limited", "ltd", "inc", "corp", "corporation", "plc", "group", "holdings", "co", "co.", "ltd.", "fund", "trust", "capital", "management", "financial", "partners", "specialty", "technology", "investment"}

    # 1. 2-word brand phrase if available (e.g. "Bain Capital", "Sixth Street", "Blue Owl", "Midcap Financial")
    if len(tokens) >= 2:
        t0 = re.sub(r'[^\w]', '', tokens[0])
        t1 = re.sub(r'[^\w]', '', tokens[1])
        if t0.lower() not in ("the", "first", "general") and t1.lower() not in suffixes:
            phrase = f"{tokens[0]} {tokens[1]}"
            if phrase not in aliases:
                aliases.append(phrase)

    # 2. Primary brand word (first word if length >= 3)
    first_word = re.sub(r'[^\w]', '', tokens[0]) if tokens else ""
    if first_word and len(first_word) >= 3 and first_word.lower() not in ("bank", "first", "the", "general", "national"):
        if first_word not in aliases:
            aliases.append(first_word)

    # 3. Short name (excluding corporate suffixes)
    short_tokens = [t for t in tokens if t.lower().strip(".,") not in suffixes]
    if short_tokens and len(short_tokens) < len(tokens):
        short_name = " ".join(short_tokens)
        if short_name and short_name not in aliases:
            aliases.append(short_name)

    return aliases


def extract_domain(url: str, source: str = "") -> str:
    """Extracts publisher domain from URL or Source string."""
    import urllib.parse
    if source and source.strip():
        s_clean = source.strip().lower()
        for k, v in KNOWN_SOURCE_DOMAINS.items():
            if k in s_clean:
                return v
    try:
        d = urllib.parse.urlparse(url).netloc.lower()
        d = d[4:] if d.startswith("www.") else d
        if d and d not in ("news.google.com", "bing.com", "google.com"):
            return d
    except Exception:
        pass
    if source:
        clean_src = re.sub(r'[^\w\.-]', '', source.lower().replace(" ", ""))
        if clean_src and not clean_src.endswith(".com"):
            clean_src += ".com"
        return clean_src
    return ""
