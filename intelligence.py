#!/usr/bin/env python3
"""
intelligence.py — Scoring, Sentiment, Classification & Query Builder
=====================================================================
Financial NLP engine with BM25 scoring, TextBlob + financial lexicon
sentiment analysis, taxonomy classification, and search query building.
"""

import re
import math
import collections
import datetime
from typing import List, Dict, Tuple, Optional, Any

from constants import (
    logger, SOURCE_RELIABILITY, TAXONOMY_MAP, NOISE_WORDS,
    FINANCIAL_LEXICON,
)

# ---------------------------------------------------------------------------
# Optional: TextBlob (lightweight, ~2MB RAM)
# ---------------------------------------------------------------------------
HAS_TEXTBLOB = False
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
    logger.info("TextBlob sentiment engine loaded.")
except ImportError:
    logger.info("TextBlob not installed – using financial lexicon only.")

HAS_VADER = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
    logger.info("VADER sentiment engine loaded (fallback).")
except ImportError:
    pass

HAS_FINVADER = False
try:
    from finvader import finvader
    HAS_FINVADER = True
    logger.info("FinVADER sentiment engine loaded.")
except ImportError:
    logger.info("FinVADER library not installed.")

HAS_FINSENTIMENT = False
_fin_analyzer = None
try:
    from fin_sentiment import SentimentAnalyzer as FinSentimentAnalyzer
    _fin_analyzer = FinSentimentAnalyzer()
    HAS_FINSENTIMENT = True
    logger.info("fin-sentiment PyTorch model engine loaded.")
except Exception as e:
    logger.info("fin-sentiment package not loaded: %s", e)


# ---------------------------------------------------------------------------
# 1. Financial & Credit Risk Lexicon
# ---------------------------------------------------------------------------
class FinancialLexicon:
    """Categorized Loughran-McDonald & Banking/BDC Credit Risk Vocabulary."""

    ASSET_QUALITY_TERMS = {
        "loan loss provisions": -0.8, "loan loss reserves": -0.7, "provisioning": -0.7,
        "provision expense": -0.8, "provisioning expense": -0.8,
        "non-performing loans": -0.9, "non performing loans": -0.9, "npl": -0.85, "npa": -0.85,
        "non-accrual": -0.85, "non accrual": -0.85, "charge-offs": -0.8, "charge offs": -0.8,
        "credit losses": -0.85, "credit cost": -0.75, "credit costs": -0.75, "impairment": -0.75,
        "impairments": -0.75, "stage 3 loans": -0.8, "delinquencies": -0.7, "delinquency": -0.7,
        "default": -0.95, "defaults": -0.95, "defaulted": -0.95, "bad loans": -0.8,
    }

    CAPITAL_LIQUIDITY_TERMS = {
        "cet1": 0.5, "tier 1 capital": 0.5, "capital adequacy": 0.5, "capital raise": 0.6,
        "capital shortfall": -0.85, "liquidity coverage ratio": 0.5, "lcr": 0.4,
        "liquidity pressure": -0.85, "liquidity concerns": -0.8, "deposit outflows": -0.85,
        "deposit outflow": -0.85, "refinancing": -0.3, "debt issuance": 0.2, "credit facility": 0.3,
        "credit line": 0.3, "borrowing costs": -0.6, "funding pressure": -0.8, "funding spreads": -0.6,
        "yield spread": -0.5, "debt yield": -0.4,
    }

    EARNINGS_NAV_TERMS = {
        "net income": 0.4, "revenue": 0.3, "earnings": 0.4, "eps": 0.4, "guidance": 0.3,
        "profit warning": -0.9, "revenue shortfall": -0.8, "nav": 0.3, "book value": 0.3,
        "markdown": -0.75, "markdowns": -0.75, "dividend cut": -0.85, "dividend reduced": -0.8,
        "dividend target": 0.1, "dividend increase": 0.7, "payout": 0.2, "share repurchase": 0.6,
        "buyback": 0.6,
    }

    GOVERNANCE_REGULATORY_TERMS = {
        "sec investigation": -0.9, "sec launch": -0.9, "investigation": -0.75,
        "occ action": -0.85, "enforcement action": -0.85,
        "subpoena": -0.8, "lawsuit": -0.75, "litigation": -0.6, "ceo resignation": -0.7,
        "cfo resignation": -0.75, "management change": -0.3, "accounting practices": -0.7,
        "compliance failure": -0.85, "regulatory fine": -0.8, "regulatory penalty": -0.8,
        "settles": 0.9, "settled": 0.9, "investigation closed": 0.9,
    }

    RATING_ACTION_TERMS = {
        "rating watch negative": -0.95, "watch negative": -0.95, "negative outlook": -0.85,
        "downgrade": -0.9, "downgraded": -0.9, "rating cut": -0.9, "creditwatch negative": -0.95,
        "rating affirmation": 0.0, "affirmed": 0.0, "rating withdrawn": -0.5,
        "rating watch positive": 0.8, "watch positive": 0.8, "positive outlook": 0.7,
        "upgrade": 0.85, "upgraded": 0.85, "upgrades": 0.85, "rating boost": 0.8,
        "fitch": 0.0, "moody's": 0.0, "s&p": 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Contextual Sentiment & Direction Parser
# ---------------------------------------------------------------------------
class ContextualSentimentParser:
    """Parses direction, context flips, negation, and magnitude in headlines."""

    @staticmethod
    def parse_signals(headline: str) -> List[Dict[str, Any]]:
        h_lower = headline.lower()
        signals = []

        # Helper: Check direction modifiers near a term
        def detect_direction(term: str, base_dim: str, base_score: float) -> Tuple[str, float, str]:
            # Regex match term in headline
            pos = h_lower.find(term.lower())
            if pos == -1:
                return "neutral", 0.0, ""

            # Check window around term (35 chars before and after)
            start = max(0, pos - 35)
            end = min(len(h_lower), pos + len(term) + 35)
            window = h_lower[start:end]

            # 1. Check Mitigators / Low Risk Negations ("remains low", "avoided", "averted", "eased", "recede")
            if re.search(r'\b(remains? low|stayed? low|avoided|averted|receded?|eased|nothin|no default|without default)\b', window):
                return "positive", abs(base_score) * 0.7, f"Risk mitigator detected near '{term}': threat remains low or averted."

            # 2. Check Favorable Direction ("declined", "decreased", "fell", "dropped", "reduced", "lower", "cut")
            if re.search(r'\b(declined?|decreased?|fell|dropped?|reduced?|lower|cut|slashed?|shrank)\b', window):
                if base_score < 0:  # e.g. "provisions declined" -> Positive for asset quality!
                    return "positive", abs(base_score) * 0.8, f"Favorable trend: '{term}' declined/reduced."
                else:  # e.g. "revenue declined" -> Negative for earnings!
                    return "negative", -abs(base_score), f"Adverse trend: '{term}' declined."

            # 3. Check Adverse Direction ("increased", "rose", "spiked", "mounted", "worsened", "grew", "surged")
            if re.search(r'\b(increased?|rose|spiked?|mounted?|worsened?|grew|surged?|soared?|higher|up)\b', window):
                if base_score < 0:  # e.g. "provisions increased" -> Severe negative!
                    return "negative", -abs(base_score) * 1.1, f"Adverse trend: '{term}' increased/spiked."
                else:  # e.g. "revenue increased" -> Positive!
                    return "positive", abs(base_score), f"Favorable trend: '{term}' increased."

            # 4. Rating Watch / Downgrade explicit phrases
            if "watch negative" in window or "negative watch" in window or "creditwatch" in window:
                return "negative", -0.95, f"Explicit rating watch negative action on '{term}'."

            # Default direction based on base_score
            dir_str = "negative" if base_score < 0 else "positive" if base_score > 0 else "neutral"
            reason_str = f"Direct term match for '{term}' with baseline score {base_score}."
            return dir_str, base_score, reason_str

        # Check Asset Quality
        for term, score in FinancialLexicon.ASSET_QUALITY_TERMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', h_lower):
                direction, severity, reason = detect_direction(term, "asset_quality", score)
                signals.append({
                    "term": term,
                    "dimension": "asset_quality",
                    "direction": direction,
                    "severity": abs(severity),
                    "score": severity,
                    "matched_phrase": term,
                    "reason": reason or f"Asset quality term '{term}' detected."
                })

        # Check Capital & Liquidity
        for term, score in FinancialLexicon.CAPITAL_LIQUIDITY_TERMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', h_lower):
                direction, severity, reason = detect_direction(term, "capital_liquidity", score)
                signals.append({
                    "term": term,
                    "dimension": "capital_liquidity",
                    "direction": direction,
                    "severity": abs(severity),
                    "score": severity,
                    "matched_phrase": term,
                    "reason": reason or f"Capital/liquidity term '{term}' detected."
                })

        # Check Earnings & NAV
        for term, score in FinancialLexicon.EARNINGS_NAV_TERMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', h_lower):
                direction, severity, reason = detect_direction(term, "earnings_nav", score)
                signals.append({
                    "term": term,
                    "dimension": "earnings_nav",
                    "direction": direction,
                    "severity": abs(severity),
                    "score": severity,
                    "matched_phrase": term,
                    "reason": reason or f"Earnings/NAV term '{term}' detected."
                })

        # Check Governance & Regulatory
        for term, score in FinancialLexicon.GOVERNANCE_REGULATORY_TERMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', h_lower):
                direction, severity, reason = detect_direction(term, "governance_regulatory", score)
                signals.append({
                    "term": term,
                    "dimension": "governance_regulatory",
                    "direction": direction,
                    "severity": abs(severity),
                    "score": severity,
                    "matched_phrase": term,
                    "reason": reason or f"Governance/regulatory term '{term}' detected."
                })

        # Check Rating Actions
        for term, score in FinancialLexicon.RATING_ACTION_TERMS.items():
            if term in ["fitch", "moody's", "s&p"] and not any(w in h_lower for w in ["downgrade", "upgrade", "watch", "outlook", "rating"]):
                continue  # Skip standalone rating agency name without action word
            if re.search(r'\b' + re.escape(term) + r'\b', h_lower):
                direction, severity, reason = detect_direction(term, "rating_action", score)
                signals.append({
                    "term": term,
                    "dimension": "rating_action",
                    "direction": direction,
                    "severity": abs(severity),
                    "score": severity,
                    "matched_phrase": term,
                    "reason": reason or f"Rating action term '{term}' detected."
                })

        return signals


# ---------------------------------------------------------------------------
# 3. Credit Risk Rule Engine
# ---------------------------------------------------------------------------
class CreditRiskRuleEngine:
    """Evaluates multi-dimensional status based on contextual signals."""

    @staticmethod
    def evaluate(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        dimensions = {
            "asset_quality": {"status": "NOT_MENTIONED", "direction": "neutral", "severity": 0.0},
            "capital_liquidity": {"status": "NOT_MENTIONED", "direction": "neutral", "severity": 0.0},
            "earnings_nav": {"status": "NOT_MENTIONED", "direction": "neutral", "severity": 0.0},
            "governance_regulatory": {"status": "NOT_MENTIONED", "direction": "neutral", "severity": 0.0},
            "rating_action": {"status": "NOT_MENTIONED", "direction": "neutral", "severity": 0.0},
        }

        # Process by dimension
        by_dim = collections.defaultdict(list)
        for s in signals:
            by_dim[s["dimension"]].append(s)

        # Asset Quality
        if "asset_quality" in by_dim:
            s_list = by_dim["asset_quality"]
            net_score = sum(s["score"] for s in s_list)
            max_sev = max(s["severity"] for s in s_list)
            if net_score < -0.15:
                dimensions["asset_quality"] = {"status": "DETERIORATING", "direction": "negative", "severity": round(max_sev, 2)}
            elif net_score > 0.15:
                dimensions["asset_quality"] = {"status": "IMPROVING", "direction": "positive", "severity": round(max_sev, 2)}
            else:
                dimensions["asset_quality"] = {"status": "STABLE", "direction": "neutral", "severity": round(max_sev, 2)}

        # Capital & Liquidity
        if "capital_liquidity" in by_dim:
            s_list = by_dim["capital_liquidity"]
            net_score = sum(s["score"] for s in s_list)
            max_sev = max(s["severity"] for s in s_list)
            if net_score < -0.15:
                dimensions["capital_liquidity"] = {"status": "STRAINED", "direction": "negative", "severity": round(max_sev, 2)}
            elif net_score > 0.15:
                dimensions["capital_liquidity"] = {"status": "STRONG", "direction": "positive", "severity": round(max_sev, 2)}
            else:
                dimensions["capital_liquidity"] = {"status": "ADEQUATE", "direction": "neutral", "severity": round(max_sev, 2)}

        # Earnings & NAV
        if "earnings_nav" in by_dim:
            s_list = by_dim["earnings_nav"]
            net_score = sum(s["score"] for s in s_list)
            max_sev = max(s["severity"] for s in s_list)
            if net_score < -0.15:
                dimensions["earnings_nav"] = {"status": "ADVERSE", "direction": "negative", "severity": round(max_sev, 2)}
            elif net_score > 0.15:
                dimensions["earnings_nav"] = {"status": "POSITIVE", "direction": "positive", "severity": round(max_sev, 2)}
            else:
                dimensions["earnings_nav"] = {"status": "NEUTRAL", "direction": "neutral", "severity": round(max_sev, 2)}

        # Governance & Regulatory
        if "governance_regulatory" in by_dim:
            s_list = by_dim["governance_regulatory"]
            net_score = sum(s["score"] for s in s_list)
            max_sev = max(s["severity"] for s in s_list)
            if net_score < -0.15:
                dimensions["governance_regulatory"] = {"status": "ADVERSE", "direction": "negative", "severity": round(max_sev, 2)}
            elif net_score > 0.15:
                dimensions["governance_regulatory"] = {"status": "POSITIVE", "direction": "positive", "severity": round(max_sev, 2)}
            else:
                dimensions["governance_regulatory"] = {"status": "NEUTRAL", "direction": "neutral", "severity": round(max_sev, 2)}

        # Rating Action
        if "rating_action" in by_dim:
            s_list = by_dim["rating_action"]
            net_score = sum(s["score"] for s in s_list)
            max_sev = max(s["severity"] for s in s_list)
            if net_score < -0.15:
                dimensions["rating_action"] = {"status": "DOWNGRADE_RISK", "direction": "negative", "severity": round(max_sev, 2)}
            elif net_score > 0.15:
                dimensions["rating_action"] = {"status": "UPGRADE", "direction": "positive", "severity": round(max_sev, 2)}
            else:
                dimensions["rating_action"] = {"status": "NEUTRAL", "direction": "neutral", "severity": round(max_sev, 2)}

        return dimensions


# ---------------------------------------------------------------------------
# 4. Master Credit Risk Intelligence Engine
# ---------------------------------------------------------------------------
class CreditRiskIntelligenceEngine:
    """Master engine producing the full multi-dimensional credit risk matrix."""

    @staticmethod
    def analyze(headline: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Step 1: Compute baseline VADER score
        baseline_vader_score = 0.0
        if HAS_VADER:
            try:
                vs = _vader.polarity_scores(headline)
                baseline_vader_score = round(vs["compound"], 3)
            except Exception:
                pass

        # Step 1b: Optional FinVADER Python library calculation
        finvader_score = None
        finvader_label = None
        finvader_enabled = (config or {}).get("finvader_enabled", False)
        if HAS_FINVADER and (finvader_enabled or config is None):
            try:
                f_score = finvader(headline, indicator='compound', use_sentibignomics=True, use_henry=True)
                finvader_score = round(float(f_score), 3)
                if finvader_score >= 0.2:
                    finvader_label = "Positive"
                elif finvader_score <= -0.2:
                    finvader_label = "Negative"
                else:
                    finvader_label = "Neutral"
            except Exception as e:
                logger.debug("FinVADER library calculation error: %s", e)

        # Step 1c: Optional fin-sentiment PyTorch package calculation
        fin_sentiment_score = None
        fin_sentiment_label = None
        fin_sentiment_enabled = (config or {}).get("fin_sentiment_enabled", False)
        if HAS_FINSENTIMENT and _fin_analyzer and (fin_sentiment_enabled or config is None):
            try:
                res = _fin_analyzer.analyze(headline)
                if res == 1:
                    fin_sentiment_label = "Negative"
                    fin_sentiment_score = -0.75
                elif res == 2:
                    fin_sentiment_label = "Positive"
                    fin_sentiment_score = 0.75
                else:
                    fin_sentiment_label = "Neutral"
                    fin_sentiment_score = 0.0
            except Exception as e:
                logger.debug("fin-sentiment calculation error: %s", e)

        # Step 2: Parse contextual financial signals
        signals = ContextualSentimentParser.parse_signals(headline)

        # Step 3: Evaluate dimensions
        dimensions = CreditRiskRuleEngine.evaluate(signals)

        # Step 4: Determine Overall Credit Risk Level & Key Risk Signal
        has_negative_rating = dimensions["rating_action"]["status"] == "DOWNGRADE_RISK"
        has_deteriorating_asset = dimensions["asset_quality"]["status"] == "DETERIORATING"
        has_strained_capital = dimensions["capital_liquidity"]["status"] == "STRAINED"
        has_adverse_gov = dimensions["governance_regulatory"]["status"] == "ADVERSE"
        has_adverse_earnings = dimensions["earnings_nav"]["status"] == "ADVERSE"

        has_positive_rating = dimensions["rating_action"]["status"] == "UPGRADE"
        has_improving_asset = dimensions["asset_quality"]["status"] == "IMPROVING"
        has_strong_capital = dimensions["capital_liquidity"]["status"] == "STRONG"
        has_positive_earnings = dimensions["earnings_nav"]["status"] == "POSITIVE"

        if has_negative_rating or has_deteriorating_asset or has_strained_capital or has_adverse_gov:
            credit_risk = "HIGH"
            overall_sentiment = "VERY_NEGATIVE" if (has_negative_rating and has_deteriorating_asset) else "NEGATIVE"
        elif has_adverse_earnings:
            credit_risk = "MEDIUM"
            overall_sentiment = "NEGATIVE"
        elif has_positive_rating or has_improving_asset or has_strong_capital:
            credit_risk = "FAVORABLE"
            overall_sentiment = "VERY_POSITIVE" if (has_positive_rating and has_strong_capital) else "POSITIVE"
        elif has_positive_earnings:
            credit_risk = "LOW"
            overall_sentiment = "POSITIVE"
        elif signals:
            credit_risk = "LOW"
            overall_sentiment = "NEUTRAL"
        else:
            credit_risk = "NEUTRAL"
            overall_sentiment = "NEUTRAL"

        # Override sentiment if finvader_enabled is explicitly set and finvader_label is available
        if finvader_enabled and finvader_label and not signals:
            overall_sentiment = finvader_label.upper()

        # Determine Key Risk Signal
        if has_negative_rating:
            key_risk_signal = "Rating Downgrade Risk"
        elif has_deteriorating_asset:
            key_risk_signal = "Asset Quality Deterioration"
        elif has_strained_capital:
            key_risk_signal = "Capital & Liquidity Stress"
        elif has_adverse_gov:
            key_risk_signal = "Governance & Legal Action"
        elif has_adverse_earnings:
            key_risk_signal = "Earnings & NAV Shortfall"
        elif has_positive_rating:
            key_risk_signal = "Rating Upgrade"
        elif has_improving_asset:
            key_risk_signal = "Asset Quality Improving"
        elif has_strong_capital:
            key_risk_signal = "Capital Structure Strengthening"
        elif signals:
            key_risk_signal = "Financial Event Tracked"
        elif finvader_enabled and finvader_label and finvader_label != "Neutral":
            key_risk_signal = f"FinVADER {finvader_label}"
        else:
            key_risk_signal = "Routine News"

        # Confidence calculation
        if signals:
            max_sev = max(s["severity"] for s in signals)
            confidence = min(0.95, round(0.65 + (max_sev * 0.3), 2))
        else:
            confidence = 0.50 if headline else 0.00

        # Build final structured output matrix
        matrix = {
            "overall_sentiment": overall_sentiment,
            "credit_risk": credit_risk,
            "key_risk_signal": key_risk_signal,
            "baseline_vader_score": baseline_vader_score,
            "finvader_score": finvader_score,
            "finvader_label": finvader_label,
            "fin_sentiment_score": fin_sentiment_score,
            "fin_sentiment_label": fin_sentiment_label,
            "asset_quality": dimensions["asset_quality"],
            "capital_liquidity": dimensions["capital_liquidity"],
            "earnings_nav": dimensions["earnings_nav"],
            "governance_regulatory": dimensions["governance_regulatory"],
            "rating_action": dimensions["rating_action"],
            "confidence": confidence,
            "signals": signals,
        }
        return matrix


# ---------------------------------------------------------------------------
# Backward Compatible Sentiment Analyzer Interface
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    """Backward-compatible adapter that uses CreditRiskIntelligenceEngine."""

    @staticmethod
    def analyze(headline: str) -> Tuple[str, float]:
        matrix = CreditRiskIntelligenceEngine.analyze(headline)
        s_map = {
            "VERY_POSITIVE": ("Very Positive", 0.85),
            "POSITIVE": ("Positive", 0.45),
            "NEUTRAL": ("Neutral", 0.0),
            "NEGATIVE": ("Negative", -0.45),
            "VERY_NEGATIVE": ("Very Negative", -0.85),
        }
        label, score = s_map.get(matrix["overall_sentiment"], ("Neutral", 0.0))
        return label, score



# ---------------------------------------------------------------------------
# BM25 Scorer
# ---------------------------------------------------------------------------
class BM25Scorer:
    """Okapi BM25 scorer for headline text keyword relevance with IDF term weighting."""

    def __init__(self, headlines: List[str] = None, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        headlines = headlines or []
        self.doc_count = len(headlines)

        total_tokens = 0
        self.df = collections.Counter()

        for h in headlines:
            tokens = [w.lower() for w in re.findall(r"\b\w+\b", h)]
            total_tokens += len(tokens)
            self.df.update(set(tokens))

        self.avgdl = (total_tokens / self.doc_count) if self.doc_count > 0 else 10.0
        if self.doc_count < 10:
            self.doc_count = 100

    def score(self, headline: str, keywords: List[str]) -> float:
        """Calculates BM25 score scaled to [0.0, 20.0]."""
        if not keywords or not headline:
            return 0.0

        h_lower = headline.lower()
        words = [w.lower() for w in re.findall(r"\b\w+\b", h_lower)]
        doc_len = len(words)
        if doc_len == 0:
            return 0.0

        word_counts = collections.Counter(words)
        raw_bm25 = 0.0

        for kw in keywords:
            kw_lower = kw.lower().strip()
            if not kw_lower:
                continue
            kw_tokens = [w.lower() for w in re.findall(r"\b\w+\b", kw_lower)]
            if not kw_tokens:
                continue

            if len(kw_tokens) > 1:
                if re.search(r"\b" + re.escape(kw_lower) + r"\b", h_lower):
                    avg_df = sum(self.df.get(t, 5) for t in kw_tokens) / len(kw_tokens)
                    idf = math.log((self.doc_count - avg_df + 0.5) / (avg_df + 0.5) + 1.0)
                    tf = 1
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
                    raw_bm25 += max(0.0, idf) * (num / den) * 1.5
            else:
                t = kw_tokens[0]
                if t in word_counts:
                    tf = word_counts[t]
                    df_val = self.df.get(t, 5)
                    idf = math.log((self.doc_count - df_val + 0.5) / (df_val + 0.5) + 1.0)
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
                    raw_bm25 += max(0.0, idf) * (num / den)

        return round(min(20.0, raw_bm25 * 3.5), 1)


# ---------------------------------------------------------------------------
# Intelligence Engine (Classification + Relevance Scoring)
# ---------------------------------------------------------------------------
class IntelEngine:
    """Financial NLP engine: noise filter, taxonomy classifier, and 7-component relevance scorer."""

    _cached_scorer: Optional[BM25Scorer] = None
    _cached_doc_count: int = -1

    @classmethod
    def get_scorer(cls, headlines: Optional[List[str]] = None) -> BM25Scorer:
        headlines = headlines or []
        if cls._cached_scorer is None or len(headlines) != cls._cached_doc_count:
            cls._cached_scorer = BM25Scorer(headlines)
            cls._cached_doc_count = len(headlines)
        return cls._cached_scorer

    @staticmethod
    def is_noise(headline: str) -> bool:
        h = headline.lower()
        return any(re.search(r"\b" + re.escape(n) + r"\b", h) for n in NOISE_WORDS)

    @classmethod
    def extract_taxonomy_keywords(cls, headline: str) -> List[Tuple[str, str]]:
        h = headline.lower()
        matches = []
        for cat, kws in TAXONOMY_MAP.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", h):
                    matches.append((kw, cat))
        return matches

    @classmethod
    def classify_event(cls, headline: str) -> Tuple[str, str]:
        """Rule-based event classification. Returns (event_category, rule_sentiment)."""
        h = headline.lower()
        prov = any(w in h for w in ["provision", "provisions", "loan loss", "reserves", "npa", "credit cost"])
        inc = any(w in h for w in ["increase", "increased", "rise", "higher", "surge", "climb", "grew", "spike", "jump"])
        dec = any(w in h for w in ["decrease", "lower", "reduced", "decline", "drop", "fell", "cut"])

        if prov and inc:
            return "Loan Loss Provisions", "Very Negative"
        if prov and dec:
            return "Credit Cost Improvement", "Positive"
        if any(w in h for w in ["bankruptcy", "insolvency", "default", "chapter 11"]):
            return "Defaults", "Very Negative"
        if any(w in h for w in ["fraud", "investigation", "lawsuit", "subpoena", "bribe"]):
            return "Fraud / Investigation", "Very Negative"
        if "downgrade" in h or "rating cut" in h:
            return "Downgrade", "Negative"
        if "upgrade" in h or "rating raised" in h:
            return "Upgrade", "Positive"
        if any(w in h for w in ["earnings beat", "record profit", "strong revenue"]):
            return "Earnings Beat", "Very Positive"
        if any(w in h for w in ["earnings miss", "net loss", "profit drop", "revenue fall"]):
            return "Earnings Miss", "Negative"
        if any(w in h for w in ["buyback", "share repurchase", "dividend increase"]):
            return "Buyback / Dividend", "Positive"
        if any(w in h for w in ["resignation", "ceo exit", "ceo steps down", "ousted"]):
            return "CEO Exit", "Negative"
        return "Neutral", ""

    @classmethod
    def analyze(cls, headline: str, source: str,
                portfolio: List[Dict], industries: List[str],
                keywords: List[str], corpus_stats: Optional[Dict[str, Any]] = None,
                provider_count: int = 1, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full analysis pipeline. Returns company, industry, event, sentiment, score.
        Enhanced 7-component relevance scoring.
        """
        h = headline.lower()

        # --- Company & Industry Match ---
        matched_company, matched_industry, company_hit = "General", "General", False
        for p in portfolio:
            for alias in p.get("aliases", [p["company"]]):
                if re.search(r"\b" + re.escape(alias.lower()) + r"\b", h):
                    matched_company, company_hit = p["company"], True
                    matched_industry = p.get("industry", "General")
                    break
            if company_hit:
                break

        # Fallback Industry match from text if company was not in portfolio
        if matched_industry == "General":
            for ind in industries:
                if re.search(r"\b" + re.escape(ind.lower()) + r"\b", h):
                    matched_industry = ind
                    break

        # --- Event Classification ---
        event, rule_sentiment = cls.classify_event(headline)

        # --- Sentiment Analysis (hybrid) ---
        if rule_sentiment:
            sentiment = rule_sentiment
        else:
            sentiment, _ = SentimentAnalyzer.analyze(headline)

        # --- 7-Component Relevance Score ---
        # 1. Company Match (0-30)
        s_comp = 30.0 if company_hit else 5.0

        # 2. BM25 Keyword Relevance (0-20)
        scorer = cls.get_scorer(corpus_stats.get("headlines") if corpus_stats else None)
        s_kw = scorer.score(headline, keywords)

        # 3. Source Reliability (0-15)
        reliability = 50.0
        for sn, sc in SOURCE_RELIABILITY.items():
            if sn in source.lower():
                reliability = float(sc)
                break
        s_src = (reliability / 100.0) * 15.0

        # 4. Freshness Decay (0-15) — NEW
        s_fresh = 10.0  # Default if no date info

        # 5. Industry Match (0-10)
        s_ind = 10.0 if matched_industry != "General" else 2.0

        # 6. Sentiment Intensity (0-5)
        s_sent = 5.0 if sentiment in ("Very Negative", "Very Positive") else (3.0 if sentiment in ("Negative", "Positive") else 1.0)

        # 7. Multi-Provider Bonus (0-5) — NEW
        s_multi = min(5.0, (provider_count - 1) * 2.5) if provider_count > 1 else 0.0

        # Run Credit Risk Intelligence Engine
        matrix = CreditRiskIntelligenceEngine.analyze(headline, config=config)

        score = round(min(100.0, max(0.0, s_comp + s_kw + s_src + s_fresh + s_ind + s_sent + s_multi)), 1)

        return {
            "company": matched_company, "industry": matched_industry,
            "event_category": event, "sentiment": sentiment,
            "relevance_score": score,
            "credit_risk": matrix["credit_risk"],
            "key_risk_signal": matrix["key_risk_signal"],
            "baseline_vader_score": matrix["baseline_vader_score"],
            "credit_risk_matrix": matrix,
        }

# ---------------------------------------------------------------------------
# 5. Deterministic Earnings & Future Reporting Parser
# ---------------------------------------------------------------------------
class DeterministicEarningsParser:
    """CPU-only deterministic regex parser for earnings dates, times, and quarterly metrics."""

    MONTH_MAP = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10,
        'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }

    @classmethod
    def detect_future_earnings(cls, headline: str, company: str = "General", url: str = "") -> Optional[Dict[str, Any]]:
        h_lower = headline.lower()
        if not any(w in h_lower for w in ["report", "results", "conference call", "announces", "earnings", "q1", "q2", "q3", "q4", "quarter"]):
            return None

        # 1. Detect Quarter
        quarter = "Q2 2026"
        q_match = re.search(r'\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b', h_lower)
        y_match = re.search(r'\b(202[4-8])\b', h_lower)
        curr_year = y_match.group(1) if y_match else str(datetime.datetime.now().year)

        if q_match:
            q_raw = q_match.group(1).lower()
            if "first" in q_raw or "q1" in q_raw:
                quarter = f"Q1 {curr_year}"
            elif "second" in q_raw or "q2" in q_raw:
                quarter = f"Q2 {curr_year}"
            elif "third" in q_raw or "q3" in q_raw:
                quarter = f"Q3 {curr_year}"
            elif "fourth" in q_raw or "q4" in q_raw:
                quarter = f"Q4 {curr_year}"

        # 2. Detect Date
        reporting_date = None
        date_precision = "EXACT"

        date_pattern = r'\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,\s*(\d{4}))?\b'

        # Check "week of August 17" first
        w_match = re.search(r'week of\s+' + date_pattern, h_lower)
        if w_match:
            m_str, day_str, year_str = w_match.group(1), w_match.group(2), w_match.group(3)
            month = cls.MONTH_MAP.get(m_str, 8)
            year = int(year_str) if year_str else int(curr_year)
            try:
                reporting_date = f"{year:04d}-{month:02d}-{int(day_str):02d}"
                date_precision = "WEEK"
            except Exception:
                pass

        if not reporting_date:
            d_match = re.search(date_pattern, h_lower)
            if d_match:
                m_str, day_str, year_str = d_match.group(1), d_match.group(2), d_match.group(3)
                month = cls.MONTH_MAP.get(m_str, 8)
                year = int(year_str) if year_str else int(curr_year)
                day = int(day_str)
                try:
                    reporting_date = f"{year:04d}-{month:02d}-{day:02d}"
                except Exception:
                    pass

        if not reporting_date:
            return None

        # 3. Detect Time & Precision
        time_str = "10:00 AM"
        time_precision = "UNKNOWN"

        t_match = re.search(r'\b(\d{1,2}:?\d{0,2})\s*(a\.?m\.?|p\.?m\.?)\b', h_lower)
        if t_match:
            time_str = f"{t_match.group(1)} {t_match.group(2).upper().replace('.', '')}"
            time_precision = "EXACT"
        elif "before market open" in h_lower or "bmo" in h_lower:
            time_str = "Before Market Open"
            time_precision = "BEFORE_MARKET_OPEN"
        elif "after close" in h_lower or "after market" in h_lower or "amc" in h_lower:
            time_str = "After Market Close"
            time_precision = "AFTER_MARKET_CLOSE"

        status = "CONFIRMED" if date_precision == "EXACT" else "PENDING_REVIEW"

        return {
            "company_name": company,
            "quarter": quarter,
            "reporting_date": reporting_date,
            "conf_call_time": time_str,
            "timezone": "ET",
            "webcast_url": url,
            "status": status,
            "date_source": "PRESS_RELEASE",
            "source_url": url,
            "source_headline": headline,
            "reporting_date_precision": date_precision,
            "reporting_time_precision": time_precision,
            "confidence": 0.95 if status == "CONFIRMED" else 0.70,
        }

    @classmethod
    def extract_quarterly_metrics(cls, headline: str, company: str = "General", url: str = "") -> Optional[Dict[str, Any]]:
        h_lower = headline.lower()
        if not any(w in h_lower for w in ["nav", "nii", "dividend", "net investment income", "book value", "per share"]):
            return None

        quarter = "Q2 2026"
        q_match = re.search(r'\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b', h_lower)
        y_match = re.search(r'\b(202[4-8])\b', h_lower)
        curr_year = y_match.group(1) if y_match else str(datetime.datetime.now().year)

        if q_match:
            q_raw = q_match.group(1).lower()
            if "first" in q_raw or "q1" in q_raw:
                quarter = f"Q1 {curr_year}"
            elif "second" in q_raw or "q2" in q_raw:
                quarter = f"Q2 {curr_year}"
            elif "third" in q_raw or "q3" in q_raw:
                quarter = f"Q3 {curr_year}"
            elif "fourth" in q_raw or "q4" in q_raw:
                quarter = f"Q4 {curr_year}"

        nav = None
        nav_m = re.search(r'\b(?:nav|book value)(?:\s+per\s+share)?\s*(?:of)?\s*\$(\d+\.\d{2})\b', h_lower)
        if nav_m:
            nav = float(nav_m.group(1))

        nii = None
        nii_m = re.search(r'\b(?:nii|net investment income)(?:\s+per\s+share)?\s*(?:of)?\s*\$(\d+\.\d{2})\b', h_lower)
        if nii_m:
            nii = float(nii_m.group(1))

        div_reg = None
        div_m = re.search(r'(?:\b(?:dividend|payout)(?:\s+of)?\s*\$(\d+\.\d{2})\b|\$(\d+\.\d{2})\s*(?:regular\s+)?(?:dividend|payout)\b)', h_lower)
        if div_m:
            div_reg = float(div_m.group(1) or div_m.group(2))

        non_accrual = None
        na_m = re.search(r'\bnon[- ]?accruals?\s*(?:of|at|rate)?\s*(\d+\.?\d*)%', h_lower)
        if na_m:
            non_accrual = float(na_m.group(1))

        if not any([nav, nii, div_reg, non_accrual]):
            return None

        return {
            "company_name": company,
            "quarter": quarter,
            "nav_per_share": nav,
            "nii_per_share": nii,
            "dividend_regular": div_reg,
            "dividend_special": 0.0,
            "non_accrual_pct": non_accrual,
            "reported_at": datetime.date.today().strftime("%Y-%m-%d"),
            "source_url": url,
            "source_headline": headline,
        }



# ---------------------------------------------------------------------------
# Query Builder (Per-provider formatting & Query-level Deduplication)
# ---------------------------------------------------------------------------
class QueryBuilder:
    """Generates search queries with alias expansion, priority sorting, and query-level deduplication."""

    @staticmethod
    def company_expression(company: str, aliases: List[str] = None, ticker: str = "") -> str:
        """
        Builds exact company group: ("Full Company Name" OR "Ticker" OR "Alias")
        Matching exact snapshot format: ("Bain Capital Specialty Finance" OR "BCSF")
        """
        terms = []
        c_clean = company.strip()
        if c_clean:
            terms.append(c_clean)

        # Include ticker if available
        if ticker and ticker.strip():
            t_clean = ticker.strip()
            if t_clean not in terms:
                terms.append(t_clean)

        # Include brand aliases if available
        if aliases:
            for a in aliases:
                a_clean = a.strip()
                if a_clean and a_clean not in terms and len(a_clean) >= 3:
                    terms.append(a_clean)

        if len(terms) == 1:
            return f'"{terms[0]}"'
        return "(" + " OR ".join(f'"{t}"' for t in terms) + ")"

    @classmethod
    def build_google(cls, company: str, aliases: List[str], keywords: List[str],
                     domains: List[str] = None, recency: str = "7d", ticker: str = "",
                     start_date: str = "", end_date: str = "") -> str:
        """
        Google News query matching snapshot format:
        ("Company Name" OR "Ticker") AND ("kw1" OR "kw2") when:7d (or after:YYYY-MM-DD before:YYYY-MM-DD)
        """
        comp_str = cls.company_expression(company, aliases, ticker)
        if keywords:
            kw_terms = [f'"{kw}"' if " " in kw else kw for kw in keywords[:6]]
            kw_str = " OR ".join(kw_terms)
            q = f"{comp_str} AND ({kw_str})"
        else:
            q = comp_str

        if domains:
            q += " AND (" + " OR ".join(f"site:{d}" for d in domains[:3]) + ")"

        if recency == "custom" and start_date and end_date:
            q += f" after:{start_date} before:{end_date}"
        elif recency and recency != "any" and recency != "custom":
            q += f" when:{recency}"
        return q

    @classmethod
    def build_bing(cls, company: str, aliases: List[str], keywords: List[str],
                   domains: List[str] = None, ticker: str = "") -> str:
        """Bing News query (simple keyword query)."""
        brand = aliases[1] if aliases and len(aliases) > 1 else company
        if keywords:
            kw_sample = keywords[0] if len(keywords) > 0 else ""
            return f'"{brand}" {kw_sample}'.strip()
        return f'"{brand}"'

    @classmethod
    def build_duckduckgo(cls, company: str, aliases: List[str], keywords: List[str],
                         ticker: str = "") -> str:
        """DuckDuckGo query phrase."""
        brand = aliases[1] if aliases and len(aliases) > 1 else company
        if keywords:
            kws = keywords[:3]
            kw_str = " ".join(kws)
            return f'"{brand}" {kw_str}'
        return f'"{brand}"'

    @classmethod
    def build_gdelt(cls, company: str, aliases: List[str], ticker: str = "") -> str:
        """GDELT query term."""
        terms = [company]
        if ticker:
            terms.append(ticker)
        return " OR ".join(f'"{t}"' for t in terms)

    @classmethod
    def build_broad(cls, company: str, aliases: List[str],
                    domains: List[str] = None, recency: str = "7d", ticker: str = "",
                    start_date: str = "", end_date: str = "") -> str:
        """Broad query matching snapshot format: ("Company Name" OR "Ticker") when:7d."""
        q = cls.company_expression(company, aliases, ticker)
        if domains:
            q += " AND (" + " OR ".join(f"site:{d}" for d in domains[:3]) + ")"
        if recency == "custom" and start_date and end_date:
            q += f" after:{start_date} before:{end_date}"
        elif recency and recency != "any" and recency != "custom":
            q += f" when:{recency}"
        return q

    @classmethod
    def get_applicable_queries_for_company(
        cls,
        company_dict: Dict[str, Any],
        categories: List[Dict[str, Any]],
        domains: List[str] = None,
        recency: str = "7d",
        start_date: str = "",
        end_date: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Computes applicable query sweeps for a company with query-level deduplication:
        1. Evaluates Universal sweeps + Industry sweeps + Company overrides.
        2. Formulates provider-specific query strings.
        3. If two categories result in the identical search query string, executes search ONCE while
           merging all category names, target dimensions, and priority tags so analytical context is 100% preserved.
        """
        company_name = company_dict.get("company") or company_dict.get("company_name", "")
        company_id = company_dict.get("id")
        industry_id = company_dict.get("industry_id")
        ticker = company_dict.get("ticker", "")
        aliases = company_dict.get("aliases", [company_name])

        # Step 1: Filter active categories matching Universal, Industry, or Company scope
        applicable_cats = []
        for cat in sorted(categories, key=lambda c: c.get("priority", 70), reverse=True):
            if not cat.get("enabled", True):
                continue
            scope = (cat.get("scope_type") or "UNIVERSAL").upper()
            c_ind = cat.get("industry_id")
            c_comp = cat.get("company_id")

            if scope == "UNIVERSAL":
                applicable_cats.append(cat)
            elif scope == "INDUSTRY" and industry_id and c_ind == industry_id:
                applicable_cats.append(cat)
            elif scope == "COMPANY" and company_id and c_comp == company_id:
                applicable_cats.append(cat)

        # Step 2: Query-level deduplication (Keyed by canonical Google query string)
        deduped_queries = {}
        for cat in applicable_cats:
            kws = cat.get("keywords", [])
            google_q = cls.build_google(company_name, aliases, kws, domains, recency, ticker=ticker, start_date=start_date, end_date=end_date)
            bing_q = cls.build_bing(company_name, aliases, kws, domains, ticker=ticker)
            ddg_q = cls.build_duckduckgo(company_name, aliases, kws, ticker=ticker)

            norm_key = google_q.strip().lower()
            if norm_key in deduped_queries:
                # Merge category context
                existing = deduped_queries[norm_key]
                if cat["name"] not in existing["category_names"]:
                    existing["category_names"].append(cat["name"])
                if cat.get("target_dimension") and cat["target_dimension"] not in existing["target_dimensions"]:
                    existing["target_dimensions"].append(cat["target_dimension"])
                existing["priority"] = max(existing["priority"], cat.get("priority", 70))
            else:
                deduped_queries[norm_key] = {
                    "category_name": cat["name"],
                    "category_names": [cat["name"]],
                    "target_dimension": cat.get("target_dimension", "Earnings / Cash Flow"),
                    "target_dimensions": [cat.get("target_dimension", "Earnings / Cash Flow")],
                    "scope_type": cat.get("scope_type", "UNIVERSAL"),
                    "industry_id": cat.get("industry_id"),
                    "priority": cat.get("priority", 70),
                    "version": cat.get("version", 1),
                    "keywords": kws,
                    "query": google_q,
                    "bing_query": bing_q,
                    "ddg_query": ddg_q,
                }

        return list(deduped_queries.values())
