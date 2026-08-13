#!/usr/bin/env python3
"""
test_credit_intelligence.py — Comprehensive Credit Risk Benchmark Suite
========================================================================
Validates direction, context flips, negation, rating actions, multi-signal
coexistence, and backward-compatible sentiment signals.
"""

import sys
from intelligence import CreditRiskIntelligenceEngine, SentimentAnalyzer

BENCHMARK_CASES = [
    # ── 1. Asset Quality & Provisions (Direction & Magnitude) ──
    {
        "headline": "Bank increases loan loss provisions by $500M",
        "expected_credit_risk": "HIGH",
        "expected_asset_quality": "DETERIORATING",
        "expected_signal": "Asset Quality Deterioration",
    },
    {
        "headline": "Loan loss provisions declined 15% YoY for BCSF",
        "expected_credit_risk": "FAVORABLE",
        "expected_asset_quality": "IMPROVING",
        "expected_signal": "Asset Quality Improving",
    },
    {
        "headline": "Credit costs dropped sharply as non-performing loans ease",
        "expected_credit_risk": "FAVORABLE",
        "expected_asset_quality": "IMPROVING",
        "expected_signal": "Asset Quality Improving",
    },
    {
        "headline": "Net charge-offs spiked to 3.5% of total portfolio",
        "expected_credit_risk": "HIGH",
        "expected_asset_quality": "DETERIORATING",
        "expected_signal": "Asset Quality Deterioration",
    },

    # ── 2. Rating Agency Actions (Downgrade Risk vs Upgrade) ──
    {
        "headline": "Fitch places BCSF on Rating Watch Negative",
        "expected_credit_risk": "HIGH",
        "expected_rating_action": "DOWNGRADE_RISK",
        "expected_signal": "Rating Downgrade Risk",
    },
    {
        "headline": "Moody's downgrades credit rating to Baa3 with negative outlook",
        "expected_credit_risk": "HIGH",
        "expected_rating_action": "DOWNGRADE_RISK",
        "expected_signal": "Rating Downgrade Risk",
    },
    {
        "headline": "S&P upgrades credit rating to A2 following capital raise",
        "expected_credit_risk": "FAVORABLE",
        "expected_rating_action": "UPGRADE",
        "expected_signal": "Rating Upgrade",
    },
    {
        "headline": "Fitch affirmed rating at BBB with stable outlook",
        "expected_credit_risk": "LOW",
        "expected_rating_action": "NEUTRAL",
    },

    # ── 3. Adversarial Context & Negation Cases (Must NOT trigger false High Risk) ──
    {
        "headline": "Default risk remains low despite rate hikes",
        "expected_credit_risk": "FAVORABLE",
        "expected_asset_quality": "IMPROVING",  # Mitigator near default term
    },
    {
        "headline": "Bank avoided a default on bond repayment",
        "expected_credit_risk": "FAVORABLE",
    },
    {
        "headline": "Provision expense declined despite weaker revenue",
        "expected_asset_quality": "IMPROVING",
        "expected_earnings_nav": "ADVERSE",  # Multi-signal coexistence!
    },
    {
        "headline": "Liquidity concerns eased following deposit inflows",
        "expected_credit_risk": "FAVORABLE",
        "expected_capital_liquidity": "STRONG",
    },
    {
        "headline": "Rating downgrade fears recede as capital remains well above minimum",
        "expected_credit_risk": "FAVORABLE",
    },

    # ── 4. Capital & Liquidity Stress vs Strength ──
    {
        "headline": "Bank faces liquidity pressure amid deposit outflows",
        "expected_credit_risk": "HIGH",
        "expected_capital_liquidity": "STRAINED",
        "expected_signal": "Capital & Liquidity Stress",
    },
    {
        "headline": "Company raises $2B in equity to strengthen CET1 capital adequacy",
        "expected_credit_risk": "FAVORABLE",
        "expected_capital_liquidity": "STRONG",
        "expected_signal": "Capital Structure Strengthening",
    },
    {
        "headline": "Refinancing debt facilities at elevated yield spreads",
        "expected_capital_liquidity": "STRAINED",
    },

    # ── 5. Earnings, NAV & Dividends ──
    {
        "headline": "BCSF re-evaluates dividend target following NAV markdown",
        "expected_credit_risk": "MEDIUM",
        "expected_earnings_nav": "ADVERSE",
        "expected_signal": "Earnings & NAV Shortfall",
    },
    {
        "headline": "Board announces 15% dividend increase and $100M share buyback",
        "expected_credit_risk": "LOW",
        "expected_earnings_nav": "POSITIVE",
    },
    {
        "headline": "Company posts net loss due to portfolio markdown",
        "expected_credit_risk": "MEDIUM",
        "expected_earnings_nav": "ADVERSE",
    },

    # ── 6. Governance & Regulatory ──
    {
        "headline": "SEC launches investigation into accounting practices",
        "expected_credit_risk": "HIGH",
        "expected_governance_regulatory": "ADVERSE",
        "expected_signal": "Governance & Legal Action",
    },
    {
        "headline": "OCC issues enforcement action and regulatory fine",
        "expected_credit_risk": "HIGH",
        "expected_governance_regulatory": "ADVERSE",
    },
    {
        "headline": "Bank settles regulatory investigation with $5M penalty",
        "expected_governance_regulatory": "POSITIVE",  # Settles / Investigation closed = mitigator
    },
    {
        "headline": "Bank appoints new CEO to lead digital transformation",
        "expected_governance_regulatory": "NOT_MENTIONED",  # Must NOT force Adverse for routine CEO appointment!
    },

    # ── 7. Neutral & Routine Corporate Headlines ──
    {
        "headline": "Bain Capital Specialty Finance Q2 Conference Call Scheduled",
        "expected_credit_risk": "NEUTRAL",
        "expected_asset_quality": "NOT_MENTIONED",
        "expected_capital_liquidity": "NOT_MENTIONED",
    },
    {
        "headline": "MarketAxess Holdings Inc. Presenting at Investor Conference",
        "expected_credit_risk": "NEUTRAL",
        "expected_asset_quality": "NOT_MENTIONED",
    },
]

def run_benchmark():
    print("====================================================================")
    print("📊 MULTI-DIMENSIONAL CREDIT RISK INTELLIGENCE ENGINE BENCHMARK TEST")
    print("====================================================================")

    passed = 0
    total = len(BENCHMARK_CASES)

    for idx, case in enumerate(BENCHMARK_CASES, 1):
        h = case["headline"]
        matrix = CreditRiskIntelligenceEngine.analyze(h)

        case_passed = True
        failures = []

        if "expected_credit_risk" in case and matrix["credit_risk"] != case["expected_credit_risk"]:
            case_passed = False
            failures.append(f"CreditRisk: got '{matrix['credit_risk']}', expected '{case['expected_credit_risk']}'")

        if "expected_asset_quality" in case and matrix["asset_quality"]["status"] != case["expected_asset_quality"]:
            case_passed = False
            failures.append(f"AssetQuality: got '{matrix['asset_quality']['status']}', expected '{case['expected_asset_quality']}'")

        if "expected_capital_liquidity" in case and matrix["capital_liquidity"]["status"] != case["expected_capital_liquidity"]:
            case_passed = False
            failures.append(f"CapitalLiquidity: got '{matrix['capital_liquidity']['status']}', expected '{case['expected_capital_liquidity']}'")

        if "expected_earnings_nav" in case and matrix["earnings_nav"]["status"] != case["expected_earnings_nav"]:
            case_passed = False
            failures.append(f"EarningsNAV: got '{matrix['earnings_nav']['status']}', expected '{case['expected_earnings_nav']}'")

        if "expected_governance_regulatory" in case and matrix["governance_regulatory"]["status"] != case["expected_governance_regulatory"]:
            case_passed = False
            failures.append(f"GovernanceRegulatory: got '{matrix['governance_regulatory']['status']}', expected '{case['expected_governance_regulatory']}'")

        if "expected_rating_action" in case and matrix["rating_action"]["status"] != case["expected_rating_action"]:
            case_passed = False
            failures.append(f"RatingAction: got '{matrix['rating_action']['status']}', expected '{case['expected_rating_action']}'")

        if case_passed:
            passed += 1
            print(f"  ✅ [{idx:02d}/{total}] {h[:60]:<60} | Risk: {matrix['credit_risk']:<8} | Signal: {matrix['key_risk_signal']}")
        else:
            print(f"  ❌ [{idx:02d}/{total}] {h[:60]:<60}")
            for f in failures:
                print(f"       └── {f}")

    print("--------------------------------------------------------------------")
    print(f"🎯 BENCHMARK SUMMARY: {passed}/{total} CASES PASSED ({round(passed/total*100, 1)}%)")
    print("====================================================================")
    return passed == total

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
