#!/usr/bin/env python3
"""
Unit test suite for DeterministicEarningsParser and Earnings DB operations.
"""
import os
import sys
import unittest
from intelligence import DeterministicEarningsParser
from database import DatabaseManager

class TestEarningsTracker(unittest.TestCase):

    def setUp(self):
        self.db_path = "test_earnings.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        if hasattr(self, "db"):
            self.db.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_detect_exact_future_earnings(self):
        h = "Bain Capital Specialty Finance, Inc. to Report Q2 2026 Financial Results on August 18 at 10:00 a.m. ET"
        res = DeterministicEarningsParser.detect_future_earnings(h, company="Bain Capital Specialty Finance", url="https://example.com/pr1")
        self.assertIsNotNone(res)
        self.assertEqual(res["quarter"], "Q2 2026")
        self.assertEqual(res["reporting_date"], "2026-08-18")
        self.assertEqual(res["status"], "CONFIRMED")
        self.assertEqual(res["reporting_date_precision"], "EXACT")
        self.assertEqual(res["source_url"], "https://example.com/pr1")

    def test_02_detect_market_timing_time_precision(self):
        h = "Blue Owl Technology Finance Corp. announces Q3 2026 results before market open on November 5"
        res = DeterministicEarningsParser.detect_future_earnings(h, company="Blue Owl", url="https://example.com/pr2")
        self.assertIsNotNone(res)
        self.assertEqual(res["quarter"], "Q3 2026")
        self.assertEqual(res["reporting_date"], "2026-11-05")
        self.assertEqual(res["reporting_time_precision"], "BEFORE_MARKET_OPEN")

    def test_03_detect_week_window_pending_review(self):
        h = "Sixth Street Specialty Lending expects to release results during the week of August 17"
        res = DeterministicEarningsParser.detect_future_earnings(h, company="Sixth Street", url="https://example.com/pr3")
        self.assertIsNotNone(res)
        self.assertEqual(res["reporting_date_precision"], "WEEK")
        self.assertEqual(res["status"], "PENDING_REVIEW")

    def test_04_extract_quarterly_metrics(self):
        h = "BCSF Reports Q2 2026 NAV of $15.20 per share, NII of $0.42, and $0.38 Dividend with non-accruals of 1.2%"
        res = DeterministicEarningsParser.extract_quarterly_metrics(h, company="BCSF")
        self.assertIsNotNone(res)
        self.assertEqual(res["nav_per_share"], 15.20)
        self.assertEqual(res["nii_per_share"], 0.42)
        self.assertEqual(res["dividend_regular"], 0.38)
        self.assertEqual(res["non_accrual_pct"], 1.2)

    def test_05_db_confirmed_over_estimated_protection(self):
        # 1. Save CONFIRMED event
        c1 = {
            "company_name": "BCSF",
            "quarter": "Q2 2026",
            "reporting_date": "2026-08-18",
            "status": "CONFIRMED",
            "date_source": "PRESS_RELEASE"
        }
        self.assertTrue(self.db.save_earnings_calendar(c1))

        # 2. Try overwriting with ESTIMATED event (should be blocked by save_earnings_calendar)
        c2 = {
            "company_name": "BCSF",
            "quarter": "Q2 2026",
            "reporting_date": "2026-08-15",
            "status": "ESTIMATED",
            "date_source": "HISTORICAL_PATTERN"
        }
        saved = self.db.save_earnings_calendar(c2)
        self.assertFalse(saved)

        # 3. Verify CONFIRMED remains intact
        events = self.db.get_earnings_calendar(company="BCSF", quarter="Q2 2026")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "CONFIRMED")
        self.assertEqual(events[0]["reporting_date"], "2026-08-18")

if __name__ == "__main__":
    unittest.main()
