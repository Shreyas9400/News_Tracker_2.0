"""
Test Suite: Industry-Aware Query Engine & Multi-Scope Intelligence Dispatcher
Tests:
1. Query Isolation (Energy vs Telecom vs BDC)
2. Null / Unassigned Industry Fallback
3. Company Specific Overrides
4. Search Query Level Deduplication & Multi-Category Context Preservation
5. Safe Industry Lifecycle (Rename display name & Status Toggling)
6. Database Schema & Migration Integrity
"""

import os
import unittest
import tempfile
import sqlite3
import json

from database import DatabaseManager
from intelligence import QueryBuilder
from constants import DEFAULT_INDUSTRIES, DEFAULT_QUERY_CATEGORIES


class TestIndustryQueryEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_intel.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_schema_and_seed(self):
        """Verifies that industries and scoped query categories are seeded with stable IDs."""
        industries = self.db.get_all_industries()
        self.assertTrue(len(industries) >= 10, "Expected at least 10 master industries")
        
        ind_ids = [i["id"] for i in industries]
        self.assertIn("BDC", ind_ids)
        self.assertIn("COMMERCIAL_BANKING", ind_ids)
        self.assertIn("ENERGY", ind_ids)
        self.assertIn("TELECOM", ind_ids)
        self.assertIn("TECH", ind_ids)

        categories = self.db.get_query_categories()
        universal_cats = [c for c in categories if c["scope_type"] == "UNIVERSAL"]
        industry_cats = [c for c in categories if c["scope_type"] == "INDUSTRY"]
        
        self.assertTrue(len(universal_cats) >= 5, "Expected universal query categories")
        self.assertTrue(len(industry_cats) >= 5, "Expected industry query categories")

    def test_02_query_isolation_energy_vs_telecom_vs_bdc(self):
        """Verifies that an Energy company gets Universal + Energy sweeps, but 0 Telecom or BDC sweeps."""
        categories = self.db.get_query_categories(enabled_only=True)

        chevron = {"company": "Chevron", "ticker": "CVX", "industry_id": "ENERGY", "aliases": ["Chevron", "CVX"]}
        verizon = {"company": "Verizon", "ticker": "VZ", "industry_id": "TELECOM", "aliases": ["Verizon", "VZ"]}
        ares = {"company": "Ares Capital", "ticker": "ARCC", "industry_id": "BDC", "aliases": ["Ares Capital", "ARCC"]}

        cvx_queries = QueryBuilder.get_applicable_queries_for_company(chevron, categories)
        vz_queries = QueryBuilder.get_applicable_queries_for_company(verizon, categories)
        arcc_queries = QueryBuilder.get_applicable_queries_for_company(ares, categories)

        # Chevron Checks
        cvx_names = [q["category_name"] for q in cvx_queries]
        self.assertTrue(any("Upstream Production" in name or "Energy" in name for name in cvx_names), "Chevron must have Energy sweeps")
        self.assertTrue(any("Default & Distress" in name or "Earnings" in name for name in cvx_names), "Chevron must have Universal sweeps")
        self.assertFalse(any("Spectrum" in name or "Subscriber" in name for name in cvx_names), "Chevron must NOT have Telecom sweeps")
        self.assertFalse(any("NAV" in name or "NII" in name for name in cvx_names), "Chevron must NOT have BDC sweeps")

        # Verizon Checks
        vz_names = [q["category_name"] for q in vz_queries]
        self.assertTrue(any("Spectrum" in name or "Subscriber" in name for name in vz_names), "Verizon must have Telecom sweeps")
        self.assertFalse(any("Upstream Production" in name or "Rig" in name for name in vz_names), "Verizon must NOT have Energy sweeps")
        self.assertFalse(any("NAV" in name or "NII" in name for name in vz_names), "Verizon must NOT have BDC sweeps")

        # Ares Capital Checks
        arcc_names = [q["category_name"] for q in arcc_queries]
        self.assertTrue(any("NAV" in name or "NII" in name for name in arcc_names), "Ares Capital must have BDC sweeps")
        self.assertFalse(any("Spectrum" in name for name in arcc_names), "Ares Capital must NOT have Telecom sweeps")
        self.assertFalse(any("Upstream Production" in name for name in arcc_names), "Ares Capital must NOT have Energy sweeps")

    def test_03_null_industry_fallback(self):
        """Verifies that an entity with NULL/unassigned industry generates all Universal sweeps cleanly."""
        categories = self.db.get_query_categories(enabled_only=True)
        unassigned_corp = {"company": "Acme Holdings", "ticker": "ACME", "industry_id": None, "aliases": ["Acme"]}

        queries = QueryBuilder.get_applicable_queries_for_company(unassigned_corp, categories)
        self.assertTrue(len(queries) >= 5, "Unassigned company must receive universal sweeps")
        for q in queries:
            self.assertEqual(q["scope_type"], "UNIVERSAL", "Only universal sweeps should apply to unassigned entity")

    def test_04_company_specific_override(self):
        """Verifies company-specific query overrides apply only to that specific company."""
        # Add JPMorgan
        self.db.add_portfolio("JPMorgan Chase", "JPM", "Commercial Banking", "USA", ["JPMorgan"], industry_id="COMMERCIAL_BANKING")
        portfolio = self.db.get_portfolio()
        jpm = next(p for p in portfolio if p["company"] == "JPMorgan Chase")
        jpm_id = jpm["id"]

        # Add BofA
        self.db.add_portfolio("Bank of America", "BAC", "Commercial Banking", "USA", ["BofA"], industry_id="COMMERCIAL_BANKING")
        bofa = next(p for p in self.db.get_portfolio() if p["company"] == "Bank of America")
        bofa_id = bofa["id"]

        # Add Company-specific override sweep for JPM only
        self.db.add_query_category(
            name="JPM Stress Test & CCAR",
            keywords=["CCAR", "stress test", "capital buffer", "Jamie Dimon"],
            scope_type="COMPANY",
            company_id=jpm_id,
            priority=85,
            target_dimension="Leverage / Capital"
        )

        all_cats = self.db.get_query_categories(enabled_only=True)
        jpm_queries = QueryBuilder.get_applicable_queries_for_company(jpm, all_cats)
        bofa_queries = QueryBuilder.get_applicable_queries_for_company(bofa, all_cats)

        jpm_names = [q["category_name"] for q in jpm_queries]
        bofa_names = [q["category_name"] for q in bofa_queries]

        self.assertIn("JPM Stress Test & CCAR", jpm_names, "JPMorgan must have its company override sweep")
        self.assertNotIn("JPM Stress Test & CCAR", bofa_names, "Bank of America must NOT have JPM override sweep")

    def test_05_query_level_deduplication(self):
        """
        Verifies that if two categories formulate the same search query,
        it executes ONCE while preserving all category names and dimensions.
        """
        # Create two categories with same keywords
        cat1 = {
            "name": "General Debt",
            "keywords": ["debt", "refinancing"],
            "scope_type": "UNIVERSAL",
            "priority": 70,
            "target_dimension": "Leverage / Capital",
            "enabled": True
        }
        cat2 = {
            "name": "Corporate Liquidity",
            "keywords": ["debt", "refinancing"],
            "scope_type": "UNIVERSAL",
            "priority": 90,
            "target_dimension": "Liquidity",
            "enabled": True
        }

        company = {"company": "TestCorp", "ticker": "TC", "industry_id": "TECH", "aliases": ["TestCorp"]}
        deduped = QueryBuilder.get_applicable_queries_for_company(company, [cat1, cat2])

        self.assertEqual(len(deduped), 1, "Duplicate queries should be merged into 1 executed query")
        first = deduped[0]
        self.assertIn("General Debt", first["category_names"])
        self.assertIn("Corporate Liquidity", first["category_names"])
        self.assertIn("Leverage / Capital", first["target_dimensions"])
        self.assertIn("Liquidity", first["target_dimensions"])
        self.assertEqual(first["priority"], 90, "Priority should resolve to highest weight (90)")

    def test_06_safe_industry_lifecycle(self):
        """Verifies industry renaming and status toggling without breaking foreign keys."""
        # 1. Add custom industry
        ok, ind_id = self.db.add_industry("AEROSPACE", "Aerospace & Defense", "STANDARD_CORP")
        self.assertTrue(ok)
        self.assertEqual(ind_id, "AEROSPACE")

        # 2. Map company to it
        self.db.add_portfolio("Lockheed Martin", "LMT", "Aerospace & Defense", "USA", ["Lockheed"], industry_id="AEROSPACE")
        p_list = self.db.get_portfolio()
        lmt = next(p for p in p_list if p["company"] == "Lockheed Martin")
        self.assertEqual(lmt["industry_id"], "AEROSPACE")

        # 3. Rename display name
        rename_ok = self.db.rename_industry("AEROSPACE", "Aerospace, Defense & Space")
        self.assertTrue(rename_ok)

        # Verify company still has stable AEROSPACE ID and updated display name
        p_list2 = self.db.get_portfolio()
        lmt2 = next(p for p in p_list2 if p["company"] == "Lockheed Martin")
        self.assertEqual(lmt2["industry_id"], "AEROSPACE")
        self.assertEqual(lmt2["industry"], "Aerospace, Defense & Space")

        # 4. Deactivate status
        self.db.toggle_industry_status("AEROSPACE", "INACTIVE")
        active_inds = self.db.get_industries(active_only=True)
        self.assertNotIn("AEROSPACE", [i["id"] for i in active_inds])

        # 5. Prevent delete when mapped
        del_ok, msg = self.db.delete_industry("AEROSPACE")
        self.assertFalse(del_ok, "Should prevent deleting industry assigned to active companies")


if __name__ == "__main__":
    unittest.main()
