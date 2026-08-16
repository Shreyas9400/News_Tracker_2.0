#!/usr/bin/env python3
"""
test_user_portfolio.py — Unit Tests for Dynamic DB Path, Versioned Schema Migrations,
3-Tier Normalized Model (Users, Entities, Portfolio), and Atomic Watchlist Bulk Import.
"""

import os
import unittest
import tempfile
import sqlite3
import json

from constants import resolve_db_path, APP_DIR, DEFAULT_CONFIG, load_config
from database import DatabaseManager


class TestDynamicDbPath(unittest.TestCase):
    def test_relative_path_resolution(self):
        resolved = resolve_db_path("custom_data/intel.db")
        expected = os.path.normpath(os.path.join(APP_DIR, "custom_data/intel.db"))
        self.assertEqual(resolved, expected)

    def test_env_var_override(self):
        tmp_db = os.path.normpath(os.path.join(tempfile.gettempdir(), "env_override.db"))
        os.environ["NEWS_INTEL_DB_PATH"] = tmp_db
        try:
            resolved = resolve_db_path("some_ignored_config.db")
            self.assertEqual(resolved, tmp_db)
        finally:
            del os.environ["NEWS_INTEL_DB_PATH"]

    def test_invalid_drive_fallback(self):
        fake_foreign_path = "Z:\\NonExistentDrive_999\\subfolder\\intel.db"
        resolved = resolve_db_path(fake_foreign_path)
        expected = os.path.normpath(os.path.join(APP_DIR, "intel.db"))
        self.assertEqual(resolved, expected)


class TestVersionedMigrationsAndNormalization(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_portfolio_users.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_schema_migrations_table_exists_and_versioned(self):
        """Verify schema_migrations table tracks applied versions up to v3."""
        c = self.db._conn()
        rows = c.execute("SELECT version, description FROM schema_migrations ORDER BY version ASC").fetchall()
        versions = [r["version"] for r in rows]
        self.assertIn(1, versions)
        self.assertIn(2, versions)
        self.assertIn(3, versions)

    def test_3tier_normalization_entity_deduplication(self):
        """Verify that when 2 users track 'Microsoft', only 1 row exists in entities table."""
        # Alice adds Microsoft
        res1 = self.db.add_portfolio(
            company="Microsoft Corporation", ticker="MSFT", industry="Technology", country="USA",
            user_aliases=["MSFT", "Microsoft"], user_name="Alice"
        )
        self.assertTrue(res1)

        # Bob also adds Microsoft Corporation
        res2 = self.db.add_portfolio(
            company="Microsoft Corporation", ticker="MSFT", industry="Technology", country="USA",
            user_aliases=["MSFT Cloud"], user_name="Bob"
        )
        self.assertTrue(res2)

        c = self.db._conn()
        entity_rows = c.execute("SELECT * FROM entities WHERE company_name='Microsoft Corporation'").fetchall()
        self.assertEqual(len(entity_rows), 1, "There should only be 1 canonical row in entities for Microsoft Corporation")

        portfolio_mappings = c.execute("SELECT * FROM portfolio WHERE entity_id=?", (entity_rows[0]["id"],)).fetchall()
        self.assertEqual(len(portfolio_mappings), 2, "There should be 2 relationship rows in portfolio (Alice & Bob)")

    def test_user_id_and_name_filtering(self):
        """Verify get_portfolio works with user_id, user_name, or All."""
        self.db.add_portfolio(company="Entity Alpha", ticker="ALP", industry="Tech", country="USA", user_aliases=[], user_name="Charlie")
        self.db.add_portfolio(company="Entity Beta", ticker="BET", industry="Tech", country="USA", user_aliases=[], user_name="Dana")

        users = self.db.get_users()
        charlie = next((u for u in users if u["name"] == "Charlie"), None)
        self.assertIsNotNone(charlie)

        # Query by user_id
        charlie_by_id = self.db.get_portfolio(user_id=charlie["id"])
        self.assertEqual(len(charlie_by_id), 1)
        self.assertEqual(charlie_by_id[0]["company"], "Entity Alpha")

        # Query by user_name
        charlie_by_name = self.db.get_portfolio(user_name="Charlie")
        self.assertEqual(len(charlie_by_name), 1)

    def test_bulk_toggle_per_user(self):
        """Verify bulk toggle operates on the selected user's entities."""
        self.db.add_portfolio(company="Alpha Corp", ticker="ALP", industry="Tech", country="USA", user_aliases=[], user_name="User1")
        self.db.add_portfolio(company="Beta Corp", ticker="BET", industry="Tech", country="USA", user_aliases=[], user_name="User2")

        # Deactivate only User1's portfolio
        self.db.bulk_toggle_portfolio(0, user_name="User1")

        u1_active = self.db.get_portfolio(user_name="User1")
        self.assertEqual(len(u1_active), 0)

        u2_active = self.db.get_portfolio(user_name="User2")
        self.assertEqual(len(u2_active), 1)

    def test_bulk_import_in_file_duplicate_and_upsert(self):
        """Verify bulk import handles in-file duplicates, updates, and creates canonical entities."""
        import_payload = [
            {"User Name": "Sarah Connor", "Entity Name": "JPMorgan Chase", "Ticker": "JPM", "Industry": "Commercial Banking", "Country": "USA", "Aliases": "JPMorgan, Chase"},
            {"User Name": "Sarah Connor", "Entity Name": "JPMorgan Chase", "Ticker": "JPM", "Industry": "Commercial Banking", "Country": "USA", "Aliases": "Chase Bank"}, # in-file duplicate
            {"User Name": "Michael Scott", "Entity Name": "JPMorgan Chase", "Ticker": "JPM", "Industry": "Commercial Banking", "Country": "USA", "Aliases": "JPM"}, # different user, same canonical entity
            {"User Name": "Michael Scott", "Entity Name": "Dunder Mifflin", "Ticker": "DMI", "Industry": "Paper", "Country": "USA", "Aliases": "Dunder"}
        ]

        result = self.db.bulk_import_portfolio(import_payload, atomic=False)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["added"], 3)

        # Canonical entity check: JPMorgan Chase should only exist once in entities table
        c = self.db._conn()
        jpm_entities = c.execute("SELECT * FROM entities WHERE company_name='JPMorgan Chase'").fetchall()
        self.assertEqual(len(jpm_entities), 1)

    def test_atomic_import_rollback_on_error(self):
        """Verify atomic=True aborts transaction and rolls back when mandatory fields are missing."""
        invalid_payload = [
            {"User Name": "Valid User", "Entity Name": "Valid Company", "Ticker": "VAL"},
            {"User Name": "Valid User", "Entity Name": "", "Ticker": "ERR"} # Missing Entity Name
        ]

        result = self.db.bulk_import_portfolio(invalid_payload, atomic=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["added"], 0)
        self.assertIn("Missing mandatory 'Entity Name / Company'", result["errors"][0])

    def test_legacy_flat_database_auto_migration(self):
        """Test migrating an old flat SQLite database without users/entities tables."""
        legacy_db_path = os.path.join(self.tmp_dir, "legacy_flat.db")
        conn = sqlite3.connect(legacy_db_path)
        conn.row_factory = sqlite3.Row
        # Create legacy table
        conn.executescript("""
            CREATE TABLE portfolio (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name    TEXT DEFAULT 'Default User',
                company_name TEXT NOT NULL,
                ticker       TEXT,
                industry     TEXT,
                country      TEXT DEFAULT 'USA',
                aliases_json TEXT,
                enabled      INTEGER DEFAULT 1
            );
            INSERT INTO portfolio(user_name, company_name, ticker, industry, country, aliases_json)
            VALUES('Legacy User A', 'Boeing Co', 'BA', 'Aerospace', 'USA', '["Boeing"]');
            INSERT INTO portfolio(user_name, company_name, ticker, industry, country, aliases_json)
            VALUES('Legacy User B', 'Boeing Co', 'BA', 'Aerospace', 'USA', '["Boeing Commercial"]');
        """)
        conn.commit()
        conn.close()

        # Opening with DatabaseManager should trigger migration v3
        migrated_db = DatabaseManager(legacy_db_path)
        try:
            users = migrated_db.get_portfolio_users()
            self.assertIn("Legacy User A", users)
            self.assertIn("Legacy User B", users)

            portfolio_a = migrated_db.get_portfolio(user_name="Legacy User A")
            self.assertEqual(len(portfolio_a), 1)
            self.assertEqual(portfolio_a[0]["company"], "Boeing Co")

            # Entity deduplication check
            c = migrated_db._conn()
            entities = c.execute("SELECT * FROM entities WHERE company_name='Boeing Co'").fetchall()
            self.assertEqual(len(entities), 1)
        finally:
            migrated_db.close()


if __name__ == "__main__":
    unittest.main()
