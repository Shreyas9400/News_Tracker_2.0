#!/usr/bin/env python3
"""
test_date_provenance_clustering.py — Unit Tests for Date Provenance (Phase 1) & Canonical Syndication Clustering (Phase 2)
========================================================================================================================
Validates:
1. DateProvenanceResolver multi-tier deterministic hierarchy (JSON-LD -> OpenGraph -> RSS -> Crawl).
2. Strict isolation of datePublished vs dateCreated.
3. Preserving published_at_raw alongside standardized UTC published_at.
4. Syndication clustering: Earliest verified timestamp on canonical event while preserving individual source records.
5. Negative testing: Unrelated stories for the same company within 24h are NOT falsely clustered.
6. Schema migration v4 execution and table integrity.
"""

import os
import unittest
import tempfile
import sqlite3
import json

from constants import DateProvenanceResolver, parse_pub_date
from database import DatabaseManager


class TestDateProvenanceResolver(unittest.TestCase):
    def test_json_ld_date_published_high_confidence(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
          <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "headline": "JPMorgan Reports Record Q2 Net Income",
            "dateCreated": "2026-08-14T06:00:00Z",
            "datePublished": "2026-08-14T14:30:00Z"
          }
          </script>
        </head>
        </html>
        """
        res = DateProvenanceResolver.resolve_best_date(html_content=html)
        self.assertEqual(res["date_source"], "JSON_LD")
        self.assertEqual(res["date_confidence"], "HIGH")
        self.assertEqual(res["published_at"], "2026-08-14T14:30:00Z")
        self.assertEqual(res["published_at_raw"], "2026-08-14T14:30:00Z")

    def test_json_ld_ignores_date_created_for_publication(self):
        """If only dateCreated exists (no datePublished), resolver does not treat it as authoritative publication."""
        html = """
        <script type="application/ld+json">
        {
          "@type": "NewsArticle",
          "headline": "Draft Internal Article",
          "dateCreated": "2026-08-10T12:00:00Z"
        }
        </script>
        """
        # Fallback should kick in because datePublished is absent
        res = DateProvenanceResolver.resolve_best_date(html_content=html, raw_date_str="Fri, 14 Aug 2026 15:00:00 +0000")
        self.assertEqual(res["date_source"], "RSS_PUBDATE")
        self.assertEqual(res["published_at"], "2026-08-14T15:00:00Z")

    def test_opengraph_meta_fallback_high_confidence(self):
        html = """
        <html>
        <head>
          <meta property="article:published_time" content="2026-08-15T09:15:00-04:00" />
        </head>
        </html>
        """
        res = DateProvenanceResolver.resolve_best_date(html_content=html)
        self.assertEqual(res["date_source"], "OPEN_GRAPH")
        self.assertEqual(res["date_confidence"], "HIGH")
        self.assertEqual(res["published_at"], "2026-08-15T13:15:00Z")
        self.assertEqual(res["published_at_raw"], "2026-08-15T09:15:00-04:00")

    def test_rss_pubdate_with_explicit_timezone_medium_confidence(self):
        raw_rss = "Fri, 14 Aug 2026 10:02:00 EST"
        res = DateProvenanceResolver.resolve_best_date(raw_date_str=raw_rss)
        self.assertEqual(res["date_source"], "RSS_PUBDATE")
        self.assertEqual(res["date_confidence"], "MEDIUM")
        self.assertEqual(res["published_at"], "2026-08-14T15:02:00Z")
        self.assertEqual(res["published_at_raw"], raw_rss)

    def test_rss_pubdate_naive_timezone_low_confidence(self):
        raw_naive = "2026-08-14 14:30:00"
        res = DateProvenanceResolver.resolve_best_date(raw_date_str=raw_naive)
        self.assertEqual(res["date_source"], "RSS_PUBDATE")
        self.assertEqual(res["date_confidence"], "LOW")
        self.assertEqual(res["published_at_raw"], raw_naive)


class TestSyndicationClustering(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_syndication.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_schema_migration_v4_applied(self):
        c = self.db._conn()
        versions = [r["version"] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version ASC").fetchall()]
        self.assertIn(4, versions)

        cols = [r["name"] for r in c.execute("PRAGMA table_info(headlines)").fetchall()]
        self.assertIn("published_at", cols)
        self.assertIn("published_at_raw", cols)
        self.assertIn("crawled_at", cols)
        self.assertIn("date_source", cols)
        self.assertIn("date_confidence", cols)
        self.assertIn("canonical_event_id", cols)
        self.assertIn("canonical_published_at", cols)

    def test_syndication_clustering_true_positive(self):
        """
        PR Newswire (Aug 14 14:02) -> Reuters (Aug 14 14:17) -> Yahoo (Aug 15 03:11)
        Should all be clustered under the same canonical_event_id.
        canonical_published_at must be the earliest (Aug 14 14:02).
        Every headline must retain its own published_at and published_at_raw!
        """
        h1 = {
            "headline": "JPMorgan Chase Reports Second-Quarter 2026 Net Income of $18.1 Billion",
            "source": "PR Newswire",
            "url": "https://www.prnewswire.com/news-releases/jpmorgan-q2-2026.html",
            "company": "JPMorgan Chase",
            "published_time": "Fri, 14 Aug 2026 10:02:00 -0400",
            "credit_risk": "FAVORABLE",
        }
        h2 = {
            "headline": "JPMorgan Posts Q2 2026 Profit of $18.1B on Investment Banking Surge - Reuters",
            "source": "Reuters",
            "url": "https://www.reuters.com/business/finance/jpmorgan-q2-results-2026.html",
            "company": "JPMorgan Chase",
            "published_time": "Fri, 14 Aug 2026 14:17:00 +0000",
            "credit_risk": "FAVORABLE",
        }
        h3 = {
            "headline": "JPMorgan Chase Earnings: Q2 2026 Net Income Hits $18.1 Billion - Yahoo Finance",
            "source": "Yahoo Finance",
            "url": "https://finance.yahoo.com/news/jpmorgan-earnings-q2-2026-syndicated.html",
            "company": "JPMorgan Chase",
            "published_time": "Sat, 15 Aug 2026 03:11:00 +0000",
            "credit_risk": "FAVORABLE",
        }

        self.db.save_headline(h1)
        self.db.save_headline(h2)
        self.db.save_headline(h3)

        c = self.db._conn()
        events = c.execute("SELECT * FROM canonical_events WHERE company_name='JPMorgan Chase'").fetchall()
        self.assertEqual(len(events), 1, "All 3 syndicated stories should be grouped under 1 canonical event")
        
        canonical_event = events[0]
        self.assertEqual(canonical_event["source_count"], 3)
        self.assertEqual(canonical_event["canonical_published_at"], "2026-08-14T14:02:00Z")

        # Verify individual headlines preserved their unique source timestamps
        sources = self.db.get_canonical_event_sources(canonical_event["canonical_id"])
        self.assertEqual(len(sources), 3)
        
        pr_news = next(s for s in sources if s["source"] == "PR Newswire")
        reuters = next(s for s in sources if s["source"] == "Reuters")
        yahoo = next(s for s in sources if s["source"] == "Yahoo Finance")

        self.assertEqual(pr_news["published_at"], "2026-08-14T14:02:00Z")
        self.assertEqual(reuters["published_at"], "2026-08-14T14:17:00Z")
        self.assertEqual(yahoo["published_at"], "2026-08-15T03:11:00Z")

    def test_negative_test_unrelated_company_stories_not_clustered(self):
        """
        Three completely unrelated stories for JPMorgan within 24 hours:
        1. Q2 Earnings beat
        2. Analyst downgrade
        3. Executive resignation
        Must produce 3 distinct canonical events!
        """
        h_earnings = {
            "headline": "JPMorgan Reports Record Q2 Net Income and Revenue",
            "source": "CNBC",
            "url": "https://cnbc.com/jpm-q2-earnings",
            "company": "JPMorgan Chase",
            "published_time": "2026-08-14T10:00:00Z",
        }
        h_downgrade = {
            "headline": "JPMorgan Downgraded to Hold at Morgan Stanley on Valuation",
            "source": "Bloomberg",
            "url": "https://bloomberg.com/jpm-downgrade",
            "company": "JPMorgan Chase",
            "published_time": "2026-08-14T16:00:00Z",
        }
        h_exec = {
            "headline": "JPMorgan Head of Global Wealth Management Resigns",
            "source": "Financial Times",
            "url": "https://ft.com/jpm-wealth-head-resigns",
            "company": "JPMorgan Chase",
            "published_time": "2026-08-15T08:00:00Z",
        }

        self.db.save_headline(h_earnings)
        self.db.save_headline(h_downgrade)
        self.db.save_headline(h_exec)

        c = self.db._conn()
        events = c.execute("SELECT * FROM canonical_events WHERE company_name='JPMorgan Chase'").fetchall()
        self.assertEqual(len(events), 3, "3 distinct news events for the same company must NOT be clustered together")
        
        for ev in events:
            self.assertEqual(ev["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
