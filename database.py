#!/usr/bin/env python3
"""
database.py — Thread-Safe SQLite Database Manager
===================================================
Handles all DB operations with mutex write locks.
Includes pagination, company sentiment aggregation,
daily top news, user-taggable sentiment, and training data export.
"""

import json
import re
import sqlite3
import threading
import datetime
import math
from typing import List, Dict, Tuple, Any, Optional

from constants import (
    logger, DEFAULT_PORTFOLIO, DEFAULT_INDUSTRIES, DEFAULT_DOMAINS,
    DEFAULT_QUERY_CATEGORIES, DEFAULT_KEYWORDS,
    parse_pub_date, auto_generate_aliases, extract_domain,
)


class DatabaseManager:
    """Thread-safe SQLite database manager with mutex lock."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_schema(self):
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS headlines (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                headline        TEXT    NOT NULL,
                source          TEXT,
                url             TEXT    UNIQUE,
                canonical_url   TEXT,
                published_time  TEXT,
                search_query    TEXT,
                company         TEXT,
                industry        TEXT,
                event_category  TEXT,
                sentiment       TEXT,
                user_sentiment  TEXT    NULL,
                relevance_score REAL,
                news_volume_status TEXT,
                providers_json  TEXT,
                domain_name     TEXT,
                is_starred      INTEGER DEFAULT 0,
                review_status   INTEGER DEFAULT 0,
                reviewed_at     DATETIME NULL,
                notification_sent INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS portfolio (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT UNIQUE NOT NULL,
                ticker       TEXT,
                industry     TEXT,
                industry_id  TEXT,
                country      TEXT,
                aliases_json TEXT,
                enabled      INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS query_categories (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT NOT NULL,
                keywords_json    TEXT NOT NULL,
                scope_type       TEXT NOT NULL DEFAULT 'UNIVERSAL',
                industry_id      TEXT NULL,
                company_id       INTEGER NULL,
                priority         INTEGER DEFAULT 70,
                target_dimension TEXT DEFAULT 'Earnings / Cash Flow',
                version          INTEGER DEFAULT 1,
                enabled          INTEGER DEFAULT 1,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS extracted_keywords (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                word       TEXT UNIQUE NOT NULL,
                category   TEXT NOT NULL,
                frequency  INTEGER DEFAULT 1,
                last_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS industries (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                status       TEXT DEFAULT 'ACTIVE',
                risk_profile TEXT DEFAULT 'STANDARD_CORP',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS keywords (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                word     TEXT UNIQUE NOT NULL,
                category TEXT DEFAULT 'general',
                enabled  INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS domains (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_name TEXT UNIQUE NOT NULL,
                enabled     INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS website_visibility (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_name TEXT UNIQUE NOT NULL,
                is_visible  INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name            TEXT NOT NULL,
                ticker                  TEXT,
                quarter                 TEXT NOT NULL,
                reporting_date          DATE,
                conf_call_time          TEXT,
                timezone                TEXT DEFAULT 'ET',
                webcast_url             TEXT,
                status                  TEXT DEFAULT 'ESTIMATED',
                date_source             TEXT DEFAULT 'HISTORICAL_PATTERN',
                source_url              TEXT,
                source_headline         TEXT,
                reporting_date_precision TEXT DEFAULT 'EXACT',
                reporting_time_precision TEXT DEFAULT 'UNKNOWN',
                confidence              REAL DEFAULT 0.75,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company_name, quarter)
            );
            CREATE TABLE IF NOT EXISTS earnings_results (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name      TEXT NOT NULL,
                quarter           TEXT NOT NULL,
                nav_per_share     REAL,
                nav_prior         REAL,
                nii_per_share     REAL,
                nii_prior         REAL,
                dividend_regular  REAL,
                dividend_special  REAL,
                non_accrual_pct   REAL,
                non_accrual_prior REAL,
                reported_at       DATE,
                source_url        TEXT,
                source_headline   TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company_name, quarter)
            );
        """)

        # Dynamic auto-migration for existing databases
        cols_h = [r["name"] for r in c.execute("PRAGMA table_info(headlines)").fetchall()]
        migrations_h = {
            "is_starred": "ALTER TABLE headlines ADD COLUMN is_starred INTEGER DEFAULT 0",
            "review_status": "ALTER TABLE headlines ADD COLUMN review_status INTEGER DEFAULT 0",
            "reviewed_at": "ALTER TABLE headlines ADD COLUMN reviewed_at DATETIME NULL",
            "notification_sent": "ALTER TABLE headlines ADD COLUMN notification_sent INTEGER DEFAULT 0",
            "user_sentiment": "ALTER TABLE headlines ADD COLUMN user_sentiment TEXT NULL",
            "domain_name": "ALTER TABLE headlines ADD COLUMN domain_name TEXT",
            "credit_risk": "ALTER TABLE headlines ADD COLUMN credit_risk TEXT DEFAULT 'NEUTRAL'",
            "key_risk_signal": "ALTER TABLE headlines ADD COLUMN key_risk_signal TEXT DEFAULT 'Routine News'",
            "baseline_vader_score": "ALTER TABLE headlines ADD COLUMN baseline_vader_score REAL DEFAULT 0.0",
            "credit_risk_matrix_json": "ALTER TABLE headlines ADD COLUMN credit_risk_matrix_json TEXT NULL",
        }
        for col, sql in migrations_h.items():
            if col not in cols_h:
                c.execute(sql)

        # Dynamic migration for portfolio
        cols_p = [r["name"] for r in c.execute("PRAGMA table_info(portfolio)").fetchall()]
        if "industry_id" not in cols_p:
            c.execute("ALTER TABLE portfolio ADD COLUMN industry_id TEXT NULL")

        # Dynamic migration for query_categories
        cols_qc = [r["name"] for r in c.execute("PRAGMA table_info(query_categories)").fetchall()]
        migrations_qc = {
            "scope_type": "ALTER TABLE query_categories ADD COLUMN scope_type TEXT DEFAULT 'UNIVERSAL'",
            "industry_id": "ALTER TABLE query_categories ADD COLUMN industry_id TEXT NULL",
            "company_id": "ALTER TABLE query_categories ADD COLUMN company_id INTEGER NULL",
            "priority": "ALTER TABLE query_categories ADD COLUMN priority INTEGER DEFAULT 70",
            "target_dimension": "ALTER TABLE query_categories ADD COLUMN target_dimension TEXT DEFAULT 'Earnings / Cash Flow'",
            "version": "ALTER TABLE query_categories ADD COLUMN version INTEGER DEFAULT 1",
            "created_at": "ALTER TABLE query_categories ADD COLUMN created_at TIMESTAMP NULL",
            "updated_at": "ALTER TABLE query_categories ADD COLUMN updated_at TIMESTAMP NULL",
        }
        for col, sql in migrations_qc.items():
            if col not in cols_qc:
                c.execute(sql)

        # Dynamic migration for industries (if previously created with integer id)
        cols_ind = [r["name"] for r in c.execute("PRAGMA table_info(industries)").fetchall()]
        if "status" not in cols_ind:
            # Check if industries has TEXT id or INTEGER id
            id_type = next((r["type"] for r in c.execute("PRAGMA table_info(industries)").fetchall() if r["name"] == "id"), "")
            if "INT" in id_type.upper():
                # Re-create industries table with TEXT primary key
                c.execute("DROP TABLE IF EXISTS industries")
                c.execute("""
                    CREATE TABLE industries (
                        id           TEXT PRIMARY KEY,
                        name         TEXT NOT NULL,
                        status       TEXT DEFAULT 'ACTIVE',
                        risk_profile TEXT DEFAULT 'STANDARD_CORP',
                        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                c.execute("ALTER TABLE industries ADD COLUMN status TEXT DEFAULT 'ACTIVE'")
                c.execute("ALTER TABLE industries ADD COLUMN risk_profile TEXT DEFAULT 'STANDARD_CORP'")
                c.execute("ALTER TABLE industries ADD COLUMN created_at TIMESTAMP NULL")
                c.execute("ALTER TABLE industries ADD COLUMN updated_at TIMESTAMP NULL")

        c.commit()
        self._seed(c)

    def _seed(self, c: sqlite3.Connection):
        with self._write_lock:
            # Seed master industries catalog
            for ind in DEFAULT_INDUSTRIES:
                c.execute(
                    "INSERT OR IGNORE INTO industries(id, name, status, risk_profile) VALUES(?,?,?,?)",
                    (ind["id"], ind["name"], ind.get("status", "ACTIVE"), ind.get("risk_profile", "STANDARD_CORP"))
                )

            # Industry mapping helper for initial portfolio seed
            ind_name_to_id = {ind["name"]: ind["id"] for ind in DEFAULT_INDUSTRIES}
            ind_name_to_id.update({
                "US Banks": "COMMERCIAL_BANKING",
                "Regional Banks": "REGIONAL_BANKS",
                "Asset Management": "FINANCE_ASSET_MGMT",
                "Commercial Banking": "COMMERCIAL_BANKING",
            })

            if c.execute("SELECT COUNT(*) FROM portfolio").fetchone()[0] == 0:
                for p in DEFAULT_PORTFOLIO:
                    i_id = ind_name_to_id.get(p["industry"], "COMMERCIAL_BANKING")
                    c.execute(
                        "INSERT INTO portfolio(company_name,ticker,industry,industry_id,country,aliases_json,enabled) VALUES(?,?,?,?,?,?,1)",
                        (p["company"], p["ticker"], p["industry"], i_id, p["country"], json.dumps(p["aliases"])),
                    )

            # Update existing portfolio entries without industry_id
            c.execute("UPDATE portfolio SET industry_id = 'COMMERCIAL_BANKING' WHERE industry_id IS NULL AND (industry LIKE '%Bank%' OR industry LIKE '%Asset%')")
            c.execute("UPDATE portfolio SET industry_id = 'BDC' WHERE industry_id IS NULL AND (industry LIKE '%BDC%' OR industry LIKE '%Credit%')")
            c.execute("UPDATE portfolio SET industry_id = 'ENERGY' WHERE industry_id IS NULL AND industry LIKE '%Energy%'")
            c.execute("UPDATE portfolio SET industry_id = 'TELECOM' WHERE industry_id IS NULL AND industry LIKE '%Tele%'")
            c.execute("UPDATE portfolio SET industry_id = 'TECH' WHERE industry_id IS NULL AND industry LIKE '%Tech%'")

            # Seed / Sync default query categories with full scope metadata
            if c.execute("SELECT COUNT(*) FROM query_categories").fetchone()[0] == 0:
                for qc in DEFAULT_QUERY_CATEGORIES:
                    c.execute(
                        "INSERT INTO query_categories(name, keywords_json, scope_type, industry_id, priority, target_dimension, version, enabled) VALUES(?,?,?,?,?,?,?,1)",
                        (
                            qc["name"],
                            json.dumps(qc["keywords"]),
                            qc.get("scope_type", "UNIVERSAL"),
                            qc.get("industry_id"),
                            qc.get("priority", 70),
                            qc.get("target_dimension", "Earnings / Cash Flow"),
                            qc.get("version", 1),
                        ),
                    )

            if c.execute("SELECT COUNT(*) FROM domains").fetchone()[0] == 0:
                for d in DEFAULT_DOMAINS:
                    c.execute("INSERT INTO domains(domain_name,enabled) VALUES(?,1)", (d,))
                    c.execute("INSERT OR IGNORE INTO website_visibility(domain_name,is_visible) VALUES(?,1)", (d,))
            if c.execute("SELECT COUNT(*) FROM keywords").fetchone()[0] == 0:
                for kw in DEFAULT_KEYWORDS:
                    c.execute("INSERT INTO keywords(word,category,enabled) VALUES(?,'general',1)", (kw,))
            c.commit()
            if c.execute("SELECT COUNT(*) FROM domains").fetchone()[0] == 0:
                for d in DEFAULT_DOMAINS:
                    c.execute("INSERT INTO domains(domain_name,enabled) VALUES(?,1)", (d,))
                    c.execute("INSERT OR IGNORE INTO website_visibility(domain_name,is_visible) VALUES(?,1)", (d,))
            if c.execute("SELECT COUNT(*) FROM keywords").fetchone()[0] == 0:
                for kw in DEFAULT_KEYWORDS:
                    c.execute("INSERT INTO keywords(word,category,enabled) VALUES(?,'general',1)", (kw,))
            c.commit()

    # -----------------------------------------------------------------------
    # Headlines CRUD
    # -----------------------------------------------------------------------
    def save_headline(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Saves headline to DB.
        Returns (saved: bool, reason: 'saved' | 'deduped' | 'error')
        """
        with self._write_lock:
            c = self._conn()
            try:
                pub_time = parse_pub_date(data.get("published_time"))
                domain = extract_domain(data["url"], data.get("source", ""))
                matrix = data.get("credit_risk_matrix", {})
                matrix_json = json.dumps(matrix) if matrix else None
                c.execute("""
                    INSERT INTO headlines(headline,source,url,canonical_url,published_time,search_query,
                        company,industry,event_category,sentiment,relevance_score,news_volume_status,
                        providers_json,domain_name,credit_risk,key_risk_signal,baseline_vader_score,credit_risk_matrix_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    data["headline"], data["source"], data["url"],
                    data.get("canonical_url", data["url"]),
                    pub_time,
                    data.get("search_query", ""), data.get("company", "General"),
                    data.get("industry", "General"), data.get("event_category", "Neutral"),
                    data.get("sentiment", "Neutral"), data.get("relevance_score", 50.0),
                    data.get("news_volume_status", "Normal"),
                    json.dumps(data.get("providers", [data["source"]])),
                    domain,
                    data.get("credit_risk", "NEUTRAL"),
                    data.get("key_risk_signal", "Routine News"),
                    data.get("baseline_vader_score", 0.0),
                    matrix_json,
                ))
                c.commit()
                if domain:
                    c.execute("INSERT OR IGNORE INTO website_visibility(domain_name,is_visible) VALUES(?,1)", (domain,))
                    c.commit()
                return True, "saved"
            except sqlite3.IntegrityError:
                row = c.execute("SELECT id, providers_json FROM headlines WHERE url=?", (data["url"],)).fetchone()
                if row:
                    providers = json.loads(row["providers_json"]) if row["providers_json"] else []
                    src = data.get("source", "Unknown")
                    if src not in providers:
                        providers.append(src)
                        c.execute("UPDATE headlines SET providers_json=? WHERE id=?", (json.dumps(providers), row["id"]))
                        c.commit()
                return False, "deduped"

    def clear_all_headlines(self):
        with self._write_lock:
            c = self._conn()
            c.execute("DELETE FROM headlines")
            c.execute("DELETE FROM extracted_keywords")
            c.execute("DELETE FROM sqlite_sequence WHERE name IN ('headlines', 'extracted_keywords')")
            c.commit()

    def toggle_star(self, headline_id: int) -> bool:
        with self._write_lock:
            c = self._conn()
            row = c.execute("SELECT is_starred FROM headlines WHERE id=?", (headline_id,)).fetchone()
            if row:
                new_val = 1 if not row["is_starred"] else 0
                c.execute("UPDATE headlines SET is_starred=? WHERE id=?", (new_val, headline_id))
                c.commit()
                return True
            return False

    def mark_reviewed(self, headline_id: int) -> bool:
        with self._write_lock:
            c = self._conn()
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("UPDATE headlines SET review_status=1, reviewed_at=? WHERE id=?", (now_str, headline_id))
            c.commit()
            return True

    def mark_all_reviewed(self) -> int:
        with self._write_lock:
            c = self._conn()
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("UPDATE headlines SET review_status=1, reviewed_at=? WHERE review_status=0", (now_str,))
            cnt = c.rowcount
            c.commit()
            return cnt

    def update_user_sentiment(self, headline_id: int, sentiment: str) -> bool:
        """User overrides auto-detected sentiment for fine-tuning training data."""
        with self._write_lock:
            c = self._conn()
            c.execute("UPDATE headlines SET user_sentiment=? WHERE id=?", (sentiment, headline_id))
            c.commit()
            return True

    # -----------------------------------------------------------------------
    # Paginated Headlines Fetch (with all filters)
    # -----------------------------------------------------------------------
    def fetch_headlines_paginated(self, page: int = 1, page_size: int = 50,
                                  q: str = "", sentiment: str = "All",
                                  recency: str = "All", industry: str = "All",
                                  company: str = "All", domain: str = "All",
                                  date_from: str = "", date_to: str = "",
                                  filter_type: str = "All") -> Dict[str, Any]:
        """Fetches headlines with full filtering and pagination."""
        base_sql = """
            SELECT h.* FROM headlines h
            LEFT JOIN website_visibility wv
              ON wv.domain_name = (
                SELECT wv2.domain_name FROM website_visibility wv2
                WHERE LOWER(h.source) LIKE '%%' || LOWER(wv2.domain_name) || '%%' LIMIT 1
              )
            WHERE (wv.is_visible IS NULL OR wv.is_visible = 1)
        """
        params: list = []

        # Filter type
        if filter_type == "starred":
            base_sql += " AND h.is_starred = 1"
        elif filter_type == "high_relevance":
            base_sql += " AND h.relevance_score >= 85"
        elif filter_type == "live":
            base_sql += " AND (h.review_status = 0 OR (h.review_status = 1 AND h.reviewed_at >= datetime('now', '-60 minutes')))"
        elif filter_type == "reviewed":
            base_sql += " AND h.review_status = 1"

        # Text search
        if q:
            base_sql += " AND (h.headline LIKE ? OR h.company LIKE ? OR h.source LIKE ? OR h.industry LIKE ? OR h.event_category LIKE ?)"
            t = f"%{q}%"
            params += [t, t, t, t, t]

        # Sentiment filter
        if sentiment and sentiment != "All":
            base_sql += " AND COALESCE(h.user_sentiment, h.sentiment) = ?"
            params.append(sentiment)

        # Industry filter (checks headline industry OR portfolio company industry)
        if industry and industry != "All":
            base_sql += " AND (h.industry = ? OR h.company IN (SELECT company_name FROM portfolio WHERE industry = ?))"
            params.append(industry)
            params.append(industry)

        # Company filter
        if company and company != "All":
            base_sql += " AND h.company = ?"
            params.append(company)

        # Domain filter
        if domain and domain != "All":
            base_sql += " AND h.domain_name LIKE ?"
            params.append(f"%{domain}%")

        # Date range filters
        if date_from:
            base_sql += " AND h.published_time >= ?"
            params.append(date_from)
        if date_to:
            base_sql += " AND h.published_time <= ?"
            params.append(date_to + " 23:59")

        # Recency filter
        if recency and recency != "All":
            days = 1 if recency == "1d" else 7 if recency == "7d" else 30 if recency == "30d" else 0
            if days > 0:
                cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
                base_sql += " AND h.published_time >= ?"
                params.append(cutoff)

        # Count total
        count_sql = f"SELECT COUNT(*) FROM ({base_sql})"
        total = self._conn().execute(count_sql, params).fetchone()[0]
        total_pages = max(1, math.ceil(total / page_size))
        page = max(1, min(page, total_pages))

        # Fetch page
        data_sql = base_sql + " ORDER BY h.published_time DESC, h.id DESC LIMIT ? OFFSET ?"
        offset = (page - 1) * page_size
        data_params = params + [page_size, offset]
        rows = self._conn().execute(data_sql, data_params).fetchall()

        items = [self._row_to_dict(r) for r in rows]
        return {"items": items, "total": total, "page": page, "pages": total_pages, "page_size": page_size}

    def _row_to_dict(self, r) -> Dict[str, Any]:
        keys = r.keys()
        return {
            "id": r["id"], "headline": r["headline"], "source": r["source"],
            "url": r["url"], "published_time": r["published_time"],
            "company": r["company"], "industry": r["industry"],
            "event_category": r["event_category"],
            "sentiment": r["user_sentiment"] if "user_sentiment" in keys and r["user_sentiment"] else r["sentiment"],
            "auto_sentiment": r["sentiment"],
            "user_sentiment": r["user_sentiment"] if "user_sentiment" in keys else None,
            "relevance_score": r["relevance_score"],
            "news_volume_status": r["news_volume_status"] if "news_volume_status" in keys else "Normal",
            "domain_name": r["domain_name"] if "domain_name" in keys else "",
            "providers": json.loads(r["providers_json"]) if r["providers_json"] else [],
            "is_starred": r["is_starred"] if "is_starred" in keys and r["is_starred"] else 0,
            "review_status": r["review_status"] if "review_status" in keys and r["review_status"] else 0,
            "reviewed_at": r["reviewed_at"] if "reviewed_at" in keys and r["reviewed_at"] else "",
            "credit_risk": r["credit_risk"] if "credit_risk" in keys and r["credit_risk"] else "NEUTRAL",
            "key_risk_signal": r["key_risk_signal"] if "key_risk_signal" in keys and r["key_risk_signal"] else "Routine News",
            "baseline_vader_score": r["baseline_vader_score"] if "baseline_vader_score" in keys and r["baseline_vader_score"] else 0.0,
            "credit_risk_matrix": json.loads(r["credit_risk_matrix_json"]) if "credit_risk_matrix_json" in keys and r["credit_risk_matrix_json"] else None,
        }

    # -----------------------------------------------------------------------
    # Dashboard Endpoints
    # -----------------------------------------------------------------------
    def dashboard_metrics(self) -> Dict[str, Any]:
        c = self._conn()
        total = c.execute("""
            SELECT COUNT(*) FROM headlines
            WHERE created_at >= datetime('now', '-24 hours')
               OR DATE(created_at, 'localtime') = DATE('now', 'localtime')
        """).fetchone()[0]

        high = c.execute("SELECT COUNT(*) FROM headlines WHERE relevance_score>=85").fetchone()[0]
        pos = c.execute("SELECT COUNT(*) FROM headlines WHERE COALESCE(user_sentiment, sentiment) IN ('Positive','Very Positive')").fetchone()[0]
        neg = c.execute("SELECT COUNT(*) FROM headlines WHERE COALESCE(user_sentiment, sentiment) IN ('Negative','Very Negative')").fetchone()[0]
        neu = c.execute("SELECT COUNT(*) FROM headlines WHERE COALESCE(user_sentiment, sentiment)='Neutral'").fetchone()[0]
        unread = c.execute("SELECT COUNT(*) FROM headlines WHERE review_status=0").fetchone()[0]
        starred = c.execute("SELECT COUNT(*) FROM headlines WHERE is_starred=1").fetchone()[0]

        tc = [{"company": r[0], "count": r[1]}
              for r in c.execute("SELECT company,COUNT(*) c FROM headlines WHERE company!='General' GROUP BY company ORDER BY c DESC LIMIT 5").fetchall()]
        ti = [{"industry": r[0], "count": r[1]}
              for r in c.execute("SELECT industry,COUNT(*) c FROM headlines WHERE industry!='General' GROUP BY industry ORDER BY c DESC LIMIT 5").fetchall()]
        return {"total": total, "high": high, "positive": pos, "negative": neg,
                "neutral": neu, "unread": unread, "starred": starred,
                "trending_companies": tc, "trending_industries": ti,
                "last_refresh": datetime.datetime.now().strftime("%H:%M:%S")}

    def get_daily_top_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns today's top headlines by relevance score."""
        c = self._conn()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        rows = c.execute("""
            SELECT * FROM headlines
            WHERE published_time >= ?
            ORDER BY relevance_score DESC, published_time DESC
            LIMIT ?
        """, (today, limit)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_company_sentiments(self, days: int = 7) -> List[Dict[str, Any]]:
        """Aggregates sentiment per company over the past N days."""
        c = self._conn()
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        rows = c.execute("""
            SELECT company,
                   COUNT(*) as total,
                   SUM(CASE WHEN COALESCE(user_sentiment, sentiment) IN ('Positive','Very Positive') THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN COALESCE(user_sentiment, sentiment) IN ('Negative','Very Negative') THEN 1 ELSE 0 END) as negative,
                   SUM(CASE WHEN COALESCE(user_sentiment, sentiment) = 'Neutral' THEN 1 ELSE 0 END) as neutral,
                   AVG(relevance_score) as avg_score
            FROM headlines
            WHERE company != 'General' AND published_time >= ?
            GROUP BY company
            ORDER BY total DESC
            LIMIT 20
        """, (cutoff,)).fetchall()

        results = []
        for r in rows:
            # Get latest sentiment for this company
            latest = c.execute("""
                SELECT COALESCE(user_sentiment, sentiment) as sent FROM headlines
                WHERE company=? AND published_time >= ?
                ORDER BY published_time DESC LIMIT 1
            """, (r["company"], cutoff)).fetchone()
            results.append({
                "company": r["company"], "total": r["total"],
                "positive": r["positive"], "negative": r["negative"],
                "neutral": r["neutral"], "avg_score": round(r["avg_score"], 1),
                "latest_sentiment": latest["sent"] if latest else "Neutral",
            })
        return results

    def get_tag_summary(self) -> List[Dict[str, Any]]:
        """Returns headline counts per event category for the dashboard tag cloud."""
        c = self._conn()
        rows = c.execute("""
            SELECT event_category, COUNT(*) as count
            FROM headlines
            WHERE event_category != 'Neutral' AND event_category != ''
            GROUP BY event_category
            ORDER BY count DESC
        """).fetchall()
        return [{"tag": r["event_category"], "count": r["count"]} for r in rows]

    def export_tagged_data(self) -> List[Dict[str, Any]]:
        """Returns all user-tagged headlines for training data export."""
        c = self._conn()
        rows = c.execute("""
            SELECT headline, sentiment as auto_sentiment, user_sentiment,
                   company, industry, event_category, source, published_time
            FROM headlines
            WHERE user_sentiment IS NOT NULL
            ORDER BY published_time DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Extracted Keywords / Taxonomy Engine
    # -----------------------------------------------------------------------
    def increment_extracted_keywords(self, pairs: List[Tuple[str, str]]):
        if not pairs:
            return
        with self._write_lock:
            c = self._conn()
            for word, cat in pairs:
                c.execute("""
                    INSERT INTO extracted_keywords(word, category, frequency, last_seen)
                    VALUES(?, ?, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(word) DO UPDATE SET
                        frequency = frequency + 1,
                        last_seen = CURRENT_TIMESTAMP
                """, (word.lower().strip(), cat))
            c.commit()

    def get_extracted_keywords(self, category: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        c = self._conn()
        if category and category != "All":
            rows = c.execute("SELECT * FROM extracted_keywords WHERE category=? ORDER BY frequency DESC, id DESC LIMIT ?", (category, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM extracted_keywords ORDER BY frequency DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "word": r["word"], "category": r["category"], "frequency": r["frequency"]} for r in rows]

    # -----------------------------------------------------------------------
    # Portfolio CRUD
    # -----------------------------------------------------------------------
    def get_portfolio(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM portfolio WHERE enabled=1").fetchall()
        return [
            {"id": r["id"], "company": r["company_name"], "ticker": r["ticker"], "industry": r["industry"],
             "industry_id": r["industry_id"], "country": r["country"],
             "aliases": json.loads(r["aliases_json"]) if r["aliases_json"] else auto_generate_aliases(r["company_name"])}
            for r in rows
        ]

    def get_all_portfolio(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM portfolio ORDER BY id").fetchall()
        return [
            {"id": r["id"], "company": r["company_name"], "ticker": r["ticker"],
             "industry": r["industry"], "industry_id": r["industry_id"], "country": r["country"],
             "aliases": json.loads(r["aliases_json"]) if r["aliases_json"] else auto_generate_aliases(r["company_name"]),
             "enabled": r["enabled"]}
            for r in rows
        ]

    def add_portfolio(self, company: str, ticker: str, industry: str, country: str, user_aliases: List[str], industry_id: str = None) -> bool:
        with self._write_lock:
            try:
                aliases = auto_generate_aliases(company, user_aliases)
                c = self._conn()
                # Resolve industry_id or display name
                if not industry_id and industry:
                    ind_row = c.execute("SELECT id FROM industries WHERE LOWER(name)=LOWER(?) OR id=?", (industry.strip(), industry.strip())).fetchone()
                    if ind_row:
                        industry_id = ind_row["id"]
                    else:
                        industry_id = re.sub(r'[^A-Z0-9_]', '', industry.upper().replace(' ', '_'))
                        c.execute("INSERT OR IGNORE INTO industries(id, name) VALUES(?,?)", (industry_id, industry.strip()))
                elif industry_id and not industry:
                    ind_row = c.execute("SELECT name FROM industries WHERE id=?", (industry_id,)).fetchone()
                    if ind_row:
                        industry = ind_row["name"]

                c.execute(
                    "INSERT INTO portfolio(company_name,ticker,industry,industry_id,country,aliases_json,enabled) VALUES(?,?,?,?,?,?,1)",
                    (company, ticker, industry, industry_id, country, json.dumps(aliases)),
                )
                c.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def update_portfolio(self, pid: int, company: str, ticker: str, industry: str, country: str, user_aliases: List[str], industry_id: str = None) -> bool:
        with self._write_lock:
            try:
                aliases = auto_generate_aliases(company, user_aliases)
                c = self._conn()
                existing = c.execute("SELECT id FROM portfolio WHERE LOWER(company_name)=LOWER(?) AND id!=?", (company, pid)).fetchone()
                if existing:
                    return False

                # Resolve industry_id or display name
                if industry_id:
                    ind_row = c.execute("SELECT name FROM industries WHERE id=?", (industry_id,)).fetchone()
                    if ind_row:
                        industry = ind_row["name"]
                elif industry:
                    ind_row = c.execute("SELECT id FROM industries WHERE LOWER(name)=LOWER(?) OR id=?", (industry.strip(), industry.strip())).fetchone()
                    if ind_row:
                        industry_id = ind_row["id"]

                c.execute(
                    "UPDATE portfolio SET company_name=?, ticker=?, industry=?, industry_id=?, country=?, aliases_json=? WHERE id=?",
                    (company, ticker, industry, industry_id, country, json.dumps(aliases), pid),
                )
                c.commit()
                return True
            except Exception as e:
                logger.error("Error updating portfolio %d: %s", pid, e)
                return False

    def delete_portfolio(self, pid: int):
        with self._write_lock:
            self._conn().execute("DELETE FROM portfolio WHERE id=?", (pid,))
            self._conn().commit()

    def toggle_portfolio(self, pid: int, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE portfolio SET enabled=? WHERE id=?", (enabled, pid))
            self._conn().commit()

    def bulk_toggle_portfolio(self, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE portfolio SET enabled=?", (enabled,))
            self._conn().commit()

    # -----------------------------------------------------------------------
    # Query Categories CRUD (Scoped & Versioned)
    # -----------------------------------------------------------------------
    def get_query_categories(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT qc.*, ind.name as industry_name, p.company_name as company_target_name FROM query_categories qc LEFT JOIN industries ind ON qc.industry_id = ind.id LEFT JOIN portfolio p ON qc.company_id = p.id"
        if enabled_only:
            sql += " WHERE qc.enabled=1"
        sql += " ORDER BY qc.priority DESC, qc.id ASC"
        rows = self._conn().execute(sql).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "keywords": json.loads(r["keywords_json"]) if r["keywords_json"] else [],
                "scope_type": r["scope_type"] or "UNIVERSAL",
                "industry_id": r["industry_id"],
                "industry_name": r["industry_name"] if r["industry_name"] else (r["industry_id"] or "Universal"),
                "company_id": r["company_id"],
                "company_target_name": r["company_target_name"],
                "priority": r["priority"] if r["priority"] is not None else 70,
                "target_dimension": r["target_dimension"] or "Earnings / Cash Flow",
                "version": r["version"] or 1,
                "enabled": r["enabled"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def add_query_category(self, name: str, keywords: List[str], scope_type: str = "UNIVERSAL",
                           industry_id: str = None, company_id: int = None, priority: int = 70,
                           target_dimension: str = "Earnings / Cash Flow") -> bool:
        with self._write_lock:
            try:
                # Sanitize scope
                scope_type = scope_type.upper() if scope_type else "UNIVERSAL"
                if scope_type == "UNIVERSAL":
                    industry_id = None
                    company_id = None
                elif scope_type == "INDUSTRY":
                    company_id = None
                elif scope_type == "COMPANY":
                    industry_id = None

                self._conn().execute(
                    """INSERT INTO query_categories(name, keywords_json, scope_type, industry_id, company_id, priority, target_dimension, version, enabled)
                       VALUES(?,?,?,?,?,?,?,1,1)""",
                    (name, json.dumps(keywords), scope_type, industry_id, company_id, priority, target_dimension),
                )
                self._conn().commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def update_query_category(self, qid: int, name: str, keywords: List[str], scope_type: str = "UNIVERSAL",
                              industry_id: str = None, company_id: int = None, priority: int = 70,
                              target_dimension: str = "Earnings / Cash Flow") -> bool:
        with self._write_lock:
            try:
                scope_type = scope_type.upper() if scope_type else "UNIVERSAL"
                if scope_type == "UNIVERSAL":
                    industry_id = None
                    company_id = None
                elif scope_type == "INDUSTRY":
                    company_id = None
                elif scope_type == "COMPANY":
                    industry_id = None

                self._conn().execute(
                    """UPDATE query_categories
                       SET name=?, keywords_json=?, scope_type=?, industry_id=?, company_id=?,
                           priority=?, target_dimension=?, version = COALESCE(version, 1) + 1, updated_at = CURRENT_TIMESTAMP
                       WHERE id=?""",
                    (name, json.dumps(keywords), scope_type, industry_id, company_id, priority, target_dimension, qid),
                )
                self._conn().commit()
                return True
            except Exception as e:
                logger.error("Error updating query category %d: %s", qid, e)
                return False

    def delete_query_category(self, qid: int):
        with self._write_lock:
            self._conn().execute("DELETE FROM query_categories WHERE id=?", (qid,))
            self._conn().commit()

    def toggle_query_category(self, qid: int, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE query_categories SET enabled=? WHERE id=?", (enabled, qid))
            self._conn().commit()

    def get_applicable_query_categories(self, company_id: int = None, industry_id: str = None) -> List[Dict[str, Any]]:
        """
        Returns applicable query categories for an entity:
        Universal + matching Industry categories + Company-specific overrides.
        """
        c = self._conn()
        sql = """
            SELECT qc.*, ind.name as industry_name
            FROM query_categories qc
            LEFT JOIN industries ind ON qc.industry_id = ind.id
            WHERE qc.enabled=1
              AND (
                qc.scope_type = 'UNIVERSAL'
                OR (qc.scope_type = 'INDUSTRY' AND ? IS NOT NULL AND qc.industry_id = ?)
                OR (qc.scope_type = 'COMPANY' AND ? IS NOT NULL AND qc.company_id = ?)
              )
            ORDER BY qc.priority DESC, qc.id ASC
        """
        rows = c.execute(sql, (industry_id, industry_id, company_id, company_id)).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "keywords": json.loads(r["keywords_json"]) if r["keywords_json"] else [],
                "scope_type": r["scope_type"],
                "industry_id": r["industry_id"],
                "industry_name": r["industry_name"] or "Universal",
                "priority": r["priority"],
                "target_dimension": r["target_dimension"],
                "version": r["version"],
            }
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # Industries CRUD (Master Catalog with Stable IDs & Status Lifecycle)
    # -----------------------------------------------------------------------
    def get_industries(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Returns industry list for dropdown selectors."""
        sql = "SELECT id, name, status, risk_profile FROM industries"
        if active_only:
            sql += " WHERE status='ACTIVE'"
        sql += " ORDER BY name ASC"
        rows = self._conn().execute(sql).fetchall()
        return [{"id": r["id"], "name": r["name"], "status": r["status"], "risk_profile": r["risk_profile"]} for r in rows]

    def get_all_industries(self) -> List[Dict[str, Any]]:
        """
        Returns full industry management matrix:
        Industry ID, Display Name, Status, Risk Profile, Active Companies Count, Associated Query Categories Count.
        """
        c = self._conn()
        rows = c.execute("""
            SELECT
                i.id,
                i.name,
                i.status,
                i.risk_profile,
                (SELECT COUNT(*) FROM portfolio p WHERE p.industry_id = i.id AND p.enabled = 1) as company_count,
                (SELECT COUNT(*) FROM query_categories qc WHERE qc.industry_id = i.id AND qc.enabled = 1) as category_count,
                i.created_at,
                i.updated_at
            FROM industries i
            ORDER BY i.name ASC
        """).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "status": r["status"] or "ACTIVE",
                "risk_profile": r["risk_profile"] or "STANDARD_CORP",
                "company_count": r["company_count"],
                "category_count": r["category_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def add_industry(self, industry_id: str, name: str, risk_profile: str = "STANDARD_CORP") -> Tuple[bool, str]:
        """Creates a new industry in the catalog."""
        with self._write_lock:
            try:
                # Normalize stable ID
                if not industry_id:
                    industry_id = re.sub(r'[^A-Z0-9_]', '', name.strip().upper().replace(' ', '_').replace('/', '_').replace('&', 'AND'))
                else:
                    industry_id = re.sub(r'[^A-Z0-9_]', '', industry_id.strip().upper())

                c = self._conn()
                existing = c.execute("SELECT id FROM industries WHERE id=?", (industry_id,)).fetchone()
                if existing:
                    return False, f"Industry with code '{industry_id}' already exists."

                c.execute(
                    "INSERT INTO industries(id, name, status, risk_profile) VALUES(?,?, 'ACTIVE', ?)",
                    (industry_id, name.strip(), risk_profile)
                )
                c.commit()
                return True, industry_id
            except Exception as e:
                logger.error("Error adding industry: %s", e)
                return False, str(e)

    def rename_industry(self, industry_id: str, new_name: str) -> bool:
        """Safely renames the industry display name and updates associated portfolio records."""
        with self._write_lock:
            try:
                c = self._conn()
                c.execute("UPDATE industries SET name=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (new_name.strip(), industry_id))
                c.execute("UPDATE portfolio SET industry=? WHERE industry_id=?", (new_name.strip(), industry_id))
                c.commit()
                return True
            except Exception as e:
                logger.error("Error renaming industry %s: %s", industry_id, e)
                return False

    def toggle_industry_status(self, industry_id: str, status: str = None) -> bool:
        """Toggles active status ('ACTIVE' vs 'INACTIVE')."""
        with self._write_lock:
            try:
                c = self._conn()
                if not status:
                    current = c.execute("SELECT status FROM industries WHERE id=?", (industry_id,)).fetchone()
                    status = "INACTIVE" if (current and current["status"] == "ACTIVE") else "ACTIVE"
                c.execute("UPDATE industries SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (status, industry_id))
                c.commit()
                return True
            except Exception as e:
                logger.error("Error toggling industry %s: %s", industry_id, e)
                return False

    def delete_industry(self, industry_id: str) -> Tuple[bool, str]:
        """
        Safely deletes custom industry if no active companies reference it.
        If companies reference it, deletion is blocked (recommends deactivation).
        """
        with self._write_lock:
            try:
                c = self._conn()
                active_comps = c.execute("SELECT COUNT(*) FROM portfolio WHERE industry_id=?", (industry_id,)).fetchone()[0]
                if active_comps > 0:
                    return False, f"Cannot delete industry '{industry_id}': {active_comps} company(ies) are currently assigned to it. Deactivate the industry instead."

                # Reassign or delete associated query categories
                c.execute("DELETE FROM query_categories WHERE industry_id=?", (industry_id,))
                c.execute("DELETE FROM industries WHERE id=?", (industry_id,))
                c.commit()
                return True, "Industry deleted successfully."
            except Exception as e:
                logger.error("Error deleting industry %s: %s", industry_id, e)
                return False, str(e)

    # -----------------------------------------------------------------------
    # Keywords CRUD
    # -----------------------------------------------------------------------
    def get_keywords(self) -> List[str]:
        return [r["word"] for r in self._conn().execute("SELECT word FROM keywords WHERE enabled=1").fetchall()]

    def get_all_keywords(self) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "word": r["word"], "category": r["category"], "enabled": r["enabled"]}
                for r in self._conn().execute("SELECT * FROM keywords ORDER BY id").fetchall()]

    def add_keyword(self, word: str) -> bool:
        with self._write_lock:
            try:
                self._conn().execute("INSERT INTO keywords(word,category,enabled) VALUES(?,'general',1)", (word,))
                self._conn().commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def delete_keyword(self, kid: int):
        with self._write_lock:
            self._conn().execute("DELETE FROM keywords WHERE id=?", (kid,))
            self._conn().commit()

    def toggle_keyword(self, kid: int, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE keywords SET enabled=? WHERE id=?", (enabled, kid))
            self._conn().commit()

    # -----------------------------------------------------------------------
    # Domains CRUD
    # -----------------------------------------------------------------------
    def get_domains(self) -> List[str]:
        return [r["domain_name"] for r in self._conn().execute("SELECT domain_name FROM domains WHERE enabled=1").fetchall()]

    def get_all_domains(self) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "domain": r["domain_name"], "enabled": r["enabled"]}
                for r in self._conn().execute("SELECT * FROM domains ORDER BY id").fetchall()]

    def add_domain(self, domain: str) -> bool:
        with self._write_lock:
            try:
                clean_d = domain.strip().lower()
                clean_d = clean_d[4:] if clean_d.startswith("www.") else clean_d
                self._conn().execute("INSERT INTO domains(domain_name,enabled) VALUES(?,1)", (clean_d,))
                self._conn().execute("INSERT OR IGNORE INTO website_visibility(domain_name,is_visible) VALUES(?,1)", (clean_d,))
                self._conn().commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def delete_domain(self, did: int):
        with self._write_lock:
            self._conn().execute("DELETE FROM domains WHERE id=?", (did,))
            self._conn().commit()

    def toggle_domain(self, did: int, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE domains SET enabled=? WHERE id=?", (enabled, did))
            self._conn().commit()

    def bulk_toggle_domains(self, enabled: int):
        with self._write_lock:
            self._conn().execute("UPDATE domains SET enabled=?", (enabled,))
            self._conn().commit()

    # -----------------------------------------------------------------------
    # Website Visibility
    # -----------------------------------------------------------------------
    def get_visibility(self) -> List[Dict[str, Any]]:
        c = self._conn()
        rows = c.execute("SELECT DISTINCT source, url FROM headlines").fetchall()
        with self._write_lock:
            for r in rows:
                d = extract_domain(r["url"], r["source"])
                if d:
                    c.execute("INSERT OR IGNORE INTO website_visibility(domain_name, is_visible) VALUES(?, 1)", (d,))
            c.commit()
        return [{"id": r["id"], "domain": r["domain_name"], "visible": r["is_visible"]}
                for r in c.execute("SELECT * FROM website_visibility ORDER BY domain_name").fetchall()]

    def add_visibility_domain(self, domain: str, visible: int = 1) -> bool:
        clean_d = domain.strip().lower()
        clean_d = clean_d[4:] if clean_d.startswith("www.") else clean_d
        if not clean_d:
            return False
        with self._write_lock:
            try:
                self._conn().execute("INSERT OR REPLACE INTO website_visibility(domain_name, is_visible) VALUES(?, ?)", (clean_d, visible))
                self._conn().commit()
                return True
            except Exception as e:
                logger.error("Error adding visibility domain: %s", e)
                return False

    def delete_visibility(self, vid: int):
        with self._write_lock:
            self._conn().execute("DELETE FROM website_visibility WHERE id=?", (vid,))
            self._conn().commit()

    def toggle_visibility(self, vid: int, visible: int):
        with self._write_lock:
            self._conn().execute("UPDATE website_visibility SET is_visible=? WHERE id=?", (visible, vid))
            self._conn().commit()

    def bulk_toggle_visibility(self, visible: int):
        with self._write_lock:
            self._conn().execute("UPDATE website_visibility SET is_visible=?", (visible,))
            self._conn().commit()

    # -----------------------------------------------------------------------
    # Corpus Stats (for BM25)
    # -----------------------------------------------------------------------
    def get_corpus_stats(self) -> Dict[str, Any]:
        rows = self._conn().execute("SELECT headline FROM headlines ORDER BY id DESC LIMIT 1000").fetchall()
        headlines = [r["headline"] for r in rows if r["headline"]]
        return {"headlines": headlines, "total_docs": len(headlines)}

    # -----------------------------------------------------------------------
    # Distinct values for filter dropdowns
    # -----------------------------------------------------------------------
    def get_distinct_companies(self) -> List[str]:
        rows = self._conn().execute("SELECT DISTINCT company FROM headlines WHERE company != 'General' ORDER BY company").fetchall()
        return [r["company"] for r in rows]

    def get_distinct_industries(self) -> List[str]:
        c = self._conn()
        rows_h = [r["industry"] for r in c.execute("SELECT DISTINCT industry FROM headlines WHERE industry IS NOT NULL AND TRIM(industry) != ''").fetchall()]
        rows_i = [r["name"] for r in c.execute("SELECT name FROM industries WHERE status='ACTIVE'").fetchall()]
        combined = sorted(list(set(rows_h + rows_i)))
        return combined

    def get_distinct_domains(self) -> List[str]:
        rows = self._conn().execute("SELECT DISTINCT domain_name FROM headlines WHERE domain_name IS NOT NULL AND domain_name != '' ORDER BY domain_name").fetchall()
        return [r["domain_name"] for r in rows]

    # -----------------------------------------------------------------------
    # Earnings Tracker CRUD
    # -----------------------------------------------------------------------
    def get_earnings_calendar(self, company: str = None, quarter: str = None, status: str = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM earnings_calendar WHERE 1=1"
        params = []
        if company and company != "All":
            sql += " AND company_name = ?"
            params.append(company)
        if quarter and quarter != "All":
            sql += " AND quarter = ?"
            params.append(quarter)
        if status and status != "All":
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY reporting_date ASC, company_name ASC"
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def save_earnings_calendar(self, data: Dict[str, Any]) -> bool:
        with self._write_lock:
            c = self._conn()
            existing = c.execute(
                "SELECT status FROM earnings_calendar WHERE company_name=? AND quarter=?",
                (data["company_name"], data["quarter"])
            ).fetchone()
            if existing and existing["status"] == "CONFIRMED" and data.get("status") == "ESTIMATED":
                return False

            src_url = data.get("source_url", "")
            src_head = data.get("source_headline", "")
            if not src_url and src_head:
                match = c.execute("SELECT url FROM headlines WHERE headline=?", (src_head,)).fetchone()
                if match and match["url"]:
                    src_url = match["url"]
                elif src_head:
                    src_url = f"https://www.google.com/search?q={urllib.parse.quote(src_head)}"

            c.execute("""
                INSERT OR REPLACE INTO earnings_calendar(
                    company_name, ticker, quarter, reporting_date, conf_call_time,
                    timezone, webcast_url, status, date_source, source_url,
                    source_headline, reporting_date_precision, reporting_time_precision,
                    confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                data["company_name"], data.get("ticker", ""), data["quarter"],
                data.get("reporting_date"), data.get("conf_call_time", ""),
                data.get("timezone", "ET"), data.get("webcast_url", ""),
                data.get("status", "ESTIMATED"), data.get("date_source", "HISTORICAL_PATTERN"),
                src_url, src_head,
                data.get("reporting_date_precision", "EXACT"),
                data.get("reporting_time_precision", "UNKNOWN"),
                data.get("confidence", 0.75),
            ))
            c.commit()
            return True

    def get_earnings_results(self, company: str = None, quarter: str = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM earnings_results WHERE 1=1"
        params = []
        if company and company != "All":
            sql += " AND company_name = ?"
            params.append(company)
        if quarter and quarter != "All":
            sql += " AND quarter = ?"
            params.append(quarter)
        sql += " ORDER BY quarter DESC, company_name ASC"
        rows = self._conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def save_earnings_results(self, data: Dict[str, Any]) -> bool:
        with self._write_lock:
            c = self._conn()
            src_url = data.get("source_url", "")
            src_head = data.get("source_headline", "")
            if not src_url and src_head:
                match = c.execute("SELECT url FROM headlines WHERE headline=?", (src_head,)).fetchone()
                if match and match["url"]:
                    src_url = match["url"]
                elif src_head:
                    src_url = f"https://www.google.com/search?q={urllib.parse.quote(src_head)}"

            c.execute("""
                INSERT OR REPLACE INTO earnings_results(
                    company_name, quarter, nav_per_share, nav_prior, nii_per_share,
                    nii_prior, dividend_regular, dividend_special, non_accrual_pct,
                    non_accrual_prior, reported_at, source_url, source_headline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["company_name"], data["quarter"],
                data.get("nav_per_share"), data.get("nav_prior"),
                data.get("nii_per_share"), data.get("nii_prior"),
                data.get("dividend_regular"), data.get("dividend_special"),
                data.get("non_accrual_pct"), data.get("non_accrual_prior"),
                data.get("reported_at"), src_url, src_head,
            ))
            c.commit()
            return True
