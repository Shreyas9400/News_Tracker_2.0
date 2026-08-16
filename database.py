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
import uuid
import sqlite3
import threading
import datetime
import math
from typing import List, Dict, Tuple, Any, Optional

from constants import (
    logger, DEFAULT_PORTFOLIO, DEFAULT_INDUSTRIES, DEFAULT_DOMAINS,
    DEFAULT_QUERY_CATEGORIES, DEFAULT_KEYWORDS,
    parse_pub_date, auto_generate_aliases, extract_domain, resolve_db_path,
    DateProvenanceResolver,
)


class DatabaseManager:
    """Thread-safe SQLite database manager with mutex lock."""

    def __init__(self, db_path: str):
        self.db_path = resolve_db_path(db_path)
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        c.commit()
        self._run_migrations(c)
        self._seed(c)

    def _run_migrations(self, c: sqlite3.Connection):
        with self._write_lock:
            applied_rows = c.execute("SELECT version FROM schema_migrations").fetchall()
            applied_versions = {r["version"] for r in applied_rows}

            # ---------------------------------------------------------------
            # Migration v1: Baseline Core Tables
            # ---------------------------------------------------------------
            if 1 not in applied_versions:
                logger.info("Applying schema migration v1: Baseline Core Schema...")
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
                c.execute("INSERT INTO schema_migrations(version, description) VALUES(1, 'Baseline Core Schema')")
                c.commit()

            # ---------------------------------------------------------------
            # Migration v2: Dynamic Column Alterations & Stable Industry IDs
            # ---------------------------------------------------------------
            if 2 not in applied_versions:
                logger.info("Applying schema migration v2: Column upgrades and scoping...")
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

                c.execute("INSERT INTO schema_migrations(version, description) VALUES(2, 'Column upgrades & Scoped queries')")
                c.commit()

            # ---------------------------------------------------------------
            # Migration v3: 3-Tier Normalized Model (Users, Entities, Portfolio)
            # ---------------------------------------------------------------
            if 3 not in applied_versions:
                logger.info("Applying schema migration v3: 3-Tier Normalized Model (users, entities, portfolio)...")
                c.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        name       TEXT UNIQUE NOT NULL,
                        email      TEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT UNIQUE NOT NULL,
                        ticker       TEXT,
                        industry     TEXT,
                        industry_id  TEXT,
                        country      TEXT DEFAULT 'USA',
                        aliases_json TEXT,
                        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Check if portfolio table needs migration from legacy flat table
                has_legacy_portfolio = False
                p_tables = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'").fetchall()
                if p_tables:
                    cols_p = [r["name"] for r in c.execute("PRAGMA table_info(portfolio)").fetchall()]
                    if "entity_id" not in cols_p:
                        has_legacy_portfolio = True

                if has_legacy_portfolio:
                    logger.info("Migrating legacy flat portfolio data into users, entities, and normalized portfolio...")
                    legacy_rows = c.execute("SELECT * FROM portfolio").fetchall()

                    # 1. Ensure Default User exists
                    c.execute("INSERT OR IGNORE INTO users(name) VALUES('Default User')")

                    # 2. Extract users and entities
                    for row in legacy_rows:
                        r_keys = row.keys()
                        user_name = (row["user_name"] if "user_name" in r_keys and row["user_name"] else "Default User").strip()
                        c.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (user_name,))

                        comp = row["company_name"].strip()
                        ticker = (row["ticker"] or "").strip().upper() if "ticker" in r_keys and row["ticker"] else None
                        industry = row["industry"] if "industry" in r_keys else None
                        industry_id = row["industry_id"] if "industry_id" in r_keys else None
                        country = row["country"] if "country" in r_keys and row["country"] else "USA"
                        aliases_json = row["aliases_json"] if "aliases_json" in r_keys else None

                        c.execute("""
                            INSERT OR IGNORE INTO entities(company_name, ticker, industry, industry_id, country, aliases_json)
                            VALUES(?, ?, ?, ?, ?, ?)
                        """, (comp, ticker, industry, industry_id, country, aliases_json))

                    # 3. Create new normalized portfolio table
                    c.execute("DROP TABLE portfolio")
                    c.execute("""
                        CREATE TABLE portfolio (
                            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id             INTEGER NOT NULL,
                            entity_id           INTEGER NOT NULL,
                            custom_aliases_json TEXT,
                            enabled             INTEGER DEFAULT 1,
                            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                            UNIQUE(user_id, entity_id)
                        );
                    """)
                    c.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id);")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_entity ON portfolio(entity_id);")

                    # 4. Populate portfolio relationships
                    for row in legacy_rows:
                        r_keys = row.keys()
                        user_name = (row["user_name"] if "user_name" in r_keys and row["user_name"] else "Default User").strip()
                        u_id = c.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (user_name,)).fetchone()[0]
                        comp = row["company_name"].strip()
                        e_id = c.execute("SELECT id FROM entities WHERE LOWER(company_name)=LOWER(?)", (comp,)).fetchone()[0]
                        enabled = row["enabled"] if "enabled" in r_keys else 1

                        c.execute("""
                            INSERT OR IGNORE INTO portfolio(user_id, entity_id, enabled)
                            VALUES(?, ?, ?)
                        """, (u_id, e_id, enabled))
                else:
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS portfolio (
                            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id             INTEGER NOT NULL,
                            entity_id           INTEGER NOT NULL,
                            custom_aliases_json TEXT,
                            enabled             INTEGER DEFAULT 1,
                            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                            UNIQUE(user_id, entity_id)
                        );
                    """)
                    c.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id);")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_entity ON portfolio(entity_id);")

                c.execute("INSERT INTO schema_migrations(version, description) VALUES(3, '3-Tier Normalized Portfolio Model')")
                c.commit()

            # ---------------------------------------------------------------
            # Migration v4: Date Provenance & Canonical Syndication Events
            # ---------------------------------------------------------------
            if 4 not in applied_versions:
                logger.info("Applying schema migration v4: Date Provenance & Canonical Syndication Events...")
                # 1. Create canonical_events table
                c.execute("""
                    CREATE TABLE IF NOT EXISTS canonical_events (
                        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                        canonical_id           TEXT UNIQUE NOT NULL,
                        entity_id              INTEGER NULL,
                        company_name           TEXT NOT NULL,
                        headline_canonical     TEXT NOT NULL,
                        canonical_published_at TEXT NOT NULL,
                        confidence             TEXT DEFAULT 'HIGH',
                        source_count           INTEGER DEFAULT 1,
                        created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                c.execute("CREATE INDEX IF NOT EXISTS idx_canonical_events_company ON canonical_events(company_name);")
                c.execute("CREATE INDEX IF NOT EXISTS idx_canonical_events_pub ON canonical_events(canonical_published_at);")

                # 2. Add provenance and clustering columns to headlines
                cols_h = [r["name"] for r in c.execute("PRAGMA table_info(headlines)").fetchall()]
                migrations_h4 = {
                    "published_at": "ALTER TABLE headlines ADD COLUMN published_at TEXT NULL",
                    "published_at_raw": "ALTER TABLE headlines ADD COLUMN published_at_raw TEXT NULL",
                    "crawled_at": "ALTER TABLE headlines ADD COLUMN crawled_at TIMESTAMP NULL",
                    "date_source": "ALTER TABLE headlines ADD COLUMN date_source TEXT DEFAULT 'RSS_PUBDATE'",
                    "date_confidence": "ALTER TABLE headlines ADD COLUMN date_confidence TEXT DEFAULT 'LOW'",
                    "canonical_event_id": "ALTER TABLE headlines ADD COLUMN canonical_event_id TEXT NULL",
                    "canonical_published_at": "ALTER TABLE headlines ADD COLUMN canonical_published_at TEXT NULL",
                    "canonical_confidence": "ALTER TABLE headlines ADD COLUMN canonical_confidence TEXT NULL",
                }
                for col, sql in migrations_h4.items():
                    if col not in cols_h:
                        c.execute(sql)

                # 3. Backfill legacy published_time and crawled_at if empty
                c.execute("UPDATE headlines SET crawled_at = CURRENT_TIMESTAMP WHERE crawled_at IS NULL")
                c.execute("UPDATE headlines SET published_at = published_time WHERE published_at IS NULL AND published_time IS NOT NULL")
                c.execute("UPDATE headlines SET published_at_raw = published_time WHERE published_at_raw IS NULL AND published_time IS NOT NULL")

                c.execute("INSERT INTO schema_migrations(version, description) VALUES(4, 'Date Provenance and Canonical Syndication Events')")
                c.commit()

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

            # Seed Default User
            c.execute("INSERT OR IGNORE INTO users(name) VALUES('Default User')")
            default_user_row = c.execute("SELECT id FROM users WHERE name='Default User'").fetchone()
            default_user_id = default_user_row[0] if default_user_row else 1

            if c.execute("SELECT COUNT(*) FROM portfolio").fetchone()[0] == 0:
                for p in DEFAULT_PORTFOLIO:
                    i_id = ind_name_to_id.get(p["industry"], "COMMERCIAL_BANKING")
                    c.execute(
                        "INSERT OR IGNORE INTO entities(company_name,ticker,industry,industry_id,country,aliases_json) VALUES(?,?,?,?,?,?)",
                        (p["company"], p["ticker"], p["industry"], i_id, p["country"], json.dumps(p["aliases"])),
                    )
                    e_row = c.execute("SELECT id FROM entities WHERE company_name=?", (p["company"],)).fetchone()
                    if e_row:
                        c.execute(
                            "INSERT OR IGNORE INTO portfolio(user_id, entity_id, enabled) VALUES(?,?,1)",
                            (default_user_id, e_row[0])
                        )

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
    # -----------------------------------------------------------------------
    # Syndication Clustering & Canonical Event Resolution (Phase 2)
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_cluster_tokens(headline: str) -> List[str]:
        if not headline:
            return []
        clean = re.sub(r'\s*[-|–—]\s*(Reuters|Bloomberg|PR Newswire|Business Wire|Yahoo Finance|CNBC|WSJ|MarketWatch|Investing\.com|Seeking Alpha|Forbes|Financial Times|The Wall Street Journal|Associated Press|AP News|Benzinga|TheStreet|Barron\'s|GlobeNewswire).*$', '', headline, flags=re.IGNORECASE)
        clean = re.sub(r'\s*\([A-Z]+:[A-Z0-9.-]+\)', '', clean)
        clean = clean.lower()
        # Normalize quarter representations
        clean = re.sub(r'\b(second-quarter|second quarter|2nd-quarter|2nd quarter|2q)\b', ' q2 ', clean)
        clean = re.sub(r'\b(first-quarter|first quarter|1st-quarter|1st quarter|1q)\b', ' q1 ', clean)
        clean = re.sub(r'\b(third-quarter|third quarter|3rd-quarter|3rd quarter|3q)\b', ' q3 ', clean)
        clean = re.sub(r'\b(fourth-quarter|fourth quarter|4th-quarter|4th quarter|4q)\b', ' q4 ', clean)
        clean = re.sub(r'\bnet\s+income\b', ' earnings ', clean)
        # Normalize billion / million amounts (e.g. 18.1B, $18.1 Billion)
        clean = re.sub(r'[\$]?(\d+(?:\.\d+)?)\s*(?:b|billion)\b', lambda m: ' ' + m.group(1).replace('.', '_') + '_billion ', clean)
        clean = re.sub(r'[\$]?(\d+(?:\.\d+)?)\s*(?:m|million)\b', lambda m: ' ' + m.group(1).replace('.', '_') + '_million ', clean)
        clean = re.sub(r'[^a-zA-Z0-9_\s]', ' ', clean)

        FINANCIAL_SYNONYMS = {
            'profit': 'earnings', 'income': 'earnings', 'net_income': 'earnings', 'results': 'earnings',
            'sales': 'revenue', 'topline': 'revenue',
        }
        stopwords = {'the', 'and', 'for', 'with', 'from', 'inc', 'corp', 'ltd', 'company', 'corporation', 'co', 'plc', 'group', 'hits', 'posts', 'reports', 'of', 'on', 'in', 'at', 'to', 'a', 'an'}

        tokens = []
        for w in clean.split():
            if len(w) > 1 and w not in stopwords:
                syn = FINANCIAL_SYNONYMS.get(w, w)
                tokens.append(syn)
        return tokens

    def _find_or_create_canonical_event(self, c: sqlite3.Connection, company: str, headline: str, published_at: str) -> Tuple[str, str, str]:
        """
        Finds matching candidate canonical event or creates a new one.
        Candidate matching requires:
          - Same Entity / Company Match
          - High Headline Similarity (Token overlap >= 0.50)
          - Temporal Proximity (within ±36-hour sliding window of publication)
        Returns (canonical_id, canonical_published_at, canonical_confidence).
        """
        company = (company or "General").strip()
        pub_iso = published_at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        clean_tokens = self._extract_cluster_tokens(headline)
        norm_headline = " ".join(clean_tokens)

        # If general/unknown company without clean tokens, create isolated event
        if company.lower() in ("general", "unknown", "") or not norm_headline:
            cid = f"evt_{uuid.uuid4().hex[:12]}"
            c.execute("""
                INSERT INTO canonical_events(canonical_id, company_name, headline_canonical, canonical_published_at, confidence, source_count)
                VALUES(?, ?, ?, ?, 'HIGH', 1)
            """, (cid, company, headline, pub_iso))
            return cid, pub_iso, "HIGH"

        # Search existing canonical events for this company within sliding ±36h temporal window
        dt_center = pub_iso.replace('T', ' ')[:19]
        recent_events = c.execute("""
            SELECT * FROM canonical_events
            WHERE LOWER(company_name) = LOWER(?)
              AND canonical_published_at >= datetime(?, '-36 hours')
              AND canonical_published_at <= datetime(?, '+36 hours')
            ORDER BY canonical_published_at ASC
        """, (company, dt_center, dt_center)).fetchall()

        best_event = None
        best_sim = 0.0

        for ev in recent_events:
            ev_tokens = self._extract_cluster_tokens(ev["headline_canonical"])
            if not ev_tokens:
                continue

            set1, set2 = set(clean_tokens), set(ev_tokens)
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            jaccard = len(intersection) / len(union) if union else 0.0
            overlap_min = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0.0
            sim = 0.5 * jaccard + 0.5 * overlap_min

            if sim > best_sim:
                best_sim = sim
                best_event = ev

        # Multi-signal confidence threshold (Requires high similarity >= 0.50)
        if best_event and best_sim >= 0.50:
            cid = best_event["canonical_id"]
            ev_pub = best_event["canonical_published_at"]
            confidence = "HIGH" if best_sim >= 0.65 else "MEDIUM"
            
            # Earliest verified timestamp wins for the canonical event without overwriting source published_at
            new_canonical_pub = ev_pub
            if pub_iso < ev_pub:
                new_canonical_pub = pub_iso

            c.execute("""
                UPDATE canonical_events
                SET canonical_published_at = ?,
                    source_count = source_count + 1,
                    confidence = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE canonical_id = ?
            """, (new_canonical_pub, confidence, cid))

            return cid, new_canonical_pub, confidence

        # No confident match -> Create a brand new canonical event
        cid = f"evt_{uuid.uuid4().hex[:12]}"
        c.execute("""
            INSERT INTO canonical_events(canonical_id, company_name, headline_canonical, canonical_published_at, confidence, source_count)
            VALUES(?, ?, ?, ?, 'HIGH', 1)
        """, (cid, company, headline, pub_iso))
        return cid, pub_iso, "HIGH"

    def get_canonical_event_sources(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Returns all individual source headlines clustered under a canonical event."""
        c = self._conn()
        rows = c.execute("""
            SELECT id, headline, source, url, published_at, published_at_raw, crawled_at,
                   date_source, date_confidence, sentiment, relevance_score
            FROM headlines
            WHERE canonical_event_id = ?
            ORDER BY published_at ASC, id ASC
        """, (canonical_id,)).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Headlines CRUD
    # -----------------------------------------------------------------------
    def save_headline(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Saves headline to DB with full date provenance resolution and candidate syndication clustering.
        Returns (saved: bool, reason: 'saved' | 'deduped' | 'error')
        """
        with self._write_lock:
            c = self._conn()
            try:
                # 1. Multi-tier date provenance resolution
                date_prov = DateProvenanceResolver.resolve_best_date(
                    raw_date_str=data.get("published_time") or data.get("published_at_raw"),
                    html_content=data.get("html_content")
                )
                published_at = date_prov["published_at"]
                published_at_raw = date_prov["published_at_raw"]
                crawled_at = date_prov["crawled_at"]
                date_source = date_prov["date_source"]
                date_confidence = date_prov["date_confidence"]
                
                # Legacy format for backward compatibility
                pub_time = published_at[:10] + " " + published_at[11:16] if len(published_at) >= 16 else published_at

                # 2. Candidate Syndication Clustering (Phase 2)
                canonical_event_id = data.get("canonical_event_id")
                canonical_published_at = data.get("canonical_published_at")
                canonical_confidence = data.get("canonical_confidence")

                if not canonical_event_id:
                    canonical_event_id, canonical_published_at, canonical_confidence = self._find_or_create_canonical_event(
                        c, data.get("company", "General"), data["headline"], published_at
                    )

                domain = extract_domain(data["url"], data.get("source", ""))
                matrix = data.get("credit_risk_matrix", {})
                matrix_json = json.dumps(matrix) if matrix else None

                c.execute("""
                    INSERT INTO headlines(headline,source,url,canonical_url,published_time,search_query,
                        company,industry,event_category,sentiment,relevance_score,news_volume_status,
                        providers_json,domain_name,credit_risk,key_risk_signal,baseline_vader_score,credit_risk_matrix_json,
                        published_at,published_at_raw,crawled_at,date_source,date_confidence,
                        canonical_event_id,canonical_published_at,canonical_confidence)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    published_at, published_at_raw, crawled_at, date_source, date_confidence,
                    canonical_event_id, canonical_published_at, canonical_confidence
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
            SELECT h.*, COALESCE(ce.source_count, 1) AS cluster_sources_count
            FROM headlines h
            LEFT JOIN canonical_events ce ON h.canonical_event_id = ce.canonical_id
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

        # Industry filter (checks headline industry OR company catalog industry)
        if industry and industry != "All":
            base_sql += " AND (h.industry = ? OR h.company IN (SELECT company_name FROM entities WHERE industry = ?))"
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
            "published_at": r["published_at"] if "published_at" in keys and r["published_at"] else r["published_time"],
            "published_at_raw": r["published_at_raw"] if "published_at_raw" in keys and r["published_at_raw"] else r["published_time"],
            "crawled_at": r["crawled_at"] if "crawled_at" in keys and r["crawled_at"] else "",
            "date_source": r["date_source"] if "date_source" in keys and r["date_source"] else "RSS_PUBDATE",
            "date_confidence": r["date_confidence"] if "date_confidence" in keys and r["date_confidence"] else "LOW",
            "canonical_event_id": r["canonical_event_id"] if "canonical_event_id" in keys else None,
            "canonical_published_at": r["canonical_published_at"] if "canonical_published_at" in keys else None,
            "canonical_confidence": r["canonical_confidence"] if "canonical_confidence" in keys else None,
            "cluster_sources_count": r["cluster_sources_count"] if "cluster_sources_count" in keys else 1,
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
    # Portfolio CRUD (3-Tier Normalized Architecture: Users, Entities, Portfolio)
    # -----------------------------------------------------------------------
    def get_portfolio(self, user_id: Optional[Any] = None, user_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns enabled portfolio companies, optionally filtered by user_id or user_name."""
        c = self._conn()
        sql = """
            SELECT p.id as portfolio_id, p.enabled, p.custom_aliases_json,
                   u.id as user_id, u.name as user_name,
                   e.id as entity_id, e.company_name, e.ticker, e.industry, e.industry_id, e.country, e.aliases_json
            FROM portfolio p
            JOIN entities e ON p.entity_id = e.id
            JOIN users u ON p.user_id = u.id
            WHERE p.enabled = 1
        """
        params = []
        if user_id is not None and str(user_id).lower() != "all":
            sql += " AND p.user_id = ?"
            params.append(int(user_id))
        elif user_name and user_name.strip() and user_name.strip().lower() != "all":
            sql += " AND LOWER(u.name) = LOWER(?)"
            params.append(user_name.strip())

        sql += " ORDER BY (CASE WHEN u.name='Default User' THEN 0 ELSE 1 END), u.name ASC, e.company_name ASC"
        rows = c.execute(sql, params).fetchall()

        results = []
        for r in rows:
            master_aliases = json.loads(r["aliases_json"]) if r["aliases_json"] else auto_generate_aliases(r["company_name"])
            custom_aliases = json.loads(r["custom_aliases_json"]) if r["custom_aliases_json"] else []
            combined_aliases = list(dict.fromkeys(master_aliases + custom_aliases))
            results.append({
                "id": r["portfolio_id"],
                "portfolio_id": r["portfolio_id"],
                "user_id": r["user_id"],
                "user_name": r["user_name"],
                "entity_id": r["entity_id"],
                "company": r["company_name"],
                "company_name": r["company_name"],
                "ticker": r["ticker"],
                "industry": r["industry"],
                "industry_id": r["industry_id"],
                "country": r["country"],
                "aliases": combined_aliases,
                "enabled": r["enabled"],
            })
        return results

    def get_all_portfolio(self, user_id: Optional[Any] = None, user_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all portfolio companies (active + inactive), optionally filtered by user_id or user_name."""
        c = self._conn()
        sql = """
            SELECT p.id as portfolio_id, p.enabled, p.custom_aliases_json,
                   u.id as user_id, u.name as user_name,
                   e.id as entity_id, e.company_name, e.ticker, e.industry, e.industry_id, e.country, e.aliases_json
            FROM portfolio p
            JOIN entities e ON p.entity_id = e.id
            JOIN users u ON p.user_id = u.id
        """
        params = []
        if user_id is not None and str(user_id).lower() != "all":
            sql += " WHERE p.user_id = ?"
            params.append(int(user_id))
        elif user_name and user_name.strip() and user_name.strip().lower() != "all":
            sql += " WHERE LOWER(u.name) = LOWER(?)"
            params.append(user_name.strip())

        sql += " ORDER BY (CASE WHEN u.name='Default User' THEN 0 ELSE 1 END), u.name ASC, e.company_name ASC"
        rows = c.execute(sql, params).fetchall()

        results = []
        for r in rows:
            master_aliases = json.loads(r["aliases_json"]) if r["aliases_json"] else auto_generate_aliases(r["company_name"])
            custom_aliases = json.loads(r["custom_aliases_json"]) if r["custom_aliases_json"] else []
            combined_aliases = list(dict.fromkeys(master_aliases + custom_aliases))
            results.append({
                "id": r["portfolio_id"],
                "portfolio_id": r["portfolio_id"],
                "user_id": r["user_id"],
                "user_name": r["user_name"],
                "entity_id": r["entity_id"],
                "company": r["company_name"],
                "company_name": r["company_name"],
                "ticker": r["ticker"],
                "industry": r["industry"],
                "industry_id": r["industry_id"],
                "country": r["country"],
                "aliases": combined_aliases,
                "enabled": r["enabled"],
            })
        return results

    def get_users(self) -> List[Dict[str, Any]]:
        """Returns structured list of registered users with tracking counts."""
        c = self._conn()
        rows = c.execute("""
            SELECT u.id, u.name, COUNT(p.id) as entity_count
            FROM users u
            LEFT JOIN portfolio p ON u.id = p.user_id
            GROUP BY u.id, u.name
            ORDER BY (CASE WHEN u.name='Default User' THEN 0 ELSE 1 END), u.name ASC
        """).fetchall()
        return [{"id": r["id"], "name": r["name"], "entity_count": r["entity_count"]} for r in rows]

    def get_portfolio_users(self) -> List[str]:
        """Backward-compatible user names list."""
        return [u["name"] for u in self.get_users()]

    def add_portfolio(self, company: str, ticker: str, industry: str, country: str,
                      user_aliases: List[str], industry_id: str = None, user_name: str = "Default User",
                      user_id: Optional[int] = None) -> bool:
        """Adds or maps a company to a user's portfolio."""
        with self._write_lock:
            try:
                c = self._conn()
                company = (company or "").strip()
                if not company:
                    return False

                # 1. Resolve User
                if user_id is not None:
                    u_row = c.execute("SELECT id, name FROM users WHERE id=?", (user_id,)).fetchone()
                    if not u_row:
                        return False
                    target_user_id = u_row["id"]
                else:
                    u_name = (user_name or "Default User").strip()
                    c.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (u_name,))
                    target_user_id = c.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (u_name,)).fetchone()[0]

                # 2. Resolve Industry
                if not industry_id and industry:
                    ind_row = c.execute("SELECT id, name FROM industries WHERE LOWER(name)=LOWER(?) OR id=?", (industry.strip(), industry.strip())).fetchone()
                    if ind_row:
                        industry_id = ind_row["id"]
                        industry = ind_row["name"]
                    else:
                        industry_id = re.sub(r'[^A-Z0-9_]', '', industry.upper().replace(' ', '_'))
                        c.execute("INSERT OR IGNORE INTO industries(id, name) VALUES(?,?)", (industry_id, industry.strip()))
                elif industry_id and not industry:
                    ind_row = c.execute("SELECT name FROM industries WHERE id=?", (industry_id,)).fetchone()
                    if ind_row:
                        industry = ind_row["name"]

                aliases = auto_generate_aliases(company, user_aliases)

                # 3. Resolve / Upsert Entity
                e_row = c.execute("SELECT id, aliases_json FROM entities WHERE LOWER(company_name)=LOWER(?)", (company,)).fetchone()
                if e_row:
                    target_entity_id = e_row["id"]
                    c.execute("""
                        UPDATE entities
                        SET ticker=COALESCE(?, ticker),
                            industry=COALESCE(?, industry),
                            industry_id=COALESCE(?, industry_id),
                            country=COALESCE(?, country),
                            aliases_json=?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                    """, (ticker or None, industry or None, industry_id or None, country or "USA", json.dumps(aliases), target_entity_id))
                else:
                    c.execute("""
                        INSERT INTO entities(company_name, ticker, industry, industry_id, country, aliases_json)
                        VALUES(?, ?, ?, ?, ?, ?)
                    """, (company, ticker or None, industry or None, industry_id or None, country or "USA", json.dumps(aliases)))
                    target_entity_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]

                # 4. Create Portfolio Mapping
                existing_p = c.execute("SELECT id FROM portfolio WHERE user_id=? AND entity_id=?", (target_user_id, target_entity_id)).fetchone()
                if existing_p:
                    return False

                c.execute("""
                    INSERT INTO portfolio(user_id, entity_id, enabled)
                    VALUES(?, ?, 1)
                """, (target_user_id, target_entity_id))
                c.commit()
                return True
            except Exception as e:
                logger.error("Error adding portfolio: %s", e)
                return False

    def update_portfolio(self, pid: int, company: str, ticker: str, industry: str, country: str,
                         user_aliases: List[str], industry_id: str = None, user_name: str = "Default User",
                         user_id: Optional[int] = None) -> bool:
        """Updates entity metadata and user portfolio relationship."""
        with self._write_lock:
            try:
                c = self._conn()
                p_row = c.execute("SELECT user_id, entity_id FROM portfolio WHERE id=?", (pid,)).fetchone()
                if not p_row:
                    return False

                old_user_id = p_row["user_id"]
                entity_id = p_row["entity_id"]

                # Resolve target user
                if user_id is not None:
                    target_user_id = user_id
                elif user_name:
                    c.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (user_name.strip(),))
                    target_user_id = c.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (user_name.strip(),)).fetchone()[0]
                else:
                    target_user_id = old_user_id

                # Resolve industry
                if industry_id:
                    ind_row = c.execute("SELECT name FROM industries WHERE id=?", (industry_id,)).fetchone()
                    if ind_row:
                        industry = ind_row["name"]
                elif industry:
                    ind_row = c.execute("SELECT id, name FROM industries WHERE LOWER(name)=LOWER(?) OR id=?", (industry.strip(), industry.strip())).fetchone()
                    if ind_row:
                        industry_id = ind_row["id"]
                        industry = ind_row["name"]

                aliases = auto_generate_aliases(company, user_aliases)

                # Update entity metadata
                c.execute("""
                    UPDATE entities
                    SET company_name=?, ticker=?, industry=?, industry_id=?, country=?, aliases_json=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                """, (company.strip(), ticker or None, industry or None, industry_id or None, country or "USA", json.dumps(aliases), entity_id))

                # Update portfolio relationship user if changed
                if target_user_id != old_user_id:
                    dup = c.execute("SELECT id FROM portfolio WHERE user_id=? AND entity_id=? AND id!=?", (target_user_id, entity_id, pid)).fetchone()
                    if dup:
                        return False
                    c.execute("UPDATE portfolio SET user_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (target_user_id, pid))

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
            self._conn().execute("UPDATE portfolio SET enabled=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (enabled, pid))
            self._conn().commit()

    def bulk_toggle_portfolio(self, enabled: int, user_id: Optional[Any] = None, user_name: Optional[str] = None):
        """Toggles active status for all companies or for a specific user."""
        with self._write_lock:
            c = self._conn()
            if user_id is not None and str(user_id).lower() != "all":
                c.execute("UPDATE portfolio SET enabled=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (enabled, int(user_id)))
            elif user_name and user_name.strip() and user_name.strip().lower() != "all":
                u_row = c.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (user_name.strip(),)).fetchone()
                if u_row:
                    c.execute("UPDATE portfolio SET enabled=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (enabled, u_row["id"]))
            else:
                c.execute("UPDATE portfolio SET enabled=?, updated_at=CURRENT_TIMESTAMP", (enabled,))
            c.commit()

    def bulk_import_portfolio(self, records: List[Dict[str, Any]], atomic: bool = True) -> Dict[str, Any]:
        """
        Batch imports portfolio records across users with 3-tier normalization,
        validation-first atomicity, and conflict deduplication.
        """
        if not records:
            return {"ok": False, "message": "No records provided", "added": 0, "updated": 0, "errors": []}

        # Step 1: In-memory Pre-validation
        parsed_records = []
        errors = []
        seen_in_batch = set()

        for idx, r in enumerate(records, start=1):
            user_name = str(r.get("user_name") or r.get("user") or r.get("analyst") or r.get("User Name") or r.get("User") or "").strip()
            company = str(r.get("company") or r.get("company_name") or r.get("entity") or r.get("entity_name") or r.get("Company Name") or r.get("Entity Name") or "").strip()
            ticker = str(r.get("ticker") or r.get("symbol") or r.get("Ticker") or "").strip().upper()
            industry = str(r.get("industry") or r.get("sector") or r.get("Industry") or "").strip()
            industry_id = r.get("industry_id") or r.get("Industry Code") or None
            country = str(r.get("country") or r.get("Country") or "USA").strip()
            raw_aliases = r.get("aliases") or r.get("Aliases") or []

            if not user_name:
                errors.append(f"Row {idx}: Missing mandatory 'User Name'")
            if not company:
                errors.append(f"Row {idx}: Missing mandatory 'Entity Name / Company'")

            if isinstance(raw_aliases, str):
                user_aliases = [a.strip() for a in re.split(r'[,;|\n]', raw_aliases) if a.strip()]
            elif isinstance(raw_aliases, list):
                user_aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
            else:
                user_aliases = []

            key = (user_name.lower(), company.lower())
            is_dup = key in seen_in_batch
            seen_in_batch.add(key)

            parsed_records.append({
                "row": idx,
                "user_name": user_name,
                "company": company,
                "ticker": ticker,
                "industry": industry,
                "industry_id": industry_id,
                "country": country,
                "user_aliases": user_aliases,
                "is_batch_duplicate": is_dup,
            })

        if errors and atomic:
            return {
                "ok": False,
                "message": f"Import aborted: {len(errors)} validation errors found.",
                "added": 0,
                "updated": 0,
                "total": len(records),
                "errors": errors,
            }

        # Step 2: Atomic Execution
        added = 0
        updated = 0

        with self._write_lock:
            c = self._conn()
            try:
                for item in parsed_records:
                    if not item["user_name"] or not item["company"]:
                        continue

                    # 1. Resolve / Create User
                    c.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (item["user_name"],))
                    u_id = c.execute("SELECT id FROM users WHERE LOWER(name)=LOWER(?)", (item["user_name"],)).fetchone()[0]

                    # 2. Resolve Industry
                    ind_id = item["industry_id"]
                    ind_name = item["industry"]
                    if not ind_id and ind_name:
                        ind_row = c.execute("SELECT id, name FROM industries WHERE LOWER(name)=LOWER(?) OR id=?", (ind_name.lower(), ind_name)).fetchone()
                        if ind_row:
                            ind_id = ind_row["id"]
                            ind_name = ind_row["name"]
                        else:
                            ind_id = re.sub(r'[^A-Z0-9_]', '', ind_name.upper().replace(' ', '_'))
                            c.execute("INSERT OR IGNORE INTO industries(id, name) VALUES(?,?)", (ind_id, ind_name))
                    elif ind_id and not ind_name:
                        ind_row = c.execute("SELECT name FROM industries WHERE id=?", (ind_id,)).fetchone()
                        if ind_row:
                            ind_name = ind_row["name"]

                    aliases = auto_generate_aliases(item["company"], item["user_aliases"])

                    # 3. Resolve / Create Canonical Entity
                    e_row = c.execute("SELECT id FROM entities WHERE LOWER(company_name)=LOWER(?)", (item["company"],)).fetchone()
                    if e_row:
                        e_id = e_row["id"]
                        c.execute("""
                            UPDATE entities
                            SET ticker=COALESCE(?, ticker),
                                industry=COALESCE(?, industry),
                                industry_id=COALESCE(?, industry_id),
                                country=COALESCE(?, country),
                                aliases_json=?,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=?
                        """, (item["ticker"] or None, ind_name or None, ind_id or None, item["country"] or "USA", json.dumps(aliases), e_id))
                    else:
                        c.execute("""
                            INSERT INTO entities(company_name, ticker, industry, industry_id, country, aliases_json)
                            VALUES(?, ?, ?, ?, ?, ?)
                        """, (item["company"], item["ticker"] or None, ind_name or None, ind_id or None, item["country"] or "USA", json.dumps(aliases)))
                        e_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]

                    # 4. Upsert User Portfolio Relationship
                    p_row = c.execute("SELECT id FROM portfolio WHERE user_id=? AND entity_id=?", (u_id, e_id)).fetchone()
                    if p_row:
                        c.execute("UPDATE portfolio SET enabled=1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (p_row["id"],))
                        updated += 1
                    else:
                        c.execute("INSERT INTO portfolio(user_id, entity_id, enabled) VALUES(?, ?, 1)", (u_id, e_id))
                        added += 1

                c.commit()
                return {
                    "ok": True,
                    "added": added,
                    "updated": updated,
                    "total": len(records),
                    "errors": errors,
                }
            except Exception as e:
                c.rollback()
                logger.error("Transactional bulk import failed: %s", e)
                return {
                    "ok": False,
                    "message": f"Database transaction failed: {str(e)}",
                    "added": 0,
                    "updated": 0,
                    "total": len(records),
                    "errors": errors + [str(e)],
                }

    # -----------------------------------------------------------------------
    # Query Categories CRUD (Scoped & Versioned)
    # -----------------------------------------------------------------------
    def get_query_categories(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT qc.*, ind.name as industry_name, e.company_name as company_target_name FROM query_categories qc LEFT JOIN industries ind ON qc.industry_id = ind.id LEFT JOIN entities e ON qc.company_id = e.id"
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
                (SELECT COUNT(DISTINCT p.entity_id) FROM portfolio p JOIN entities e ON p.entity_id = e.id WHERE e.industry_id = i.id AND p.enabled = 1) as company_count,
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
        """Safely renames the industry display name and updates associated entity records."""
        with self._write_lock:
            try:
                c = self._conn()
                c.execute("UPDATE industries SET name=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (new_name.strip(), industry_id))
                c.execute("UPDATE entities SET industry=?, updated_at=CURRENT_TIMESTAMP WHERE industry_id=?", (new_name.strip(), industry_id))
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
                active_comps = c.execute("SELECT COUNT(*) FROM entities WHERE industry_id=?", (industry_id,)).fetchone()[0]
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
