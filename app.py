#!/usr/bin/env python3
"""
app.py — News Intelligence Terminal Entry Point & HTTP Server
==============================================================
Slim entry point that imports modular components and serves
the API + static frontend.

Version: 2.0.0
"""

import os
import sys
import json
import queue
import time
import threading
import webbrowser
import datetime
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

from constants import (
    logger, APP_PORT, APP_VERSION, APP_DIR,
    load_config, save_config, DEFAULT_CONFIG,
)
from database import DatabaseManager
from intelligence import QueryBuilder
from providers import FetcherEngine


# ---------------------------------------------------------------------------
# HTTP Request Handler (API + Static Files)
# ---------------------------------------------------------------------------
class RequestHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests to API handlers and serves static files."""

    server_version = f"NewsIntel/{APP_VERSION}"

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def _json(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _serve_static(self, filepath: str, content_type: str):
        """Serves a static file from the static/ directory."""
        full_path = os.path.join(APP_DIR, filepath)
        if not os.path.exists(full_path):
            self.send_error(404)
            return
        with open(full_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ===== GET Routes =====
    def do_GET(self):
        path = self.path.split("?")[0]
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        # --- Static Files ---
        if path == "/" or path == "/index.html":
            self._serve_static("static/index.html", "text/html; charset=utf-8")

        elif path.endswith(".css"):
            self._serve_static(path.lstrip("/"), "text/css")

        elif path.endswith(".js"):
            self._serve_static(path.lstrip("/"), "application/javascript")

        # --- Dashboard APIs ---
        elif path == "/api/dashboard":
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.dashboard_metrics(user_name=user_name))

        elif path == "/api/dashboard/daily":
            limit = int(qs.get("limit", ["10"])[0])
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.get_daily_top_news(limit, user_name=user_name))

        elif path == "/api/dashboard/company_sentiments":
            days = int(qs.get("days", ["7"])[0])
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.get_company_sentiments(days, user_name=user_name))

        elif path == "/api/dashboard/tag_summary":
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.get_tag_summary(user_name=user_name))

        # --- Headlines (Paginated) ---
        elif path == "/api/headlines":
            page = int(qs.get("page", ["1"])[0])
            page_size = int(qs.get("page_size", [str(self.server.app_config.get("page_size", 50))])[0])
            q = qs.get("q", [""])[0]
            sentiment = qs.get("sentiment", ["All"])[0]
            recency = qs.get("recency", ["All"])[0]
            industry = qs.get("industry", ["All"])[0]
            company = qs.get("company", ["All"])[0]
            domain = qs.get("domain", ["All"])[0]
            date_from = qs.get("date_from", [""])[0]
            date_to = qs.get("date_to", [""])[0]
            ft = qs.get("filter_type", ["All"])[0]
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.fetch_headlines_paginated(
                page, page_size, q, sentiment, recency, industry, company, domain, date_from, date_to, ft, user_name=user_name
            ))

        # --- Taxonomy ---
        elif path == "/api/taxonomy/keywords":
            cat = qs.get("category", ["All"])[0]
            self._json(self.server.app_db.get_extracted_keywords(cat))

        # --- Query Inspector ---
        elif path == "/api/queries":
            portfolio = self.server.app_db.get_portfolio()
            categories = self.server.app_db.get_query_categories(enabled_only=False)
            domains = self.server.app_db.get_domains() if self.server.app_config.get("domain_filter_enabled") else []
            recency = self.server.app_config.get("recency_window", "7d")
            s_date = self.server.app_config.get("custom_start_date", "")
            e_date = self.server.app_config.get("custom_end_date", "")
            queries = []

            for comp in portfolio:
                company = comp["company"]
                ticker = comp.get("ticker", "")
                aliases = comp.get("aliases", [company])
                ind_id = comp.get("industry_id")
                ind_name = comp.get("industry") or (ind_id or "General")

                # Get applicable scoped queries with query-level deduplication
                applicable_cat_queries = QueryBuilder.get_applicable_queries_for_company(
                    comp, categories, domains=domains, recency=recency, start_date=s_date, end_date=e_date
                )

                broad_q = QueryBuilder.build_broad(company, aliases, domains, recency, ticker=ticker, start_date=s_date, end_date=e_date)
                bing_broad = QueryBuilder.build_bing(company, aliases, [], ticker=ticker)
                ddg_broad = QueryBuilder.build_duckduckgo(company, aliases, [], ticker=ticker)
                gf_fin_url = f"https://finance.google.com/finance/company_news?q={urllib.parse.quote(ticker)}&output=rss" if ticker else ""
                gf_news_q = f'"{ticker}" {company.split()[0]}' if ticker else company.split()[0]

                queries.append({
                    "company": company,
                    "ticker": ticker,
                    "industry": ind_name,
                    "industry_id": ind_id,
                    "aliases": aliases,
                    "broad_query": broad_q,
                    "provider_queries": {
                        "google_news": {"broad": broad_q, "label": "Google News RSS"},
                        "google_finance": {
                            "finance_rss_url": gf_fin_url,
                            "news_rss_query": gf_news_q,
                            "label": "Google Finance RSS",
                        },
                        "bing": {"broad": bing_broad, "label": "Bing News RSS"},
                        "ddg": {"broad": ddg_broad, "label": "DuckDuckGo HTML"},
                    },
                    "category_queries": applicable_cat_queries,
                })
            self._json(queries)

        elif path == "/api/query_categories":
            self._json(self.server.app_db.get_query_categories())

        # --- CRUD Lists ---
        elif path == "/api/portfolio":
            user_id = qs.get("user_id", [None])[0]
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.get_all_portfolio(user_id=user_id, user_name=user_name))

        elif path == "/api/portfolio/users":
            self._json(self.server.app_db.get_users())

        elif path == "/api/industries":
            self._json(self.server.app_db.get_all_industries())

        elif path == "/api/keywords":
            self._json(self.server.app_db.get_all_keywords())

        elif path == "/api/domains":
            self._json(self.server.app_db.get_all_domains())

        elif path == "/api/visibility":
            self._json(self.server.app_db.get_visibility())

        elif path.startswith("/api/canonical_events/") and path.endswith("/sources"):
            cid = path.split("/")[3]
            sources = self.server.app_db.get_canonical_event_sources(cid)
            self._json({"canonical_id": cid, "sources": sources})

        elif path == "/api/earnings/calendar":
            comp = qs.get("company", ["All"])[0]
            qtr = qs.get("quarter", ["All"])[0]
            st = qs.get("status", ["All"])[0]
            user_name = qs.get("user", ["All"])[0]
            events = self.server.app_db.get_earnings_calendar(company=comp, quarter=qtr, status=st, user_name=user_name)

            # Dynamic countdown calculation (calculated dynamically, never persisted)
            now = datetime.datetime.now()
            for ev in events:
                r_date = ev.get("reporting_date")
                if not r_date:
                    ev["countdown"] = "TBD"
                    continue
                try:
                    dt = datetime.datetime.strptime(r_date, "%Y-%m-%d")
                    delta_days = (dt.date() - now.date()).days
                    if delta_days == 0:
                        ev["countdown"] = "REPORTING TODAY"
                    elif delta_days > 0:
                        ev["countdown"] = f"⏰ T-{delta_days} Days"
                    else:
                        ev["countdown"] = f"REPORTED ({abs(delta_days)}d ago)"
                except Exception:
                    ev["countdown"] = "TBD"
            self._json(events)

        elif path == "/api/earnings/results":
            comp = qs.get("company", ["All"])[0]
            qtr = qs.get("quarter", ["All"])[0]
            user_name = qs.get("user", ["All"])[0]
            results = self.server.app_db.get_earnings_results(company=comp, quarter=qtr, user_name=user_name)

            # Calculate NII coverage ratio & NAV deltas
            for res in results:
                nav_curr = res.get("nav_per_share")
                nav_prior = res.get("nav_prior")
                if nav_curr and nav_prior and nav_prior > 0:
                    res["nav_change_pct"] = round(((nav_curr - nav_prior) / nav_prior) * 100.0, 2)
                else:
                    res["nav_change_pct"] = None

                nii = res.get("nii_per_share")
                div = res.get("dividend_regular")
                if nii and div and div > 0:
                    res["nii_coverage_pct"] = round((nii / div) * 100.0, 1)
                else:
                    res["nii_coverage_pct"] = None
            self._json(results)

        elif path == "/api/settings":
            self._json(self.server.app_config)

        elif path == "/api/status":
            self._json({"running": self.server.app_fetcher.is_running})

        # --- Filter Dropdowns ---
        elif path == "/api/filters/companies":
            user_name = qs.get("user", ["All"])[0]
            self._json(self.server.app_db.get_distinct_companies(user_name=user_name))

        elif path == "/api/filters/industries":
            self._json(self.server.app_db.get_distinct_industries())

        elif path == "/api/filters/domains":
            self._json(self.server.app_db.get_distinct_domains())

        # --- Training Data Export ---
        elif path == "/api/export/training_data":
            self._json(self.server.app_db.export_tagged_data())

        else:
            self.send_error(404)

    # ===== POST Routes =====
    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_json()

        if path == "/api/refresh":
            self.server.app_fetcher.start()
            self._json({"ok": True})

        elif path == "/api/earnings/trigger_sweep":
            res = self.server.app_fetcher.run_earnings_sweep()
            self._json({"ok": True, "result": res})

        elif path == "/api/headlines/clear":
            self.server.app_db.clear_all_headlines()
            self._json({"ok": True})

        elif path.startswith("/api/headlines/") and path.endswith("/star"):
            hid = int(path.split("/")[3])
            self._json({"ok": self.server.app_db.toggle_star(hid)})

        elif path.startswith("/api/headlines/") and path.endswith("/review"):
            hid = int(path.split("/")[3])
            self._json({"ok": self.server.app_db.mark_reviewed(hid)})

        elif path.startswith("/api/headlines/") and path.endswith("/sentiment"):
            hid = int(path.split("/")[3])
            sentiment = body.get("sentiment", "")
            self._json({"ok": self.server.app_db.update_user_sentiment(hid, sentiment)})

        elif path == "/api/headlines/review_all":
            cnt = self.server.app_db.mark_all_reviewed()
            self._json({"ok": True, "count": cnt})

        elif path == "/api/queries/test":
            test_q = body.get("query", "").strip()
            if not test_q:
                self._json({"items": []})
                return
            items = self.server.app_fetcher.parse_single_feed(
                f"https://news.google.com/rss/search?q={urllib.parse.quote(test_q)}&hl=en-US&gl=US&ceid=US:en",
                self.server.app_config.get("user_agents", DEFAULT_CONFIG["user_agents"])
            )
            self._json({"items": items[:15]})

        elif path == "/api/queries/test_provider":
            provider = body.get("provider", "google").strip()
            query = body.get("query", "").strip()
            if not query:
                self._json({"items": []})
                return
            items = self.server.app_fetcher.parse_single_feed_provider(
                provider, query,
                self.server.app_config.get("user_agents", DEFAULT_CONFIG["user_agents"])
            )
            self._json({"items": items[:15]})

        elif path == "/api/portfolio":
            ok = self.server.app_db.add_portfolio(
                body.get("company", ""), body.get("ticker", ""),
                body.get("industry", ""), body.get("country", ""),
                body.get("aliases", []), industry_id=body.get("industry_id"),
                user_name=body.get("user_name", "Default User"),
                user_id=body.get("user_id"))
            self._json({"ok": ok})

        elif path.startswith("/api/portfolio/") and path.endswith("/update"):
            pid = int(path.split("/")[3])
            ok = self.server.app_db.update_portfolio(
                pid, body.get("company", ""), body.get("ticker", ""),
                body.get("industry", ""), body.get("country", ""),
                body.get("aliases", []), industry_id=body.get("industry_id"),
                user_name=body.get("user_name", "Default User"),
                user_id=body.get("user_id"))
            self._json({"ok": ok})

        elif path == "/api/portfolio/bulk_toggle":
            self.server.app_db.bulk_toggle_portfolio(
                body.get("enabled", 1),
                user_id=body.get("user_id"),
                user_name=body.get("user_name"))
            self._json({"ok": True})

        elif path == "/api/portfolio/import":
            items = body.get("items", [])
            atomic = body.get("atomic", True)
            res = self.server.app_db.bulk_import_portfolio(items, atomic=atomic)
            self._json(res)

        elif path.startswith("/api/portfolio/") and path.endswith("/toggle"):
            pid = int(path.split("/")[3])
            self.server.app_db.toggle_portfolio(pid, body.get("enabled", 1))
            self._json({"ok": True})

        elif path == "/api/query_categories":
            ok = self.server.app_db.add_query_category(
                name=body.get("name", ""),
                keywords=body.get("keywords", []),
                scope_type=body.get("scope_type", "UNIVERSAL"),
                industry_id=body.get("industry_id"),
                company_id=body.get("company_id"),
                priority=int(body.get("priority", 70)),
                target_dimension=body.get("target_dimension", "Earnings / Cash Flow")
            )
            self._json({"ok": ok})

        elif path.startswith("/api/query_categories/") and path.endswith("/update"):
            qid = int(path.split("/")[3])
            ok = self.server.app_db.update_query_category(
                qid,
                name=body.get("name", ""),
                keywords=body.get("keywords", []),
                scope_type=body.get("scope_type", "UNIVERSAL"),
                industry_id=body.get("industry_id"),
                company_id=body.get("company_id"),
                priority=int(body.get("priority", 70)),
                target_dimension=body.get("target_dimension", "Earnings / Cash Flow")
            )
            self._json({"ok": ok})

        elif path.startswith("/api/query_categories/") and path.endswith("/toggle"):
            qid = int(path.split("/")[3])
            self.server.app_db.toggle_query_category(qid, body.get("enabled", 1))
            self._json({"ok": True})

        elif path == "/api/visibility/add":
            ok = self.server.app_db.add_visibility_domain(body.get("domain", ""), body.get("visible", 0))
            self._json({"ok": ok})

        elif path == "/api/visibility/bulk_toggle":
            self.server.app_db.bulk_toggle_visibility(body.get("visible", 1))
            self._json({"ok": True})

        elif path == "/api/earnings/calendar":
            ok = self.server.app_db.save_earnings_calendar(body)
            self._json({"ok": ok})

        elif path == "/api/earnings/results":
            ok = self.server.app_db.save_earnings_results(body)
            self._json({"ok": ok})

        elif path.startswith("/api/visibility/") and path.endswith("/toggle"):
            vid = int(path.split("/")[3])
            self.server.app_db.toggle_visibility(vid, body.get("visible", 1))
            self._json({"ok": True})

        elif path == "/api/industries":
            ok, msg = self.server.app_db.add_industry(
                body.get("id", ""), body.get("name", ""), body.get("risk_profile", "STANDARD_CORP")
            )
            self._json({"ok": ok, "message": msg})

        elif path.startswith("/api/industries/") and path.endswith("/rename"):
            iid = urllib.parse.unquote(path.split("/")[3])
            ok = self.server.app_db.rename_industry(iid, body.get("name", ""))
            self._json({"ok": ok})

        elif path.startswith("/api/industries/") and (path.endswith("/toggle") or path.endswith("/status")):
            iid = urllib.parse.unquote(path.split("/")[3])
            ok = self.server.app_db.toggle_industry_status(iid, body.get("status"))
            self._json({"ok": ok})

        elif path == "/api/keywords":
            ok = self.server.app_db.add_keyword(body.get("word", ""))
            self._json({"ok": ok})

        elif path.startswith("/api/keywords/") and path.endswith("/toggle"):
            kid = int(path.split("/")[3])
            self.server.app_db.toggle_keyword(kid, body.get("enabled", 1))
            self._json({"ok": True})

        elif path == "/api/domains":
            ok = self.server.app_db.add_domain(body.get("domain", ""))
            self._json({"ok": ok})

        elif path == "/api/domains/bulk_toggle":
            self.server.app_db.bulk_toggle_domains(body.get("enabled", 1))
            self._json({"ok": True})

        elif path.startswith("/api/domains/") and path.endswith("/toggle"):
            did = int(path.split("/")[3])
            self.server.app_db.toggle_domain(did, body.get("enabled", 1))
            self._json({"ok": True})

        elif path == "/api/settings":
            for k, v in body.items():
                self.server.app_config[k] = v
            save_config(self.server.app_config)
            self._json({"ok": True})

        else:
            self.send_error(404)

    # ===== DELETE Routes =====
    def do_DELETE(self):
        path = self.path.split("?")[0]

        if path.startswith("/api/portfolio/"):
            pid = int(path.split("/")[3])
            self.server.app_db.delete_portfolio(pid)
            self._json({"ok": True})

        elif path.startswith("/api/query_categories/"):
            qid = int(path.split("/")[3])
            self.server.app_db.delete_query_category(qid)
            self._json({"ok": True})

        elif path.startswith("/api/visibility/"):
            vid = int(path.split("/")[3])
            self.server.app_db.delete_visibility_domain(vid)
            self._json({"ok": True})

        elif path.startswith("/api/industries/"):
            iid = urllib.parse.unquote(path.split("/")[3])
            ok, msg = self.server.app_db.delete_industry(iid)
            self._json({"ok": ok, "message": msg})

        elif path.startswith("/api/keywords/"):
            kid = int(path.split("/")[3])
            self.server.app_db.delete_keyword(kid)
            self._json({"ok": True})

        elif path.startswith("/api/domains/"):
            did = int(path.split("/")[3])
            self.server.app_db.delete_domain(did)
            self._json({"ok": True})

        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
def main():
    config = load_config()
    db = DatabaseManager(config["db_path"])
    event_q: queue.Queue = queue.Queue()
    fetcher = FetcherEngine(db, config, event_q)

    server = HTTPServer(("127.0.0.1", APP_PORT), RequestHandler)
    server.app_db = db
    server.app_config = config
    server.app_fetcher = fetcher

    url = f"http://localhost:{APP_PORT}"
    logger.info("Starting News Intelligence Terminal at %s", url)
    print(f"\n  📰  News Intelligence Terminal v{APP_VERSION}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  🌐  Open in browser: {url}")
    print(f"  🔧  Workers: {config.get('max_workers', 4)} parallel threads")
    print(f"  🔍  Providers: Google={'ON' if config.get('google_enabled') else 'OFF'} | "
          f"Bing={'ON' if config.get('bing_enabled') else 'OFF'} | "
          f"DDG={'ON' if config.get('duckduckgo_enabled') else 'OFF'} | "
          f"GDELT={'ON' if config.get('gdelt_enabled') else 'OFF'}")
    print(f"  📅  Recency Window: {config.get('recency_window', '7d')}")
    print(f"  💾  Database: {config['db_path']}")
    print(f"  ⏹  Press Ctrl+C to stop\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    def auto_refresh():
        interval = config.get("refresh_interval_minutes", 5) * 60
        while True:
            time.sleep(interval)
            logger.info("Auto-refresh triggered.")
            fetcher.start()

    threading.Thread(target=auto_refresh, daemon=True, name="AutoRefresh").start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
