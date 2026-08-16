#!/usr/bin/env python3
"""
providers.py — Multi-Provider News Fetcher with Async Workers
==============================================================
All 5 search providers (DuckDuckGo, Bing, GDELT, Google, RSS) +
ThreadPoolExecutor-based parallel fetcher with smart retry (max 3).
"""

import re
import json
import time
import queue
import random
import datetime
import threading
import collections
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from constants import logger, DEFAULT_CONFIG, parse_pub_date, DateProvenanceResolver
from intelligence import IntelEngine, QueryBuilder


# ---------------------------------------------------------------------------
# Base Fetch Helper with Smart Retry (max 3, no looping)
# ---------------------------------------------------------------------------
import ssl

# Fast unverified SSL context for high-performance HTTP requests
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_with_retry(url: str, user_agents: List[str], provider_name: str,
                     max_retries: int = 2, timeout: int = 5) -> Optional[bytes]:
    """
    Ultra-fast simple HTTP request wrapper using standard urllib.
    Non-blocking: fast timeout (default 5s) so threads never hang.
    """
    for attempt in range(1, max_retries + 1):
        try:
            ua = random.choice(user_agents) if user_agents else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            req = urllib.request.Request(url, headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 403, 503):
                if attempt == max_retries:
                    return None
                time.sleep(1.0)
            else:
                return None
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(0.5)
    return None


def parse_rss_items(raw: bytes) -> List[Dict[str, Any]]:
    """Parses RSS XML feed into standardized item dicts with date provenance."""
    items = []
    try:
        root = ET.fromstring(raw)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
            if title and link:
                date_prov = DateProvenanceResolver.resolve_best_date(raw_date_str=pub_date)
                items.append({
                    "title": title, "link": link,
                    "pubDate": pub_date,
                    "published_at": date_prov["published_at"],
                    "published_at_raw": date_prov["published_at_raw"],
                    "date_source": date_prov["date_source"],
                    "date_confidence": date_prov["date_confidence"],
                    "source": item.findtext("source") or "RSS",
                })
    except Exception as e:
        logger.debug("RSS parse error: %s", e)
    return items


# ---------------------------------------------------------------------------
# Provider: Google News RSS
# ---------------------------------------------------------------------------
class GoogleNewsProvider:
    name = "Google News"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("google_enabled", True)

    def fetch(self, query: str, user_agents: List[str]) -> List[Dict[str, str]]:
        eq = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={eq}&hl=en-US&gl=US&ceid=US:en"
        raw = fetch_with_retry(url, user_agents, self.name, max_retries=self.config.get("max_retries", 3))
        if not raw:
            return []
        return parse_rss_items(raw)

    def get_delay(self) -> float:
        """Random delay within configured range to avoid bot detection."""
        delay_range = self.config.get("google_delay_range", [3, 8])
        base = random.uniform(delay_range[0], delay_range[1])
        jitter = base * random.uniform(-0.3, 0.3)
        return max(1.0, base + jitter)


# ---------------------------------------------------------------------------
# Provider: Google Finance RSS (ticker-based company news)
# ---------------------------------------------------------------------------
class GoogleFinanceProvider:
    name = "Google Finance"
    # Google Finance company news RSS — highly targeted, ticker-based
    # Format: https://finance.google.com/finance/company_news?q=TICKER&output=rss
    # Also fetches Google News RSS filtered by ticker for non-US companies
    NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    FINANCE_RSS = "https://finance.google.com/finance/company_news?q={ticker}&output=rss"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("google_finance_enabled", True)

    def fetch(self, company: str, aliases: List[str], ticker: str, user_agents: List[str],
              recency: str = "7d") -> List[Dict[str, str]]:
        items = []
        uas = user_agents

        # 1. Google Finance RSS (ticker-based) — most targeted source
        if ticker:
            fin_url = self.FINANCE_RSS.format(ticker=urllib.parse.quote(ticker))
            raw = fetch_with_retry(fin_url, uas, self.name, max_retries=2, timeout=6)
            if raw and len(raw) > 100:
                parsed = parse_rss_items(raw)
                for it in parsed:
                    it["source"] = f"Google Finance ({ticker})"
                items.extend(parsed)

        # 2. Google News RSS with ticker as search term
        if ticker:
            q_ticker = f'"{ticker}" {company.split()[0]}'
            url2 = self.NEWS_RSS.format(query=urllib.parse.quote(q_ticker))
            if recency and recency != "any":
                url2 += f"&tbs=qdr:{recency[0]}"
            raw2 = fetch_with_retry(url2, uas, self.name, max_retries=2, timeout=6)
            if raw2 and len(raw2) > 100:
                parsed2 = parse_rss_items(raw2)
                for it in parsed2:
                    it["source"] = f"Google Finance"
                items.extend(parsed2)

        # Deduplicate by URL
        seen, deduped = set(), []
        for it in items:
            if it["link"] not in seen:
                seen.add(it["link"])
                deduped.append(it)
        return deduped

    def build_query_preview(self, company: str, aliases: List[str], ticker: str, recency: str = "7d") -> Dict[str, str]:
        """Returns human-readable query preview for the Query Inspector."""
        fin_url = self.FINANCE_RSS.format(ticker=urllib.parse.quote(ticker)) if ticker else "N/A (no ticker)"
        news_q = f'"{ticker}" {company.split()[0]}' if ticker else company.split()[0]
        return {
            "finance_rss_url": fin_url,
            "news_rss_query": news_q,
        }

    def get_delay(self) -> float:
        return random.uniform(1.0, 2.5)


# ---------------------------------------------------------------------------
# Provider: Yahoo Finance RSS (ticker-targeted earnings & corporate releases)
# ---------------------------------------------------------------------------
class YahooFinanceProvider:
    """Fetches real-time corporate news & earnings announcements via Yahoo Finance RSS."""
    RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("yahoo_finance_enabled", True)
        self.name = "Yahoo Finance"

    def fetch(self, ticker: str, uas: List[str]) -> List[Dict[str, Any]]:
        if not self.enabled or not ticker:
            return []
        url = self.RSS_URL.format(ticker=urllib.parse.quote(ticker))
        raw = fetch_with_retry(url, uas, self.name, max_retries=2, timeout=6)
        if not raw or len(raw) < 100:
            return []
        items = parse_rss_items(raw)
        for it in items:
            it["source"] = f"Yahoo Finance ({ticker})"
        return items

    def get_delay(self) -> float:
        return random.uniform(0.5, 1.5)


# ---------------------------------------------------------------------------
# Provider: Bing News (HTML scrape — extracts news links from bing.com/news)
# ---------------------------------------------------------------------------
class BingNewsProvider:
    name = "Bing News"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("bing_enabled", True)

    def fetch(self, query: str, user_agents: List[str]) -> List[Dict[str, str]]:
        """Scrape bing.com/news/search HTML page for news article links."""
        items = []
        eq = urllib.parse.quote(query)
        url = f"https://www.bing.com/news/search?q={eq}&form=NSBCLK"
        ua = random.choice(user_agents) if user_agents else "Mozilla/5.0"
        req = urllib.request.Request(url, headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        try:
            with urllib.request.urlopen(req, timeout=8, context=SSL_CTX) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Extract external links with their text — Bing wraps news in <a> tags to external sites
            external_links = re.findall(
                r'<a[^>]*href="(https?://(?!www\.bing\.com|www\.microsoft\.com|go\.microsoft\.com|account\.microsoft)[^"]+)"[^>]*>([\s\S]*?)</a>',
                html
            )
            seen_urls = set()
            for href, inner_html in external_links:
                # Strip HTML tags to get clean title text
                title = re.sub(r'<[^>]+>', ' ', inner_html).strip()
                title = re.sub(r'\s+', ' ', title)
                if not title or len(title) < 12 or href in seen_urls:
                    continue
                # Skip non-news links (images, trackers, etc.)
                if any(skip in href.lower() for skip in ['.jpg', '.png', '.gif', 'javascript:', '#', '/search?', 'bing.com']):
                    continue
                seen_urls.add(href)
                items.append({
                    "title": title,
                    "link": href,
                    "pubDate": "",
                    "source": "Bing News",
                })
        except Exception as e:
            logger.debug("Bing HTML scrape error: %s", e)

        # Also fetch RSS as supplementary source (high-quality structured data)
        try:
            rss_url = f"https://www.bing.com/news/search?q={eq}&format=rss"
            raw = fetch_with_retry(rss_url, user_agents, "Bing RSS", max_retries=2)
            if raw:
                rss_items = parse_rss_items(raw)
                for it in rss_items:
                    if it["link"] not in seen_urls:
                        seen_urls.add(it["link"])
                        items.append(it)
        except Exception:
            pass

        return items

    def get_delay(self) -> float:
        return random.uniform(1.0, 2.5)


# ---------------------------------------------------------------------------
# Provider: DuckDuckGo (Human Cookie-Session HTML Scrape)
# ---------------------------------------------------------------------------
class DuckDuckGoProvider:
    name = "DuckDuckGo"
    _lock = threading.Lock()

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("duckduckgo_enabled", True)

    @classmethod
    def reset_rate_limit(cls):
        pass

    @staticmethod
    def _decode_uddg(href: str) -> str:
        try:
            if "uddg=" in href:
                qs = urllib.parse.urlparse("https:" + href if href.startswith("//") else href).query
                params = urllib.parse.parse_qs(qs)
                decoded = params.get("uddg", [""])[0]
                return urllib.parse.unquote(decoded) if decoded else href
        except Exception:
            pass
        return href

    def fetch(self, query: str, user_agents: List[str]) -> List[Dict[str, str]]:
        """
        Simulates an authentic human browser session on html.duckduckgo.com:
        1. Maintains an http.cookiejar session processor.
        2. Visits landing page to collect session cookies.
        3. Submits clean HTML POST query with authentic headers & cookies.
        """
        items = []
        with DuckDuckGoProvider._lock:
            try:
                import http.cookiejar
                cookie_jar = http.cookiejar.CookieJar()
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=SSL_CTX),
                    urllib.request.HTTPCookieProcessor(cookie_jar)
                )
                ua = random.choice(user_agents) if user_agents else "Mozilla/5.0"

                # Step 1: Establish human session cookies by visiting landing page
                try:
                    req_land = urllib.request.Request("https://html.duckduckgo.com/html/", headers={
                        "User-Agent": ua,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    })
                    with opener.open(req_land, timeout=5) as resp:
                        _ = resp.read()
                except Exception:
                    pass

                # Step 2: Clean query string (strip strict outer quotes if wrapping whole query)
                clean_q = query.strip()
                if clean_q.startswith('"') and clean_q.endswith('"') and clean_q.count('"') == 2:
                    clean_q = clean_q[1:-1]
                if "news" not in clean_q.lower():
                    clean_q += " news"

                url = "https://html.duckduckgo.com/html/"
                data = urllib.parse.urlencode({"q": clean_q, "b": "", "kl": "us-en"}).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://html.duckduckgo.com",
                    "Referer": "https://html.duckduckgo.com/html/",
                })

                with opener.open(req, timeout=6) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")

                matches = re.findall(r'<a[^>]+class=["\']result__a["\'][^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', html)
                if not matches:
                    matches = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*class=["\']result__a["\'][^>]*>([^<]+)</a>', html)

                seen = set()
                for href, title in matches:
                    real_url = self._decode_uddg(href)
                    title = re.sub(r'\s+', ' ', title).strip()
                    if not title or not real_url or real_url in seen or "duckduckgo.com" in real_url:
                        continue
                    seen.add(real_url)
                    items.append({
                        "title": title,
                        "link": real_url,
                        "pubDate": "",
                        "source": "DuckDuckGo",
                    })
            except Exception as e:
                logger.debug("DDG human HTML scraper error: %s", e)

        return items

    def get_delay(self) -> float:
        return random.uniform(1.5, 3.0)



# ---------------------------------------------------------------------------
# Provider: GDELT (Plain HTTP RSS — avoids SSL handshake timeout)
# ---------------------------------------------------------------------------
class GDELTProvider:
    name = "GDELT"
    API_RSS = "http://api.gdeltproject.org/api/v2/doc/doc"
    _lock = threading.Lock()

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("gdelt_enabled", True)

    def fetch(self, query: str, user_agents: List[str], days: int = 7) -> List[Dict[str, str]]:
        items = []
        with self._lock:  # Query GDELT sequentially to prevent 429 rate-limiting
            time.sleep(0.8)
            try:
                timespan = f"{min(days, 30)}days"
                params = {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": "25",
                    "timespan": timespan,
                    "format": "rss",
                    "sort": "DateDesc",
                }
                url = f"{self.API_RSS}?{urllib.parse.urlencode(params)}"
                ua = random.choice(user_agents) if user_agents else "Mozilla/5.0"
                req = urllib.request.Request(url, headers={
                    "User-Agent": ua,
                    "Accept": "application/rss+xml,text/xml,*/*",
                })
                try:
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        raw = resp.read()
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        logger.debug("GDELT rate-limited (429), skipping cleanly.")
                    else:
                        logger.debug("GDELT HTTP error %s", e.code)
                    return []
                except Exception as e:
                    logger.debug("GDELT fetch error: %s", e)
                    return []

                if not raw or len(raw) < 50:
                    return []

                parsed = parse_rss_items(raw)
                for it in parsed[:25]:
                    items.append({
                        "title": it["title"],
                        "link": it["link"],
                        "pubDate": it.get("pubDate", ""),
                        "source": it.get("source", "GDELT"),
                    })
            except Exception as e:
                logger.debug("GDELT provider error: %s", e)
        return items

    def get_delay(self) -> float:
        return random.uniform(1.0, 2.0)


# ---------------------------------------------------------------------------
# Provider: Direct RSS Feeds (CNBC, Yahoo Finance, MarketWatch, Reuters, AP)
# ---------------------------------------------------------------------------
class RSSFeedProvider:
    name = "RSS Feeds"

    # Curated list of live financial/news RSS feeds (validated active URLs)
    FEEDS = [
        ("CNBC Finance",    "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
        ("CNBC Top News",   "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Yahoo Finance",   "https://finance.yahoo.com/news/rssindex"),
        ("MarketWatch",     "http://feeds.marketwatch.com/marketwatch/topstories/"),
        ("Reuters Business","https://feeds.reuters.com/reuters/businessNews"),
        ("AP Business",     "https://rsshub.app/apnews/topics/business"),
        ("Seeking Alpha",   "https://seekingalpha.com/feed.xml"),
        ("FT Markets",      "https://www.ft.com/rss/home/uk"),
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("rss_enabled", True)

    def fetch(self, company: str, aliases: List[str], user_agents: List[str]) -> List[Dict[str, str]]:
        # Build search terms: company name + all aliases, min 4 chars
        all_names = list({n.lower() for n in ([company] + (aliases or [])) if len(n) >= 4})
        results = []
        seen_urls = set()
        for feed_name, url in self.FEEDS:
            raw = fetch_with_retry(url, user_agents, f"RSS:{feed_name}", max_retries=1, timeout=5)
            if not raw or len(raw) < 100:
                continue
            items = parse_rss_items(raw)
            for it in items:
                if it["link"] in seen_urls:
                    continue
                title_lower = it["title"].lower()
                if any(name in title_lower for name in all_names):
                    it["source"] = feed_name
                    seen_urls.add(it["link"])
                    results.append(it)
        return results

    def get_delay(self) -> float:
        return random.uniform(0.1, 0.5)


# ---------------------------------------------------------------------------
# Fetcher Engine (Async Workers with ThreadPoolExecutor)
# ---------------------------------------------------------------------------
class FetcherEngine:
    """
    Multi-provider, multi-threaded news fetcher.
    Uses ThreadPoolExecutor for parallel per-company fetching.
    Smart error retry (max 3, no looping).
    """

    def __init__(self, db, config: Dict[str, Any], event_queue: queue.Queue):
        self.db = db
        self.config = config
        self.q = event_queue
        self._running = False
        self._lock = threading.Lock()

        # Initialize providers
        self.google = GoogleNewsProvider(config)
        self.google_finance = GoogleFinanceProvider(config)
        self.yahoo = YahooFinanceProvider(config)
        self.bing = BingNewsProvider(config)
        self.duckduckgo = DuckDuckGoProvider(config)
        self.gdelt = GDELTProvider(config)
        self.rss = RSSFeedProvider(config)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        threading.Thread(target=self._cycle, daemon=True, name="FetcherMain").start()

    def _cycle(self):
        start_ts = datetime.datetime.now().strftime("%H:%M:%S")
        logger.info("🚀 Starting parallel discovery cycle...")

        # Reset DDG rate-limit flag so each cycle gets a fresh chance
        DuckDuckGoProvider.reset_rate_limit()

        # Initialize thread-safe cycle metrics
        self.stats = {
            "start_time": start_ts,
            "providers": {"DuckDuckGo": 0, "Bing News": 0, "GDELT": 0, "Google News": 0, "Google Finance": 0, "RSS Feeds": 0},
            "raw_total": 0,
            "deduped": 0,
            "low_score": 0,
            "noise": 0,
            "saved": 0,
            "companies": collections.defaultdict(lambda: {"sourced": 0, "saved": 0, "deduped": 0}),
        }

        try:
            portfolio = self.db.get_portfolio()
            if not portfolio:
                logger.info("No portfolio companies configured. Skipping.")
                return

            max_workers = self.config.get("max_workers", 4)
            logger.info("Using %d worker threads for %d companies", max_workers, len(portfolio))

            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as pool:
                futures = {pool.submit(self._fetch_company, comp): comp["company"] for comp in portfolio}

                for future in as_completed(futures):
                    company = futures[future]
                    try:
                        result = future.result()
                        logger.info("✅ %s: fetched %d new headlines", company, result)
                    except Exception as e:
                        logger.error("❌ %s: worker failed: %s", company, e)

            # Print Terminal Analysis Summary Report
            self._log_terminal_analysis(start_ts)

        except Exception as e:
            logger.error("❌ Cycle error: %s", e)
        finally:
            self._running = False
            self.q.put({"type": "DONE", "ts": datetime.datetime.now().strftime("%H:%M:%S")})
            logger.info("✅ Discovery cycle complete.")

    def run_earnings_sweep(self) -> Dict[str, Any]:
        """Dedicated trigger for Earnings & Future Reporting Intelligence Sweep."""
        start_ts = datetime.datetime.now().strftime("%H:%M:%S")
        portfolio = self.db.get_portfolio()
        uas = self.config.get("user_agents", DEFAULT_CONFIG["user_agents"])
        recency = self.config.get("recency_window", "7d")

        logger.info("📅 Starting Dedicated Earnings & Future Reporting Sweep for %d portfolio entities...", len(portfolio))

        total_sourced = 0
        total_events_saved = 0

        # Step 1: Parse existing ingested headlines in DB
        try:
            from intelligence import DeterministicEarningsParser
            all_db_headlines = self.db._conn().execute("SELECT headline, source, url, company FROM headlines").fetchall()
            for row in all_db_headlines:
                h_text = row["headline"]
                c_name = row["company"] or "General"
                h_url = row["url"] or ""
                ev = DeterministicEarningsParser.detect_future_earnings(h_text, company=c_name, url=h_url)
                if ev and self.db.save_earnings_calendar(ev):
                    total_events_saved += 1
                met = DeterministicEarningsParser.extract_quarterly_metrics(h_text, company=c_name, url=h_url)
                if met and self.db.save_earnings_results(met):
                    total_events_saved += 1
        except Exception as e:
            logger.debug("DB headline parse error: %s", e)

        # Step 2: Live multi-provider search for earnings releases
        for comp in portfolio:
            company_name = comp["company"]
            ticker = comp.get("ticker", "")
            aliases = comp.get("aliases", [company_name])

            # Query 1: Yahoo Finance RSS feed (ticker-targeted)
            items_y = []
            if ticker:
                logger.info("🔍 [Yahoo Finance RSS] Fetching ticker earnings: %s", ticker)
                items_y = self.yahoo.fetch(ticker, uas)
                total_sourced += len(items_y)

            # Query 2: Google News search for earnings releases
            q_g = f'("{company_name}" OR "{ticker}") AND ("earnings release" OR "conference call" OR "financial results")'
            logger.info("🔍 [Google News] Searching Earnings: %s", q_g)
            items_g = self.google.fetch(q_g, uas)
            total_sourced += len(items_g)

            # Query 3: Bing News search for results announcements
            q_b = f'("{company_name}" OR "{ticker}") AND ("announces results" OR "reports quarter" OR "NAV per share")'
            logger.info("🔍 [Bing News] Searching Earnings: %s", q_b)
            items_b = self.bing.fetch(q_b, uas)
            total_sourced += len(items_b)

            all_items = items_y + items_g + items_b

            for it in all_items:
                try:
                    from intelligence import DeterministicEarningsParser
                    ev = DeterministicEarningsParser.detect_future_earnings(it["title"], company=company_name, url=it["link"])
                    if ev and self.db.save_earnings_calendar(ev):
                        total_events_saved += 1
                    met = DeterministicEarningsParser.extract_quarterly_metrics(it["title"], company=company_name, url=it["link"])
                    if met and self.db.save_earnings_results(met):
                        total_events_saved += 1
                except Exception:
                    pass

        end_ts = datetime.datetime.now().strftime("%H:%M:%S")
        logger.info("✅ Earnings Sweep Complete [%s ➔ %s]: %d items sourced | %d events saved.", start_ts, end_ts, total_sourced, total_events_saved)

        return {
            "total_sourced": total_sourced,
            "total_events_saved": total_events_saved,
            "companies_swept": len(portfolio)
        }

    def _log_terminal_analysis(self, start_ts: str):
        """Prints a comprehensive terminal analysis summary report at the end of each run."""
        s = self.stats
        end_ts = datetime.datetime.now().strftime("%H:%M:%S")
        prov_lines = "\n".join(f"     - {p:<12}: {cnt} items" for p, cnt in s["providers"].items())

        comp_lines = []
        for comp, stats in sorted(s["companies"].items()):
            comp_lines.append(f"     - {comp:<35}: {stats['sourced']:>3} sourced | {stats['deduped']:>3} deduped | {stats['saved']:>3} new saved")
        comp_summary = "\n".join(comp_lines) if comp_lines else "     - None"

        report = f"""
  ====================================================================
  📊 DISCOVERY CYCLE TERMINAL ANALYSIS REPORT [{start_ts} ➔ {end_ts}]
  ====================================================================
  • Search Provider Breakdown (Raw Sourced):
{prov_lines}
  --------------------------------------------------------------------
  • Total Raw Items Sourced     : {s['raw_total']}
  • Duplicates Filtered (URL)   : {s['deduped']}
  • Below Min Score (< {self.config.get('min_relevance_ingest', 40.0)})    : {s['low_score']}
  • Noise Headlines Filtered    : {s['noise']}
  • Total New Headlines Saved   : {s['saved']}
  --------------------------------------------------------------------
  • Per-Company Sourced Breakdown:
{comp_summary}
  ====================================================================
"""
        print(report)

    def _fetch_company(self, comp: Dict[str, Any]) -> int:
        """Fetches news from ALL enabled providers for one company. Runs in worker thread."""
        # Initial random stagger (0.2s - 1.5s) per worker thread to avoid hitting providers simultaneously
        time.sleep(random.uniform(0.2, 1.5))

        company_name = comp["company"]
        ticker = comp.get("ticker", "")
        aliases = comp.get("aliases", [company_name])
        uas = self.config.get("user_agents", DEFAULT_CONFIG["user_agents"])
        domains = self.db.get_domains() if self.config.get("domain_filter_enabled") else []
        recency = self.config.get("recency_window", "7d")
        s_date = self.config.get("custom_start_date", "")
        s_date = self.config.get("custom_start_date", "")
        e_date = self.config.get("custom_end_date", "")
        all_categories = self.db.get_query_categories(enabled_only=True)

        # Smart query generation: Universal + Company's Industry + Company Overrides with query-level deduplication
        applicable_queries = QueryBuilder.get_applicable_queries_for_company(
            comp, all_categories, domains=domains, recency=recency, start_date=s_date, end_date=e_date
        )

        total_saved = 0

        # Provider 1: DuckDuckGo (DDGS news search — ONE broad query per company)
        # DDG rate-limits aggressively (403 after ~1-2 queries). Global lock ensures
        # sequential access. If rate-limited, skips cleanly for remaining companies.
        if self.duckduckgo.enabled:
            q_broad_ddg = QueryBuilder.build_duckduckgo(company_name, aliases, [], ticker=ticker)
            items = self.duckduckgo.fetch(q_broad_ddg, uas)
            total_saved += self._process_items(items, company_name, q_broad_ddg, "DuckDuckGo")
            time.sleep(self.duckduckgo.get_delay())

        # Provider 2: Bing News (Always runs Broad Search + Scoped Category sweeps)
        if self.bing.enabled:
            # Broad Bing search
            q_broad_b = QueryBuilder.build_bing(company_name, aliases, [], ticker=ticker)
            items = self.bing.fetch(q_broad_b, uas)
            total_saved += self._process_items(items, company_name, q_broad_b, "Bing News")
            time.sleep(self.bing.get_delay())

            # Scoped Category Bing sweeps
            for cat_item in applicable_queries:
                q = cat_item.get("bing_query") or QueryBuilder.build_bing(company_name, aliases, cat_item.get("keywords", []), domains, ticker=ticker)
                items = self.bing.fetch(q, uas)
                total_saved += self._process_items(items, company_name, q, "Bing News")
                time.sleep(self.bing.get_delay())

        # Provider 3: GDELT (Fast non-blocking)
        if self.gdelt.enabled:
            q = QueryBuilder.build_gdelt(company_name, aliases, ticker=ticker)
            days = 1 if recency == "1d" else 7 if recency == "7d" else 30
            items = self.gdelt.fetch(q, uas, days=days)
            total_saved += self._process_items(items, company_name, q, "GDELT")
            time.sleep(self.gdelt.get_delay())

        # Provider 4: Google News (Always runs Broad Search + Scoped Category sweeps)
        if self.google.enabled:
            # Always run Broad Google search (gets 50-100 top material headlines for company)
            q_broad_g = QueryBuilder.build_broad(company_name, aliases, domains, recency, ticker=ticker, start_date=s_date, end_date=e_date)
            items = self.google.fetch(q_broad_g, uas)
            total_saved += self._process_items(items, company_name, q_broad_g, "Google News")
            time.sleep(self.google.get_delay())

            # Scoped Category Google sweeps (Deduplicated query execution)
            for cat_item in applicable_queries:
                q = cat_item.get("query") or QueryBuilder.build_google(company_name, aliases, cat_item.get("keywords", []), domains, recency, ticker=ticker, start_date=s_date, end_date=e_date)
                items = self.google.fetch(q, uas)
                total_saved += self._process_items(items, company_name, q, "Google News")
                time.sleep(self.google.get_delay())

        # Provider 5: Google Finance RSS (ticker-targeted)
        if self.google_finance.enabled and ticker:
            items = self.google_finance.fetch(company_name, aliases, ticker, uas, recency)
            total_saved += self._process_items(items, company_name, f"GoogleFinance:{ticker}", "Google Finance")
            time.sleep(self.google_finance.get_delay())

        # Provider 6: Direct RSS Feeds
        if self.rss.enabled:
            items = self.rss.fetch(company_name, aliases, uas)
            total_saved += self._process_items(items, company_name, company_name, "RSS Feeds")
            time.sleep(self.rss.get_delay())

        return total_saved

    def _process_items(self, items: List[Dict], company: str, query: str, provider_key: str) -> int:
        """Processes fetched items: noise filter, scoring, DB save."""
        if not items:
            return 0

        # Standardize provider key for metrics summary
        p_name = ("DuckDuckGo" if "Duck" in provider_key
                  else "Bing News" if "Bing" in provider_key
                  else "GDELT" if "GDELT" in provider_key
                  else "Google Finance" if "Finance" in provider_key
                  else "Google News" if "Google" in provider_key
                  else "RSS Feeds")

        portfolio = self.db.get_portfolio()
        industries = self.db.get_distinct_industries()
        keywords = self.db.get_keywords()
        corpus_stats = self.db.get_corpus_stats()
        saved_count = 0

        recency = self.config.get("recency_window", "7d")
        days = 1 if recency == "1d" else 7 if recency == "7d" else 30 if recency == "30d" else 0
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d") if days > 0 else ""
        min_ingest_score = float(self.config.get("min_relevance_ingest", 40.0))

        for it in items:
            # Metrics: raw items sourced
            if hasattr(self, "stats"):
                self.stats["raw_total"] += 1
                self.stats["providers"][p_name] += 1
                self.stats["companies"][company]["sourced"] += 1

            if IntelEngine.is_noise(it["title"]):
                if hasattr(self, "stats"):
                    self.stats["noise"] += 1
                continue
            pub_formatted = parse_pub_date(it.get("pubDate"))
            if cutoff and pub_formatted < cutoff:
                continue

            intel = IntelEngine.analyze(
                it["title"], it.get("source", provider_key),
                portfolio, industries, keywords,
                corpus_stats=corpus_stats,
                config=self.config,
            )

            if intel["relevance_score"] < min_ingest_score:
                if hasattr(self, "stats"):
                    self.stats["low_score"] += 1
                continue

            final_company = intel["company"]
            if final_company == "General":
                if intel["relevance_score"] < 50.0:
                    continue
                final_company = company

            # Extract taxonomy keywords
            extracted_pairs = IntelEngine.extract_taxonomy_keywords(it["title"])
            if extracted_pairs:
                self.db.increment_extracted_keywords(extracted_pairs)

            data = {
                "headline": it["title"], "source": it.get("source", provider_key),
                "url": it["link"],
                "published_time": it.get("pubDate") or it.get("published_at_raw") or pub_formatted,
                "published_at_raw": it.get("published_at_raw") or it.get("pubDate") or pub_formatted,
                "published_at": it.get("published_at"),
                "date_source": it.get("date_source"),
                "date_confidence": it.get("date_confidence"),
                "search_query": query,
                "company": final_company,
                "industry": intel["industry"], "event_category": intel["event_category"],
                "sentiment": intel["sentiment"], "relevance_score": intel["relevance_score"],
                "providers": [provider_key],
                "credit_risk": intel.get("credit_risk", "NEUTRAL"),
                "key_risk_signal": intel.get("key_risk_signal", "Routine News"),
                "baseline_vader_score": intel.get("baseline_vader_score", 0.0),
                "credit_risk_matrix": intel.get("credit_risk_matrix", {}),
            }
            saved, reason = self.db.save_headline(data)
            if saved:
                saved_count += 1
                if hasattr(self, "stats"):
                    self.stats["saved"] += 1
                    self.stats["companies"][final_company]["saved"] += 1
                self.q.put({"type": "NEW", "data": data})

                # Deterministic Earnings Announcement & Metrics Parser
                try:
                    from intelligence import DeterministicEarningsParser
                    earn_event = DeterministicEarningsParser.detect_future_earnings(it["title"], company=final_company, url=it["link"])
                    if earn_event:
                        self.db.save_earnings_calendar(earn_event)
                    earn_metrics = DeterministicEarningsParser.extract_quarterly_metrics(it["title"], company=final_company, url=it["link"])
                    if earn_metrics:
                        self.db.save_earnings_results(earn_metrics)
                except Exception:
                    pass
            elif reason == "deduped":
                if hasattr(self, "stats"):
                    self.stats["deduped"] += 1
                    self.stats["companies"][final_company]["deduped"] += 1

        return saved_count

    def parse_single_feed(self, url: str, user_agents: List[str]) -> List[Dict[str, str]]:
        """Public method for query testing UI (Google News RSS)."""
        raw = fetch_with_retry(url, user_agents, "TestQuery", max_retries=2)
        if not raw:
            return []
        return parse_rss_items(raw)

    def parse_single_feed_provider(self, provider: str, query: str,
                                    user_agents: List[str]) -> List[Dict[str, str]]:
        """Route a test query to the correct provider for the Query Inspector tester."""
        if provider == "google_finance":
            # query = ticker symbol for Google Finance RSS
            ticker = query.strip()
            items = self.google_finance.fetch("", [], ticker, user_agents)
            return items[:15]
        elif provider == "bing":
            raw = fetch_with_retry(
                f"https://www.bing.com/news/search?q={urllib.parse.quote(query)}&format=rss",
                user_agents, "BingTest", max_retries=2)
            return parse_rss_items(raw)[:15] if raw else []
        elif provider == "ddg":
            return self.duckduckgo.fetch(query, user_agents)[:15]
        else:  # default: google news
            raw = fetch_with_retry(
                f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en",
                user_agents, "GoogleTest", max_retries=2)
            return parse_rss_items(raw)[:15] if raw else []
