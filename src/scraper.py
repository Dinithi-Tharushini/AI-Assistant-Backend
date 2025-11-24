import re
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


BLOCKED_PATTERNS = [
    r"/contact",
    r"/contacts",
    r"/login",
    r"/signup",
    r"/register",
    r"/logout",
    r"/cart",
    r"/checkout",
    r"/feedback",
    r"[?&]feedback=",
    r"/support",
    r"/help",
    r"mailto:",
    r"tel:",
]

# Default text snippets we want to exclude from scraped page content
CONTENT_BLOCK_DEFAULTS = [
    r"\bfeedback\b",
    r"how\s+can\s+we\s+help\s+you\??",
    r"help\s+us\s+improve",
    r"chat\s+with\s+us",
    r"subscribe\s+to\s+our\s+newsletter",
    r"accept\s+cookies",
    r"cookie\s+preferences",
    r"feedbackform",
    r"^solutions$",
    r"^learn\s*more$",
    r"^countries$",
    r"^careers$",
    r"^news(room)?$",
    r"^contact$",
    r"a\s+world\s+of\s+possibilities",
]


@dataclass
class ScrapeConfig:
    max_pages: int = 50
    max_depth: int = 3
    request_timeout: int = 15
    delay_seconds: float = 0.5
    exclude_url_patterns: list[str] = field(default_factory=list)
    block_text_patterns: list[str] = field(default_factory=list)


class WebScraper:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=200,
            length_function=len,
        )

    def _is_same_domain(self, base: str, url: str) -> bool:
        try:
            return urlparse(base).netloc == urlparse(url).netloc
        except Exception:
            return False

    def _is_blocked(self, url: str, cfg: ScrapeConfig) -> bool:
        patterns = BLOCKED_PATTERNS + (cfg.exclude_url_patterns or [])
        return any(re.search(p, url, re.IGNORECASE) for p in patterns)

    def _clean_html(self, html: str) -> str:
        """Return HTML string after removing non-content elements (scripts, footers, widgets)."""
        soup = BeautifulSoup(html, 'lxml')
        # Remove scripts/styles and common chrome
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        # Remove footer/header/navigation explicitly
        for tag in soup.find_all('footer'):
            tag.decompose()
        for tag in soup.find_all('header'):
            tag.decompose()
        for tag in soup.find_all('nav'):
            tag.decompose()
        # Remove elements that look like footers by id/class/role
        try:
            for tag in soup.find_all(True, id=re.compile(r'footer', re.I)):
                tag.decompose()
            for tag in soup.find_all(True, class_=re.compile(r'footer', re.I)):
                tag.decompose()
            for tag in soup.find_all(True, attrs={'role': re.compile(r'contentinfo', re.I)}):
                tag.decompose()
            # Remove common header/nav/menu containers
            for tag in soup.find_all(True, attrs={'role': re.compile(r'navigation', re.I)}):
                tag.decompose()
            for tag in soup.find_all(True, id=re.compile(r'(header|nav|menu|top-bar|breadcrumb|breadcrumbs|hamb)', re.I)):
                tag.decompose()
            for tag in soup.find_all(True, class_=re.compile(r'(header|nav|menu|top-bar|breadcrumb|breadcrumbs|hamb)', re.I)):
                tag.decompose()
        except Exception:
            pass
        # Remove specific feedback/help widgets (e.g., Advantis floating buttons)
        try:
            # Bootstrap modal trigger buttons
            for tag in soup.find_all(attrs={"data-bs-toggle": re.compile(r"modal", re.I)}):
                target = (tag.get("data-bs-target") or tag.get("data-target") or "").lower()
                if "feedback" in target or "help" in target:
                    tag.decompose()
            # Images used as text for feedback/help widgets
            for img in soup.find_all("img", src=True):
                src = img.get("src", "").lower()
                if "feedback" in src or "help-text" in src:
                    img.decompose()
            # Containers with ids/classes mentioning feedback/help
            for tag in soup.find_all(True, id=re.compile(r"(feedback|help)", re.I)):
                tag.decompose()
            for tag in soup.find_all(True, class_=re.compile(r"(feedback|help)", re.I)):
                tag.decompose()
        except Exception:
            pass

        return str(soup)

    def _extract_text(self, html: str) -> str:
        """Extract text with newlines from HTML for further filtering."""
        soup = BeautifulSoup(html, 'lxml')
        text = soup.get_text(separator='\n')
        return text

    def _filter_text(self, text: str, cfg: ScrapeConfig) -> str:
        # Drop lines matching blocked phrases
        block_patterns = [re.compile(p, re.IGNORECASE) for p in (CONTENT_BLOCK_DEFAULTS + (cfg.block_text_patterns or []))]
        lines = [ln.strip() for ln in text.split('\n')]
        kept: list[str] = []
        for ln in lines:
            if not ln:
                continue
            if any(p.search(ln) for p in block_patterns):
                continue
            kept.append(ln)
        cleaned = ' '.join(kept)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _dedupe_lines_global(self, text: str, seen: set[str]) -> str:
        """Remove lines already seen in this crawl (site-wide boilerplate)."""
        out: list[str] = []
        for raw in text.split('. '):  # sentence-ish split
            ln = raw.strip()
            if len(ln) < 30:
                continue
            sig = re.sub(r"\W+", "", ln.lower())
            if sig in seen:
                continue
            seen.add(sig)
            out.append(ln)
        return '. '.join(out)

    def _fetch(self, url: str, timeout: int) -> str:
        resp = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'RAG-Assistant-Bot/1.0 (+https://example.com)'
        })
        resp.raise_for_status()
        return resp.text

    def crawl(self, start_url: str, config: ScrapeConfig | None = None) -> dict:
        cfg = config or ScrapeConfig()
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        texts: list[str] = []
        metas: list[dict] = []
        base_domain = urlparse(start_url).netloc

        seen_signatures: set[str] = set()

        while queue and len(visited) < cfg.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            if depth > cfg.max_depth:
                continue
            if self._is_blocked(url, cfg):
                continue
            if not self._is_same_domain(start_url, url):
                continue

            try:
                html = self._fetch(url, cfg.request_timeout)
            except Exception:
                continue

            visited.add(url)
            cleaned_html = self._clean_html(html)
            cleaned_text = self._extract_text(cleaned_html)
            cleaned_text = self._filter_text(cleaned_text, cfg)
            cleaned_text = self._dedupe_lines_global(cleaned_text, seen_signatures)
            if cleaned_text and len(cleaned_text) > 50:
                texts.append(cleaned_text)
                metas.append({ 'source_url': url, 'cleaned_html': cleaned_html })

            # discover links
            try:
                soup = BeautifulSoup(html, 'lxml')
                for a in soup.find_all('a', href=True):
                    href = a['href'].strip()
                    next_url = urljoin(url, href)
                    if next_url.startswith('http') and next_url not in visited and not self._is_blocked(next_url, cfg):
                        if self._is_same_domain(start_url, next_url):
                            queue.append((next_url, depth + 1))
            except Exception:
                pass

            time.sleep(cfg.delay_seconds)

        return { 'texts': texts, 'metadata': metas, 'visited': list(visited) }

    def scrape_and_index(self, start_url: str, config: ScrapeConfig | None = None, *, index: bool = False, include_text: bool = True) -> dict:
        """Crawl site and optionally index into the vector store.

        index=False by default so you can inspect scraped results first.
        include_text=True returns the scraped text per page for verification.
        """
        result = self.crawl(start_url, config)

        if index and result['texts']:
            # Split long pages into chunks before embedding to keep payloads small
            chunk_texts: list[str] = []
            chunk_meta: list[dict] = []
            for text, meta in zip(result['texts'], result['metadata']):
                chunks = self.text_splitter.split_text(text)
                for idx, chunk in enumerate(chunks):
                    chunk_texts.append(chunk)
                    # Keep only lightweight metadata (url + chunk index)
                    chunk_meta.append({
                        'source_url': meta.get('source_url'),
                        'chunk': idx,
                    })
            if chunk_texts:
                self.vector_store.add_texts(chunk_texts, chunk_meta)

        items = []
        if include_text:
            for text, meta in zip(result['texts'], result['metadata']):
                item = { 'url': meta.get('source_url'), 'text': text, 'length': len(text) }
                if 'cleaned_html' in meta:
                    item['html'] = meta['cleaned_html']
                items.append(item)

        return {
            'pages_scraped': len(result['texts']),
            'visited': result['visited'],
            'items': items
        }


