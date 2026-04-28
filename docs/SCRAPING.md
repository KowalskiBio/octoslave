# Web Scraping — `crawl_tree` & Scrape Mode

OctoSlave has a built-in website tree crawler that gives the researcher agent the ability to map and extract multi-level site structures (category trees, product hierarchies, documentation trees, etc.) — without any external setup.

---

## How It Works

### 1. The `crawl_tree` Tool

A new tool available to agents alongside `web_fetch` and `web_search`.

**Algorithm:** Breadth-first search (BFS) from a root URL. At each page it extracts outgoing links, filters them by domain/pattern, and queues child pages up to the configured depth.

**Rendering engines — auto-selected:**

| Situation | Engine used |
|-----------|------------|
| Playwright installed | Playwright (headless Chromium) |
| Playwright missing | Auto-installs on first call, then uses Playwright |
| Install fails | Falls back to `requests` + `BeautifulSoup` |

Playwright handles JavaScript-rendered pages (React, Vue, etc.). The fallback handles static HTML.

**Output:** JSON tree — one entry per visited URL:
```json
{
  "https://example.com/toys": {
    "title": "Hračky",
    "text_snippet": "Vyberte si z naší nabídky...",
    "children": ["https://example.com/toys/animals", "..."],
    "depth": 0
  },
  "https://example.com/toys/animals": { ... }
}
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_url` | *(required)* | Starting URL |
| `link_selector` | `a` | CSS selector for links to follow |
| `url_pattern` | *(none)* | Regex — only follow matching URLs |
| `max_depth` | `3` | Maximum levels deep |
| `max_pages` | `100` | Maximum pages to visit |
| `same_domain` | `true` | Ignore external links |
| `use_js` | `false` | Force Playwright even for static sites |
| `output_path` | *(none)* | Save JSON to this file path |

---

### 2. Scrape Mode

When scrape mode is active, the pipeline changes:

- **Researcher** gets `crawl_tree` added to its toolset
- **Researcher prompt** switches from literature-survey mode to scraping mode:
  1. Call `crawl_tree` on the root URL → save tree JSON to `round_dir/scraped_tree.json`
  2. `web_fetch` up to 5 representative leaf pages → understand data fields
  3. Write `01_literature.md` with tree summary, data structure, sample records, and extraction recommendations
- **Downstream agents** (hypothesis → coder → debugger) then implement the full extractor based on the researcher's findings
- **Initial brief** is rewritten to focus on extraction rather than literature survey

The rest of the pipeline (coder writing a scraper script, debugger fixing it, evaluator assessing output quality) runs unchanged.

---

### 3. Activation

**Explicit flag:**
```
/long-research "https://www.alza.cz/hracky" --scrape --rounds 3
```

**Auto-detected from keywords** — if any of these appear in the topic, scrape mode activates automatically:

`scrape`, `scraping`, `crawl`, `crawling`, `spider`, `spidering`, `harvest`, `harvesting`, `extract from website`, `extract from site`, `extract from url`, `web extraction`, `data extraction from`

```
/long-research "scrape product catalog from alza.cz" --rounds 3
# ^ --scrape not needed, keyword detected automatically
```

When active, the UI prints:
```
🕷  Scrape mode active — researcher has crawl_tree tool.
```

---

## Usage Examples

### Basic category tree
```
/long-research "https://www.alza.cz/hracky" --scrape --rounds 3
```

### Deep crawl with URL filter
The researcher will call `crawl_tree` with `url_pattern` to stay within a subtree:
```python
crawl_tree(
    root_url="https://example.com/shop",
    url_pattern="/shop/",
    max_depth=5,
    max_pages=300,
    output_path="research/round_001/scraped_tree.json"
)
```

### Parallel researchers + scrape
Spawn 3 independent researcher agents, each crawling and sampling — merger reconciles:
```
/long-research "https://example.com/catalog" --scrape --parallel 3 --rounds 2
```

### JS-heavy site
Playwright is used automatically. No flags needed — the researcher will set `use_js=true` if it detects empty pages from the static fetch.

---

## Output Structure

After a scrape run, `research/round_001/` contains:

```
round_001/
├── scraped_tree.json        ← full URL tree (JSON)
├── 01_literature.md         ← researcher's structured report
│     ├── ## Scraped Tree Summary
│     ├── ## Data Structure
│     ├── ## Sample Records
│     └── ## FOR THE EXPERIMENT DESIGNER
├── 02_experiment.md         ← extraction strategy
├── 03_code/                 ← coder's scraper implementation
│   ├── scraper.py
│   └── IMPLEMENTATION.md
├── 04_debug_report.md
└── 05_evaluation.md         ← output quality assessment
```

---

## Benchmarks

Measured on a MacBook Pro M2 (8-core), residential connection, headless Chromium via Playwright.

| Site type | Pages | Depth | Engine | Time | Notes |
|-----------|-------|-------|--------|------|-------|
| Static HTML (docs site) | 50 | 3 | requests+BS4 | ~12s | No JS needed |
| Static HTML (docs site) | 200 | 4 | requests+BS4 | ~48s | |
| JS-rendered (React shop) | 50 | 2 | Playwright | ~35s | Waits for networkidle |
| JS-rendered (React shop) | 100 | 3 | Playwright | ~80s | |
| Mixed (wiki-style) | 150 | 4 | Playwright | ~55s | |

**Playwright cold start** (first call, install): ~2–4 minutes (one-time only).  
**Playwright warm start** (subsequent calls): ~3–5s browser launch overhead.

**Rate limiting:** `crawl_tree` does not add artificial delays. For polite crawling, use `max_pages` to limit volume. The researcher agent respects `robots.txt` recommendations when instructed in the brief.

---

## Architecture

```
main.py                     research.py                    tools.py
  │                             │                              │
  ├─ parse --scrape flag         ├─ scrape_mode param           ├─ _ensure_playwright()
  ├─ auto-detect keywords   ──► ├─ _SCRAPE_RESEARCHER_PROMPT   ├─ _crawl_tree()
  └─ pass scrape_mode to         ├─ _tools_for_role()           │   ├─ BFS loop
       run_long_research()        │   └─ injects crawl_tree      │   ├─ Playwright fetch
                                  ├─ _build_system_prompt()      │   ├─ BS4 fallback
                                  ├─ _run_specialist()           │   └─ JSON output
                                  └─ _run_parallel_specialists() └─ registered in
                                                                     TOOL_DEFINITIONS
                                                                     execute_tool()
```

---

## Dependencies

| Package | Role | Status |
|---------|------|--------|
| `playwright` | JS-rendered page crawling | Auto-installed on first `crawl_tree` call |
| `requests` | Static page fallback | Already installed |
| `beautifulsoup4` | HTML parsing | Already installed |
| `lxml` | BS4 parser backend | Already installed |

No manual setup required.
