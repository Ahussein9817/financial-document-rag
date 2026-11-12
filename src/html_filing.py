"""Split an EDGAR 10-K HTML/iXBRL document into pages and read its own reporting metadata.

EDGAR serves 10-Ks as HTML, which has no fixed pagination. We approximate "pages" with the
filing's own print page-break markers (CSS `page-break-before/after: always`,
`break-before/after: page`). These pages are NOT the pagination of any PDF rendering and are
not comparable to the page numbers of the historical PDF corpus. If a document has no
page-break markers the whole text is one page (flagged by the caller through `n_pages == 1`).

Standard library only; offline.
"""

import html as html_lib
import re
from html.parser import HTMLParser

PAGE_BREAK = re.compile(
    r"(?:page-break-(?:before|after)|break-(?:before|after))\s*:\s*(?:always|page)", re.IGNORECASE
)
_SKIP = {"script", "style", "head", "title"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        if tag in ("p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        if tag in ("td", "th"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(fragment: str) -> str:
    p = _Text()
    p.feed(fragment)
    text = "".join(p.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def split_html_pages(html: str) -> list[str]:
    """Ordered non-empty page texts. A page break marker inside a tag ends the current page
    at the end of that tag (so `<hr style="page-break-after:always">` closes the page it follows)."""
    pages, last = [], 0
    for m in PAGE_BREAK.finditer(html):
        end = html.find(">", m.end())
        end = len(html) if end == -1 else end + 1
        pages.append(html[last:end])
        last = end
    pages.append(html[last:])
    texts = [html_to_text(p) for p in pages]
    return [t for t in texts if t]


_NONNUM = re.compile(r"<(/?)ix:nonNumeric\b[^>]*>", re.IGNORECASE)


def _dei(html: str, name: str) -> str | None:
    """Text content of the first <ix:nonNumeric name="dei:<name>"> element, tags stripped.

    Real filings nest other nonNumeric elements inside (e.g. the fiscal-year-end tag inside
    the period-end tag), so the element's end is found by depth counting.
    """
    m = re.search(rf'<ix:nonNumeric\b[^>]*name="dei:{name}"[^>]*>', html)
    if not m:
        return None
    depth = 1
    for t in _NONNUM.finditer(html, m.end()):
        depth += -1 if t.group(1) else 1
        if depth == 0:
            raw = html_lib.unescape(re.sub(r"<[^>]+>", "", html[m.end():t.start()]))
            return re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
    return None


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def parse_date(text: str) -> str | None:
    """'December 31, 2024' -> '2024-12-31'."""
    m = re.search(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", text or "")
    if not m:
        return None
    name = m.group(1).lower()
    month = _MONTHS.get(name) or next((v for k, v in _MONTHS.items() if len(name) == 3 and k.startswith(name)), None)
    if not month:
        return None
    return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"


def read_filing_metadata(html: str) -> dict:
    """Reporting facts from the filing itself.

    Prefers the iXBRL cover tags (dei:DocumentType, DocumentPeriodEndDate,
    DocumentFiscalYearFocus). Falls back to the cover sentence 'for the fiscal year ended
    <date>'. `fiscal_year` is None (never guessed) if neither is found.
    """
    form_text = _dei(html, "DocumentType")
    form = form_text.split()[0] if form_text else None
    period = parse_date(_dei(html, "DocumentPeriodEndDate") or "")
    fy_text = _dei(html, "DocumentFiscalYearFocus")
    fy_m = re.search(r"\b(\d{4})\b", fy_text or "")
    fy_focus = fy_m.group(1) if fy_m else None
    source = None
    fiscal_year = int(fy_focus) if fy_focus and fy_focus.isdigit() else None
    if fiscal_year is not None:
        source = "ixbrl_dei"
    text = html_to_text(html[:600_000])
    if period is None:
        m = re.search(r"fiscal\s+year\s+ended\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", text, re.IGNORECASE)
        period = parse_date(m.group(1)) if m else None
        if period and fiscal_year is None:
            fiscal_year, source = int(period[:4]), "cover_text_period_end"
    elif fiscal_year is None:
        fiscal_year, source = int(period[:4]), "ixbrl_period_end"
    return {"form_type": form, "period_of_report": period, "fiscal_year": fiscal_year,
            "fiscal_year_source": source}


def split_blocks(text: str, block_chars: int = 4000) -> list[str]:
    """Deterministic fallback locator: consecutive blocks of whole lines totalling about
    `block_chars` characters (a single longer line becomes its own block)."""
    blocks, cur, size = [], [], 0
    for line in text.split("\n"):
        if cur and size + len(line) > block_chars:
            blocks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def locate(html: str, min_pages: int = 20, block_chars: int = 4000) -> tuple[list[str], str]:
    """Locator units for a filing and their kind.

    ('page_break', pages) when the filing has at least `min_pages` print page breaks;
    otherwise ('text_block', blocks): fixed-size text blocks. A 10-K is hundreds of pages,
    so fewer than `min_pages` natural pages means the markers are missing or unusable.
    The unit number is used as the `pN` in passage ids either way; `locator_kind` records
    which it is, because a block number is not a printed page number.
    """
    pages = split_html_pages(html)
    if len(pages) >= min_pages:
        return pages, "page_break"
    return split_blocks("\n".join(pages), block_chars), "text_block"
