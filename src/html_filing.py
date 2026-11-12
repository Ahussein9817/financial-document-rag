"""Split an EDGAR 10-K HTML/iXBRL document into pages and read its own reporting metadata.

EDGAR serves 10-Ks as HTML, which has no fixed pagination. We approximate "pages" with the
filing's own print page-break markers (CSS `page-break-before/after: always`,
`break-before/after: page`). These pages are NOT the pagination of any PDF rendering and are
not comparable to the page numbers of the historical PDF corpus. If a document has no
page-break markers the whole text is one page (flagged by the caller through `n_pages == 1`).

Standard library only; offline.
"""

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


_DEI = r'name="dei:{name}"[^>]*>(?:\s*<[^>]+>)*\s*([^<]+?)\s*<'


def _dei(html: str, name: str) -> str | None:
    m = re.search(_DEI.format(name=name), html)
    return m.group(1).strip() if m else None


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def parse_date(text: str) -> str | None:
    """'December 31, 2024' -> '2024-12-31'."""
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text or "")
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"


def read_filing_metadata(html: str) -> dict:
    """Reporting facts from the filing itself.

    Prefers the iXBRL cover tags (dei:DocumentType, DocumentPeriodEndDate,
    DocumentFiscalYearFocus). Falls back to the cover sentence 'for the fiscal year ended
    <date>'. `fiscal_year` is None (never guessed) if neither is found.
    """
    form = _dei(html, "DocumentType")
    period = parse_date(_dei(html, "DocumentPeriodEndDate") or "")
    fy_focus = _dei(html, "DocumentFiscalYearFocus")
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
