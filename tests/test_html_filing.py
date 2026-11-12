import html_filing as hf

HTML = """<html><head><title>x</title><style>p{color:red}</style></head><body>
<p>Cover: for the fiscal year ended <b>December 31, 2024</b></p>
<hr style="page-break-after:always">
<div><p>Net income was $ 58,471 million</p><table><tr><td>EPS</td><td>19.75</td></tr></table></div>
<div style="break-before: page"><p>Risk factors</p></div>
</body></html>"""


def test_split_html_pages_on_print_breaks_and_skips_style():
    pages = hf.split_html_pages(HTML)
    assert len(pages) == 3
    assert "fiscal year ended December 31, 2024" in pages[0] and "color:red" not in pages[0]
    assert "58,471 million" in pages[1] and "EPS 19.75" in pages[1]
    assert pages[2] == "Risk factors"


def test_split_without_markers_is_single_page_and_deterministic():
    h = "<p>one</p><p>two</p>"
    assert hf.split_html_pages(h) == ["one\ntwo"] == hf.split_html_pages(h)
    assert hf.split_html_pages("") == []


def test_parse_date():
    assert hf.parse_date("December 31, 2024") == "2024-12-31"
    assert hf.parse_date("Smarch 3, 2024") is None and hf.parse_date("") is None


def test_metadata_prefers_ixbrl_tags_over_filing_date():
    h = ('<ix:nonNumeric name="dei:DocumentType" contextRef="c">10-K</ix:nonNumeric>'
         '<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="c">December 31, 2024</ix:nonNumeric>'
         '<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c"><span>2024</span></ix:nonNumeric>'
         "<p>Filed February 2025</p>")
    m = hf.read_filing_metadata(h)
    assert m == {"form_type": "10-K", "period_of_report": "2024-12-31", "fiscal_year": 2024,
                 "fiscal_year_source": "ixbrl_dei"}


def test_metadata_fallback_to_cover_sentence_and_unresolved_is_none():
    m = hf.read_filing_metadata("<p>FOR THE FISCAL YEAR ENDED December 31, 2023</p>")
    assert m["fiscal_year"] == 2023 and m["period_of_report"] == "2023-12-31"
    assert m["fiscal_year_source"] == "cover_text_period_end"
    n = hf.read_filing_metadata("<p>Annual report, filed February 2025</p>")
    assert n["fiscal_year"] is None and n["period_of_report"] is None   # never guessed


def test_period_end_without_focus_tag_uses_period_year():
    h = '<ix:nonNumeric name="dei:DocumentPeriodEndDate">June 30, 2024</ix:nonNumeric>'
    m = hf.read_filing_metadata(h)
    assert m["fiscal_year"] == 2024 and m["fiscal_year_source"] == "ixbrl_period_end"
