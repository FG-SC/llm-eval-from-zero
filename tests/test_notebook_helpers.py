"""Model answers render as math, plain dollars stay money, and the '####' answer line stays visible."""
import re

import pandas as pd

from src.analyze import _tail
from src.notebook import styled, to_markdown
from src.style import wrap
from src.words import WORDS, new_words


def test_inline_and_display_math_are_converted():
    text = "So each pair costs \\(\\$5 + \\$3 = \\$8\\).\n\\[\n8 \\times 8 = 64\n\\]\n#### 64"
    md = to_markdown(text)
    assert "$\\$5 + \\$3 = \\$8$" in md
    assert "$$8 \\times 8 = 64$$" in md
    assert md.rstrip().endswith("#### 64")


def test_plain_dollars_are_escaped():
    assert to_markdown("It costs $5, then $2 more.") == "It costs \\$5, then \\$2 more."


def test_answer_line_is_not_a_heading():
    md = to_markdown("Total: 8 x 8 = 64\n#### 64\n  # note")
    assert "\n\\#### 64" in md and "\n  \\# note" in md
    assert not re.search(r"(?m)^\s*#", md)


def test_dollars_the_model_already_escaped_stay_money():
    # '\$5' from the model must not become '\\$5' (an escaped backslash followed by a math '$')
    assert to_markdown("first \\$5, then \\$3") == "first \\$5, then \\$3"


def test_line_breaks_are_kept():
    assert to_markdown("Total: 8\n#### 8") == "Total: 8  \n\\#### 8"


def test_html_like_text_is_shown():
    md = to_markdown("write #### <number>")
    assert "&lt;number>" in md and "<" not in md  # an escaped '<' is enough: no tag can start
    assert "#### &lt;number&gt;" in styled(pd.DataFrame({"prompt": ["#### <number>"]})).to_html()


def test_tail_never_cuts_a_formula():
    answer = "Each pair costs 8.\n\\[\n8 \\times 8 = 64\n\\]\n#### 64"
    assert _tail(answer, 3) == "\\[\n8 \\times 8 = 64\n\\]\n#### 64"
    assert _tail("a\nb\nc\nd", 3) == "b\nc\nd"


def test_titles_wrap_into_balanced_lines():
    title = "The model got 91 of 100 right. Any true score from 83.8% to 95.2% fits that result"
    lines = wrap(title, 80).split("<br>")
    assert len(lines) == 2 and min(len(l) for l in lines) > 30
    assert wrap("short title", 80) == "short title"


def test_empty_answer():
    assert to_markdown("") == "" and to_markdown(None) == ""


def test_new_words_never_start_math_and_stay_short():
    for section, (title, words) in WORDS.items():
        box = new_words(section)
        assert box.startswith(f"**{title} ({len(words)})**")
        assert "$" not in box.replace("\\$", "")
        assert len(words) <= 8


def test_status_mentions_the_key_only_for_live_calls(monkeypatch):
    from src.notebook import status

    monkeypatch.setenv("GROQ_API_KEY", "x")
    summary = {"runs": {"r1_c0": {}}}
    assert "needs an API key" in status(summary) and "found" not in status(summary)
    assert "found" in status(summary, live=True)
