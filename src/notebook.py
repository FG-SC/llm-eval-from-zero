"""Notebook helpers: a one-line setup and readable displays (full table text, rendered math)."""
import os
import re
import subprocess
import sys
from pathlib import Path

_MATH = re.compile(r"(\\\(.*?\\\)|\\\[.*?\\\])", re.DOTALL)
_LINE_HASH = re.compile(r"(?m)^([ \t]*)#")


NEEDED = {"pandas": "pandas", "plotly": "plotly", "scipy": "scipy", "statsmodels": "statsmodels", "yaml": "PyYAML",
          "dotenv": "python-dotenv", "lm_eval": "lm_eval", "IPython": "ipython"}


def check_environment() -> None:
    """Stop with a clear message if this notebook runs on a Python without the project's packages
    (the usual cause: the notebook kernel is not the project's virtual environment)."""
    import importlib.util

    missing = [pkg for module, pkg in NEEDED.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            f"This notebook is running on {sys.executable}, which does not have: {', '.join(missing)}.\n"
            f"Fix: pick the project's .venv as the notebook kernel (VS Code: 'Select Kernel' at the top right), "
            f"or install the packages for this Python:\n    \"{sys.executable}\" -m pip install -r requirements.txt")


def setup() -> dict:
    """Point the kernel at the project, set display options and load the saved results."""
    root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import pandas as pd
    import plotly.io as pio

    # interactive figures in Jupyter, plus a static image so GitHub and nbviewer can show them too
    # (the static image needs Chrome; without it, e.g. on Colab, keep the interactive figure only)
    pio.renderers.default = "plotly_mimetype"
    try:
        import plotly.graph_objects as go

        pio.to_image(go.Figure(), format="png", width=10, height=10)
        pio.renderers.default = "plotly_mimetype+png"
        pio.renderers["png"].width, pio.renderers["png"].height = 900, None  # height: each chart's own
    except Exception:  # noqa: BLE001 - any kaleido/Chrome failure just means no static copy
        pass
    pd.set_option("display.max_colwidth", None)  # never cut a question or an answer in half
    pd.set_option("display.width", 200)
    pd.set_option("styler.html.mathjax", False)  # "$5 ... $3" in a question is money, not math
    pd.set_option("display.html.use_mathjax", False)

    from src import analyze
    from src.settings import prepare_environment

    prepare_environment()
    return analyze.load()


def status(summary: dict, live: bool = False) -> str:
    """The setup cell's report, in words. It mentions the API key only when a live call is switched on."""
    from src.analyze import RUNS

    done = [f"Day {RUNS[r]['day']} ({RUNS[r]['label'].lower()})" for r in summary["runs"]]
    finished = f"Finished experiments: {'; '.join(done)}." if done else "No experiment has finished yet."
    if live:
        found = "found" if os.environ.get("GROQ_API_KEY") else "NOT found: add it to .env"
        calls = f"Live calls are on, so they need a Groq key ({found})."
    else:
        calls = "Live calls are off, so nothing here needs an API key."
    return f"Loaded the saved results. {finished}  \n{calls}"


def rerun(name: str) -> dict:
    """Re-run one evaluation through the harness (cached answers are free), then rebuild the summary."""
    from src import analyze

    subprocess.run([sys.executable, "-m", "src.run", "--run", name], check=True)
    summary = analyze.build()
    analyze.write(summary)
    return summary


def styled(df):
    """A DataFrame as a left-aligned table whose long text wraps instead of being cut. Cell text is
    escaped: '#### <number>' is text, not an HTML tag."""
    return (df.style.format(escape="html").hide(axis="index")
            .set_properties(**{"text-align": "left", "white-space": "pre-wrap", "vertical-align": "top"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))


def show_table(df) -> None:
    from IPython.display import display

    display(styled(df))


def to_markdown(text: str) -> str:
    """Model answers use LaTeX (\\( \\) and \\[ \\]). Outside math, '$' is money (the model may already
    write it as '\\$'), '<' is not HTML, a line starting with '#' is the answer line '#### 64' (not a
    heading), and every line break is kept."""
    parts = []
    for piece in _MATH.split(text or ""):
        if piece.startswith("\\("):
            parts.append("$" + piece[2:-2].strip() + "$")
        elif piece.startswith("\\["):
            parts.append("\n$$" + piece[2:-2].strip() + "$$\n")
        else:
            piece = piece.replace("\\$", "$").replace("$", "\\$").replace("<", "&lt;")
            parts.append(_LINE_HASH.sub(r"\1\\#", piece).replace("\n", "  \n"))
    return "".join(parts)


def show_answer(text: str) -> None:
    from IPython.display import Markdown, display

    display(Markdown(to_markdown(text)))


def say(markdown: str) -> None:
    """Display a Markdown block built from the results (so prose can quote numbers without typing them)."""
    from IPython.display import Markdown, display

    display(Markdown(markdown))
