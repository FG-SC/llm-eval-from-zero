"""One visual style for every chart in the notebook."""
import textwrap

import plotly.graph_objects as go

FONT = '"Segoe UI", Arial, sans-serif'  # Arial has the same width if Segoe UI is missing (measured)
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILLION, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#6B7280"
INK, MUTED, GRID, PAPER = "#111827", "#4B5563", "#E5E7EB", "#FFFFFF"

# font sizes and margins in px
BASE, TITLE, SUBTITLE, FOOTER = 14, 20, 14, 12
HEIGHT, WRAP = 580, 80
TOP, BOTTOM, SIDE, FOOT_SHIFT = 130, 110, 40, 78


def wrap(text: str, width: int) -> str:
    """Plotly titles and annotations do not wrap by themselves (never break inside hyphenated words).
    Lines are balanced: the narrowest width that needs no more lines, so no word is left alone."""
    if not text:
        return ""
    lines = textwrap.wrap(text, width, break_on_hyphens=False)
    for narrower in range(max(len(word) for word in text.split()), width):
        candidate = textwrap.wrap(text, narrower, break_on_hyphens=False)
        if len(candidate) == len(lines):
            lines = candidate
            break
    return "<br>".join(lines)


def footer(n=None, model=None, date=None) -> str:
    """Where a chart's numbers come from: exam size, model and the day of the run."""
    parts = [f"n={n}" if n else None, model.split("/")[-1] if model else None, date]
    return " · ".join(p for p in parts if p)


def apply(fig: go.Figure, title: str, subtitle: str = "", foot: str = "", height: int = HEIGHT) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=BASE, color=INK),
        # title and subtitle are centred in a tall top margin, so 3-line titles neither clip nor touch the plot
        title=dict(text=wrap(title, WRAP), x=0.02, xanchor="left", font=dict(size=TITLE),
                   subtitle=dict(text=wrap(subtitle, int(WRAP * 1.3)), font=dict(size=SUBTITLE, color=MUTED))),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, height=height,
        margin=dict(l=SIDE, r=SIDE, t=TOP, b=BOTTOM),
        showlegend=fig.layout.showlegend if fig.layout.showlegend is not None else False,
        hoverlabel=dict(font=dict(family=FONT, size=BASE)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(size=BASE))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font=dict(size=BASE))
    if foot:
        fig.add_annotation(text=foot, xref="paper", yref="paper", x=0.5, y=0, yshift=-FOOT_SHIFT,
                           xanchor="center", yanchor="top", showarrow=False, font=dict(size=FOOTER, color=MUTED))
    return fig


def pending(fig: go.Figure) -> go.Figure:
    """Diagonal watermark for charts whose run has not finished: never show made-up numbers."""
    fig.add_annotation(text="<b>PENDING</b>", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       textangle=-30, font=dict(size=90, color="rgba(213,94,0,0.25)"))
    return fig
