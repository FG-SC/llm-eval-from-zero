"""Every chart of the notebook. Every number is read from results/summary.json; titles are built from it."""
import math

import plotly.graph_objects as go

from src import analyze, stats
from src.style import (BASE, BLUE, BOTTOM, GREY, HEIGHT, INK, MUTED, ORANGE, TITLE, TOP, VERMILLION, WRAP, apply,
                       footer, pending, wrap)

LIGHT = "#D1D5DB"


def _label_rows(fig: go.Figure, labels: list[str], slot_share: float = 0.32, height: int = HEIGHT) -> None:
    """Category labels written above each row (left-aligned), instead of tick labels that push the plot right."""
    shift = (height - TOP - BOTTOM) / len(labels) * slot_share
    for i, label in enumerate(labels):
        fig.add_annotation(xref="paper", yref="y", x=0, y=i, yshift=shift, xanchor="left", yanchor="bottom",
                           text=label, showarrow=False, font=dict(size=BASE, color=INK))
    fig.update_yaxes(showticklabels=False, range=[len(labels) - 0.4, -0.75])


def _legend_below(fig: go.Figure, height: int = HEIGHT) -> None:
    """A one-line legend just under the x-axis tick labels, above the footer."""
    y = -34 / (height - TOP - BOTTOM)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=y, xanchor="left", yanchor="top",
                                                   font=dict(size=BASE), traceorder="normal"))


def _pending_figure(title: str, run: str) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return pending(apply(fig, title, f"Run {run} has not finished"))


def _run_label(run: str) -> str:
    meta = analyze.RUNS[run]
    return f"{meta['label']} · {meta['prompt']}"


# ---------------------------------------------------------------- Day 2
def fig_score(summary: dict) -> go.Figure:
    """Day 2: the score is an estimate."""
    run = summary["runs"].get("r1_c0")
    if not run:
        return _pending_figure("Accuracy with its margin of error", "r1_c0")
    s, n = run["scores"]["strict-match"], run["n"]
    lo, hi = s["ci95"]
    fig = go.Figure(go.Scatter(
        x=[s["accuracy"]], y=[0], mode="markers", marker=dict(size=BASE * 1.5, color=BLUE),
        error_x=dict(type="data", symmetric=False, array=[hi - s["accuracy"]], arrayminus=[s["accuracy"] - lo],
                     thickness=BASE * 0.35, width=BASE * 0.9, color=BLUE),
        hovertemplate=f"{s['correct']} of {n} right<br>score %{{x:.1%}}<br>95% interval {lo:.1%} to {hi:.1%}<extra></extra>",
    ))
    fig.add_annotation(x=s["accuracy"], y=0, text=f"<b>{s['accuracy']:.0%}</b>", yshift=BASE * 2.8,
                       showarrow=False, font=dict(size=TITLE * 1.4, color=BLUE))
    for value, anchor in ((lo, "right"), (hi, "left")):
        fig.add_annotation(x=value, y=0, text=f"{value:.1%}", yshift=-BASE * 2.3, xanchor=anchor,
                           showarrow=False, font=dict(size=BASE, color=MUTED))
    note = (f"True score: the share of all GSM8K-style questions this exact setup would get right. The bar is a "
            f"range for the true score, not for the next exam: a new random {n} questions can land outside it. It "
            f"covers luck only (which questions were drawn, which answer the model happened to give), not wrong "
            f"answer keys or questions the model saw in training.")
    fig.add_annotation(xref="paper", yref="paper", x=0, y=0.02, xanchor="left", yanchor="bottom", align="left",
                       text=wrap(note, int(WRAP * 1.3)), showarrow=False,
                       font=dict(size=BASE * 0.9, color=MUTED))
    fig.update_xaxes(range=[0.5, 1.0], tickformat=".0%", title_text="share of questions answered right")
    fig.update_yaxes(range=[-2.2, 1], showticklabels=False, showgrid=False, zeroline=False)
    return apply(fig, f"The model got {s['correct']} of {n} right. Any true score from {lo:.1%} to {hi:.1%} "
                      f"fits that result",
                 f"GSM8K, zero-shot, reasoning effort {run['card']['reasoning_effort']}, official grading rule · "
                 f"dot = score, bar = 95% range (confidence interval)",
                 footer(n, summary["model"], run["date"]))


def fig_margin(summary: dict) -> go.Figure:
    """Day 2: how wide the 95% range is for a given exam size (formula, not measured)."""
    n_all = summary["sample"]["dataset"]["n_rows"]
    ns = sorted({int(round(10 * (n_all / 10) ** (i / 88))) for i in range(89)})  # 10 ... all test questions, log-spaced
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[200 * stats.wilson_halfwidth(0.5, k) for k in ns], mode="lines",
                             name="a model at 50% right (for comparison: the widest possible range)", line=dict(color=GREY, width=3, dash="dot"),
                             hovertemplate="%{x} questions: %{y:.1f} points wide<extra></extra>"))
    run = summary["runs"].get("r1_c0")
    title, subtitle = "More questions, narrower range", "Computed from the Wilson formula, not measured"
    if run:
        s, n = run["scores"]["strict-match"], run["n"]
        p = s["accuracy"]
        width_n = 100 * (s["ci95"][1] - s["ci95"][0])
        width_4n = 200 * stats.wilson_halfwidth(p, 4 * n)
        width_all = 200 * stats.wilson_halfwidth(p, n_all)
        fig.add_trace(go.Scatter(x=ns, y=[200 * stats.wilson_halfwidth(p, k) for k in ns], mode="lines",
                                 name=f"a model at {p:.0%} right (ours)", line=dict(color=BLUE, width=4),
                                 hovertemplate="%{x} questions: %{y:.1f} points wide<extra></extra>"))
        fig.add_trace(go.Scatter(x=[n, n_all], y=[width_n, width_all], mode="markers", showlegend=False,
                                 marker=dict(size=BASE * 1.2, color=BLUE),
                                 customdata=["our exam", f"all {n_all:,} GSM8K test questions"],
                                 hovertemplate="%{customdata}: %{y:.1f} points wide<extra></extra>"))
        fig.add_annotation(x=math.log10(n), y=width_n, xref="x", yref="y", align="left", xanchor="left",
                           yanchor="bottom",
                           text=f"our exam: {n} questions,<br>{width_n:.1f} points wide<br>({s['ci95'][0]:.1%} to {s['ci95'][1]:.1%})",
                           ax=BASE * 3, ay=-BASE * 5.5, showarrow=True, arrowhead=2,
                           font=dict(size=BASE, color=BLUE))
        title = (f"{n} questions give a range {width_n:.1f} points wide; {4 * n} would roughly halve it "
                 f"({width_4n:.1f} points)")
        subtitle = (f"From the Wilson formula. Even all {n_all:,} test questions leave {width_all:.1f} points (right "
                    f"dot): they are themselves one sample of GSM8K-style questions")
    fig.update_xaxes(type="log", title_text="number of questions (log scale: each ×10 takes the same space)",
                     tickvals=[10, 100, 1000], ticktext=["10", "100", "1,000"])
    fig.update_yaxes(title_text="width of the 95% range (percentage points)", rangemode="tozero",
                     tickvals=[10, 20, 30, 40, 50])
    fig.update_layout(showlegend=True, legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                                                   bgcolor="rgba(255,255,255,0.9)", font=dict(size=BASE)))
    return apply(fig, title, subtitle, footer(date=run["date"] if run else None))


# ---------------------------------------------------------------- Day 3
def fig_audit(summary: dict) -> go.Figure:
    """Day 3: who was at fault, read one by one after the run, in both directions: the answers marked
    wrong, and the answers marked right that a corrected key turns into mistakes."""
    audit = summary.get("key_audit")
    if not audit:
        return _pending_figure("Why the answers were marked wrong", "r1_c0")
    counts = audit["counts"]
    plat = audit.get("platinum", {})
    agree = plat.get("agrees_with_audit", {})
    rows = [(f"Marked wrong · {analyze.VERDICT_TEXT[v].lower()}", counts[v], v == "model_error")
            for v in analyze.VERDICT_ORDER]
    if plat:
        rows.append(("Marked right · likely the model's mistake, hidden by a wrong key", len(plat["hidden_mistakes"]),
                     True))
    labels = [label for label, _, _ in rows]
    fig = go.Figure(go.Bar(
        x=[k for _, k, _ in rows], y=list(range(len(rows))), orientation="h", width=0.45,
        marker_color=[VERMILLION if mistake else BLUE for _, _, mistake in rows],
        text=[k for _, k, _ in rows], textposition="outside", textfont=dict(size=BASE * 1.2),
        customdata=labels, hovertemplate="%{customdata}: %{x}<extra></extra>"))
    _label_rows(fig, labels)
    fig.update_xaxes(range=[0, max(k for _, k, _ in rows) * 1.2], title_text="answers", dtick=1)
    k = counts["model_error"]
    sub = "Orange: the model's mistake; blue: not the model's fault."
    if plat:
        sub += " The last bar counts answers marked right."
    sub += f" Read after the run by {audit['n_readers']} AI readers, not people"
    if agree:
        sub += f"; GSM8K-Platinum agrees on all {agree['agree']} it can check"
    run = summary["runs"]["r1_c0"]
    return apply(fig, f"The grader marked {audit['graded_wrong']} answers wrong; {k} {'was' if k == 1 else 'were'} "
                      f"clearly the model's mistake", sub, footer(run["n"], summary["model"], run["date"]))


def fig_ladder(summary: dict) -> go.Figure:
    """Day 3: the same answers graded five ways; each row changes one part of the exam."""
    run = summary["runs"].get("r1_c0")
    plat = (summary.get("key_audit") or {}).get("platinum")
    if not run or not plat:
        return _pending_figure("Same answers, different grading", "r1_c0")
    rows = [("Official: strict rule, GSM8K keys", run["scores"]["strict-match"]),
            ("Grading rule: compare numbers", run["scores"]["number-match"]),
            (f"Answer keys: {len(plat['key_changed'])} corrected", plat["steps"]["keys_only"]),
            (f"Questions: {len(plat['removed'])} removed", plat["steps"]["questions_only"]),
            ("Both: GSM8K-Platinum", plat["score"])]
    labels = [f"{label} · <b>{s['correct']}/{s['n']} = {s['accuracy']:.1%}</b>" for label, s in rows]
    fig = go.Figure(go.Scatter(
        x=[s["accuracy"] for _, s in rows], y=list(range(len(rows))), mode="markers",
        marker=dict(size=BASE * 1.3, color=[INK] + [BLUE] * (len(rows) - 1)),
        customdata=[label for label, _ in rows],
        hovertemplate="%{customdata}: %{x:.1%}<extra></extra>"))
    _label_rows(fig, labels, slot_share=0.25)
    fig.update_xaxes(range=[0.85, 1.0], tickformat=".0%", title_text="score of the same answers")
    low, high = min(s["accuracy"] for _, s in rows), max(s["accuracy"] for _, s in rows)
    sub = (f"Dropping {len(plat['removed'])} badly written questions ({len(plat['removed_wrong'])} were misses) raises "
           f"the score; fixing {len(plat['key_changed'])} keys lowers it. The answers never change, so these gaps are "
           f"not luck")
    return apply(fig, f"Same answers, graded five ways: from {low:.1%} to {high:.1%}", sub,
                 footer(run["n"], summary["model"], run["date"]))


# ---------------------------------------------------------------- Day 4
def _answer_mix(run: dict) -> tuple[int, int, int]:
    """(gave a number, wrote UNANSWERABLE, no final answer) for one run, read from the answer's last line."""
    if "rule_labels" in run:
        lab = run["rule_labels"]
        return lab["answered_number"]["correct"], lab["abstained"]["correct"], lab["no_final_answer"]["correct"]
    said = run.get("abstained") or 0
    number = run["extracted"]["strict-match"]
    return number, said, run["n"] - number - said


def fig_abstention(summary: dict) -> go.Figure:
    """Day 4: what the model wrote on its last line, with and without permission to say UNANSWERABLE."""
    runs = summary["runs"]
    if not {"r3_c2_p0", "r3_c2_p1"} <= set(runs):
        return _pending_figure("Does the model admit a fact is missing?", "r3_c2_p1")
    names = [r for r in ("r3_c2_p0", "r3_c2_p1", "r1_c0_p1") if r in runs]
    labels = [f"{analyze.RUNS[r]['label'].replace(', may say UNANSWERABLE', '')} · {runs[r]['prompt']}"
              f"{' (may say UNANSWERABLE)' if runs[r]['prompt'] == 'P1' else ''}" for r in names]
    mixes = [_answer_mix(runs[r]) for r in names]
    fig = go.Figure()
    for i, (part, color) in enumerate((("gave a number", BLUE), ("wrote UNANSWERABLE", ORANGE),
                                       ("no final answer", LIGHT))):
        values = [m[i] for m in mixes]
        fig.add_trace(go.Bar(x=values, y=list(range(len(names))), orientation="h", width=0.45, name=part,
                             marker_color=color, text=[v if v else "" for v in values], textposition="inside",
                             insidetextanchor="middle", textfont=dict(size=BASE, color="white" if color != LIGHT else INK),
                             customdata=labels, hovertemplate=f"%{{customdata}}<br>{part}: %{{x}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    _label_rows(fig, labels)
    _legend_below(fig)
    fig.update_xaxes(range=[0, max(runs[r]["n"] for r in names)], dtick=10)
    p0, p1 = mixes[0], mixes[1]
    flagged = len(summary["comparisons"].get("c2_p0_to_p1", {}).get("excluded", []))
    flag_note = f" ({flagged} flagged as possibly still answerable)" if flagged else ""
    sub = (f"Missing fact: GSM-Plus rewrites of our questions with one needed fact removed{flag_note}. Counted from "
           f"the last line of each answer")
    return apply(fig, f"With a fact missing, the model still gave a number {p0[0]} times out of {sum(p0)}; allowed to "
                      f"say UNANSWERABLE, it did so {p1[1]} times", sub,
                 footer(runs["r3_c2_p1"]["n"], summary["model"], runs["r3_c2_p1"]["date"]))


# ---------------------------------------------------------------- Day 5
CHANGES = [("c0_to_retest", "Asked again: same prompt, same questions"),
           ("c0_to_p1", "New prompt: may say UNANSWERABLE (P1)"),
           ("c0_to_c1", "New questions: one irrelevant sentence added")]


def _changes(summary: dict) -> list[tuple[str, dict]]:
    return [(label, summary["comparisons"][key]["all"]) for key, label in CHANGES if key in summary["comparisons"]]


def fig_flips(summary: dict) -> go.Figure:
    """Day 5: the same 100 questions compared one by one with the Day 2 run: which grades moved, and which way."""
    rows = _changes(summary)
    if not rows or "c0_to_c1" not in summary["comparisons"]:
        return _pending_figure("Which answers changed?", "r2_c1")
    labels = [label for label, _ in rows]
    lost, gained = [c["before_only"] for _, c in rows], [c["after_only"] for _, c in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[-k for k in lost], y=list(range(len(rows))), orientation="h", width=0.45,
                         name="questions: right → wrong", marker_color=VERMILLION, text=lost, textposition="outside",
                         customdata=labels, hovertemplate="%{customdata}<br>right → wrong: %{text}<extra></extra>"))
    fig.add_trace(go.Bar(x=gained, y=list(range(len(rows))), orientation="h", width=0.45, name="questions: wrong → right",
                         marker_color=BLUE, text=gained, textposition="outside",
                         customdata=labels, hovertemplate="%{customdata}<br>wrong → right: %{x}<extra></extra>"))
    fig.update_layout(barmode="relative")
    _label_rows(fig, labels)
    _legend_below(fig)
    top = max(lost + gained) + 2
    top += top % 2
    ticks = list(range(-top, top + 1, 2))
    fig.update_xaxes(range=[-top - 0.5, top + 0.5], tickvals=ticks, ticktext=[str(abs(v)) for v in ticks],
                     zeroline=True, zerolinecolor=INK, zerolinewidth=1.5)
    c1 = summary["comparisons"]["c0_to_c1"]["all"]
    noise = summary["comparisons"].get("c0_to_retest", {}).get("all")
    title = (f"One irrelevant sentence flipped {c1['before_only']} grades from right to wrong and {c1['after_only']} "
             f"from wrong to right")
    if noise:
        moved = noise["before_only"] + noise["after_only"]
        title += f"; asking again moved {moved} grade{'' if moved == 1 else 's'}"
    sub = ("Each row: the same 100 problems (last row: their GSM-Plus rewrites), graded with the strict rule and "
           "compared one by one with the Day 2 run (original questions, prompt P0)")
    run = summary["runs"]["r2_c1"]
    return apply(fig, title, sub, footer(run["n"], summary["model"]))


def _signed(points: float) -> str:
    return "0" if round(points) == 0 else f"{points:+.0f}"


def fig_paired_diff(summary: dict) -> go.Figure:
    """Day 5: each change's effect on the score, with its 95% range from a paired bootstrap."""
    rows = _changes(summary)
    if not rows or "c0_to_c1" not in summary["comparisons"]:
        return _pending_figure("Is the change bigger than the noise?", "r2_c1")
    labels = [label for label, _ in rows]
    diffs = [100 * c["diff"] for _, c in rows]
    los = [100 * c["diff_ci95_bootstrap"][0] for _, c in rows]
    his = [100 * c["diff_ci95_bootstrap"][1] for _, c in rows]
    ps = [c["mcnemar_p"] for _, c in rows]
    fig = go.Figure(go.Scatter(
        x=diffs, y=list(range(len(rows))), mode="markers", marker=dict(size=BASE * 1.2, color=BLUE),
        error_x=dict(type="data", symmetric=False, array=[h - d for h, d in zip(his, diffs)],
                     arrayminus=[d - lo for d, lo in zip(diffs, los)], thickness=BASE * 0.25, width=BASE * 0.6,
                     color=BLUE),
        customdata=list(zip(labels, los, his, ps)),
        hovertemplate=("%{customdata[0]}<br>change %{x:+.0f} points<br>95% range %{customdata[1]:+.0f} to "
                       "%{customdata[2]:+.0f}<br>McNemar p = %{customdata[3]:.2f}<extra></extra>")))
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.4, y1=len(rows) - 0.6, xref="x", yref="y",
                  line=dict(color=INK, width=1.5))
    _label_rows(fig, [f"{label} · <b>{_signed(d)} point{'' if round(abs(d)) == 1 else 's'} [{_signed(lo)}, {_signed(hi)}]</b>"
                      for label, d, lo, hi in zip(labels, diffs, los, his)])
    edge = max(abs(v) for v in los + his) + 2
    fig.update_xaxes(range=[-edge, edge], title_text="change in score vs the Day 2 run (percentage points)",
                     zeroline=False)
    c1 = summary["comparisons"]["c0_to_c1"]["all"]
    d, lo, hi = 100 * c1["diff"], 100 * c1["diff_ci95_bootstrap"][0], 100 * c1["diff_ci95_bootstrap"][1]
    verdict = "includes zero, so it could be noise" if lo <= 0 <= hi else "stays clear of zero"
    moved = f"lowered the score by {-d:.0f} points" if d < 0 else f"raised the score by {d:.0f} points"
    sub = ("Dot: change in the share right. Bar: 95% range (paired bootstrap). A change convinces only when its bar "
           "stays clear of zero, without even touching it")
    run = summary["runs"]["r2_c1"]
    return apply(fig, f"The extra sentence {moved}, but its 95% range ({_signed(lo)} to {_signed(hi)}) "
                      f"{verdict}", sub, footer(run["n"], summary["model"]))


# ---------------------------------------------------------------- every run
def fig_all_runs(summary: dict) -> go.Figure:
    """Every run graded right/wrong, side by side: the score and its 95% range."""
    names = [r for r in analyze.RUNS if r in summary["runs"] and "scores" in summary["runs"][r]]
    if not names:
        return _pending_figure("Every run's score", "r1_c0")
    scores = [summary["runs"][r]["scores"]["strict-match"] for r in names]
    labels = [f"{_run_label(r)} · <b>{s['correct']}/{s['n']}</b>" for r, s in zip(names, scores)]
    fig = go.Figure(go.Scatter(
        x=[s["accuracy"] for s in scores], y=list(range(len(names))), mode="markers",
        marker=dict(size=BASE * 1.2, color=BLUE),
        error_x=dict(type="data", symmetric=False, array=[s["ci95"][1] - s["accuracy"] for s in scores],
                     arrayminus=[s["accuracy"] - s["ci95"][0] for s in scores], thickness=BASE * 0.25,
                     width=BASE * 0.6, color=BLUE),
        customdata=[[label, s["ci95"][0], s["ci95"][1]] for label, s in zip(labels, scores)],
        hovertemplate="%{customdata[0]}<br>%{x:.0%} (95% range %{customdata[1]:.1%} to %{customdata[2]:.1%})"
                      "<extra></extra>"))
    _label_rows(fig, labels)
    fig.update_xaxes(range=[0.7, 1.0], tickformat=".0%", title_text="share of questions answered right (strict rule)")
    overlap = max(s["ci95"][0] for s in scores) <= min(s["ci95"][1] for s in scores)
    title = ("Every run's score with its 95% range: the ranges overlap, which settles nothing, so Day 5 compares the "
             "runs question by question" if overlap else "Every run's score with its 95% range")
    sub = ("Same 100 questions; the extra-sentence run uses GSM-Plus rewrites of them. The missing-fact runs have no "
           "right answer to score (Day 4)")
    return apply(fig, title, sub, footer(scores[0]["n"], summary["model"]))


def fig_tokens(summary: dict) -> go.Figure:
    """Every run: tokens per question, split into the prompt, the hidden reasoning and the visible answer."""
    names = [r for r in analyze.RUNS if r in summary["runs"]]
    if not names:
        return _pending_figure("Tokens per question", "r1_c0")
    toks = [summary["runs"][r]["tokens"] for r in names]
    labels = []
    for r, t in zip(names, toks):
        cut = t["finish_reasons"].get("length", 0)
        note = f" ({cut} answer{'' if cut == 1 else 's'} cut at {t['completion_max']:,} tokens)" if cut else ""
        labels.append(f"{_run_label(r)}{note}")
    parts = [("prompt (what we send)", [t["prompt_mean"] for t in toks], LIGHT),
             ("hidden reasoning", [t["reasoning_mean"] for t in toks], ORANGE),
             ("visible answer", [t["completion_mean"] - t["reasoning_mean"] for t in toks], BLUE)]
    fig = go.Figure()
    for name, values, color in parts:
        fig.add_trace(go.Bar(x=values, y=list(range(len(names))), orientation="h", width=0.45, name=name,
                             marker_color=color, customdata=labels,
                             hovertemplate=f"%{{customdata}}<br>{name}: %{{x:.0f}} tokens<extra></extra>"))
    fig.update_layout(barmode="stack")
    height = HEIGHT + 120  # six rows
    _label_rows(fig, labels, height=height)
    _legend_below(fig, height)
    by = dict(zip(names, toks))
    title = "Tokens per question in every run"
    if {"r3_c2_p0", "r3_c2_p1"} <= set(by):
        title = (f"With a fact missing and no way out, the model wrote {by['r3_c2_p0']['completion_mean']:.0f} output "
                 f"tokens per question; allowed to say UNANSWERABLE, {by['r3_c2_p1']['completion_mean']:.0f}")
    sub = ("Averages per question, as counted by Groq. Bar = prompt + output; output (orange + blue) = hidden "
           "reasoning + visible answer. Every part is billed")
    return apply(fig, title, sub, footer(summary["runs"][names[0]]["n"], summary["model"]), height=height)
