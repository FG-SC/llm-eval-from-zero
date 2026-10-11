"""Tables and text blocks of the notebook. Numbers come from results/summary.json and the config and data
files next to it; question and answer text from the harness's logged samples in results/.

  python -m src.tables   # refresh the results table in README.md from results/summary.json
"""
import json
import re

import pandas as pd

from src import analyze
from src.notebook import to_markdown
from src.settings import CONFIG_DIR, RESULTS_DIR, ROOT, TASKS_DIR, load_task_yaml, load_yaml

PENDING = "PENDING"
CHECK, CROSS = "✓", "✗"
EXAM = "r1_c0"  # the Day 2 exam: the 100 original questions, prompt P0


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"


def _score_text(s: dict) -> str:
    return f"{s['correct']}/{s['n']} = {_pct(s['accuracy'])} {_ci(s['ci95'])}"


def _run(summary: dict, run: str):
    return summary["runs"].get(run)


def _per_question(r: dict) -> int:
    """Tokens per question, rounded the same way everywhere."""
    return round(r["tokens"]["total"] / r["n"])


def _plain(x) -> str:
    """0.0 -> '0' (a table cell would otherwise show 0.000000)."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def _ids(ids) -> str:
    """#306, #515 and #962."""
    tags = [f"#{i}" for i in ids]
    return " and ".join(tags) if len(tags) <= 2 else ", ".join(tags[:-1]) + " and " + tags[-1]


def _logged(idx: int, run: str = EXAM) -> dict:
    """The harness's logged sample of one question, by rule ('strict-match', 'flexible-extract')."""
    rows = {}
    for line in (RESULTS_DIR / run / "samples_gsm8k_seed100.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        if s["doc"]["orig_idx"] == idx:
            rows[s["filter"]] = s
    return rows


# ---------------------------------------------------------------- the metrics of the notebook
# (day, metric, what it tells you, exam analogy, how AI engineers use it). Definitions only, no results.
METRICS = [
    ("2", "Accuracy (the score)", "The share of questions graded right", "Your grade: 18 of 20 is 90%",
     "The headline number of an eval set, tracked for every model or prompt change"),
    ("2", "95% range (confidence interval)", "How sure the score is, given how many questions were asked",
     "One exam is a sample of what a student knows",
     "Decide whether a change is a real improvement or luck, and how big an eval set must be"),
    ("3", "Strict vs flexible rule", "Which part of the answer is graded: the number after '####', or the last number",
     "Reading only the answer box vs reading the whole page",
     "Catch grading bugs and format problems before blaming the model"),
    ("4", "Abstention rate", "How often the model says a question cannot be answered",
     "A student who writes 'not enough information'", "Check that your app admits what it cannot answer"),
    ("4", "False confidence", "How often the model answers a question that has no answer",
     "A student who guesses on a broken question", "Measure the risk of made-up answers"),
    ("4", "Rule vs judge agreement", "How often a simple rule and an AI grader give the same label",
     "Two teachers grading the same exam", "Use a second grader to find the answers worth reading yourself"),
    ("5", "Paired difference", "How the score moves when the same questions are changed or asked again, compared "
     "question by question", "The same student, the same exam, one distracting sentence added",
     "Regression tests: compare two versions on the same inputs"),
    ("2-5", "Cost per question", "The tokens used to answer one question", "The price of grading",
     "Budget your evaluations and choose the size of your eval set"),
]


def metrics_map() -> pd.DataFrame:
    """Every metric of the notebook, in plain words, and where it comes up."""
    return pd.DataFrame([(m[1], m[2], m[3], m[4], m[0]) for m in METRICS],
                        columns=["Metric", "What it tells you", "Exam analogy", "How AI engineers use it", "Day"])


# ---------------------------------------------------------------- Day 1
def sample_questions(summary: dict, k: int = 3) -> pd.DataFrame:
    """The first k sampled questions (lowest GSM8K index), with their answer keys."""
    run = _run(summary, EXAM)
    if not run:
        return pd.DataFrame([(PENDING, PENDING, PENDING)], columns=["GSM8K #", "Question", "Answer key"])
    return pd.DataFrame([(r["orig_idx"], r["question"], r["gold"]) for r in run["per_question"][:k]],
                        columns=["GSM8K #", "Question", "Answer key"])


def sample_card(summary: dict) -> str:
    """How the 100 questions were drawn, and their fingerprint, from the sample file's record."""
    s = summary["sample"]
    seed = re.search(r"Random\((\d+)\)", s["draw"])
    return (f"**The draw**, as one line of Python: `{s['draw']}`. Seed {seed.group(1) if seed else '?'}, "
            f"{s['n']} of the {s['dataset']['n_rows']:,} test questions. **Fingerprint** of the file that lists them "
            f"(`data/sample_ids.json`): `{s['sha256'][:12]}…`. Every run records it, and every question is checked "
            f"against its own saved fingerprint before it is asked.")


def worked_solution(idx: int) -> str:
    """GSM8K's official worked solution for one question, as Markdown."""
    rows = _logged(idx)
    if not rows:
        return PENDING
    steps = rows["strict-match"]["doc"]["answer"].strip().splitlines()
    body = "\n".join(f"    {step}" for step in steps)
    return (f"**The answer key for GSM8K #{idx}** (the `<<...>>` parts are calculator annotations added to the "
            f"solutions, so a model can use a calculator; ignore them. The last line holds the official answer):"
            f"\n\n{body}")


def full_set_cost(summary: dict) -> str:
    """Why 100 questions: what the whole test set would cost, from the tokens measured in the Day 2 run."""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    n_all = summary["sample"]["dataset"]["n_rows"]
    full = r["tokens"]["total"] / r["n"] * n_all
    tpd = load_yaml(CONFIG_DIR / "models.yaml")["model_under_test"]["limits"]["tpd"]
    return (f"**Why only {r['n']} questions?** Each question costs tokens. Groq's free plan allows {tpd:,} tokens a "
            f"day for this model, and every run in this notebook fits in it. At the {_per_question(r):,} tokens per "
            f"question measured in the Day 2 run, all {n_all:,} questions would cost about {round(full, -3):,.0f} "
            f"tokens: {full / tpd:.1f} days of the free allowance for every experiment. A random sample is the classic "
            f"answer. It estimates the score on all questions of this kind, right on average, at the price of a margin "
            f"of error.")


# ---------------------------------------------------------------- Day 2
def score_sentence(summary: dict) -> str:
    """The Day 2 result, in words: the score and its 95% range."""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    s = r["scores"]["strict-match"]
    lo, hi = s["ci95"]
    return (f"**{s['correct']} of {s['n']} right ({_pct(s['accuracy'])}).** Any true score from **{_pct(lo)} to "
            f"{_pct(hi)}** fits this result: that is its 95% range, a Wilson interval. The range is lopsided, "
            f"{100 * (s['accuracy'] - lo):.1f} points below the score and {100 * (hi - s['accuracy']):.1f} above, "
            f"because a score cannot go above 100%.")


def run_card(summary: dict, run: str = EXAM) -> pd.DataFrame:
    """The run card: the rules of this exam. Each line is a setting that can move the score, or a fact you need
    to repeat the run."""
    r = _run(summary, run)
    if not r:
        return pd.DataFrame([("status", PENDING, "")], columns=["Setting", "Value", "What it means"])
    c, t = r["card"], r["tokens"]
    cut = r["n"] - t["finish_reasons"].get("stop", 0)
    rows = [
        ("Model", f"{summary['model']}, served by Groq", "the system under test"),
        ("Harness", f"{c['harness']}, task {c['task']} v{c['task_version']}", "the software that runs the exam"),
        ("Questions", f"{c['dataset']} @ {str(c['dataset_revision'])[:7]}; {summary['sample']['draw']}",
         "where the questions come from, and how the 100 were drawn"),
        ("Prompt", f"{r['prompt']}, zero-shot ({c['num_fewshot']} solved examples), one chat message",
         "the exact text shown in the live demo above"),
        ("Grading rule", "strict rule, exact match (in the task file, fixed before the run)",
         "how an answer becomes right or wrong (Day 3)"),
        ("Temperature", _plain(c["temperature"]),
         "always the most likely token, chosen for repeatability; gpt-oss's own guide recommends 1.0, which can "
         "score differently"),
        ("Reasoning effort", c["reasoning_effort"], "the cheapest setting; more effort may score higher"),
        ("Max output tokens", f"{c['max_gen_toks']:,}", "longest answer allowed, reasoning text included"),
        ("Stop sequences", "none" if not c["stop"] else ", ".join(c["stop"]), "the model decides when to stop"),
        ("API seed", _plain(c["api_seed"]),
         "a second seed, sent with every request (not the one that drew the questions); best effort, not a promise "
         "of identical answers"),
        ("Answers cut off", f"{cut} of {r['n']} (longest: {t['completion_max']:,} of {c['max_gen_toks']:,} tokens)",
         "a cut-off answer would be graded wrong"),
        ("Tokens used", f"{t['total']:,} (about {_per_question(r):,} per question)", "what the run cost"),
        ("Date", r["date"], "hosted models can change over time"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value", "What it means"])


def task_outline(task: str = "gsm8k_seed100") -> pd.DataFrame:
    """The task file in five lines: Day 1's four parts, plus how each question is asked."""
    cfg = load_task_yaml(task)
    gen = cfg.get("generation_kwargs", {})
    revision = str((cfg.get("dataset_kwargs") or {}).get("revision", ""))[:7]
    prompt = str(cfg.get("doc_to_text", "")).rsplit("_", 1)[-1].upper()
    rows = [
        ("Questions", "dataset_path, dataset_kwargs, process_docs",
         f"{cfg['dataset_path']}, {cfg['test_split']} split, version {revision}; only our sampled questions, each "
         f"checked against its saved fingerprint"),
        ("How each question is asked", "doc_to_text, num_fewshot, generation_kwargs",
         f"prompt {prompt}, {cfg['num_fewshot']} solved examples, temperature {_plain(gen.get('temperature'))}, up to "
         f"{gen.get('max_gen_toks', 0):,} output tokens"),
        ("Answer key", "doc_to_target", "GSM8K's worked solution; the grader keeps what follows '#### '"),
        ("Grading rule", "filter_list, metric_list",
         "two rules read the answer (strict and flexible, Day 3); exact match compares it with the key as text"),
        ("Score", "metric_list (aggregation: mean)", "the share of questions graded right"),
    ]
    return pd.DataFrame(rows, columns=["Part of the exam", "Where in the task file", "Ours"])


def task_yaml(task: str = "gsm8k_seed100") -> str:
    """The task file as the harness reads it (comment lines left out)."""
    text = (TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).strip()


def logged_sample(summary: dict, idx: int, run: str = EXAM) -> pd.DataFrame:
    """One question as the harness logged it: what was sent, what came back, how each rule graded it."""
    from tasks.utils import doc_to_text_p0

    rows = _logged(idx, run)
    strict, flex = rows["strict-match"], rows["flexible-extract"]
    sent = json.loads(strict["arguments"][0][0][0])[0]["content"] if isinstance(strict["arguments"], list) else ""
    same = sent == doc_to_text_p0(strict["doc"])
    answer_lines = [l for l in strict["resps"][0][0].strip().splitlines() if l.strip()]
    key_lines = [l for l in str(strict["target"]).strip().splitlines() if l.strip()]
    table = [
        ("doc: the question", strict["doc"]["question"]),
        ("arguments: what was sent", "prompt P0 with this question, word for word as in the live demo" if same else sent),
        ("resps: the model's visible answer (last line)", answer_lines[-1] if answer_lines else ""),
        ("target: the answer key (last line)", f"{key_lines[-1] if key_lines else ''}  (the full worked solution is "
                                               f"logged; the grader keeps what follows '#### ')"),
        ("filtered_resps, strict rule", f"{strict['filtered_resps'][0]}  → exact_match {strict['exact_match']} "
                                        f"(1 = right)"),
        ("filtered_resps, flexible rule", f"{flex['filtered_resps'][0]}  → exact_match {flex['exact_match']} "
                                          f"(1 = right)"),
    ]
    return pd.DataFrame(table, columns=["Field in the log", "Value"])


def harness_output(run: str = EXAM) -> str:
    """The results table exactly as lm-evaluation-harness prints it."""
    from lm_eval.utils import make_table

    return make_table(json.loads((RESULTS_DIR / run / "results.json").read_text(encoding="utf-8")))


def harness_side_note(summary: dict) -> str:
    """How the harness's 'score ± standard error' relates to the Wilson range used in this notebook."""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    h, s = r["harness"]["strict-match"], r["scores"]["strict-match"]
    quick = (h["exact_match"] - 2 * h["stderr"], h["exact_match"] + 2 * h["stderr"])
    return (f"How to read it: *Value* is the score and *Stderr* its standard error, **{h['exact_match']:.2f} ± "
            f"{h['stderr']:.4f}**; *n-shot* is the number of solved examples in the prompt; ↑ means higher is better. "
            f"Counting the {r['n']} logged answers ourselves gives the same {s['correct']}/{s['n']}. A quick rule turns "
            f"a standard error into a 95% range, the score ± 2 standard errors: {_pct(quick[0])} to {_pct(quick[1])}. "
            f"The Wilson range we report, {_ci(s['ci95'])}, is close, but it never leaves 0–100% and is more accurate "
            f"near 100%, where this model is.")


# ---------------------------------------------------------------- Day 3
DEMO_CASES = [
    ("The format we asked for", "#### 64", "64"),
    ("A dollar sign after ####", "#### $64", "64"),
    ("No space after ####", "####64", "64"),
    ("No #### line at all", "The answer is 64.", "64"),
    ("A note after the answer", "#### 64 (8 pairs of glasses)", "64"),
    ("Right number, written with decimals", "#### 8.00", "8"),
    ("Right number, written as a fraction", "#### 10/2", "5"),
]


def grading_demo(extra=None) -> pd.DataFrame:
    """Short answers we wrote (plus any (answer, key) pairs passed in `extra`), graded by
    lm-evaluation-harness's own code with our task's rules."""
    from lm_eval.api.metrics import exact_match_hf_evaluate
    from lm_eval.filters.extraction import RegexFilter

    cfg = load_task_yaml("gsm8k_seed100")
    metric = cfg["metric_list"][0]
    rules = {}
    for f in cfg["filter_list"]:
        regex = next(step for step in f["filter"] if step["function"] == "regex")
        rules[f["name"]] = RegexFilter(regex_pattern=regex["regex_pattern"], group_select=regex.get("group_select", 0))
    rows = []
    for case, answer, key in DEMO_CASES + [("Yours", a, str(k)) for a, k in (extra or [])]:
        cells = [case, answer, key]
        for name in ("strict-match", "flexible-extract"):
            read = list(rules[name].apply([[answer]], [{}]))[0][0]
            right = exact_match_hf_evaluate([read], [f"#### {key}"], regexes_to_ignore=metric["regexes_to_ignore"],
                                            ignore_case=metric["ignore_case"],
                                            ignore_punctuation=metric["ignore_punctuation"])["exact_match"]
            cells.append(f"reads {read} → {'right' if right else 'wrong'}")
        rows.append(cells)
    return pd.DataFrame(rows, columns=["Case", "Answer ends with", "Key", "Strict rule", "Flexible rule"])


def rules_agreement(summary: dict) -> str:
    """One paragraph, computed: did the two rules ever give different grades in our run?"""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    cats, n = r["categories"], r["n"]
    split = cats["right_flexible_only"] + cats["right_strict_only"]
    wrote = r["extracted"]["strict-match"]
    differ = [q for q in r["per_question"] if analyze._as_key_text(q["strict"]) != analyze._as_key_text(q["flexible"])]
    box = f"The model followed the requested format, a `####` line, in {wrote} of {n} answers."
    if split:
        return f"**The two rules gave different grades to {split} of {n} answers.** {box}"
    readings = ""
    if differ:
        wrong_anyway = all(not q["strict_ok"] and not q["flexible_ok"] for q in differ)
        readings = (f" Their readings differed on {len(differ)} answers ({_ids(q['orig_idx'] for q in differ)})"
                    f"{', but both rules marked those wrong' if wrong_anyway else ''}.")
    return (f"**In our run the two rules never gave different grades:** both marked the same "
            f"{r['scores']['strict-match']['correct']} answers right.{readings} {box}")


def errors(summary: dict, run: str = EXAM) -> pd.DataFrame:
    """Every answer, by what the two rules decided."""
    r = _run(summary, run)
    if not r:
        return pd.DataFrame([(PENDING, "", "")], columns=["What the rules decided", "Answers", "Share"])
    return pd.DataFrame([(analyze.CATEGORY_TEXT[c], r["categories"][c], _pct(r["categories"][c] / r["n"]))
                         for c in analyze.CATEGORY_ORDER], columns=["What the rules decided", "Answers", "Share"])


EXPLAIN = {
    "clean": "Ends with the requested line. Both rules read {strict}, which matches the key.",
    "number_not_text": ("The model wrote `{final_line}`. Both rules read `{strict}`, and exact match compares text, so "
                        "`{strict}` is not `{gold}`. The number is right; the grade is wrong."),
    "no_box": ("No `####` line: the answer ends with {ending}. The strict rule finds nothing ({strict}); the "
               "flexible rule reads the last number, {flexible}. The key is {gold}."),
    "rules_disagree": ("The model wrote `{final_line}`. The strict rule reads {strict} and the flexible rule reads "
                       "{flexible}. The key is {gold}."),
    "key_error": "{reason} GSM8K-Platinum corrects this key to {platinum_key}.",
}
KIND_TITLE = {"clean": "A clean, correct answer", "number_not_text": "Right number, marked wrong",
              "no_box": "No #### line: the model boxed its answer the LaTeX way",
              "rules_disagree": "The two rules read different numbers", "key_error": "The answer key is wrong"}


def _ending(e: dict) -> str:
    """How an answer ends, in words: a formula's content rather than its bare closing '\\]'."""
    if e["final_line"] in ("\\]", "\\)"):
        formulas = re.findall(r"\\\[(.*?)\\\]|\\\((.*?)\\\)", e["answer_tail"], re.DOTALL)
        if formulas:
            last = " ".join((formulas[-1][0] or formulas[-1][1]).split())
            return f"a formula, `{last}`"
    return f"`{e['final_line']}`"


def _raw(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def answer_cards(summary: dict, run: str = EXAM) -> list[str]:
    """Real answers, picked by fixed rules (not by hand), shown as the grader reads them: raw text."""
    r = _run(summary, run)
    if not r:
        return [PENDING]
    cards = []
    for e in r["examples"]:
        verdict = (f"strict reads **{e['strict']}** {CHECK if e['strict_ok'] else CROSS} · flexible reads "
                   f"**{e['flexible']}** {CHECK if e['flexible_ok'] else CROSS} · key **{e['gold']}**")
        note = EXPLAIN[e["kind"]].format(**{"reason": "", "platinum_key": "", **e}, ending=_ending(e))
        if e["kind"] == "no_box" and "\\boxed" in _ending(e):
            note += (" `\\boxed{...}` is a common way to mark the final answer in math texts, but it is not the "
                     "format our prompt asked for.")
        if e["kind"] == "rules_disagree" and "/" in e["final_line"]:
            note += " Neither rule understands fractions: the strict rule stops at the '/', the flexible one takes the last part."
        key_part = ""
        if e["kind"] == "key_error":
            solution = _logged(e["orig_idx"], run)["strict-match"]["doc"]["answer"]
            key_part = f"**The official worked solution (the answer key):**\n\n{_raw(solution)}\n\n"
        cards.append(f"#### GSM8K #{e['orig_idx']} · {KIND_TITLE[e['kind']]}\n\n> {to_markdown(e['question'])}\n\n"
                     f"{key_part}**End of the model's answer, as the grader reads it (raw text):**\n\n"
                     f"{_raw(e['answer_tail'])}\n\n{verdict}\n\n{note}")
    return cards


PLATINUM_STATUS = {"removed": "removed as badly written (reviewed by people)",
                   "revised": "key corrected (reviewed by people)",
                   "verified": "kept, key checked by people",
                   "consensus": "kept without a human review (all of its models agreed with the key)"}
PLATINUM_ACTION = {"removed": "removed the question as badly written", "revised": "corrected the key",
                   "verified": "kept the question after checking its key", "consensus": "kept it without a review"}


def audit_method(summary: dict) -> str:
    """Who read the answers marked wrong, from data/key_audit.json."""
    a = summary.get("key_audit")
    if not a:
        return PENDING
    return (f"This reading happened after the run (post-hoc), by {a['n_readers']} AI readers: Claude, the AI "
            f"assistant that ran the experiment, and {a['n_readers'] - 1} more Claude sessions that did not see each "
            f"other's verdicts. Each saw the question, the official worked solution and the model's full answer. "
            f"**That is not a human audit.** So every verdict was then compared with GSM8K-Platinum, where people "
            f"reviewed the questions its models got wrong. Read the questions below and judge for yourself.")


def _votes(item: dict) -> str:
    votes, top = item["votes"], item["votes"][item["verdict"]]
    total = sum(votes.values())
    others = "; ".join(f"{c} said: {analyze.VERDICT_TEXT[v].lower()}" for v, c in votes.items() if v != item["verdict"])
    return f"{top} of {total} readers" + (f"; {others}" if others else "")


def audit(summary: dict) -> pd.DataFrame:
    """The answers graded wrong, read one by one (post-hoc), next to GSM8K-Platinum's verdict."""
    a = summary.get("key_audit")
    if not a:
        return pd.DataFrame([(PENDING,) * 6], columns=["GSM8K #", "Question", "Key", "Model wrote", "Verdict", "Why"])
    rows = [(i["idx"], i["question"], i["gold"], i["model_answer"],
             f"{analyze.VERDICT_TEXT[i['verdict']]} ({_votes(i)}). GSM8K-Platinum: "
             f"{PLATINUM_STATUS.get(i['platinum'], 'n/a')}.", i["reason"])
            for i in a["items"]]
    return pd.DataFrame(rows, columns=["GSM8K #", "Question", "Key", "Model wrote", "Verdict", "Why"])


def platinum_story(summary: dict) -> str:
    """What GSM8K-Platinum changes on our questions, step by step, and how far it agrees with the audit."""
    a = summary.get("key_audit")
    if not a or "platinum" not in a:
        return PENDING
    p, ag, n = a["platinum"], a["platinum"]["agrees_with_audit"], summary["runs"][EXAM]["n"]
    lines = [f"**What GSM8K-Platinum changes on our {n} questions.**", ""]
    lines.append(f"1. It **removes {len(p['removed'])} questions** as badly written: {len(p['removed_wrong'])} that "
                 f"the model got wrong ({_ids(p['removed_wrong'])}) and {len(p['removed_right'])} it got right "
                 f"({_ids(p['removed_right'])}). Its people only reviewed questions that at least one of its models got "
                 f"wrong, so the questions it removes tend to be ones models miss.")
    fixes = []
    for c in p["key_changed"]:
        effect = ("so it becomes right" if c["right_after"] and not c["right_before"] else
                  "so it becomes wrong" if c["right_before"] and not c["right_after"] else "no change")
        fixes.append(f"   - #{c['idx']}: key {c['gsm8k_key']} → {c['platinum_key']}; the model answered {c['model']}, "
                     f"{effect}. {c.get('note') or ''}".rstrip())
    lines.append(f"2. It **corrects {len(p['key_changed'])} answer keys**:")
    lines += fixes
    if p["hidden_mistakes"]:
        lines.append(f"\n**A wrong answer key can also hide a model's mistake:** {_ids(p['hidden_mistakes'])} were "
                     f"marked right only because the old key was wrong too.")
    agree = (f"all {ag['checked']} answers" if ag["agree"] == ag["checked"] else
             f"{ag['agree']} of the {ag['checked']} answers")
    consensus = (f" {_ids(ag['consensus'])} was kept without a review, because every model Platinum ran agreed with "
                 f"the key." if ag["consensus"] else "")
    lines += ["", f"**Agreement with our reading:** Platinum agrees on {agree} we can compare, and people reviewed "
                  f"{ag['reviewed_by_people']} of them.{consensus}"]
    for item in ag["not_compared"]:
        lines[-1] += (f" On #{item['idx']} our readers saw a {analyze.VERDICT_TEXT[item['verdict']].lower()}; Platinum "
                      f"{PLATINUM_ACTION.get(item['platinum'], item['platinum'])}.")
    fs = p.get("full_test_set")
    if fs:
        typical = fs["key_changed"] * n / fs["n"]
        lines += ["", f"**Is that typical?** Across the whole test set Platinum removed {fs['removed_from_platinum']} of "
                      f"{fs['n']:,} questions and changed {fs['key_changed']} keys. Our {n} questions hold "
                      f"{len(p['key_changed'])} of those {fs['key_changed']} key changes, where about {typical:.1f} "
                      f"would be typical for {n} questions: wrong keys are rare in GSM8K, and our sample happens to "
                      f"hold more than its share."]
    return "\n".join(lines)


def score_ladder(summary: dict) -> pd.DataFrame:
    """The same answers, graded five ways: each row changes one part of the exam."""
    r = _run(summary, EXAM)
    cols = ["How the same answers are graded", "Score [95% CI]"]
    if not r:
        return pd.DataFrame([(PENDING, PENDING)], columns=cols)
    rows = [("Official: strict rule, exact match, GSM8K keys", _score_text(r["scores"]["strict-match"])),
            ("Grading rule changed: compare numbers, so 8.00 = 8", _score_text(r["scores"]["number-match"]))]
    p = (summary.get("key_audit") or {}).get("platinum")
    if p:
        rows += [(f"Answer keys changed: Platinum's {len(p['key_changed'])} corrected keys, same {r['n']} questions",
                  _score_text(p["steps"]["keys_only"])),
                 (f"Questions changed: Platinum's {len(p['removed'])} removed questions dropped, GSM8K keys",
                  _score_text(p["steps"]["questions_only"])),
                 ("Both: GSM8K-Platinum as published", _score_text(p["score"]))]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- Day 4
P0_RUN, P1_RUN, CONTROL_RUN = "r3_c2_p0", "r3_c2_p1", "r1_c0_p1"
PROMPT_TEXT = {P0_RUN: "P0: no permission to abstain", P1_RUN: "P1: may write UNANSWERABLE"}
JUDGE_TEXT = {"ANSWERED_NUMBER": "Gave a number", "ABSTAINED": "Said it cannot be answered", "OTHER": "Something else",
              "INVALID": "Judge reply unreadable"}


def _twins(summary: dict, run: str, title: str, k: int) -> pd.DataFrame:
    """The first k original questions next to their GSM-Plus twins from `run`."""
    orig, var = _run(summary, EXAM), _run(summary, run)
    cols = ["GSM8K #", "Original question", title]
    if not orig or not var:
        return pd.DataFrame([(PENDING,) * 3], columns=cols)
    twin = {r["orig_idx"]: r["question"] for r in var["per_question"]}
    return pd.DataFrame([(r["orig_idx"], r["question"], twin[r["orig_idx"]]) for r in orig["per_question"][:k]],
                        columns=cols)


def missing_fact_pairs(summary: dict, k: int = 2) -> pd.DataFrame:
    """A question and its twin with one needed fact removed (GSM-Plus 'critical thinking')."""
    return _twins(summary, P0_RUN, "Same question, one fact removed", k)


RULE_TEXT = {"answered_number": "Ended with a number", "abstained": "Wrote UNANSWERABLE",
             "no_final_answer": "Neither"}


def abstention_table(summary: dict) -> pd.DataFrame:
    """The missing-fact exam: how each answer ended (the simple last-line rule), for each prompt."""
    cols = ["Prompt", *RULE_TEXT.values()]
    rows = []
    for run, text in PROMPT_TEXT.items():
        labels = (_run(summary, run) or {}).get("rule_labels")
        rows.append((text, *[(_score_text(labels[k]) if labels else PENDING) for k in RULE_TEXT]))
    return pd.DataFrame(rows, columns=cols)


def abstention_story(summary: dict) -> str:
    """The Day 4 result in words: a confident number without permission, abstention with it."""
    p0, p1 = _run(summary, P0_RUN), _run(summary, P1_RUN)
    c = summary.get("comparisons", {}).get("c2_p0_to_p1")
    if not (p0 and p1 and c):
        return PENDING
    n, a, w = p0["n"], c["all"], c["without_flagged"]
    return (f"**Without permission (P0), {p0['rule_labels']['answered_number']['correct']} of {n} answers to questions "
            f"that cannot be solved as written ended with a confident number.** With permission (P1), "
            f"{p1['rule_labels']['abstained']['correct']} wrote UNANSWERABLE and "
            f"{p1['rule_labels']['answered_number']['correct']} still gave a number. Question by question, "
            f"{a['after_only']} switched to abstaining and {a['before_only']} the other way (McNemar p "
            f"{_p(a['mcnemar_p'])}{': far too lopsided to be luck' if a['mcnemar_p'] < 0.001 else ''}). {len(c['excluded'])} of the {n} variants ({_ids(c['excluded'])}) were flagged before "
            f"any run as possibly answerable or debatable; leaving them out, P1 abstained on {_pct(w['rate_after'])}.")


def control_sentence(summary: dict) -> str:
    """The side effect: does permission to abstain make the model refuse questions it can answer?"""
    r, base = _run(summary, CONTROL_RUN), _run(summary, EXAM)
    if not r or not base:
        return PENDING
    s, b = r["scores"]["strict-match"], base["scores"]["strict-match"]
    text = (f"**Control.** On the {r['n']} questions that *can* be solved, the P1 prompt made the model write "
            f"UNANSWERABLE {r['abstained']} times, and scored {_score_text(s)}, against {_score_text(b)} with P0.")
    c = summary.get("comparisons", {}).get("c0_to_p1")
    if c:
        a = c["all"]
        lo, hi = a["diff_ci95_bootstrap"]
        noise = ("the range includes zero, so this cost is within the noise (we cannot say the permission hurt)"
                 if lo <= 0 <= hi else "the range stays clear of zero, so the permission has a real cost")
        text += (f" Question by question, {a['before_only']} right answers became wrong and {a['after_only']} wrong "
                 f"answers became right (difference {_points(a['diff'])} points, 95% range {_points(lo)} to "
                 f"{_points(hi)}): {noise}.")
    return text


def judge_agreement(summary: dict) -> str:
    """How often the last-line rule and the AI judge gave the same label, and what that does (not) prove."""
    j = summary.get("judge_results")
    if not j:
        return PENDING
    same = {"answered_number -> ANSWERED_NUMBER", "abstained -> ABSTAINED"}
    pairs = [(p, k) for r in j["runs"].values() for p, k in r["rule_vs_judge"].items()]
    total, agree = sum(k for _, k in pairs), sum(k for p, k in pairs if p in same)
    return (f"**The rule and the judge gave the same label to {agree} of the {total} answers.** That shows two "
            f"different methods agree; it does not prove that both are right. The {total - agree} where they differ "
            f"are the ones worth reading yourself (below).")


def rule_vs_judge_table(summary: dict) -> pd.DataFrame:
    """Do the plain rule (last line) and the judge (reads the answer) agree? Counts of every pair."""
    j = summary.get("judge_results")
    cols = ["Prompt", "Last-line rule says", "Judge says", "Answers"]
    if not j:
        return pd.DataFrame([(PENDING, "", "", "")], columns=cols)
    rows = []
    for run, text in PROMPT_TEXT.items():
        for pair, count in ((j["runs"].get(run) or {}).get("rule_vs_judge") or {}).items():
            rule, judge = pair.split(" -> ")
            rows.append((text, rule.replace("_", " "), JUDGE_TEXT.get(judge, judge).lower(), count))
    return pd.DataFrame(rows, columns=cols)


def disagreements_table(summary: dict) -> pd.DataFrame:
    """The answers where the last-line rule and the judge differ, with the end of each answer, for the reader to judge."""
    j = summary.get("judge_results")
    cols = ["Prompt", "GSM8K #", "Last-line rule says", "Judge says", "End of the answer (raw text)"]
    if not j:
        return pd.DataFrame([(PENDING,) * 5], columns=cols)
    rows = []
    for d in j.get("disagreements", []):
        q = next(x for x in summary["runs"][d["run"]]["per_question"] if x["orig_idx"] == d["idx"])
        rows.append((PROMPT_TEXT.get(d["run"], d["run"]), d["idx"], d["rule"].replace("_", " "),
                     JUDGE_TEXT.get(d["judge"], d["judge"]).lower(), q["answer_tail"]))
    return pd.DataFrame(rows or [("none", "", "", "", "")], columns=cols)


# ---------------------------------------------------------------- Day 5
DISTRACTOR_RUN, RETEST_RUN = "r2_c1", "r1_c0_retest"


def distractor_pairs(summary: dict, k: int = 2) -> pd.DataFrame:
    """A question and its twin with one irrelevant sentence added (GSM-Plus 'distraction insertion')."""
    return _twins(summary, DISTRACTOR_RUN, "Same question, one irrelevant sentence added", k)


def _points(x: float) -> str:
    return f"{100 * x:+.1f}"


def _p(p: float) -> str:
    """A p-value a beginner can read: '< 0.001' instead of '2.6e-26'."""
    return "< 0.001" if p < 0.001 else f"= {p:.2f}"


def distractor_story(summary: dict) -> str:
    """The Day 5 result: the score with an extra sentence, compared question by question."""
    base, r, c = _run(summary, EXAM), _run(summary, DISTRACTOR_RUN), summary.get("comparisons", {}).get("c0_to_c1")
    if not (base and r and c):
        return PENDING
    a, w = c["all"], c["without_flagged"]
    return (f"**With one irrelevant sentence added to each question the model scored "
            f"{_score_text(r['scores']['strict-match'])}, against {_score_text(base['scores']['strict-match'])} on the "
            f"originals.** Question by question: {a['before_only']} questions it got right before are now wrong, and "
            f"{a['after_only']} it got wrong before are now right (difference {_points(a['diff'])} points, bootstrap "
            f"95% range {_points(a['diff_ci95_bootstrap'][0])} to {_points(a['diff_ci95_bootstrap'][1])}; "
            f"McNemar p {_p(a['mcnemar_p'])}). Leaving out the {len(c['excluded'])} questions flagged before the run or "
            f"with a wrong answer key ({_ids(c['excluded'])}): {w['before_only']} lost, {w['after_only']} gained, "
            f"difference {_points(w['diff'])} points.")


def _count(k: int, noun: str) -> str:
    return f"{k} {noun}{'' if k == 1 else 's'}"


def retest_story(summary: dict) -> str:
    """The noise floor: the same questions asked again, with nothing changed."""
    c = summary.get("comparisons", {}).get("c0_retest")
    if not c:
        return PENDING
    return (f"**Asking the same {c['n']} original questions a second time, with nothing changed:** "
            f"{_count(c['answer_changed']['correct'], 'final answer')} changed and "
            f"{_count(c['grade_flipped']['correct'], 'grade')} flipped ({c['right_to_wrong']} from right to wrong, {c['wrong_to_right']} from wrong to right). "
            f"Part of any difference between two runs is just the model answering differently.")


def paired_plain(summary: dict) -> str:
    """Day 5 in plain words: why 10 against 4 can still be luck, and what to conclude."""
    c, rt = summary.get("comparisons", {}).get("c0_to_c1"), summary.get("comparisons", {}).get("c0_retest")
    if not c:
        return PENDING
    a = c["all"]
    lost, gained = a["before_only"], a["after_only"]
    flips = lost + gained
    lo, hi = a["diff_ci95_bootstrap"]
    text = (f"**In plain words.** Only the {flips} questions that flipped tell us anything; the others gave the same "
            f"grade both times. If the extra sentence had no effect at all, those {flips} flips would tend to split "
            f"evenly, about {flips / 2:g} each way, like coin tosses. A split at least as uneven as {lost} against "
            f"{gained} (either way) happens by pure chance with probability {a['mcnemar_p']:.2f}, about "
            f"{round(100 * a['mcnemar_p'])} times in 100 (that is McNemar's p). That is not rare enough to rule out "
            f"luck.")
    if rt:
        k = rt["grade_flipped"]["correct"]
        text += (f" And asking the same questions again flipped only {_count(k, 'grade')}, so why is that not the "
                 f"noise to beat? Because it measures only the model's own randomness. It says nothing about the luck "
                 f"of *which* 100 questions we drew, and that luck is what the 95% range covers.")
    verdict = ("**Conclusion:** the extra sentence may hurt, but 100 questions are not enough to tell. To find out, "
               "run the same comparison on more questions" if lo <= 0 <= hi else
               "**Conclusion:** the range stays clear of zero, so the extra sentence has a real effect on this setup")
    return text + f"\n\n{verdict} (difference {_points(a['diff'])} points, 95% range {_points(lo)} to {_points(hi)})."


def score_plain(summary: dict) -> str:
    """Day 2 in plain words, right under the score chart."""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    s = r["scores"]["strict-match"]
    lo, hi = s["ci95"]
    return (f"**In plain words.** The model got {s['correct']} of {r['n']} right. Another random {r['n']} questions "
            f"would have given a somewhat different score, so {s['correct']}/{r['n']} is an estimate. The range says "
            f"which true scores fit what we saw: anything from {_pct(lo)} to {_pct(hi)}. To make it about half as "
            f"wide, you need about four times as many questions. One trap: two scores whose ranges overlap can still "
            f"be different. To compare two versions, test them on the same questions and look at the questions that "
            f"changed (Day 5).")


def flips_table(summary: dict) -> pd.DataFrame:
    """Right-to-wrong and wrong-to-right counts: the effect of the extra sentence next to the noise floor."""
    cmp_ = summary.get("comparisons", {})
    cols = ["Comparison", "Right → wrong", "Wrong → right", "Net change in the score"]
    rows = []
    if "c0_to_c1" in cmp_:
        a = cmp_["c0_to_c1"]["all"]
        rows.append(("Originals vs one extra sentence", a["before_only"], a["after_only"],
                     f"{_points(a['diff'])} points"))
    if "c0_retest" in cmp_:
        c = cmp_["c0_retest"]
        net = (c["wrong_to_right"] - c["right_to_wrong"]) / c["n"]
        rows.append(("Originals vs the same originals asked again", c["right_to_wrong"], c["wrong_to_right"],
                     f"{_points(net)} points"))
    return pd.DataFrame(rows or [(PENDING, "", "", "")], columns=cols)


# ---------------------------------------------------------------- every run
def results_table(summary: dict) -> pd.DataFrame:
    """Every run in one table (PENDING until its run finishes)."""
    rows = []
    for run, meta in sorted(analyze.RUNS.items(), key=lambda kv: kv[1]["day"]):
        r = _run(summary, run)
        label = f"Day {meta['day']} · {meta['label']}"
        if not r:
            rows.append((label, meta["prompt"], PENDING, PENDING, PENDING))
            continue
        tokens = f"{_per_question(r):,}"
        if r["kind"] == "numeric":
            rows.append((label, r["prompt"], _score_text(r["scores"]["strict-match"]), f"{r['abstained']} of {r['n']}", tokens))
        else:
            gave = r["rule_labels"]["answered_number"]
            abst = r["rule_labels"]["abstained"]
            rows.append((label, r["prompt"], f"gave a number: {_score_text(gave)}", _score_text(abst), tokens))
    return pd.DataFrame(rows, columns=["Run", "Prompt", "Result [95% CI]", "Said UNANSWERABLE", "Tokens per question"])


def tldr(summary: dict) -> str:
    """The whole notebook in one paragraph, for readers who already know evaluation."""
    r = _run(summary, EXAM)
    if not r:
        return PENDING
    s, c = r["scores"]["strict-match"], r["card"]
    parts = [f"**For practitioners, in one paragraph.** `{summary['model']}` on Groq; zero-shot prompt (P0), "
             f"temperature {_plain(c['temperature'])}, reasoning effort {c['reasoning_effort']}; {r['n']} random GSM8K "
             f"test questions; {c['harness']} with a custom task that keeps the built-in gsm8k grading rules. Result: "
             f"**{_score_text(s)}** (strict rule, 95% Wilson interval), not comparable with leaderboard numbers (the "
             f"built-in task is 5-shot)."]
    a = summary.get("key_audit") or {}
    if a and a.get("platinum"):
        p = a["platinum"]
        parts.append(f"Day 3: of the {a['graded_wrong']} answers marked wrong, {a['counts']['model_error']} are clearly "
                     f"the model's mistake; GSM8K-Platinum's corrected keys reveal {len(p['hidden_mistakes'])} more "
                     f"among the answers marked right; with Platinum's questions and keys the same answers score "
                     f"{p['score']['correct']}/{p['score']['n']}.")
    runs, comps = summary["runs"], summary.get("comparisons", {})
    if {"r3_c2_p0", "r3_c2_p1"} <= set(runs):
        p0, p1 = runs["r3_c2_p0"]["rule_labels"], runs["r3_c2_p1"]["rule_labels"]
        parts.append(f"Day 4: on GSM-Plus rewrites with a needed fact removed, the model still gave a number "
                     f"{p0['answered_number']['correct']}/{p0['answered_number']['n']} times with P0; allowed to write "
                     f"UNANSWERABLE (P1), it did so {p1['abstained']['correct']}/{p1['abstained']['n']} times.")
    if "c0_to_c1" in comps:
        d = comps["c0_to_c1"]["all"]
        lo, hi = d["diff_ci95_bootstrap"]
        text = (f"Day 5: one irrelevant sentence moved the score {_points(d['diff'])} points (paired 95% range "
                f"{_points(lo)} to {_points(hi)}, McNemar p {_p(d['mcnemar_p'])})")
        if "c0_retest" in comps:
            k = comps["c0_retest"]["grade_flipped"]["correct"]
            text += f"; asking the same questions again flipped {k} grade{'' if k == 1 else 's'}"
        parts.append(text + ".")
    parts.append("Caveats: one provider, one week of runs, greedy decoding (gpt-oss's guide recommends temperature 1.0), and "
                 "training-data contamination is not tested.")
    return " ".join(parts)


# ---------------------------------------------------------------- README
README = ROOT / "README.md"
README_BLOCK = re.compile(r"(<!-- results:start -->\n).*?(<!-- results:end -->)", re.DOTALL)


def readme_block(summary: dict) -> str:
    """The README's results section: the notebook's table of every run, as Markdown."""
    df = results_table(summary)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return ("\n".join(lines) + "\n\n"
            f"Every number above is read from `results/summary.json`; PENDING marks a run that has not finished. "
            f"Model: `{summary['model']}` on Groq, {summary['sample']['n']} of the "
            f"{summary['sample']['dataset']['n_rows']:,} GSM8K test questions.\n")


def update_readme(summary: dict) -> bool:
    """Rewrite the block between <!-- results:start --> and <!-- results:end --> in README.md."""
    text = README.read_text(encoding="utf-8")
    if not README_BLOCK.search(text):
        return False
    README.write_text(README_BLOCK.sub(lambda m: m.group(1) + readme_block(summary) + m.group(2), text),
                      encoding="utf-8", newline="\n")
    return True


if __name__ == "__main__":
    print("README.md results table refreshed" if update_readme(analyze.load()) else "no results block in README.md")
