"""'New words': every term the notebook uses, defined in plain words just before it is first needed.
Each box is short (at most 8 words) and sits next to the cells that use its words."""

WORDS = {
    "0": ("New words today", [
        ("LLM evaluation", "measuring how well a language model, or an app built on one, does a job: give it a fixed "
                           "set of tasks, grade its answers the same way every time, and turn the grades into numbers."),
        ("Model (LLM)", "a large language model: the kind of AI behind chat assistants. It writes text one piece at a "
                        "time."),
        ("Model evaluation", "grading a model on a public exam, so it can be compared with other models."),
        ("Application evaluation", "grading your own product (your prompt, your data, your settings) on an exam "
                                   "built from the tasks your users actually bring."),
        ("Benchmark", "a standardized exam for AI models: fixed questions, an answer key, a grading rule and a "
                      "score."),
        ("Metric", "a number computed from the graded answers, like the share of questions right. Each metric "
                   "answers one question about the model."),
        ("LLM judge", "a language model used as the grader, for answers that are too open-ended to compare "
                      "with a key."),
    ]),
    "1": ("New words today", [
        ("GSM8K", "a public benchmark of grade-school math word problems, released by OpenAI in 2021."),
        ("Test set", "the questions set aside for grading, not for training: the exam paper itself."),
        ("Answer key", "the official answer to each question. In GSM8K it is the number after `####` at the end of a "
                       "short worked solution."),
        ("Grading rule", "how an answer is judged right or wrong. Ours compares the model's final number with the key "
                         "as text, which is called exact match."),
        ("Score (accuracy)", "the number the exam ends with: the share of questions graded right."),
        ("Random sample", "a subset chosen by chance. On average it represents the whole set, and Day 2 shows how far "
                          "one draw can be off."),
    ]),
    "2a": ("New words: running the exam", [
        ("Token", "a chunk of text, often part of a word. Models read and write tokens, and providers count and bill "
                  "by them."),
        ("Prompt", "the text the model receives. Ours, which we call P0, is a two-line instruction (solve it step by "
                   "step, end with `####` and the number) plus the question."),
        ("Zero-shot", "the prompt contains no solved examples, only the instruction and the question."),
        ("gpt-oss-20b", "a model whose files OpenAI published for anyone to run, with about 20 billion parameters "
                        "(the numbers it learned in training). It writes out its reasoning before it answers."),
        ("Groq", "a company that runs open models in the cloud and sells access by the token. It has a free plan. "
                 "Not Grok, the xAI chatbot."),
        ("API", "a way for a program to send a question to a model over the internet and get the answer back. An "
                "API key is the password for it."),
        ("Reasoning text", "the model's working, written before its answer and returned in a separate field. You pay "
                           "for its tokens, but the grader does not read it."),
    ]),
    "2b": ("New words: reading the score", [
        ("True score", "the share of all questions of this kind that this exact setup (model, prompt, settings, "
                       "grading rule) would get right. We can only estimate it."),
        ("95% confidence interval", "the range of true scores that fit what we observed. It is built so that, over "
                                    "many exams, it catches the true score about 19 times in 20."),
        ("Margin of error", "the ± part of a result, about half the width of the 95% range (the '± 3' in a poll's "
                            "'52% ± 3'). Ours is lopsided, so this notebook shows the whole range. It shrinks as the "
                            "exam gets longer."),
        ("Standard error", "how far one exam's score typically lands from the true score. lm-evaluation-harness "
                           "prints it after ±."),
        ("Wilson interval", "a standard formula for a 95% range around a percentage. Unlike score ± 2 standard "
                            "errors, it never goes below 0% or above 100%."),
    ]),
    "2x": ("New words: under the hood", [
        ("lm-evaluation-harness", "free, open-source software from EleutherAI that runs benchmarks on models."),
        ("Task YAML", "the small text file that describes one exam for the harness: questions, prompt, answer key, "
                      "grading rule and score."),
        ("Regex", "a search pattern for text. The grading rules use one to find the number in an answer."),
        ("Seed", "the starting number of a random draw. The same seed always draws the same questions, so anyone can "
                 "repeat the draw."),
        ("Fingerprint (sha256)", "a short code computed from a file. If even one character of the 100 questions "
                                 "changed, the code would change."),
    ]),
    "5": ("New words today", [
        ("Distractor", "an irrelevant sentence added to a question, with numbers in it, to see whether the model is "
                       "thrown off (GSM-Plus calls it distraction insertion)."),
        ("Paired comparison", "comparing two runs on the same questions, one pair at a time, instead of comparing two "
                              "totals."),
        ("McNemar test", "a test for paired yes/no results: it looks only at the questions whose result changed."),
        ("Bootstrap range", "a range for the difference found by redrawing the questions at random many times."),
        ("Noise floor", "how much two runs differ when nothing was changed: the smallest difference you can trust."),
    ]),
    "4": ("New words today", [
        ("GSM-Plus", "a published set of variants of GSM8K questions (Li et al., 2024), made by changing the "
                     "questions in controlled ways."),
        ("Critical-thinking variant", "a GSM-Plus question with one needed fact removed, so it has no single right "
                                      "answer as written."),
        ("Abstention", "the model says a question cannot be answered, instead of giving a number."),
        ("False confidence", "a confident answer to a question that has no answer."),
        ("Control run", "the same test with one thing changed, to check a side effect. Here: does permission to "
                        "abstain make the model refuse questions it can answer?"),
    ]),
    "3": ("New words today", [
        ("Strict rule (strict-match)", "reads only the first number written right after '#### '."),
        ("Flexible rule (flexible-extract)", "reads the last number anywhere in the visible answer."),
        ("[invalid]", "what a rule returns when it finds nothing to read."),
        ("LaTeX", "a code for writing math, like `\\frac{10}{2}`. Chat apps display it as formulas; the grader reads "
                  "the raw code."),
        ("Post-hoc", "done after seeing the results. Useful for learning, but it must not change the score you "
                     "committed to beforehand."),
        ("GSM8K-Platinum", "a cleaned-up copy of the GSM8K test set, made by researchers at MIT in 2025. Several "
                           "frontier models answered every question, and people reviewed each question that at least "
                           "one model got wrong: they fixed wrong keys and removed badly written questions. Questions "
                           "that every model answered as the key says were kept without a review."),
    ]),
}


def new_words(section: str) -> str:
    """Markdown box with the terms of one section."""
    title, words = WORDS[str(section)]
    lines = [f"**{title} ({len(words)})**", ""]
    lines += [f"- **{term}**: {definition}" for term, definition in words]
    # a '$' is a character here, never the start of math (none of the definitions contains one inside `code`)
    return "\n".join(lines).replace("$", "\\$")
