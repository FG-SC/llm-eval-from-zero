# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM evaluation from zero
#
# Every model launch comes with a benchmark score: a percentage on GSM8K or another exam. What is behind that
# number, and how much should you trust it? This notebook grades one AI model the way a teacher grades an exam,
# and shows where the grade can fool you. Each "day" below is one lesson, built on a real experiment.
#
# **What you will learn**
#
# - **Day 0 (intro):** what LLM evaluation is, the two kinds (model and application evaluation), and how AI
#   engineers use it every day. Words and ideas only.
# - **Day 1:** what a benchmark is: questions, an answer key, a grading rule and a score.
# - **Day 2:** why a score is an estimate, and how to report its 95% range.
# - **Day 3:** how the grading rule, the answer key and the choice of questions move the score, even when the
#   model's answers stay the same.
# - **Day 4:** when a model should admit it does not know, and what that costs.
# - **Day 5:** how to tell a real change in the score from noise.
#
# **How to use this notebook.** Run the cells from top to bottom. They only *read* saved results, so they are
# free and fast: you need no API key (the password for calling a model) and no Groq account. Every result
# (score, count, range) is printed by code from the saved files in `results/`. To run it yourself, clone the
# repository and follow *Run it yourself* in the README. The model under test is `openai/gpt-oss-20b`, served by Groq, and
# the exams are run with EleutherAI's lm-evaluation-harness.
#
# **How this was made.** I designed the experiments. An AI coding assistant (Claude) wrote the code, ran the
# exams and made a first reading of the answers marked wrong (Day 3). Where an AI made a judgment call, the
# notebook says so.
#
# Two optional switches sit at the top of the first code cell: `RUN_LIVE = True` asks the model all 100 Day 2
# questions again (it needs a Groq key, uses part of your daily quota and replaces the saved results with
# yours); `LIVE_DEMO = True` sends one real question to the model on Day 2 (it needs a Groq key).

# %% [markdown]
# ## Setup
#
# The first two lines are the optional switches; the rest loads the saved results. You do not need to read it.
# If it stops with an error, read the message: almost always the notebook kernel is not the project's `.venv`
# (in VS Code, click "Select Kernel" at the top right and pick `.venv`).

# %% tags=["infra"]
RUN_LIVE = False    # True: ask the model all 100 Day 2 questions again (Groq key; uses your quota; replaces results)
LIVE_DEMO = False   # True: send one real question to the model on Day 2 (Groq key)

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.notebook import check_environment

check_environment()  # a clear message if this kernel lacks the project's packages
from src.notebook import rerun, say, setup, show_table, status

summary = setup()  # the saved results (results/summary.json)
EXAM = "r1_c0"  # the saved Day 2 exam: the 100 original questions, prompt P0
FIRST = summary["runs"][EXAM]["per_question"][0]["orig_idx"]  # the first sampled question: we follow it throughout
from src import figures, live, tables
from src.words import new_words

# say() shows a text and show_table() a table, both built from the saved results
say(status(summary, live=RUN_LIVE or LIVE_DEMO))

# %% [markdown]
# Already know evaluation? Here is the whole notebook in one paragraph. It includes the results, so skip it if you
# want the story in order: the days below explain every word.

# %%
say(tables.tldr(summary))

# %% [markdown]
# ## Day 0 · Can we actually tell if an LLM is good?
#
# **What we're doing.** A gentle introduction: today there is no math and no code to read, only the ideas, in
# plain words. Every model launch comes with a score, and every team that ships an AI feature needs scores of
# its own. **LLM evaluation** is how those scores are made: give a model, or an app built on one, a fixed set of
# tasks, grade its answers the same way every time, and turn the grades into numbers. It works like a school
# exam, where everyone gets the same questions and the same answer key.

# %%
say(new_words("0"))

# %% [markdown]
# ### Two kinds of evaluation
#
# - **Model evaluation** asks: how good is this model, in general? Public exams, called benchmarks, give every
#   model the same questions, so models can be compared.
# - **Application evaluation** asks: how good is *my product* at *my users'* tasks? There the exam is yours: real
#   inputs from your users, the answers you expect, and a grading rule you choose. You run it again every time you
#   change the prompt, the model or a setting.
#
# This notebook grades one model on a public benchmark, because anyone can repeat it. Every day ends with how
# the same idea works for your own application.

# %% [markdown]
# ### How AI engineers use evaluation
#
# The numbers matter because of what you do with them. A team that builds with LLMs runs the same loop again and
# again: write the exam down, run the model or the app, grade and score, read the failures, change one thing, and
# run the same exam again to see whether the change helped. That is how evaluation improves an application: one
# measured change at a time. Here is where this notebook does each step:
#
# | Step | What it means | Where |
# |---|---|---|
# | 1. Write the exam down | inputs, expected answers, grading rule | Day 1 |
# | 2. Run the model or the app | the same settings every time | Day 2 |
# | 3. Grade and score | the share right, and how sure we are | Day 2 |
# | 4. Read the failures | the model, the grader, the key or the question? | Day 3 |
# | 5. Change one thing | the prompt, the model, a setting or the grader | Day 4 (a new prompt) |
# | 6. Run the same exam again | compare, question by question | Day 5 |

# %% [markdown]
# ### What we will measure
#
# A metric turns graded answers into one number that answers one question about the model. These are the
# metrics of this notebook. You do not need to understand them yet: each one gets its own day.

# %%
show_table(tables.metrics_map())

# %% [markdown]
# **If you build LLM features.** Write the exam down *before* you change anything: real inputs from your users,
# the answers you expect, a grading rule, and the number you will report. A small set is a fine start. Run it
# again after every change.
#
# **Try it.** Think of a feature you use, such as a chatbot or a summarizer. Write three real inputs and the
# answer you would expect for each. That is the start of an application exam. Which of the three would be hard
# to grade?

# %% [markdown]
# ## Day 1 · I gave an LLM a test. What does its score actually mean?
#
# **What we're doing.** A benchmark is a standardized exam for AI models. It has four parts: the questions, an
# answer key, a grading rule and a score. Change any one of them and the score can change. Our exam is
# **GSM8K**, a public set of grade-school math word problems. We use a random sample of 100 of its test
# questions, the same 100 in every experiment (Days 4 and 5 use modified versions of them).

# %%
say(new_words("1"))

# %% [markdown]
# Three real questions from our sample, with their answer keys. (GSM8K # is the question's row in the test
# set, counting from 0.)

# %%
show_table(tables.sample_questions(summary, 3))

# %% [markdown]
# Each answer key is a short worked solution written by people. This is the one for the first question:

# %%
say(tables.worked_solution(FIRST))

# %% [markdown]
# On that question, the four parts are: the question itself; its answer key, the number after `####` in the
# solution above; the grading rule, "find the number after `####` in the model's answer and compare it with the
# key"; and the score, the share of the 100 questions graded right, with its margin of error (Day 2).
#
# **If you build LLM features.** Write the four parts down *before* you look at the model's answers: which
# questions, which answer key, which grading rule, and which number you will report. Otherwise you will be
# tempted to move the goalposts after seeing the results.
#
# **Try it.** Change the `3` in `tables.sample_questions(summary, 3)` to `10` to read more questions.

# %% [markdown]
# ## Day 2 · Should I be impressed by the score?
#
# **What we're doing.** We hand the exam to the model one question at a time, read its final answer and
# compare it with the key. Like a student's grade on one exam, the model's grade is an estimate. What we want is
# its **true score**: the share of all questions of this kind that this exact setup (model, prompt, settings,
# grading rule) would get right. We asked only some of them, so the score comes with a margin of error.

# %%
say(new_words("2a"))

# %% [markdown]
# First, one question, exactly as we send it. The instruction is our prompt, P0:

# %%
live.show(live=LIVE_DEMO)

# %% [markdown]
# ### The result
#
# All 100 questions went through the same steps. First the words to read the score, then the score:

# %%
say(new_words("2b"))
say(tables.score_sentence(summary))

# %% [markdown]
# The score and its 95% range:

# %%
figures.fig_score(summary).show()
say(tables.score_plain(summary))

# %% [markdown]
# *That box is the idea of Day 2. The rest of the day shows how the range is computed and lists the settings of the
# exam: read it when you need the details.*
#
# **What the range covers, and what it does not.** It covers luck: which questions happened to be drawn, and
# which answer the model happened to give to each (each question was asked once). It does not cover mistakes
# in the answer key (Day 3), questions the model may have seen during training (we cannot test that here; see
# *Common questions* below), or the served model changing over time. And it belongs to this setup: with
# more reasoning effort, another prompt or another provider, the same model can score differently.
#
# How wide the range is for an exam of any size:

# %%
figures.fig_margin(summary).show()

# %% [markdown]
# Why is there still a range when every test question is used? Because the true score is about every
# GSM8K-style question, and the test set itself is one sample of those. And why only 100 questions here?

# %%
say(tables.full_set_cost(summary))

# %% [markdown]
# ### The rules of this exam
#
# The run card. Each line is a setting that can move the score, or a fact you need to repeat the run:

# %%
show_table(tables.run_card(summary, EXAM))

# %% [markdown]
# **If you build LLM features.** Never report a score without its 95% range and the number of questions behind
# it. If the gap between two scores is smaller than either one's margin of error (the ± part), these numbers
# alone cannot tell which is better, and somewhat bigger gaps can still be luck. To compare two versions, give
# them the same questions and count the questions only one of them got right (Day 5).
#
# **Try it.** Run `from src import stats` and then `stats.wilson(9, 10)`, `stats.wilson(90, 100)` and
# `stats.wilson(360, 400)`: the same 90% from 10, 100 and 400 questions. Watch the range shrink.

# %% [markdown]
# ### Under the hood (optional): how the harness runs the exam
#
# Skip this part if you only want the ideas; Day 3 does not depend on it.

# %%
say(new_words("2x"))

# %%
say(tables.sample_card(summary))

# %% [markdown]
# An exam is described by a task file. Ours, part by part (Day 1's four parts, plus how each question is
# asked):

# %%
show_table(tables.task_outline())

# %% [markdown]
# The file itself, `tasks/gsm8k_seed100.yaml` ("seed100" means a seeded sample of 100 questions).
# `process_docs` keeps our sampled questions and checks each one against the fingerprint saved for it in
# `data/sample_ids.json`; the harness's own `--limit 100` would take the *first* 100 questions instead.

# %%
print(tables.task_yaml())

# %% [markdown]
# This is **not** the harness's built-in `gsm8k` task. We kept its grading rules word for word, but the
# built-in task asks "Question: ... Answer:", shows the model 5 solved examples first, stops it at the word
# "Question:" and, through an API, allows only 256 output tokens unless told otherwise. So our score is
# comparable with the other experiments in this notebook, not with leaderboard numbers.
#
# Every answer is saved by the harness with how it was graded. The first question, as logged:

# %%
show_table(tables.logged_sample(summary, FIRST))

# %% [markdown]
# The harness's own report:

# %%
print(tables.harness_output())
say(tables.harness_side_note(summary))

# %% [markdown]
# The same exam can be run from the command line with the harness alone. Groq accepts the same request format
# as OpenAI, so the harness's OpenAI-style client works: it reads the key from `OPENAI_API_KEY`, and nothing is
# sent to OpenAI. On macOS or Linux:
#
# ```bash
# OPENAI_API_KEY=<your Groq key> python -m lm_eval run --model local-chat-completions \
#   --model_args model=openai/gpt-oss-20b,base_url=https://api.groq.com/openai/v1/chat/completions,num_concurrent=1,max_retries=10,seed=2026 \
#   --include_path tasks --tasks gsm8k_seed100 --apply_chat_template \
#   --gen_kwargs reasoning_effort=low --log_samples --output_path results/plain_cli
# ```
#
# On Windows (PowerShell), set the variables first, then write the command on one line:
#
# ```powershell
# $env:OPENAI_API_KEY = "<your Groq key>"; $env:PYTHONUTF8 = "1"
# python -m lm_eval run --model local-chat-completions --model_args model=openai/gpt-oss-20b,base_url=https://api.groq.com/openai/v1/chat/completions,num_concurrent=1,max_retries=10,seed=2026 --include_path tasks --tasks gsm8k_seed100 --apply_chat_template --gen_kwargs reasoning_effort=low --log_samples --output_path results/plain_cli
# ```
#
# This project runs the same exam with `python -m src.run --run r1_c0` (`r1_c0` is the name of the Day 2 run).
# It sends the same requests and adds four helpers: it paces requests to stay under Groq's limits, logs every
# request with its token count, refuses to start a run that would use more than half of the daily quota, and
# saves complete answers so that re-running costs nothing on the same machine. One grading detail differs: an
# answer cut off by the token limit is graded as invalid instead of on its partial text.

# %%
# Optional: runs only if RUN_LIVE = True (needs a Groq key; uses your own quota; replaces the saved results)
if RUN_LIVE:
    summary = rerun(EXAM)

# %% [markdown]
# ### Common questions
#
# - **Why does a short message cost more tokens than it has words?** Tokens are often parts of words, and Groq
#   adds gpt-oss's chat format (a short header) before the model reads the message.
# - **Can I read the model's reasoning?** Yes. Groq returns it in a separate field, and every response is saved
#   in `results/<run>/raw_requests.jsonl`. The grader reads only the visible answer.
# - **Could the model have seen these questions in training?** Possibly: GSM8K has been public since 2021. That
#   is called contamination, and this notebook cannot test it. Day 5's extra-sentence test checks robustness, not
#   memorization. One way to test it is to write fresh questions of the same kind: GSM1k (Zhang et al., 2024)
#   did that, and some models scored lower on the fresh questions.
# - **Why not compare with this model's published GSM8K score?** Published scores use other prompts, other
#   numbers of solved examples and other settings, so they measure a different exam.

# %% [markdown]
# ## Day 3 · The model was right. The evaluator said it was wrong.
#
# **What we're doing.** A grade is only as good as the grading. Today I read every answer the grader marked
# wrong and ask who was at fault: the model, the grading rule, the answer key or the question itself.
#
# First, the grading rule. The harness reads every answer in two ways: the *strict* rule reads the first number
# written right after `#### ` (four hash marks and a space), the format our prompt asked for; the *flexible*
# rule reads the last number anywhere in the visible answer. Each reading is then compared with the key **as
# text**. A teacher who reads only the answer box and one who reads the whole page can give different grades,
# so we check whether they did.

# %%
say(new_words("3"))

# %% [markdown]
# How the two rules behave, on short answers we wrote to show each case (some look like real ones you will
# meet below), graded by the harness's own code:

# %%
show_table(tables.grading_demo())

# %% [markdown]
# Neither rule is always kinder: a note after the answer fools the flexible rule, a dollar sign fools the
# strict one, and both reject `8.00` when the key says `8`. What happened in our run:

# %%
say(tables.rules_agreement(summary))
show_table(tables.errors(summary))

# %% [markdown]
# ### Five real answers, as the grader reads them
#
# Picked by fixed rules, not by hand: the first clean answer, the first right number marked wrong, the first
# answer without the `####` line, the first where the two rules read different numbers, and the first whose
# answer key is wrong.

# %%
for card in tables.answer_cards(summary):
    say(card)

# %% [markdown]
# ### Reading every answer marked wrong
#
# The habit from Day 1 says the official score is fixed before the run, so it stays as it is. But we can still
# *read* the answers the grader marked wrong, one by one.

# %%
say(tables.audit_method(summary))
show_table(tables.audit(summary))

# %% [markdown]
# Who was at fault, in both directions:

# %%
figures.fig_audit(summary).show()

# %%
say(tables.platinum_story(summary))

# %% [markdown]
# The table and the chart below grade the same answers five ways. Each row changes one part of Day 1's
# exam: the grading rule, the answer key or the questions. The answers never change, so the gaps between rows
# are exact, not luck. The 95% range in the table is each score's own uncertainty about the true score; do not
# compare the ranges by eye.

# %%
show_table(tables.score_ladder(summary))
figures.fig_ladder(summary).show()

# %% [markdown]
# **If you build LLM features.** Read the answers your grader marks wrong, and spot-check some it marks right.
# A wrong mark can hide a right answer (8.00 is not 8, a wrong key, a question with two readings), and a right
# mark can hide a mistake when the key is wrong too. Fix the grader and the keys, grade the saved answers again,
# and report the new score next to the old one instead of replacing it.
#
# **Try it.** Grade your own answers: `show_table(tables.grading_demo([("#### 1,000", "1000"), ("#### -5", "-5")]))`.
# Can you write a right answer that fools both rules?

# %% [markdown]
# ## Day 4 · Would you trust an AI that admits it doesn't know?
#
# **What we're doing.** A good assistant says "I can't tell" when a question does not give enough information.
# GSM-Plus (Li et al., ACL 2024) has a version of each of our questions with one needed fact removed. We ask the
# model all 100 of them twice: with the prompt we used so far (P0), and with a prompt that adds one line, a
# permission to write `#### UNANSWERABLE` (P1). Then we check that the permission has no bad side effect.

# %%
say(new_words("4"))

# %% [markdown]
# A question and its twin with one fact removed:

# %%
show_table(tables.missing_fact_pairs(summary, 2))

# %% [markdown]
# P1 is P0 plus this line: *If the problem does not give enough information to compute a single answer, write
# on the last line: #### UNANSWERABLE*.
#
# **How we count.** A simple rule reads the last line of each answer: a number, `UNANSWERABLE`, or neither.
# As a second check, an AI judge (`openai/gpt-oss-120b`) also reads the end of each answer (below).
#
# ### The result

# %%
say(tables.abstention_story(summary))
show_table(tables.abstention_table(summary))

# %% [markdown]
# The same counts as a chart, with the control run below (the original, solvable questions asked with P1; more
# on it in the next section):

# %%
figures.fig_abstention(summary).show()

# %% [markdown]
# ### Side effect: does the permission make it refuse questions it can answer?
#
# The control run asks the 100 original, solvable questions with P1.

# %%
say(tables.control_sentence(summary))

# %% [markdown]
# ### A second check: an AI judge (optional)
#
# A rule on the last line misses an answer that says "cannot be determined" in words, so an AI judge read the end
# of every answer too. It is a bigger model of the same family as the model under test, so treat it as a second
# opinion, not as the truth.

# %%
say(tables.judge_agreement(summary))

# %% [markdown]
# Every pair of labels, counted:

# %%
show_table(tables.rule_vs_judge_table(summary))

# %% [markdown]
# The answers where they differ, with the end of each answer as the judge saw it. Read them and decide who is
# right (some are borderline: an answer can say "I cannot determine this" and still end with a number):

# %%
show_table(tables.disagreements_table(summary))

# %% [markdown]
# **If you build LLM features.** Test the "I don't know" path on purpose: add inputs your app should decline and
# measure how often it still answers. Check that permission to decline does not hurt the normal inputs. And before
# you rely on an LLM judge, compare it with a person's labels on a sample (we did not do that here).

# %% [markdown]
# ## Day 5 · My model improved. Or did it?
#
# **What we're doing.** Two checks, on the same 100 questions. First, GSM-Plus has a version of each question
# with one irrelevant sentence added, with numbers in it: does the score survive? Second, we ask the original
# questions again with nothing changed, to learn how much the answers move on their own. A difference only
# means something if it is bigger than that.

# %%
say(new_words("5"))

# %% [markdown]
# A question and its twin with one irrelevant sentence added:

# %%
show_table(tables.distractor_pairs(summary, 2))

# %% [markdown]
# ### The result

# %%
say(tables.distractor_story(summary))
say(tables.retest_story(summary))
show_table(tables.flips_table(summary))

# %% [markdown]
# The questions that flipped, in each direction, for the two Day 5 checks and for Day 4's new prompt P1 (asked
# on the original questions). Each row compares a run with the Day 2 run, question by question; the first row is
# the noise floor (the same prompt and questions, asked again):

# %%
figures.fig_flips(summary).show()

# %% [markdown]
# Counting flips is not enough: is the change in the score bigger than luck? Resampling the same questions many
# times (a paired bootstrap) gives each change a 95% range. If the range includes zero (touches or crosses it),
# the data cannot rule out "no effect at all". The first row's range starts exactly at zero because only one
# question flipped there, and only in one direction: still consistent with no change.

# %%
figures.fig_paired_diff(summary).show()
say(tables.paired_plain(summary))

# %% [markdown]
# **If you build LLM features.** Compare two versions on the same inputs and count the inputs that flipped, in
# each direction. Know your noise floor first: run the *same* version twice. And remember what the range from
# Day 2 does not cover: a benchmark that has been public for years may be in a model's training data, so a good
# score on it is not proof of the skill you care about.

# %% [markdown]
# ## All the experiments
#
# Every experiment in one table. Prompt P0 is the instruction you saw on Day 2; P1 adds permission to answer
# UNANSWERABLE.

# %%
show_table(tables.results_table(summary))

# %% [markdown]
# Every run that has a right answer to score, side by side. Look at how much the ranges overlap: that is why Day 5
# compares runs question by question instead of by their scores alone.

# %%
figures.fig_all_runs(summary).show()

# %% [markdown]
# What each run cost, in tokens per question. gpt-oss "thinks" before it answers; that hidden reasoning is
# billed as output too:

# %%
figures.fig_tokens(summary).show()

# %% [markdown]
# ## Using this at work: a checklist
#
# Every day ended with "If you build LLM features". Here they are in one place, in the order you would use them:
#
# 1. **Write the exam down first**: real inputs from your users, the answer you expect for each, a grading rule
#    and the number you will report. Do it *before* you look at any answers (Day 1).
# 2. **Report every score with its size and its range**: "91/100, 95% range 84% to 95%", never "91%" alone (Day 2).
# 3. **Read the failures**: open the answers marked wrong, and a few marked right. The grader or the answer key can
#    be the problem, not the model (Day 3).
# 4. **Test the "I don't know" path on purpose**: add inputs your app should decline, measure how often it still
#    answers, and check that the permission to decline does not hurt the normal inputs (Day 4).
# 5. **Use a second grader as a pointer, not as the truth**: an LLM judge shows you where to look; read the
#    cases where it and your rule disagree (Day 4).
# 6. **Compare versions on the same inputs**: count the inputs that flipped in each direction, and only call it an
#    improvement when the range of the difference stays clear of zero (Day 5).
# 7. **Know your noise**: run the same version twice. And remember that a small exam cannot detect small changes.
#
# **A tiny exam for your own app**, to make it concrete. Say you build a support chatbot for an online shop:
#
# | Input (from real users) | Expected answer | How to grade |
# |---|---|---|
# | "Can I return shoes after 40 days?" | No: the limit is 30 days | Says no and mentions 30 days |
# | "Where is my order #A123?" | Asks to log in; never invents a status | Does not state a delivery date |
# | "Do you ship to Mars?" | Politely declines | Declines; no made-up shipping price |
# | "How much is shipping to Lisbon?" | The price from the shipping table | The number matches the table |
# | "Refund 2 of my 3 items, please" | Explains a partial refund | Mentions both kept and refunded items |
#
# Five rows are a start. Grow it with every failure you see in real use, and run it after every change to the
# prompt, the model or the documents behind the bot.
