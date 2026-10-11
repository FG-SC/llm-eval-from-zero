"""Day 4 text and tables: PENDING without data, and every number taken from the saved counts."""
from src import tables


def score(k, n=100):
    return {"correct": k, "n": n, "accuracy": k / n, "ci95": [max(0, k / n - 0.1), min(1, k / n + 0.1)]}


def rule(number, abstained, neither):
    return {"answered_number": score(number), "abstained": score(abstained), "no_final_answer": score(neither)}


SUMMARY = {
    "runs": {"r3_c2_p0": {"n": 100, "rule_labels": rule(97, 0, 3),
                          "per_question": [{"orig_idx": 4, "answer_tail": "cannot say, so\n#### 5"}]},
             "r3_c2_p1": {"n": 100, "rule_labels": rule(14, 86, 0), "per_question": []}},
    "comparisons": {"c2_p0_to_p1": {"all": {"n": 100, "before_only": 0, "after_only": 86, "mcnemar_p": 1e-20},
                                    "without_flagged": {"n": 97, "rate_after": 86 / 97}, "excluded": [3, 9]}},
    "judge_results": {"runs": {
        "r3_c2_p0": {"rule_vs_judge": {"answered_number -> ANSWERED_NUMBER": 96, "answered_number -> ABSTAINED": 1,
                                       "no_final_answer -> OTHER": 3}},
        "r3_c2_p1": {"rule_vs_judge": {"abstained -> ABSTAINED": 86, "answered_number -> ANSWERED_NUMBER": 14}}},
        "disagreements": [{"run": "r3_c2_p0", "idx": 4, "rule": "answered_number", "judge": "ABSTAINED"}]},
}


def test_everything_is_pending_without_data():
    empty = {"runs": {}, "comparisons": {}}
    assert tables.abstention_story(empty) == tables.PENDING
    assert tables.control_sentence(empty) == tables.PENDING
    for table in (tables.abstention_table, tables.rule_vs_judge_table, tables.missing_fact_pairs,
                  tables.disagreements_table):
        assert tables.PENDING in table(empty).to_string()


def test_story_numbers_come_from_the_rule_counts():
    text = tables.abstention_story(SUMMARY)
    assert "97 of 100 answers" in text and "86 wrote UNANSWERABLE and 14 still gave a number" in text
    assert "86 switched to abstaining and 0 the other way" in text
    assert "#3 and #9" in text and "abstained on 88.7%" in text


def test_tables_have_one_row_per_prompt_and_pair():
    t = tables.abstention_table(SUMMARY)
    assert list(t["Prompt"]) == ["P0: no permission to abstain", "P1: may write UNANSWERABLE"]
    assert "97/100" in t.iloc[0]["Ended with a number"] and "86/100" in t.iloc[1]["Wrote UNANSWERABLE"]
    pairs = tables.rule_vs_judge_table(SUMMARY)
    assert len(pairs) == 5 and pairs["Answers"].sum() == 200


def test_disagreements_show_the_end_of_the_answer():
    t = tables.disagreements_table(SUMMARY)
    assert len(t) == 1 and t.iloc[0]["GSM8K #"] == 4 and t.iloc[0]["Judge says"] == "said it cannot be answered"
    assert t.iloc[0]["End of the answer (raw text)"].endswith("#### 5")
