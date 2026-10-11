"""Day 5 text and tables: PENDING without data, and every number taken from the comparisons."""
from src import tables


def score(k, n):
    return {"correct": k, "n": n, "accuracy": k / n, "ci95": [0.8, 0.95]}


def pair(before, after, diff, n=100):
    return {"n": n, "before_only": before, "after_only": after, "mcnemar_p": 0.25, "diff": diff,
            "diff_ci95_bootstrap": [diff - 0.05, diff + 0.05]}


SUMMARY = {"runs": {"r1_c0": {"scores": {"strict-match": score(91, 100)}},
                    "r2_c1": {"scores": {"strict-match": score(88, 100)}}},
           "comparisons": {
               "c0_to_c1": {"all": pair(6, 3, -0.03), "without_flagged": pair(5, 3, -0.02, 94), "excluded": [980, 981]},
               "c0_retest": {"n": 100, "answer_changed": score(7, 100), "grade_flipped": score(4, 100),
                             "right_to_wrong": 3, "wrong_to_right": 1}}}


def test_pending_without_data():
    empty = {"runs": {}, "comparisons": {}}
    assert tables.distractor_story(empty) == tables.PENDING
    assert tables.retest_story(empty) == tables.PENDING
    assert tables.PENDING in tables.flips_table(empty).to_string()
    assert tables.PENDING in tables.distractor_pairs(empty).to_string()


def test_distractor_story_uses_the_comparison():
    text = tables.distractor_story(SUMMARY)
    assert "88/100 = 88.0%" in text and "91/100 = 91.0%" in text
    assert "6 questions it got right before are now wrong, and 3 it got wrong before are now right" in text
    assert "difference -3.0 points" in text and "(#980 and #981)" in text and "5 lost, 3 gained" in text


def test_retest_story_and_flips_table():
    text = tables.retest_story(SUMMARY)
    assert "7 final answers changed and 4 grades flipped (3 from right to wrong, 1 from wrong to right)" in text
    t = tables.flips_table(SUMMARY)
    assert list(t["Right → wrong"]) == [6, 3] and list(t["Wrong → right"]) == [3, 1]
    assert list(t["Net change in the score"]) == ["-3.0 points", "-2.0 points"]
