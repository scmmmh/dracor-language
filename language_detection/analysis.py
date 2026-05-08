"""Analysis functionality."""

import os
from csv import DictReader


def generate_basic_statistics(corpus: str, filename: str):
    total_works_count = 0
    for work_filename in os.listdir(os.path.join(corpus, "tei")):
        if work_filename.endswith(".xml"):
            total_works_count = total_works_count + 1
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    correct_snippet_count = 0
    total_snippet_count = 0
    works = {}
    for row in data:
        if row["file"] not in works:
            works[row["file"]] = {
                "correct_snippet_count": 0,
                "total_snippet_count": 0,
            }
        if row["is_multilingual"] == "yes":
            correct_snippet_count = correct_snippet_count + 1
            works[row["file"]]["correct_snippet_count"] = (
                works[row["file"]]["correct_snippet_count"] + 1
            )
        total_snippet_count = total_snippet_count + 1
        works[row["file"]]["total_snippet_count"] = (
            works[row["file"]]["total_snippet_count"] + 1
        )
    works_completely_correct = 0
    works_partially_correct = 0
    for counts in works.values():
        if counts["correct_snippet_count"] > 0:
            if counts["correct_snippet_count"] == counts["total_snippet_count"]:
                works_completely_correct = works_completely_correct + 1
            else:
                works_partially_correct = works_partially_correct + 1

    print(f"Total number of snippets: {total_snippet_count}")
    print(
        f"Correctly identified snippets: {correct_snippet_count} ({correct_snippet_count / total_snippet_count})"
    )
    print(f"Total number of works: {len(works)}")
    print(f"Total number of works [complete corpus]: {total_works_count}")
    print(
        f"Correctly identified works: {works_completely_correct} ({works_completely_correct / len(works)})"
    )
    print(
        f"Correctly identified works: {works_completely_correct} ({works_completely_correct / total_works_count})"
    )
    print(
        f"Partially correctly identified works: {works_completely_correct + works_partially_correct} {(works_completely_correct + works_partially_correct) / len(works)}"
    )
    print(
        f"Partially correctly identified works [complete corpus]: {works_completely_correct + works_partially_correct} {(works_completely_correct + works_partially_correct) / total_works_count}"
    )
