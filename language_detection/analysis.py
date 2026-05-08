"""Analysis functionality."""

import os
from csv import DictReader

import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn

from language_detection.util import get_filename_year_mapping, get_filenames


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


def generate_snippet_accuracy_data(corpus: str, filename: str):
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    data_points = []
    for threshold in range(100, 0, -1):
        filtered = [row for row in data if float(row["confidence"]) >= threshold / 100]
        correct = [row for row in filtered if row["is_multilingual"] == "yes"]
        data_points.append(
            {
                "threshold": threshold / 100,
                "correct": len(correct) / len(filtered),
                "identified": len(correct),
            }
        )
        print(threshold, len(correct) / len(filtered))
        if len(filtered) == len(data):
            break
    df = DataFrame(data_points)
    df.plot(x="threshold", y="correct", label="Fraction correct")
    plt.savefig(os.path.join("analysis", corpus, "snippet-threshold-accuracy.png"))
    plt.close()
    df.plot(x="threshold", y="identified", label="Number of snippets identified")
    plt.savefig(os.path.join("analysis", corpus, "snippet-threshold-identified.png"))
    plt.close()


def generate_temporal_distribution(corpus: str, filename: str):
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        is_multilingual = False
        for row in data:
            if row["file"] == work_filename and row["is_multilingual"] == "yes":
                is_multilingual = True
                break
        data_points.append({"year": year, "is_multilingual": is_multilingual})
    df = DataFrame(data_points)
    seaborn.kdeplot(data=df, x="year", label="All works")
    seaborn.kdeplot(data=df[df.is_multilingual], x="year", label="Multilingual works")
    plt.title("Distribution of works over time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(os.path.join("analysis", corpus, "distribution-works-time.png"))
    plt.close()
