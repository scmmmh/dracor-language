"""Analysis functionality."""

import os
from collections import Counter
from csv import DictReader

import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from lxml import etree
from pandas import DataFrame, Series
from scipy.stats import chi2_contingency
import seaborn

from language_detection.detector import text_for_node
from language_detection.processing import get_speaker, lines_for_tree, NAMESPACES
from language_detection.util import get_filename_year_mapping


def kdeplot(data: Series, label: str):
    """Generate a KDE plot."""
    np_data = data.to_numpy().copy()
    np_data.sort()
    kde = FFTKDE(kernel="gaussian", bw="ISJ").fit(np_data).evaluate()
    plt.plot(kde[0], kde[1], label=label)


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

    # Generate snippet length numbers
    print("----------------------")
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        max_snippet_length = 0
        for row in data:
            if row["file"] == work_filename and row["is_multilingual"] == "yes":
                max_snippet_length = max(
                    max_snippet_length, len(row["line"].split(" "))
                )
        data_points.append({"year": year, "max_snippet_length": max_snippet_length})
    df = DataFrame(data_points)
    for threshold in range(1, 11):
        works_count = df.max_snippet_length[df.max_snippet_length >= threshold].count()
        print(
            f"Min {threshold} words: {works_count} ({works_count / total_works_count:.2})"
        )

    # Generate snippet count numbers
    print("----------------------")
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        multilingual_lines = []
        for row in data:
            if (
                row["file"] == work_filename
                and row["is_multilingual"] == "yes"
                and row["line"] not in multilingual_lines
            ):
                multilingual_lines.append(row["line"])
        data_points.append(
            {"year": year, "multilingual_lines": len(multilingual_lines)}
        )
    df = DataFrame(data_points)
    for threshold in range(1, 11):
        works_count = df.multilingual_lines[df.multilingual_lines >= threshold].count()
        print(
            f"Min {threshold} lines: {works_count} ({works_count / total_works_count:.2})"
        )


def generate_snippet_accuracy_data(corpus: str, filename: str):
    print("----------------------")
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
    # Generate graph for distribution of multilingual texts over time
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

    kdeplot(df.year, "All works")
    kdeplot(df.year[df.is_multilingual], "Multilingual works")
    plt.title("Distribution of works over time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(os.path.join("analysis", corpus, "distribution-works-time.png"))
    plt.close()

    # Generate graphs for distribution of multilingual texts over time, for different snippet lengths
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        max_snippet_length = 0
        for row in data:
            if row["file"] == work_filename and row["is_multilingual"] == "yes":
                max_snippet_length = max(
                    max_snippet_length, len(row["line"].split(" "))
                )
        data_points.append({"year": year, "max_snippet_length": max_snippet_length})
    df = DataFrame(data_points)
    kdeplot(df.year, label="All works")
    for threshold in range(1, 11):
        kdeplot(
            df.year[df.max_snippet_length >= threshold],
            label=f"Multilingual (min {threshold})",
        )
    plt.title("Distribution of works over time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(
        os.path.join(
            "analysis",
            corpus,
            "distribution-works-time-min-words.png",
        )
    )
    plt.close()

    # Generate graphs for distribution of multilingual texts over time, for different snippet counts
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        multilingual_lines = []
        for row in data:
            if (
                row["file"] == work_filename
                and row["is_multilingual"] == "yes"
                and row["line"] not in multilingual_lines
            ):
                multilingual_lines.append(row["line"])
        data_points.append(
            {"year": year, "multilingual_lines": len(multilingual_lines)}
        )
    df = DataFrame(data_points)
    kdeplot(df.year, label="All works")
    for threshold in range(1, 11):
        kdeplot(
            df.year[df.multilingual_lines >= threshold],
            label=f"Multilingual (min {threshold})",
        )
    plt.title("Distribution of works over time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(
        os.path.join("analysis", corpus, "distribution-works-time-min-lines.png")
    )
    plt.close()

    # Generate distribution of multilingual works relative to confidence threshold
    data_points = []
    for threshold in range(100, 70, -1):
        multilingual = 0
        for work_filename, year in filename_year_mapping.items():
            is_multilingual = False
            for row in data:
                if (
                    row["file"] == work_filename
                    and row["is_multilingual"] == "yes"
                    and float(row["confidence"]) >= (threshold / 100)
                ):
                    is_multilingual = True
                    break
            if is_multilingual:
                multilingual = multilingual + 1
        data_points.append({"threshold": threshold, "count": multilingual})
    df = DataFrame(data_points)
    df.plot(x="threshold", y="count", label="Multilingual texts")
    plt.savefig(os.path.join("analysis", corpus, "work-threshold-count.png"))
    plt.close()

    # Generate distribution of incorrect multilingual works relative to confidence threshold
    data_points = []
    for threshold in range(100, 70, -1):
        multilingual = 0
        for work_filename, year in filename_year_mapping.items():
            is_multilingual = False
            is_false_multilingual = False
            for row in data:
                if (
                    row["file"] == work_filename
                    and row["is_multilingual"] == "yes"
                    and float(row["confidence"]) >= (threshold / 100)
                ):
                    is_multilingual = True
                if (
                    row["file"] == work_filename
                    and row["is_multilingual"] == "no"
                    and float(row["confidence"]) >= (threshold / 100)
                ):
                    is_false_multilingual = True
            if not is_multilingual and is_false_multilingual:
                multilingual = multilingual + 1
        data_points.append({"threshold": threshold, "count": multilingual})
    df = DataFrame(data_points)
    df.plot(x="threshold", y="count", label="Incorrect Multilingual texts")
    plt.savefig(os.path.join("analysis", corpus, "work-threshold-error-count.png"))
    plt.close()

    # Generate graph for distribution of language use over time
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    filename_year_mapping = get_filename_year_mapping(corpus)
    data_points = []
    for work_filename, year in filename_year_mapping.items():
        work = {
            "year": year,
            "has_english": False,
            "has_french": False,
            "has_italian": False,
            "has_latin": False,
            "has_spanish": False,
        }
        for row in data:
            if row["file"] == work_filename and row["is_multilingual"] == "yes":
                if row["language"] == "Language.ENGLISH":
                    work["has_english"] = True
                elif row["language"] == "Language.FRENCH":
                    work["has_french"] = True
                elif row["language"] == "Language.ITALIAN":
                    work["has_italian"] = True
                elif row["language"] == "Language.LATIN":
                    work["has_latin"] = True
                elif row["language"] == "Language.SPANISH":
                    work["has_spanish"] = True
        data_points.append(work)
    df = DataFrame(data_points)
    kdeplot(df.year, label="All works")
    kdeplot(df.year[df.has_english], label="English use")
    kdeplot(df.year[df.has_french], label="French use")
    kdeplot(df.year[df.has_italian], label="Italian use")
    kdeplot(df.year[df.has_latin], label="Latin use")
    kdeplot(df.year[df.has_spanish], label="Spanish use")
    plt.title("Distribution of works over time")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(os.path.join("analysis", corpus, "distribution-languages-time.png"))
    plt.close()


def generate_multilingual_speaker_distribution(corpus: str, filename: str):
    """Generate a distribution of the number of unique speakers per work."""
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    speakers = {}
    for row in data:
        if not row["is_multilingual"]:
            continue
        with open(os.path.join(corpus, "tei", row["file"])) as in_f:
            doc = etree.parse(in_f)
        all_lines = list(lines_for_tree(doc.getroot()))
        for line in all_lines:
            if text_for_node(line) == row["line"]:
                speaker = get_speaker(line)
                if speaker is not None:
                    if row["file"] not in speakers:
                        speakers[row["file"]] = set()
                    speakers[row["file"]].add(text_for_node(speaker))
    counts = Counter([len(values) for values in speakers.values()])
    seaborn.barplot(
        x=[item[0] for item in counts.most_common()],
        y=[item[1] for item in counts.most_common()],
    )
    plt.title("Distribution of number of multilingual speakers")
    plt.xlabel("Number of unique multilingual speakers")
    plt.xlabel("Number of multilingual texts")
    plt.savefig(
        os.path.join("analysis", corpus, "distribution-multilingual-speakers.png")
    )
    plt.close()
    print(f"Number of speaker counts: {counts.most_common()}")


def generate_work_type_distribution(corpus: str, filename: str):
    """Generate a distribution of the number of multilingual works per work category."""
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    multilingual_works = set()
    for row in data:
        if not row["is_multilingual"]:
            continue
        multilingual_works.add(row["file"])
    data = []
    for filename in os.listdir(os.path.join(corpus, "tei")):
        if filename.endswith(".xml"):
            with open(os.path.join(corpus, "tei", filename)) as in_f:
                doc = etree.parse(in_f)
                genre = doc.xpath(
                    "//tei:term[@type='genreTitle']/text()", namespaces=NAMESPACES
                )
                if len(genre) > 0:
                    data.append({"genre": genre[0], "kind": "All"})
                else:
                    data.append({"genre": "Unspecified", "kind": "All"})
    for filename in multilingual_works:
        with open(os.path.join(corpus, "tei", filename)) as in_f:
            doc = etree.parse(in_f)
            genre = doc.xpath(
                "//tei:term[@type='genreTitle']/text()", namespaces=NAMESPACES
            )
            if len(genre) > 0:
                data.append({"genre": genre[0], "kind": "Multilingual"})
            else:
                data.append({"genre": "Unspecified", "kind": "Multilingual"})
    df = DataFrame(data)
    seaborn.catplot(data=df, x="genre", hue="kind", kind="count")
    plt.title("Distribution per genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of works")
    plt.savefig(os.path.join("analysis", corpus, "distribution-works-genre.png"))
    plt.close()
