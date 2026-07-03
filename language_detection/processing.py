"""Processing functionality."""

import os
from csv import DictReader, DictWriter
from random import sample
from typing import Literal

from lxml import etree
from scipy.stats import chi2_contingency

from language_detection.detector import text_for_node

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}


def extract_above_threshold(
    corpus: str, lower_threshold: float, upper_threshold: None | float = None
):
    with open(os.path.join("results", corpus, "full.csv")) as in_f:
        reader = DictReader(in_f)
        if upper_threshold is not None:
            filename = f"subset-{lower_threshold}-{upper_threshold}.csv"
        else:
            filename = f"subset-{lower_threshold}.csv"
        with open(os.path.join("results", corpus, filename), "w") as out_f:
            writer = DictWriter(
                out_f,
                fieldnames=[
                    "language",
                    "confidence",
                    "line",
                    "file",
                    "date",
                    "date_type",
                    "is_multilingual",
                ],
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            for row in reader:
                if float(row["confidence"]) >= lower_threshold and (
                    upper_threshold is None
                    or float(row["confidence"]) < upper_threshold
                ):
                    writer.writerow(row)


def extract_random_temporal_sample(
    corpus: str, size: int, filter: None | Literal["not-multilingual"] = None
) -> None:
    """Extract a random sample of works from the total sample of works.

    If the filter is set to "not-multilingual", then all works that have a multiingual text-snippet are excluded.

    The sampling is run 30 times and the one with the year distribution most similar to the overal corpus is output.
    """
    total_corpus = set()
    file_year_map = {}
    for work_filename in os.listdir(os.path.join(corpus, "tei")):
        if work_filename.endswith(".xml"):
            with open(os.path.join(corpus, "tei", work_filename)) as in_f:
                doc = etree.parse(in_f)
            root = doc.getroot()
            bodies = root.xpath("tei:text/tei:body", namespaces=NAMESPACES)
            if len(bodies) == 0:
                continue
            date = {}
            for event_type in ("written", "print", "premiere"):
                dates = root.xpath(
                    f'tei:standOff/tei:listEvent/tei:event[@type="{event_type}"]',
                    namespaces=NAMESPACES,
                )
                if len(dates) > 0:
                    date = {"type": dates[0].attrib["type"]}
                    if "when" in dates[0].attrib:
                        date["when"] = dates[0].attrib["when"]
                    elif "notAfter" in dates[0].attrib:
                        date["when"] = dates[0].attrib["notAfter"]
                    break
            year = int(date["when"][:4])
            file_year_map[work_filename] = year
            total_corpus.add(work_filename)
    if filter == "not-multilingual":
        with open(os.path.join("results", corpus, "full.csv")) as in_f:
            reader = DictReader(in_f)
            for line in reader:
                if line["is_multilingual"] == "yes" and line["file"] in total_corpus:
                    total_corpus.remove(line["file"])
    total_corpus = list(total_corpus)
    selected_corpus = None
    pvalue = 0
    for _ in range(0, 30):
        sample_corpus = sample(total_corpus, size)
        sample_years = {}
        corpus_years = {}
        for work_filename in sample_corpus:
            if file_year_map[work_filename] not in sample_years:
                sample_years[file_year_map[work_filename]] = 1
            else:
                sample_years[file_year_map[work_filename]] = (
                    sample_years[file_year_map[work_filename]] + 1
                )
        for work_filename in total_corpus:
            if file_year_map[work_filename] not in corpus_years:
                corpus_years[file_year_map[work_filename]] = 1
            else:
                corpus_years[file_year_map[work_filename]] = (
                    corpus_years[file_year_map[work_filename]] + 1
                )
        table = [[], []]
        for year in corpus_years.keys():
            table[0].append(corpus_years[year])
            if year in sample_years:
                table[1].append(sample_years[year])
            else:
                table[1].append(0)
        tmp_pvalue = chi2_contingency(table).pvalue
        if tmp_pvalue > pvalue:
            selected_corpus = sample_corpus
            pvalue = tmp_pvalue
    with open(os.path.join("results", corpus, "sample.csv"), "w") as out_f:
        writer = DictWriter(out_f, fieldnames=["file", "is_multilingual"])
        writer.writeheader()
        for work_filename in selected_corpus:
            writer.writerow({"file": work_filename, "is_multilingual": ""})


def lines_for_tree(node):
    """Extract all the lines to be processed.

    These are either l or p tags."""
    if node.tag in ("{http://www.tei-c.org/ns/1.0}l", "{http://www.tei-c.org/ns/1.0}p"):
        text = text_for_node(node)
        if text.strip():
            yield node
    else:
        for child in node:
            for line in lines_for_tree(child):
                yield line


def get_speaker(line):
    speaker = line.xpath("../tei:speaker", namespaces=NAMESPACES)
    if len(speaker) > 0:
        return speaker[0]
    else:
        speaker = line.xpath("../../tei:speaker", namespaces=NAMESPACES)
        if len(speaker) > 0:
            return speaker[0]
    return None


def extract_multilingual_snippets(corpus: str, filename: str, min_words: int):
    with open(os.path.join("results", corpus, filename)) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    snippets = []
    for row in data:
        if row["is_multilingual"] != "yes":
            continue
        if len(row["line"].split(" ")) < min_words:
            continue
        with open(os.path.join(corpus, "tei", row["file"])) as in_f:
            doc = etree.parse(in_f)
        all_lines = list(lines_for_tree(doc.getroot()))
        for line in all_lines:
            if text_for_node(line) == row["line"]:
                line_idx = all_lines.index(line)
                snippet = [
                    f"{text_for_node(get_speaker(all_lines[idx]))}: {text_for_node(all_lines[idx])}"
                    for idx in range(
                        max((0, line_idx - 3)), min((line_idx + 4, len(all_lines)))
                    )
                ]
                snippets.append(
                    {
                        "language": row["language"],
                        "confidence": row["confidence"],
                        "line": row["line"],
                        "file": row["file"],
                        "snippet": "\n".join(snippet),
                    }
                )
    with open(
        os.path.join("results", corpus, f"multilingual-snippets-{min_words}.csv"), "w"
    ) as out_f:
        writer = DictWriter(
            out_f, fieldnames=["language", "confidence", "line", "file", "snippet"]
        )
        writer.writeheader()
        for row in snippets:
            writer.writerow(row)
