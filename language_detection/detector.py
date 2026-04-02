"""Language detection functionality."""

import os
from csv import DictWriter
from multiprocessing import cpu_count, Pool

from lingua import Language, LanguageDetectorBuilder
from lxml import etree

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}

detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.ITALIAN,
    Language.LATIN,
).build()


def text_for_node(node):
    """Extract the text for a XML node."""
    if node.tag in ("{http://www.tei-c.org/ns/1.0}stage",):
        return ""
    buffer = []
    if node.text:
        buffer.append(node.text)
    for child in node:
        buffer.append(text_for_node(child))
        if child.tail:
            buffer.append(child.tail)
    return "".join(buffer)


def lines_for_tree(node):
    """Extract all the lines to be processed.

    These are either l or p tags."""
    if node.tag in ("{http://www.tei-c.org/ns/1.0}l", "{http://www.tei-c.org/ns/1.0}p"):
        text = text_for_node(node)
        if text.strip():
            yield text.strip()
    else:
        for child in node:
            for line in lines_for_tree(child):
                yield line


def fix_classification(language: Language, line: str):
    for marker in (
        "hihi",
        "hoho",
        "hiho",
        "hehehe",
        "ridir",
        "ridid",
        "hoiho",
        "hollodri",
        "dudeldumdei",
    ):
        if marker in line.lower():
            return Language.GERMAN
    return language


def identify_languages(
    corpus: str, filename: str, default_language: Language
) -> None | dict:
    """Identify the foreign language snippets in a single file."""
    if filename.endswith(".xml"):
        with open(os.path.join(corpus, "tei", filename)) as in_f:
            rows = []
            doc = etree.parse(in_f)
            root = doc.getroot()
            bodies = root.xpath("tei:text/tei:body", namespaces=NAMESPACES)
            if len(bodies) == 0:
                return None
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
            body = bodies[0]
            languages = set()
            buffer = []
            for line in lines_for_tree(body):
                confidences = detector.compute_language_confidence_values(line)
                if (
                    fix_classification(confidences[0].language, line)
                    != default_language
                ):
                    rows.append(
                        {
                            "language": confidences[0].language,
                            "confidence": confidences[0].value,
                            "line": line,
                            "file": filename,
                            "date": date["when"],
                            "date_type": date["type"],
                        }
                    )
                    languages.add(fix_classification(confidences[0].language, line))
                else:
                    languages.add(default_language)
                buffer.append(line)
            if detector.detect_language_of("".join(buffer)) == default_language:
                document_correct = True
            else:
                document_correct = False
            return {
                "languages": languages,
                "document_correct": document_correct,
                "foreign_texts": rows,
            }


def extract_foreign_language_snippets(corpus: str):
    """Extract all identified foreign language snippets."""
    if corpus == "engdracor":
        default_language = Language.ENGLISH
    elif corpus == "fredracor":
        default_language = Language.FRENCH
    elif corpus == "gerdracor":
        default_language = Language.GERMAN
    else:
        default_language = None
    confident_language_counts = []
    document_correct = 0
    document_incorrect = 0
    rows = []
    with Pool(max(cpu_count() - 1, 1)) as pool:
        for result in pool.starmap(
            identify_languages,
            [
                (corpus, filename, default_language)
                for filename in os.listdir(os.path.join(corpus, "tei"))
            ],
        ):
            if result is not None:
                if result["document_correct"]:
                    document_correct = document_correct + 1
                else:
                    document_incorrect = document_incorrect + 1
                rows.extend(result["foreign_texts"])
                confident_language_counts.append(len(result["languages"]))

    print(document_correct, document_incorrect)

    with open(os.path.join("results", corpus, "full.csv"), "w") as out_f:
        writer = DictWriter(
            out_f,
            fieldnames=["language", "confidence", "line", "file", "date", "date_type"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
