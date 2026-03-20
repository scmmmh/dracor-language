import os
from collections import Counter

from lingua import Language, LanguageDetectorBuilder
from lxml import etree
from rich.progress import track

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}

detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
    Language.ITALIAN,
).build()


def text_for_node(node):
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
    if node.tag in ("{http://www.tei-c.org/ns/1.0}l", "{http://www.tei-c.org/ns/1.0}p"):
        text = text_for_node(node)
        if text.strip():
            yield text.strip()
    else:
        for child in node:
            for line in lines_for_tree(child):
                yield line


def calculate_stats(corpus: str):
    if corpus == "engdracor":
        default_language = Language.ENGLISH
    elif corpus == "fredracor":
        default_language = Language.FRENCH
    elif corpus == "gerdracor":
        default_language = Language.GERMAN
    else:
        default_language = None
    simple_language_counts = []
    confident_language_counts = []
    foreign_texts = []
    total_lines = 0
    document_correct = 0
    document_incorrect = 0
    for filename in track(
        os.listdir(os.path.join(corpus, "tei")), description="Processing files"
    ):
        if filename.endswith(".xml"):
            with open(os.path.join(corpus, "tei", filename)) as in_f:
                doc = etree.parse(in_f)
                root = doc.getroot()
                bodies = root.xpath("tei:text/tei:body", namespaces=NAMESPACES)
                if len(bodies) == 0:
                    continue
                body = bodies[0]
                languages = set()
                high_confidence_languages = set()
                buffer = []
                for line in lines_for_tree(body):
                    total_lines = total_lines + 1
                    confidences = detector.compute_language_confidence_values(line)
                    if confidences[0].value >= 0.8:
                        if confidences[0].language != default_language:
                            foreign_texts.append((confidences[0].language, line))
                        high_confidence_languages.add(confidences[0].language)
                    else:
                        high_confidence_languages.add(default_language)
                    languages.add(confidences[0].language)
                    buffer.append(line)
                if detector.detect_language_of("".join(buffer)) == default_language:
                    document_correct = document_correct + 1
                else:
                    document_incorrect = document_incorrect + 1
                simple_language_counts.append(len(languages))
                confident_language_counts.append(len(high_confidence_languages))

    print(Counter(simple_language_counts))
    print(document_correct, document_incorrect)
    print(Counter(confident_language_counts))
    print(len(foreign_texts), total_lines)
