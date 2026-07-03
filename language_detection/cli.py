from typing import Literal

from typer import Typer

from language_detection import analysis, processing
from language_detection.classification import merge_manual_classifications
from language_detection.detector import extract_foreign_language_snippets
from language_detection.processing import (
    extract_above_threshold,
    extract_random_temporal_sample,
)

app = Typer(help="Language detection in Dracor corpora")


@app.command()
def extract_foreign_snippets(corpus: str):
    """Extract all foreign language snippets from the texts in the corpus."""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    extract_foreign_language_snippets(corpus)


@app.command()
def merge_classification(corpus: str):
    """Merge all manual classification data."""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    merge_manual_classifications(corpus)


@app.command()
def extract_threshold_subset(
    corpus: str, lower_threshold: float, upper_threshold: None | float = None
):
    """Extract all foreign language snippets within given thresholds."""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    extract_above_threshold(corpus, lower_threshold, upper_threshold)


@app.command()
def extract_random_sample(
    corpus: str, size: int, filter: None | Literal["not-multilingual"] = None
):
    """Extract all foreign language snippets within given thresholds."""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    extract_random_temporal_sample(corpus, size, filter)


@app.command()
def calculate_statistics(corpus: str, filename: str):
    """Calculate the statistics for the file in the corpus."""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    analysis.generate_basic_statistics(corpus, filename)
    analysis.generate_snippet_accuracy_data(corpus, filename)
    analysis.generate_temporal_distribution(corpus, filename)
    analysis.generate_multilingual_speaker_distribution(corpus, filename)
    analysis.generate_work_type_distribution(corpus, filename)
    analysis.generate_multilingual_line_stats(corpus, filename)
    analysis.generate_gender_distribution(corpus, filename)


@app.command()
def extract_multilingual_snippets(corpus: str, filename: str, min_words: int = 0):
    """Extract multilingual snippets from texts"""
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    processing.extract_multilingual_snippets(corpus, filename, min_words)
