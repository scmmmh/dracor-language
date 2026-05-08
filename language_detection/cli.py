from typing import Literal

from typer import Typer

from language_detection.analysis import (
    generate_basic_statistics,
    generate_snippet_accuracy_data,
    generate_temporal_distribution,
)
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
    generate_basic_statistics(corpus, filename)
    generate_snippet_accuracy_data(corpus, filename)
    generate_temporal_distribution(corpus, filename)
