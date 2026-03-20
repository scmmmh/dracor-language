from typer import Typer

from language_detection.detector import calculate_stats

app = Typer(help="Language detection in Dracor corpora")


@app.command()
def detect_language(corpus: str):
    if corpus not in ("engdracor", "fredracor", "gerdracor"):
        return
    calculate_stats(corpus)
