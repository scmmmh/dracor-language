"""Processing functionality."""

import os
from csv import DictReader, DictWriter


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
