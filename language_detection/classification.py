"""Manual classification processing code."""

import os
from csv import DictReader, DictWriter


def merge_manual_classifications(corpus: str):
    """Merge all manual classification data into the full data-set."""
    with open(os.path.join("results", corpus, "full.csv")) as in_f:
        reader = DictReader(in_f)
        data = list(reader)
    for filename in os.listdir(os.path.join("results", corpus)):
        if filename.startswith("classified-multilingual-"):
            with open(os.path.join("results", corpus, filename)) as in_f:
                reader = DictReader(in_f)
                for line in reader:
                    for row in data:
                        if line["file"] == row["file"] and line["line"] == row["line"]:
                            if line["is_multilingual"] == "no":
                                row["is_multilingual"] = "no"
                            else:
                                row["is_multilingual"] = "yes"
    with open(os.path.join("results", corpus, "full.csv"), "w") as out_f:
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
        for row in data:
            writer.writerow(row)
