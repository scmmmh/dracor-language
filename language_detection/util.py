import json
import os

from lxml import etree

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}


def json_file_cache(cache_filename: str):
    def decorator(func):
        def wrapper(corpus: str, *args, **kwargs):
            if os.path.exists(os.path.join("results", corpus, cache_filename)):
                with open(os.path.join("results", corpus, cache_filename)) as in_f:
                    return json.load(in_f)
            tmp = func(corpus, *args, **kwargs)
            with open(os.path.join("results", corpus, cache_filename), "w") as out_f:
                json.dump(tmp, out_f)
            return tmp

        return wrapper

    return decorator


@json_file_cache("_filenames_list.json")
def get_filenames(corpus: str) -> list[str]:
    filenames = []
    for work_filename in os.listdir(os.path.join(corpus, "tei")):
        if work_filename.endswith(".xml"):
            filenames.append(work_filename)
    return filenames


@json_file_cache("_filename_year_mapping.json")
def get_filename_year_mapping(corpus: str) -> dict[str, int]:
    """Get the filename - year distribution"""
    file_year_map = {}
    for work_filename in get_filenames(corpus):
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
    return file_year_map
