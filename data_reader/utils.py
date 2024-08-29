from enum import Enum
import pandas as pd


class PAGE(Enum):
    MAIN = 1
    CHAT = 2
    STATS = 3
    VISUAL = 4


def read_file(file):
    if file.type == "application/jsonl" or file.type == "application/json":
        text = pd.read_json(file, lines=True)
    elif file.type == "text/csv":
        text = pd.read_csv(file)
    elif file.type == "application/xml" or file.type == "test/xml":
        text = pd.read_xml(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file.type == "application/vnd.ms-excel":
        text = pd.read_excel(file)
    else:
        return None

    return text
