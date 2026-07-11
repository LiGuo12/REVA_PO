import json
import csv
from typing import Union, Optional


def jsonl_to_csv(jsonl_path: str, csv_path: str, encoding: str = "utf-8") -> int:
    """
    Convert a JSONL file (one JSON object per line) into a CSV with columns: id, pred, gt.

    Each line should look like: {"id": ..., "pred": ..., "gt": ...}

    Returns:
        number of rows written (excluding header).
    """
    n = 0
    with open(jsonl_path, "r", encoding=encoding) as fin, open(csv_path, "w", newline="", encoding=encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "pred", "gt"])
        writer.writeheader()

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

            # Be tolerant to missing keys, but keep them explicit
            row = {
                "id": obj.get("id", ""),
                "pred": obj.get("pred", ""),
                "gt": obj.get("gt", ""),
            }

            # If pred/gt are not strings (e.g., list/dict), dump them into JSON string
            for k in ("pred", "gt"):
                if isinstance(row[k], (dict, list)):
                    row[k] = json.dumps(row[k], ensure_ascii=False)

            writer.writerow(row)
            n += 1

    return n

rows = jsonl_to_csv("/mnt/ssd4tb/code/code/checkpoint/stage_3/iu_xray/new/result/test_final_test_epoch0009.jsonl", "/mnt/ssd4tb/code/code/checkpoint/stage_3/iu_xray/new/result/test_final_test_epoch0009.csv")
print("written rows:", rows)