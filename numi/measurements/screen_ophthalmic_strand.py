#!/usr/bin/env python3
"""Execute the fixed 8-0 Nylon held-out screen from the pinned PLOS S4 workbook."""
import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

EXPECTED_SHA = "49a0f7f4875bc6354367c3efd832ec26642355965fe0cae31ffc950280045978"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLUMNS = [("A", "B"), ("D", "E"), ("G", "H"), ("J", "K"), ("M", "N")]
NAMES = [f"8-0NS{i}" for i in range(1, 6)]

def evaluate(series):
    for name, pairs in zip(NAMES, series):
        if len(pairs) < 20 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in pairs):
            raise ValueError(f"insufficient or nonfinite samples: {name}")
    train = series[0]
    mx = sum(x for x, _ in train) / len(train)
    my = sum(y for _, y in train) / len(train)
    denominator = sum((x - mx)**2 for x, _ in train)
    if not denominator:
        raise ValueError("development extensions have zero variance")
    slope = sum((x - mx)*(y - my) for x, y in train) / denominator
    intercept = my - slope * mx
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise ValueError("invalid development fit")
    heldout = []
    for name, pairs in zip(NAMES[1:], series[1:]):
        denom = sum(abs(y) for _, y in pairs)
        if denom == 0:
            raise ValueError(f"zero force denominator: {name}")
        error = sum(abs(intercept + slope*x - y) for x, y in pairs) / denom
        heldout.append({"specimen": name, "count": len(pairs),
                        "relative_absolute_error": error, "passed": error <= 0.10})
    return {"development_specimen": NAMES[0], "development_count": len(train),
            "intercept_N": intercept, "slope_N_per_mm": slope,
            "heldout": heldout, "passed": all(x["passed"] for x in heldout)}

def parse(workbook):
    raw = workbook.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_SHA:
        raise ValueError(f"workbook SHA mismatch: {observed}")
    with zipfile.ZipFile(workbook) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = wb.find(NS+"sheets")
        names = [s.get("name") for s in sheets]
        if len(names) < 4 or names[3] != "8-0 Nylon":
            raise ValueError("unexpected sheet ordering")
        strings = ["".join(item.itertext()) for item in
                   ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS+"si")]
        result = [[] for _ in COLUMNS]
        found = set()
        with z.open("xl/worksheets/sheet4.xml") as f:
            for _, row in ET.iterparse(f, events=("end",)):
                if row.tag != NS+"row":
                    continue
                number = int(row.get("r"))
                values = {}
                for c in row.findall(NS+"c"):
                    match = re.fullmatch(r"([A-Z]+)([0-9]+)", c.get("r", ""))
                    if not match or int(match[2]) != number:
                        raise ValueError("invalid cell coordinate")
                    col = match[1]
                    if col not in {v for pair in COLUMNS for v in pair}:
                        continue
                    if c.find(NS+"f") is not None:
                        raise ValueError("formula in selected cells")
                    v = c.find(NS+"v")
                    if v is not None:
                        values[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
                if number <= 3:
                    for i, (xcol, ycol) in enumerate(COLUMNS):
                        if number == 1 and values.get(xcol) != NAMES[i]:
                            raise ValueError("unexpected specimen identity")
                        if number == 2 and (values.get(xcol), values.get(ycol)) != ("Extension", "Load"):
                            raise ValueError("unexpected measurement names")
                        if number == 3 and (values.get(xcol), values.get(ycol)) != ("(mm)", "(N)"):
                            raise ValueError("unexpected units")
                    found.add(number)
                else:
                    for i, (xcol, ycol) in enumerate(COLUMNS):
                        x, y = values.get(xcol), values.get(ycol)
                        if (x is None) != (y is None):
                            raise ValueError(f"unpaired values at row {number}")
                        if x is not None:
                            x, y = float(x), float(y)
                            if not math.isfinite(x) or not math.isfinite(y):
                                raise ValueError(f"nonfinite value at row {number}")
                            if 2.0 <= x <= 4.0:
                                result[i].append((x, y))
                row.clear()
        if found != {1, 2, 3}:
            raise ValueError("missing headers")
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = evaluate(parse(args.workbook))
    result.update({"workbook_sha256": EXPECTED_SHA,
                   "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "interval_mm": [2.0, 4.0], "threshold": 0.10,
                   "plan_commit": "40921e133", "evidence_scope": "physical data screen, not Numi simulation"})
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
