#!/usr/bin/env python3
"""Emit the frozen single-pair prediction for the synthetic skin study."""

import hashlib
import json
import math
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: predict_synthetic_skin_traction.py MODEL.json")
    model_path = Path(sys.argv[1])
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    prediction = model["predicted_difference_m"]
    if not all(math.isfinite(float(prediction[key])) for key in ("minimum", "maximum")):
        raise SystemExit("model prediction bounds must be finite")
    if prediction["minimum"] > prediction["maximum"]:
        raise SystemExit("model prediction interval is reversed")
    print(json.dumps({
        "schema": "numi.science.prediction.v1",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "prediction": {
            "estimand": "paired_difference_mean",
            "minimum": prediction["minimum"],
            "maximum": prediction["maximum"],
        },
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
