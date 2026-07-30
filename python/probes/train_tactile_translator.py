#!/usr/bin/env python3
"""Train a real-camera-to-metric-depth translator from paired calibration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metalrobo.tactile import TactileObservationContract
from metalrobo.tactile_translator import (
    TactileTranslatorTrainer,
    TranslatorTrainingConfig,
    load_calibration_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sensor_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    contract = TactileObservationContract.from_json(
        args.contract.read_text(encoding="utf-8")
    )
    records = load_calibration_manifest(
        args.manifest,
        sensor_id=args.sensor_id,
    )
    trainer = TactileTranslatorTrainer(
        contract,
        args.sensor_id,
        TranslatorTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            input_channels=args.input_channels,
            seed=args.seed,
        ),
    )
    history = trainer.train(records)
    contract_path = trainer.save(args.output)
    print(
        json.dumps(
            {
                "records": len(records),
                "last_metrics": history[-1],
                "translator_contract": str(contract_path),
                "real_hardware_transfer_claimed": False,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
