#!/bin/sh
set -eu
cd "$(dirname "$0")/../overlay"
PYTHONDONTWRITEBYTECODE=1 python3 matter/tools/test_base_cauchy_fallback_reference.py
