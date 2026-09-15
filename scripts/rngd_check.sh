#!/usr/bin/env bash
set -euo pipefail

echo "RNGD correctness gate failed: the hardware correctness suite is not implemented." >&2
echo "A compiler artifact or static schedule does not validate device outputs." >&2
exit 1
