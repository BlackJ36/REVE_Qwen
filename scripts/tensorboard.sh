#!/bin/bash
# Launch TensorBoard for training visualization
set -e

LOGDIR=${LOGDIR:-output/logs}
PORT=${PORT:-6006}

echo "=== Starting TensorBoard ==="
echo "Log dir: ${LOGDIR}"
echo "URL: http://localhost:${PORT}"

uv run tensorboard --logdir=${LOGDIR} --port=${PORT} --bind_all
