#!/bin/bash

# Path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_PATH="$PROJECT_ROOT/src/tests/data/DSC02111.ARW"

echo "--- CLI Example ---"
echo "Using image: $IMAGE_PATH"
echo

# 1. Extract a physical TRNG seed
echo "[1/2] Extracting physical seed (TRNG)..."
python3 -m src extract "$IMAGE_PATH"
echo

# 2. Generate a stream of random values
echo "[2/2] Generating random stream (CSPRNG)..."
# Using --type bytes and -l 32
python3 -m src generate "$IMAGE_PATH" --type bytes -l 32
echo

# 3. Generating a random string
echo "[3/3] Generating a random alphanumeric string..."
python3 -m src generate "$IMAGE_PATH" --type string -l 20 --charset alpha
echo
