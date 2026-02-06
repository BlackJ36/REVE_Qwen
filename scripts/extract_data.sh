#!/bin/bash
# Extract downloaded datasets into the correct directory structure
# Requires: p7zip-full (sudo apt install p7zip-full)
set -e

cd /home/fxxk/git/REVE_Qwen/data

echo "=== Extracting Benchmark (7z files) ==="
mkdir -p benchmark_raw
cd Benchmark
for f in S*.mat.7z; do
    if [ -f "$f" ]; then
        # Extract, 7z will create S1.mat etc.
        # Need to rename to S01.mat format
        7z x -y "$f" -o../benchmark_raw/
    fi
done
cd ..

# Rename S1.mat -> S01.mat, S2.mat -> S02.mat, etc.
cd benchmark_raw
for f in S[0-9].mat; do
    if [ -f "$f" ]; then
        num=${f:1:1}
        mv "$f" "S0${num}.mat"
        echo "Renamed $f -> S0${num}.mat"
    fi
done
cd ..

echo ""
echo "=== Extracting BETA (tar.gz files) ==="
mkdir -p beta_raw
cd BETA
for f in S*.tar.gz; do
    if [ -f "$f" ]; then
        echo "Extracting $f..."
        tar -xzf "$f" -C ../beta_raw/
    fi
done
cd ..

# Handle potential nested directories and rename files
cd beta_raw
# If files are in subdirectories, move them up
find . -name "S*.mat" -type f -exec mv {} . \; 2>/dev/null || true
# Rename S1.mat -> S01.mat format
for f in S[0-9].mat; do
    if [ -f "$f" ]; then
        num=${f:1:1}
        mv "$f" "S0${num}.mat"
        echo "Renamed $f -> S0${num}.mat"
    fi
done
cd ..

echo ""
echo "=== Verification ==="
echo "Benchmark files:"
ls benchmark_raw/*.mat 2>/dev/null | wc -l
echo "BETA files:"
ls beta_raw/*.mat 2>/dev/null | wc -l

echo ""
echo "=== Done! ==="
