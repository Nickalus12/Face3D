#!/bin/bash
# Run mutation testing on critical Face3D pipeline modules
# Usage: bash scripts/run_mutation_tests.sh [module]
# Modules: color_correction, frame_filter, orientation, colmap_io, all (default)

set -e
cd "$(dirname "$0")/.."

MODULE="${1:-all}"

run_mutmut() {
    local path="$1"
    local name="$2"
    echo ""
    echo "============================================"
    echo "  Mutating: $name"
    echo "  Path:     $path"
    echo "============================================"
    mutmut run \
        --paths-to-mutate="$path" \
        --tests-dir=tests/ \
        --runner="python -m pytest tests/test_mutation_targets.py -x -q --tb=line"
    echo ""
    mutmut results
    echo ""
}

case "$MODULE" in
    color_correction)
        run_mutmut "src/capture/color_correction.py" "Color Correction (S-Log3 EOTF)"
        ;;
    frame_filter)
        run_mutmut "src/capture/frame_filter.py" "Frame Filter (blur/exposure)"
        ;;
    orientation)
        run_mutmut "src/sensors/orientation.py" "Orientation (Madgwick/Complementary)"
        ;;
    colmap_io)
        run_mutmut "src/utils/colmap_io.py" "COLMAP I/O (binary read/write)"
        ;;
    all)
        run_mutmut "src/capture/color_correction.py" "Color Correction (S-Log3 EOTF)"
        run_mutmut "src/capture/frame_filter.py" "Frame Filter (blur/exposure)"
        run_mutmut "src/sensors/orientation.py" "Orientation (Madgwick/Complementary)"
        run_mutmut "src/utils/colmap_io.py" "COLMAP I/O (binary read/write)"
        ;;
    *)
        echo "Unknown module: $MODULE"
        echo "Usage: $0 [color_correction|frame_filter|orientation|colmap_io|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Mutation Testing Complete ==="
echo "To see surviving mutants: mutmut show <id>"
echo "To generate HTML report:  mutmut html"
