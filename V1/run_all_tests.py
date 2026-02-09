#!/usr/bin/env python3
"""
Master Test Script for All QPINN Models
Runs training and inference for all 6 reaction-diffusion models (1D and 2D)
Excludes RD_example folder
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent

# Models to test (exclude RD_example)
MODELS = [
    "Brussel",
    "Gierer_Meinhardt",
    "GrayScott",
    "Lengyel_Epstein",
    "Schnakenberg",
    "Thomas",
]

SPATIAL_DIMS = ["1D", "2D"]


def get_training_file(model: str, spatial_dim: str) -> Path:
    """Get path to training file"""
    model_name = model.lower()
    if spatial_dim == "1D":
        if model == "Brussel":
            filename = "brussel_1d_training.py"
        elif model == "Gierer_Meinhardt":
            filename = "gierer_meinhardt_1d_training.py"
        elif model == "GrayScott":
            filename = "grayscott_1d_training.py"
        elif model == "Lengyel_Epstein":
            filename = "lengyel_epstein_1d_training.py"
        elif model == "Schnakenberg":
            filename = "schnakenberg_1d_training.py"
        elif model == "Thomas":
            filename = "thomas_1d_training.py"
    else:  # 2D
        if model == "Brussel":
            filename = "brussel_2d_training.py"
        elif model == "Gierer_Meinhardt":
            filename = "gierer_meinhardt_2d_training.py"
        elif model == "GrayScott":
            filename = "grayscott_2d_training.py"
        elif model == "Lengyel_Epstein":
            filename = "lengyel_epstein_2d_training.py"
        elif model == "Schnakenberg":
            filename = "schnakenberg_2d_training.py"
        elif model == "Thomas":
            filename = "thomas_2d_training.py"

    return BASE_DIR / model / f"{spatial_dim}_models" / filename


def get_inference_file(model: str, spatial_dim: str) -> Path:
    """Get path to inference file"""
    model_name = model.lower()
    if spatial_dim == "1D":
        if model == "Brussel":
            filename = "brussel_1d_inference.py"
        elif model == "Gierer_Meinhardt":
            filename = "gierer_meinhardt_1d_inference.py"
        elif model == "GrayScott":
            filename = "grayscott_1d_inference.py"
        elif model == "Lengyel_Epstein":
            filename = "lengyel_epstein_1d_inference.py"
        elif model == "Schnakenberg":
            filename = "schnakenberg_1d_inference.py"
        elif model == "Thomas":
            filename = "thomas_1d_inference.py"
    else:  # 2D
        if model == "Brussel":
            filename = "brussel_2d_inference.py"
        elif model == "Gierer_Meinhardt":
            filename = "gierer_meinhardt_2d_inference.py"
        elif model == "GrayScott":
            filename = "grayscott_2d_inference.py"
        elif model == "Lengyel_Epstein":
            filename = "lengyel_epstein_2d_inference.py"
        elif model == "Schnakenberg":
            filename = "schnakenberg_2d_inference.py"
        elif model == "Thomas":
            filename = "thomas_2d_inference.py"

    return BASE_DIR / model / f"{spatial_dim}_models" / filename


def run_script(script_path: Path, script_type: str) -> Tuple[bool, str, float]:
    """
    Run a Python script and return success status, output, and execution time
    
    Args:
        script_path: Path to the script
        script_type: "training" or "inference"
    
    Returns:
        (success, output, execution_time)
    """
    if not script_path.exists():
        return False, f"❌ File not found: {script_path}", 0.0

    print(f"\n{'='*80}")
    print(f"Running: {script_path.name}")
    print(f"Type: {script_type}")
    print(f"{'='*80}")

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        execution_time = time.time() - start_time

        if result.returncode == 0:
            return True, result.stdout, execution_time
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, error_msg, execution_time

    except subprocess.TimeoutExpired:
        return False, "❌ Timeout (>1 hour)", time.time() - start_time
    except Exception as e:
        return False, f"❌ Error: {str(e)}", time.time() - start_time


def main():
    """Main test runner"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "🚀 MASTER TEST SCRIPT - ALL QPINN MODELS" + " "*18 + "║")
    print("╚" + "="*78 + "╝")

    # Organize tests
    tests: List[Tuple[str, str, str, str]] = []  # (model, spatial_dim, test_type, file_path)

    for model in MODELS:
        for spatial_dim in SPATIAL_DIMS:
            training_file = get_training_file(model, spatial_dim)
            inference_file = get_inference_file(model, spatial_dim)

            tests.append((model, spatial_dim, "training", str(training_file)))
            tests.append((model, spatial_dim, "inference", str(inference_file)))

    print(f"\n📋 Total tests to run: {len(tests)}")
    print(f"   - Models: {len(MODELS)}")
    print(f"   - Spatial dimensions: {len(SPATIAL_DIMS)}")
    print(f"   - Test types: 2 (training + inference)")

    # Results tracking
    results = {
        "passed": [],
        "failed": [],
        "timing": {},
    }

    total_start = time.time()

    # Run tests
    for i, (model, spatial_dim, test_type, file_path) in enumerate(tests, 1):
        test_label = f"{model} {spatial_dim} {test_type}"
        print(f"\n[{i}/{len(tests)}] {test_label}")

        success, output, exec_time = run_script(Path(file_path), test_type)
        results["timing"][test_label] = exec_time

        if success:
            results["passed"].append(test_label)
            print(f"✅ {test_label} - {exec_time:.2f}s")
        else:
            results["failed"].append(test_label)
            print(f"❌ {test_label} - {exec_time:.2f}s")
            print(f"\nError details:\n{output[-500:]}")  # Last 500 chars

    total_time = time.time() - total_start

    # Print summary
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "📊 TEST SUMMARY" + " "*38 + "║")
    print("╚" + "="*78 + "╝")

    print(f"\n✅ Passed: {len(results['passed'])}/{len(tests)}")
    for test in results["passed"]:
        print(f"   ✓ {test}")

    if results["failed"]:
        print(f"\n❌ Failed: {len(results['failed'])}/{len(tests)}")
        for test in results["failed"]:
            print(f"   ✗ {test}")

    print(f"\n⏱️  Execution Times:")
    sorted_times = sorted(results["timing"].items(), key=lambda x: x[1], reverse=True)
    for test_label, exec_time in sorted_times[:5]:
        print(f"   {test_label}: {exec_time:.2f}s")

    print(f"\n📈 Statistics:")
    print(f"   Total tests: {len(tests)}")
    print(f"   Passed: {len(results['passed'])}")
    print(f"   Failed: {len(results['failed'])}")
    print(f"   Success rate: {len(results['passed'])/len(tests)*100:.1f}%")
    print(f"   Total execution time: {total_time/60:.2f} minutes")
    print(f"   Average per test: {total_time/len(tests):.2f}s")

    # Exit code
    exit_code = 0 if len(results["failed"]) == 0 else 1

    print("\n" + "="*78)
    if exit_code == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {len(results['failed'])} TEST(S) FAILED")
    print("="*78 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
