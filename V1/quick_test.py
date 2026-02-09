#!/usr/bin/env python3
"""
Quick Test Script - Tests one model (Brussel 1D) for validation
Run this to quickly verify everything works before running full test suite
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

def run_test(script_name: str, description: str):
    """Run a single test and report results"""
    script_path = BASE_DIR / script_name
    
    if not script_path.exists():
        print(f"❌ File not found: {script_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            timeout=600,
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS ({elapsed:.2f}s)")
            return True
        else:
            print(f"\n❌ {description} - FAILED ({elapsed:.2f}s)")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False


def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "🚀 QUICK TEST - BRUSSEL 1D (Training + Inference)" + " "*4 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Brussel/1D_models/brussel_1d_training.py", "Brussel 1D Training"),
        ("Brussel/1D_models/brussel_1d_inference.py", "Brussel 1D Inference"),
    ]
    
    results = []
    total_start = time.time()
    
    for script, description in tests:
        success = run_test(script, description)
        results.append((description, success))
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("📊 QUICK TEST SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nResult: {passed}/{total} passed")
    print(f"Total time: {total_time:.2f}s")
    print("="*70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
