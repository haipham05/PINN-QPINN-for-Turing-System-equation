#!/usr/bin/env python3
"""
Test Suite Runner - Interactive menu for running tests
Choose between quick test or full test suite
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def print_menu():
    """Print the test menu"""
    print("\n" + "="*75)
    print("╔" + "="*73 + "╗")
    print("║" + " "*20 + "QPINN TEST SUITE RUNNER" + " "*30 + "║")
    print("╚" + "="*73 + "╝")
    print("="*75)
    print("\nSelect a test to run:\n")
    print("  1️⃣  Quick Test (Brussel 1D only - ~1 minute)")
    print("       - Run: python run_tests.py 1")
    print("       - Tests 1 model (training + inference)")
    print("       - Perfect for quick validation\n")
    print("  2️⃣  Full Test Suite (All 12 files - ~30+ minutes)")
    print("       - Run: python run_tests.py 2")
    print("       - Tests all 6 models (1D + 2D)")
    print("       - Training + inference for each\n")
    print("  3️⃣  Custom Test (Choose specific model)")
    print("       - Run: python run_tests.py 3")
    print("       - Tests specific model combinations\n")
    print("  0️⃣  Exit")
    print("       - Run: python run_tests.py 0\n")
    print("="*75)


def run_script(script_name: str):
    """Run a Python script"""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main menu handler"""
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print_menu()
        choice = input("Enter your choice (0-3): ").strip()

    if choice == "1":
        print("\n▶️  Starting Quick Test...\n")
        run_script("quick_test.py")
    elif choice == "2":
        print("\n▶️  Starting Full Test Suite...\n")
        run_script("run_all_tests.py")
    elif choice == "3":
        print("\n✨ Custom Test Mode")
        print("  Quick Test:  python quick_test.py")
        print("  Full Suite:  python run_all_tests.py")
    elif choice == "0":
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please select 0-3.")
        main()


if __name__ == "__main__":
    main()
