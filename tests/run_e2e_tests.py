"""
Voice Sampler Unified E2E Test Runner CLI
Executes End-to-End test suites across Tiers 1-4 with clear reports and exit codes.

Usage:
    python tests/run_e2e_tests.py [--all] [--tier {1,2,3,4}] [--url URL] [--mock] [-v]
"""

import argparse
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_suite_for_tier(tier: int) -> unittest.TestSuite:
    """Discovers and builds test suite for the requested tier."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if tier == 1:
        from tests.e2e.test_tier1_features import (
            TestTier1Health,
            TestTier1SettingsGPU,
            TestTier1SamplesCRUD,
            TestTier1AudioStreaming,
            TestTier1TranscriptEditing,
            TestTier1WhisperTranscription,
            TestTier1F5TTSTraining,
            TestTier1RVCTraining,
            TestTier1TTSInference,
            TestTier1STSInference,
            TestTier1ModelRegistry,
            TestTier1GeneratedAudioPlayback,
        )
        for cls in [
            TestTier1Health,
            TestTier1SettingsGPU,
            TestTier1SamplesCRUD,
            TestTier1AudioStreaming,
            TestTier1TranscriptEditing,
            TestTier1WhisperTranscription,
            TestTier1F5TTSTraining,
            TestTier1RVCTraining,
            TestTier1TTSInference,
            TestTier1STSInference,
            TestTier1ModelRegistry,
            TestTier1GeneratedAudioPlayback,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    elif tier == 2:
        from tests.e2e.test_tier2_boundary import (
            TestTier2EmptyInputs,
            TestTier2MissingResources,
            TestTier2PitchAndIndexBoundaries,
            TestTier2NonCUDAHandling,
            TestTier2AudioFormatsAndByteRanges,
        )
        for cls in [
            TestTier2EmptyInputs,
            TestTier2MissingResources,
            TestTier2PitchAndIndexBoundaries,
            TestTier2NonCUDAHandling,
            TestTier2AudioFormatsAndByteRanges,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    elif tier == 3:
        from tests.e2e.test_tier3_combos import TestTier3CrossFeatureCombos
        suite.addTests(loader.loadTestsFromTestCase(TestTier3CrossFeatureCombos))

    elif tier == 4:
        from tests.e2e.test_tier4_scenarios import TestTier4PolishVoiceCloningScenarios
        suite.addTests(loader.loadTestsFromTestCase(TestTier4PolishVoiceCloningScenarios))

    return suite


def run_tier(tier_num: int, tier_name: str, verbosity: int = 1) -> Dict[str, Any]:
    """Runs a single test tier and collects execution metrics."""
    suite = build_suite_for_tier(tier_num)
    test_count = suite.countTestCases()

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    duration = time.time() - start_time

    passed = test_count - len(result.failures) - len(result.errors) - len(result.skipped)

    return {
        "tier": tier_num,
        "name": tier_name,
        "total": test_count,
        "passed": passed,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration": duration,
        "success": result.wasSuccessful(),
    }


def main():
    parser = argparse.ArgumentParser(description="Voice Sampler E2E Test Suite Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific test tier (1-4)")
    parser.add_argument("--all", action="store_true", default=True, help="Run all tiers (default)")
    parser.add_argument("--url", type=str, default=None, help="Target backend URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--mock", action="store_true", help="Force embedded mock server mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test execution")
    args = parser.parse_args()

    if args.url:
        os.environ["VOICE_SAMPLER_TEST_URL"] = args.url
    if args.mock:
        os.environ["VOICE_SAMPLER_FORCE_MOCK"] = "1"

    tier_definitions = [
        (1, "Tier 1: Feature Coverage (>=5 tests per feature)"),
        (2, "Tier 2: Boundary & Corner Cases"),
        (3, "Tier 3: Cross-Feature Integration Combinations"),
        (4, "Tier 4: Real-World Polish Voice Cloning Scenarios"),
    ]

    selected_tiers = tier_definitions
    if args.tier:
        selected_tiers = [td for td in tier_definitions if td[0] == args.tier]

    print("=" * 80)
    print(" VOICE SAMPLER -- END-TO-END (E2E) TEST SUITE")
    print("=" * 80)
    mode_str = "Embedded Mock / Simulation ASGI Server" if args.mock or "VOICE_SAMPLER_FORCE_MOCK" in os.environ else (args.url or "Auto-Detect")
    print(f" Target Mode  : {mode_str}")
    print(f" Python Ver   : {sys.version.split()[0]}")
    print(f" Platform     : {sys.platform}")
    print(f" Active Tiers : {[t[0] for t in selected_tiers]}")
    print("=" * 80)
    print()

    results = []
    overall_success = True
    verbosity = 2 if args.verbose else 1

    for tier_num, tier_name in selected_tiers:
        print(f"[>] Running Tier {tier_num}: {tier_name}...")
        tier_metrics = run_tier(tier_num, tier_name, verbosity=verbosity)
        results.append(tier_metrics)
        if not tier_metrics["success"]:
            overall_success = False
        print(f"[OK] Completed Tier {tier_num} in {tier_metrics['duration']:.2f}s\n")

    # Render Summary Table
    print("=" * 80)
    print(" E2E TEST EXECUTION SUMMARY REPORT")
    print("=" * 80)
    header = f"{'Tier':<8}{'Tier Name':<42}{'Total':<8}{'Passed':<8}{'Failed':<8}{'Time':<8}"
    print(header)
    print("-" * 80)

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_time = 0.0

    for r in results:
        total_tests += r["total"]
        total_passed += r["passed"]
        total_failed += (r["failures"] + r["errors"])
        total_time += r["duration"]
        tier_label = f"Tier {r['tier']}"
        time_str = f"{r['duration']:.2f}s"
        print(f"{tier_label:<8}{r['name'][:40]:<42}{r['total']:<8}{r['passed']:<8}{r['failures'] + r['errors']:<8}{time_str:<8}")

    print("-" * 80)
    total_time_str = f"{total_time:.2f}s"
    print(f"{'TOTAL':<8}{'All Selected Tiers':<42}{total_tests:<8}{total_passed:<8}{total_failed:<8}{total_time_str:<8}")
    print("=" * 80)

    if overall_success:
        print("\n [SUCCESS] ALL E2E TESTS PASSED (100% SUCCESS RATE)\n")
        sys.exit(0)
    else:
        print(f"\n [FAILURE] E2E TEST SUITE ENCOUNTERED {total_failed} FAILURES/ERRORS\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
