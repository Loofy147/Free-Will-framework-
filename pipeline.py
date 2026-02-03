import subprocess
import sys
import json

def run_step(name, cmd):
    print(f"\n>>> RUNNING STEP: {name}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {name}")
        print(result.stderr)
        return False
    print(f"SUCCESS: {name}")
    # Extract key metrics from output
    for line in result.stdout.split('\n'):
        if any(k in line for k in ["FWI", "Verified", "Healthy", "Synergy", "Correlation", "HALTED", "Capability"]):
            print(f"    {line.strip()}")
    return True

def main():
    print("="*80)
    print(" FULL-SPECTRUM VOLITION PIPELINE")
    print("="*80)

    # 1. Verification
    if not run_step("Formal Safety Verification", "python /app/verify_formal.py"):
        sys.exit(1)

    # 2. Integrated System & Benchmarks
    if not run_step("Integrated Volition Benchmark", "python /app/integrated_framework.py"):
        sys.exit(1)

    # 3. Final Demo (Social + Individual)
    if not run_step("Social Volition Transition", "python /app/social_volition.py"):
        sys.exit(1)

    # Check for mission status
    if os.path.exists('GLOBAL_MISSION_STATUS.json'):
        print("\n[ANALYSIS] GLOBAL_MISSION_STATUS.json found.")
        with open('GLOBAL_MISSION_STATUS.json', 'r') as f:
            data = json.load(f)
            bench = data.get('global_benchmark', {})
            print(f"    Final Mean FWI: {bench.get('mean_individual_fwi', 0):.4f}")
            print(f"    Collective Synergy: {bench.get('synergy_ratio', 0):.2f}x")
            print(f"    Neuro-Correlate Correlation: {bench.get('fwi_bold_correlation', 0):.4f}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("Individual, Social, and Biological Agency Unified.")
    print("="*80)

if __name__ == "__main__":
    import os
    main()
