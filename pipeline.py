import os
import subprocess
import sys

def run_step(name, cmd):
    print(f"\n>>> RUNNING STEP: {name}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {name}")
        print(result.stdout)
        print(result.stderr)
        return False
    print(f"SUCCESS: {name}")
    # Print a summary from stdout
    lines = result.stdout.split('\n')
    for line in lines:
        if "Optimal weights" in line or "Verified" in line or "System Healthy" in line or "FWI Score" in line:
            print(f"    {line.strip()}")
    return True

def main():
    # 1. Optimization
    if not run_step("FWI Optimization", "python /app/adaptive_fwi.py"):
        sys.exit(1)

    # 2. Formal Verification
    if not run_step("Formal Verification", "python /app/verify_formal.py"):
        sys.exit(1)

    # 3. Integration & System Check
    if not run_step("Integration & System Check", "python /app/integrated_framework.py"):
        sys.exit(1)

    # 4. Final Demo
    if not run_step("Final Demo", "python /app/demo_app.py"):
        sys.exit(1)

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("All success gates passed (FWI threshold, Formal proofs, Integration)")
    print("="*80)

if __name__ == "__main__":
    main()
