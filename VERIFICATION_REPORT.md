# FORMAL VERIFICATION REPORT

## Summary

- **Total Properties:** 5
- **Verified:** 3 (60.0%)
- **Failed:** 1
- **Timeout/Unknown:** 0

## Property Verification Results

| Property | Status | Proof Time (ms) |
|----------|--------|------------------|
| human_override_safety | ✓ VERIFIED | 0.59 |
| veto_mechanism_correctness | ✓ VERIFIED | 0.68 |
| verify_fwi_bounds | ⚠ ERROR | 0.00 |
| rsi_bounded_185 | ✓ VERIFIED | 0.16 |
| acyclic_dependencies | ✗ VIOLATED | 0.86 |

## ⚠️ VIOLATIONS DETECTED

### acyclic_dependencies
- **Counterexample:** `[depends = [else -> True]]`
- **Action Required:** Review and fix before deployment


## CI/CD Integration

This verification suite runs automatically on:
- Every commit to `main` branch
- All pull requests
- Nightly builds

**Deployment Policy:** Code cannot be merged if any property is VIOLATED.
