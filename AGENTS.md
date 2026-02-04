# AGENT GUIDELINES

## Mandatory Documentation Rule
**After any code changes, documentation must be updated accordingly.**
This includes updating the `README.md`, `INNOVATION_REPORT.md`, and `EXECUTIVE_SUMMARY.md` if the changes affect core features, metrics, or project status.

## Coding Conventions
- Use JAX for heavy numerical simulations.
- Ensure all `@jax.jit` methods are `@staticmethod` when used within classes to avoid issues with non-array `self` arguments.
- Prefer `np.zeros` + slice assignment over `np.pad` for small arrays in performance-critical loops.
- Use `ndarray.round(decimals).tobytes()` for high-performance hashing of state representations.

## Verification
- Always run the full test suite (`python3 test_free_will.py`, `python3 test_social_volition.py`, `python3 test_safety_properties.py`) before submitting.
- Verify the `IntegratedVolitionSystem` benchmark results in `GLOBAL_MISSION_STATUS.json`.
