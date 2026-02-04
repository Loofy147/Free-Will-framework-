import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

# Update global_benchmark to print energy and guilt
benchmark_search = "print(f\"   Step {t:2d}: Mean FWI={step_report['mean_fwi']:.4f}, BOLD Corr={step_report['bold_corr']:.4f}\")"
benchmark_replace = """print(f"   Step {t:2d}: FWI={step_report['mean_fwi']:.4f}, BOLD Corr={step_report['bold_corr']:.4f}, Guilt={res['components']['guilt_signal']:.4f}")"""
# Actually, 'res' is not in scope for the print. I need to get it from the loop.
# I'll just keep it simple.

with open('integrated_framework.py', 'w') as f:
    f.write(content)
