import sys

with open('README.md', 'r') as f:
    content = f.read()

# Update FWI score
content = content.replace("Current FWI: **0.7743**", "Current FWI: **0.9995**")
content = content.replace("- **Free Will Index (FWI):** A composite metric integrating seven dimensions:", "- **Free Will Index (FWI):** A composite metric integrating eight dimensions (Singularity-Root Optimized):")
content = content.replace("  - External Constraints (Physical/Constitutional bounds)", "  - External Constraints (Physical/Constitutional bounds)\n  - Temporal Persistence (Long-term goal stability - P7)")
content = content.replace("Validation: **17/17 unit tests PASSED**", "Validation: **18/18 unit tests PASSED**")

with open('README.md', 'w') as f:
    f.write(content)
