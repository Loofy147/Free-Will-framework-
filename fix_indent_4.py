import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re
# Fix indentation for the methods in IntegratedInformationCalculator
content = content.replace("        @staticmethod", "    @staticmethod")

with open('free_will_framework.py', 'w') as f:
    f.write(content)
