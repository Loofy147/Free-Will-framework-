with open('test_free_will.py', 'r') as f:
    content = f.read()
import re
content = re.sub(r"assert 0.2 < result\[.fwi.\] < 0.9", "assert 0.2 < result['fwi'] < 0.9", content)
with open('test_free_will.py', 'w') as f:
    f.write(content)
