import numpy as np
from free_will_framework import IntegratedInformationCalculator

calc = IntegratedInformationCalculator()
# 5-node cycle
conn = np.zeros((5, 5))
for i in range(5):
    conn[i, (i+1)%5] = 1.0
    conn[(i+1)%5, i] = 1.0

state1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
state2 = np.zeros(5)

phi1 = calc.compute_phi(conn, state1)
phi2 = calc.compute_phi(conn, state2)

print(f"Phi1 (state1): {phi1}")
print(f"Phi2 (state2): {phi2}")

if phi1 == phi2:
    print("BUG VERIFIED: Phi is identical for different states because of buggy cache!")
else:
    print("BUG FIXED: Phi is different for different states.")
