import numpy as np
from free_will_framework import IntegratedInformationCalculator

calc = IntegratedInformationCalculator()
# Weakly connected graph
conn = np.eye(5) * 0.5
conn[0, 1] = 0.1
conn[1, 0] = 0.1

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
