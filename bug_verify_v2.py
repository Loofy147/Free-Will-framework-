import numpy as np
from free_will_framework import IntegratedInformationCalculator

calc = IntegratedInformationCalculator()
conn = np.ones((5, 5))
state1 = np.ones(5) * 10
state2 = np.zeros(5)

phi1 = calc.compute_phi(conn, state1)
phi2 = calc.compute_phi(conn, state2)

print(f"Phi1 (state1): {phi1}")
print(f"Phi2 (state2): {phi2}")

if phi1 == phi2:
    print("BUG VERIFIED: Phi is identical for different states because of buggy cache!")
else:
    print("BUG NOT VERIFIED: Phi is different.")
