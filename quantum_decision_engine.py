import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

def _von_neumann_entropy(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return -float(np.sum(vals * np.log(vals)))

def _partial_trace_B(rho_AB: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
    rho_reshaped = rho_AB.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.einsum('jiki->jk', rho_reshaped)

class QuantumPolicyState:
    def __init__(self, n_actions: int, initial: str = "uniform"):
        self.n = n_actions
        if initial == "uniform":
            psi = np.ones(n_actions, dtype=complex) / np.sqrt(n_actions)
            self.rho = np.outer(psi, psi.conj())
        elif initial == "classical":
            self.rho = np.eye(n_actions, dtype=complex) / n_actions
        else:
            self.rho = np.eye(n_actions, dtype=complex) / n_actions

    @property
    def measurement_entropy(self) -> float:
        probs = np.real(np.diag(self.rho))
        probs = np.maximum(probs, 0)
        if probs.sum() > 1e-15:
            probs /= probs.sum()
        return -float(np.sum(probs * np.log(probs + 1e-15)))

    @property
    def entropy(self) -> float:
        return _von_neumann_entropy(self.rho)

    @property
    def coherence(self) -> float:
        return float(np.sum(np.abs(self.rho)) - np.sum(np.abs(np.diag(self.rho))))

    def interfere(self, utilities: np.ndarray):
        phases = np.exp(1j * utilities * np.pi)
        U = np.diag(phases)
        self.rho = U @ self.rho @ U.conj().T

    def unitary_evolve(self, H: np.ndarray, dt: float = 0.1):
        from scipy.linalg import expm
        U = expm(-1j * H * dt)
        self.rho = U @ self.rho @ U.conj().T

    def decohere(self, gamma: float, dt: float = 0.1):
        # Phase damping: off-diagonals decay
        decay = np.exp(-gamma * dt)
        mask = np.full((self.n, self.n), decay)
        np.fill_diagonal(mask, 1.0)
        self.rho = self.rho * mask

    def measure(self) -> int:
        probs = np.real(np.diag(self.rho))
        probs = np.maximum(probs, 0); probs /= probs.sum()
        outcome = int(np.random.choice(self.n, p=probs))
        # Collapse
        self.rho = np.zeros((self.n, self.n), dtype=complex)
        self.rho[outcome, outcome] = 1.0
        return outcome

    def is_valid(self) -> bool:
        tr = np.trace(self.rho)
        herm = np.allclose(self.rho, self.rho.conj().T)
        return bool(np.abs(tr - 1.0) < 1e-9 and herm)

class EntangledAgentPair:
    def __init__(self, n_A: int, n_B: int):
        self.n_A = n_A
        self.n_B = n_B
        # Max entangled state: 1/sqrt(n) sum |ii>
        n = min(n_A, n_B)
        psi = np.zeros(n_A * n_B, dtype=complex)
        for i in range(n):
            psi[i * n_B + i] = 1.0 / np.sqrt(n)
        self.rho_joint = np.outer(psi, psi.conj())

    @property
    def rho_A(self): return _partial_trace_B(self.rho_joint, self.n_A, self.n_B)
    @property
    def rho_B(self):
        rho_reshaped = self.rho_joint.reshape(self.n_A, self.n_B, self.n_A, self.n_B)
        return np.einsum('ijik->jk', rho_reshaped)

    def measure_A(self) -> Tuple[int, np.ndarray]:
        rho_A = self.rho_A
        probs = np.real(np.diag(rho_A))
        probs = np.maximum(probs, 0); probs /= probs.sum()
        outcome = int(np.random.choice(self.n_A, p=probs))
        proj = np.zeros((self.n_A, self.n_A), dtype=complex)
        proj[outcome, outcome] = 1.0
        P = np.kron(proj, np.eye(self.n_B, dtype=complex))
        self.rho_joint = P @ self.rho_joint @ P.conj().T
        self.rho_joint /= np.trace(self.rho_joint)
        return outcome, self.rho_B

@dataclass
class DecisionRecord:
    agent_id: str
    initial_entropy: float
    post_interference_entropy: float
    post_decoherence_entropy: float
    final_action: int
    duration_ms: float

class QuantumDecisionEngine:
    def __init__(self, n_actions: int = 20, decoherence_rate: float = 0.1):
        self.n_actions = n_actions
        self.gamma = decoherence_rate
        self.agents: Dict[str, QuantumPolicyState] = {}

    def add_agent(self, agent_id: str, initial: str = "uniform"):
        self.agents[agent_id] = QuantumPolicyState(self.n_actions, initial)

    def run_decision_cycle(self, agent_id: str, utilities: np.ndarray) -> DecisionRecord:
        t0 = time.time()
        agent = self.agents[agent_id]
        h0 = agent.measurement_entropy
        agent.interfere(utilities)
        H = np.diag(utilities.astype(complex)) + 0.5 * np.ones((self.n_actions, self.n_actions), dtype=complex) / self.n_actions
        agent.unitary_evolve(H, dt=0.5)
        h1 = agent.measurement_entropy
        agent.decohere(self.gamma, dt=1.0)
        h2 = agent.measurement_entropy
        action = agent.measure()
        return DecisionRecord(agent_id, h0, h1, h2, action, (time.time()-t0)*1000)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        engine = QuantumDecisionEngine()
        q_red, c_red = [], []
        for _ in range(100):
            u = np.random.rand(20)
            engine.add_agent("Q")
            rec = engine.run_decision_cycle("Q", u)
            q_red.append(rec.initial_entropy - rec.post_interference_entropy)
            p_c = np.exp(u * 5); p_c /= p_c.sum()
            h_c = -np.sum(p_c * np.log(p_c + 1e-15))
            c_red.append(2.9957 - h_c)
        print(f"Quantum Advantage: {np.mean(q_red)/np.mean(c_red):.2f}x")

def demo_social_entanglement():
    print("\n[DEMO] Social Entanglement")
    pair = EntangledAgentPair(10, 10)
    print(f"Initial entanglement (Phi_social): {pair.rho_A.shape}") # placeholder
    outcome_A, rho_B = pair.measure_A()
    print(f"Agent A measured: {outcome_A}")
    probs_B = np.real(np.diag(rho_B))
    outcome_B = np.argmax(probs_B)
    print(f"Agent B collapsed to: {outcome_B}")
    print(f"Correlation: {outcome_A == outcome_B}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo_social_entanglement()
