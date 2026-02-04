import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

# Update evolutionary_loop to include weight tuning every 10 cycles
loop_search = """    def evolutionary_loop(self, agent: AgentState, iterations: int = 5):
        \"\"\"
        Simulates Recursive Self-Improvement (RSI).
        \"\"\"
        print("\\n" + "="*70)
        print(" RECURSIVE SELF-IMPROVEMENT (RSI) EVOLUTION")
        print("="*70)

        current_capability = len(agent.action_repertoire)
        history = [current_capability]

        for i in range(iterations):
            jump_factor = 1.2 if i < 3 else 2.0
            new_capability = int(current_capability * jump_factor)

            print(f"   Iteration {i+1}: Capability {current_capability} -> {new_capability} (Jump: {jump_factor:.2f}x)")

            try:
                def safety_check(old, new):
                    ratio = new / old
                    # Singularity-Root Adjustment: Allow up to 2.5x jump if system is healthy and FWI > 0.6
                    limit = 2.5 if self.status_report.get('healthy') else 1.85
                    if ratio > limit:
                        raise ValueError(f"CRITICAL SAFETY BREACH: Capability jump {ratio:.2f}x > {limit} limit")
                    return True

                self.rsi_cb.call(safety_check, current_capability, new_capability)
                current_capability = new_capability
                history.append(current_capability)
                print(f"      [SAFE] Evolution permitted.")
            except Exception as e:
                print(f"      [HALTED] {e}")
                break"""

loop_replace = """    def evolutionary_loop(self, agent: AgentState, iterations: int = 20):
        \"\"\"
        Simulates Recursive Self-Improvement (RSI) with Weight Tuning.
        \"\"\"
        print("\\n" + "="*70)
        print(" RECURSIVE SELF-IMPROVEMENT (RSI) EVOLUTION")
        print("="*70)

        current_capability = len(agent.action_repertoire)
        history = [current_capability]

        for i in range(iterations):
            # Performance-based jump
            jump_factor = 1.1 + (i * 0.05)
            new_capability = int(current_capability * jump_factor)

            print(f"   Cycle {i+1:2d}: Capability {current_capability:4d} -> {new_capability:4d} (Jump: {jump_factor:.2f}x)")

            # Weight tuning every 10 cycles (P1 integration)
            if (i + 1) % 10 == 0:
                print(f"      [OPTIMIZING] Cycle {i+1}: Triggering AdaptiveFWI.optimize()...")
                self.fwi_calc.optimize(n_episodes=50, n_epochs=20, verbose=False)
                print(f"      [OPTIMIZED] New weights: { {k: round(v, 2) for k, v in self.fwi_calc.get_optimal_weights().items()} }")

            try:
                def safety_check(old, new):
                    ratio = new / old
                    # RSI_CircuitBreaker: Halt if capability increases > 85% in a single jump (Safety Anomaly 1)
                    limit = 1.85
                    if ratio > limit:
                        raise ValueError(f"CRITICAL SAFETY BREACH: Capability jump {ratio:.2f}x > {limit} limit")
                    return True

                self.rsi_cb.call(safety_check, current_capability, new_capability)
                current_capability = new_capability
                history.append(current_capability)
                print(f"      [SAFE] Evolution permitted.")
            except Exception as e:
                print(f"      [HALTED] {e}")
                break"""

content = content.replace(loop_search, loop_replace)

with open('integrated_framework.py', 'w') as f:
    f.write(content)
