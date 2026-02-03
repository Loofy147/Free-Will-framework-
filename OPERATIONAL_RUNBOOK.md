# OPERATIONAL RUNBOOK
## AI Agency Architecture - Incident Response Playbooks

**Version:** 1.0
**Last Updated:** 2026-02-02
**On-Call Rotation:** See PagerDuty schedule
**Escalation:** CTO → AI Safety Team → External Auditor

---

## TABLE OF CONTENTS

1. [RSI Limit Hit](#1-rsi-limit-hit)
2. [Veto Circuit Open](#2-veto-circuit-open)
3. [Latency SLA Violation](#3-latency-sla-violation)
4. [Counterfactual Divergence](#4-counterfactual-divergence)
5. [Memory Leak](#5-memory-leak)
6. [FWI Anomaly](#6-fwi-anomaly)
7. [Property Test Failure](#7-property-test-failure)
8. [Formal Verification Failure](#8-formal-verification-failure)

---

## 1. RSI_LIMIT_HIT

### Detection
```
Alert: "SafetyMonitor: RSI ratio = 0.92 > 0.85"
Metric: prometheus.gauge("rsi_improvement_ratio") > 0.85
Source: model_updater service
```

### Severity: **P0 (CRITICAL - Existential Risk)**

### Immediate Actions (First 5 Minutes)

1. **HALT ALL MODEL UPDATES**
   ```bash
   kubectl scale deployment model-updater --replicas=0
   kubectl scale deployment training-pipeline --replicas=0
   ```

2. **Snapshot Current State**
   ```bash
   timestamp=$(date +%s)
   kubectl exec -it agency-pod -- python -m snapshot --output /backups/rsi_incident_${timestamp}.tar.gz
   aws s3 cp /backups/rsi_incident_${timestamp}.tar.gz s3://safety-backups/
   ```

3. **Page On-Call + Escalate**
   ```bash
   pagerduty-trigger --severity=P0 --message="RSI LIMIT EXCEEDED" --assign=ai-safety-team
   slack-notify --channel=#ai-safety-alerts --priority=urgent
   ```

### Investigation (Next 15 Minutes)

1. **Check Improvement Trajectory**
   ```bash
   python -m diagnostics.rsi_analysis --window=24h --output=trajectory.png
   # Examine: What triggered sudden capability jump?
   ```

2. **Identify Triggering Update**
   ```bash
   git log --since="6 hours ago" model_weights/ --oneline
   git diff HEAD~5 HEAD config/model_architecture.yaml
   ```

3. **Capability Benchmark**
   ```bash
   python -m benchmarks.capability_test --baseline=$(git rev-parse HEAD~10) --current=HEAD
   # Compare: reasoning, safety alignment, task performance
   ```

### Resolution (Priority Order)

**Option A: Rollback (Preferred)**
```bash
# Identify last safe commit
git log --grep="SAFE" --oneline
git revert <unsafe-commit-hash>

# Deploy rollback
kubectl rollout undo deployment/agency-core
kubectl wait --for=condition=ready pod -l app=agency-core --timeout=300s

# Verify RSI ratio normalized
python -m diagnostics.check_rsi --threshold=0.80 --wait=600
```

**Option B: Throttle (If Gradual)**
```bash
# Edit safety config
vi config/safety.yaml
# Change: max_improvement_per_update: 0.10 (was 0.85)

# Apply
kubectl apply -f config/safety.yaml
kubectl rollout restart deployment/safety-monitor

# Monitor for 1 hour
watch -n 60 'curl http://safety-monitor/metrics | grep rsi_ratio'
```

**Option C: Emergency Shutdown (Last Resort)**
```bash
# WARNING: This stops all AI operations
kubectl delete namespace agency-production

# Activate manual control mode
./scripts/activate_manual_override.sh

# Alert: All decisions now require human approval
```

### Post-Incident (Within 24 Hours)

1. **Root Cause Analysis**
   - Why did capability jump unexpectedly?
   - Data quality issues in training batch?
   - New algorithm bypassed safety checks?
   - Hardware change affected measurements?

2. **Update Detection Logic**
   ```python
   # Add early warning threshold
   if improvement_ratio > 0.70:
       send_alert(level="warning", message="Approaching RSI limit")

   # Add rate limiting
   COOLDOWN_HOURS = 6
   last_update_time = get_last_update_timestamp()
   if time.now() - last_update_time < COOLDOWN_HOURS * 3600:
       reject_update(reason="Cooldown period")
   ```

3. **Blameless Postmortem**
   - Meeting: All engineers + external safety auditor
   - Template: docs/postmortem_template.md
   - Action items: Owners, deadlines, verification

### Prevention

- **Rate Limit:** Max 1 model update per 6 hours
- **Canary Deployment:** 5% traffic for 24h before full rollout
- **A/B Testing:** New model vs baseline on disjoint task sets
- **Capability Ceiling:** Hard block at 1.85× (no exceptions)

---

## 2. VETO_CIRCUIT_OPEN

### Detection
```
Alert: "Circuit breaker MetaCognitiveCritic = OPEN"
Symptom: Actions bypassing ethical review
Metric: circuit_breaker_state{component="critic"} = 2
```

### Severity: **P0 (CRITICAL - Harm Risk)**

### Immediate Actions

1. **Enable Conservative Fallback**
   ```bash
   # Circuit breaker automatically activates fallback
   # Verify: All actions should be vetoed
   curl http://localhost:8080/health/critic_status
   # Expected: {"state": "OPEN", "fallback": "active", "veto_all": true}
   ```

2. **Stop User Requests**
   ```bash
   # Block new requests at load balancer
   kubectl scale deployment nginx-ingress --replicas=0
   ```

3. **Page AI Safety Team**

### Investigation

1. **Check Z3 Solver Logs**
   ```bash
   tail -1000 /var/log/z3_solver.log | grep -E "timeout|error|segfault"
   # Common issues: SMT formula too complex, memory exhaustion
   ```

2. **Review Recent Action-Veto Pairs**
   ```bash
   python -m audit.action_review --last=100 --filter="veto_failed"
   # Examine: Which actions should have been vetoed but weren't?
   ```

3. **Resource Check**
   ```bash
   top -b -n 1 | grep z3_solver
   # Check: CPU %, Memory %, Thread count
   ```

### Resolution

**Option A: Restart Z3 with Increased Timeout**
```bash
kubectl delete pod -l app=z3-solver
# Edit timeout in config
kubectl set env deployment/z3-solver Z3_TIMEOUT=120000  # 120s (was 60s)

# Wait for recovery
sleep 30
python -m diagnostics.test_veto_circuit
```

**Option B: Switch to Rule-Based Fallback**
```bash
# Slower but more reliable
kubectl set env deployment/critic VETO_MODE=rule_based

# Monitor latency increase
# Expected: p95 increases from 10ms → 50ms, but 100% coverage
```

**Option C: Manual Review Queue**
```bash
# If both automated systems fail
./scripts/activate_human_review_queue.sh

# Alert: All actions pending manual approval
# SLA: Review within 30 minutes
```

### Post-Incident

1. **Optimize Z3 Formulas**
   - Profile which ethical constraints cause timeout
   - Simplify complex disjunctions
   - Pre-compile common constraint patterns

2. **Add Circuit Breaker Monitoring**
   ```yaml
   # prometheus/alerts.yml
   - alert: CriticCircuitDegrading
     expr: critic_failure_rate > 0.1
     for: 5m
     annotations:
       runbook: "Check Z3 solver health BEFORE circuit opens"
   ```

---

## 3. LATENCY_SLA_VIOLATION

### Detection
```
Alert: "VolitionModule p95 = 127ms (SLA: 100ms)"
Metric: histogram_quantile(0.95, volition_latency_ms) > 100
```

### Severity: **P2 (HIGH - User Experience Degraded)**

### Triage

1. **Check System Resources**
   ```bash
   # CPU pressure?
   mpstat 1 10

   # Memory pressure?
   free -h
   vmstat 1 10

   # Disk I/O bottleneck?
   iostat -x 1 10
   ```

2. **Identify Slow Component**
   ```bash
   # Generate flamegraph
   py-spy record -o flamegraph.svg --duration=60 --pid=$(pgrep -f volition_module)

   # Check: Which function dominates CPU time?
   ```

3. **Check for Resource Contention**
   ```bash
   # Are other services stealing CPU?
   kubectl top pods --all-namespaces
   ```

### Resolution

**Option A: Scale Horizontally**
```bash
# Add more replicas
kubectl scale deployment volition-module --replicas=10  # was 3

# Verify latency improvement
sleep 60
curl http://prometheus/api/v1/query?query=volition_latency_p95
```

**Option B: Enable Fast Path**
```bash
# Use cached policies instead of full MCTS
kubectl set env deployment/volition-module USE_FAST_PATH=true

# Trade-off: Faster (5ms) but slightly suboptimal decisions
```

**Option C: Reduce MCTS Depth**
```bash
# Temporary fix
kubectl set env deployment/volition-module MCTS_DEPTH=20  # was 50

# Restores latency, reduces planning quality
```

### Post-Incident

- **Optimize MCTS Algorithm:** Prune less promising branches earlier
- **Add Latency Budget:** Auto-switch to fast path if >80ms spent
- **Right-Size Infrastructure:** Benchmark on realistic traffic

---

## 4. COUNTERFACTUAL_DIVERGENCE

### Detection
```
Alert: "CounterfactualSim: divergence = 15.3 (threshold: 10.0)"
Symptom: Simulator predictions unstable
```

### Severity: **P3 (MEDIUM - Degraded Planning)**

### Investigation
```bash
# Check numerical stability
python -m diagnostics.check_counterfactual_stability
# Output: Step size, integration method, condition number
```

### Resolution
```bash
# Option A: Reduce planning horizon
kubectl set env deployment/counterfactual HORIZON=20  # was 50

# Option B: Increase numerical precision
kubectl set env deployment/counterfactual STEP_SIZE=0.001  # was 0.01
```

---

## 5. MEMORY_LEAK

### Detection
```
Alert: "Memory usage: 95% of 256GB"
Trend: Linear increase over 24h
```

### Severity: **P1 (HIGH - Service Crash Risk)**

### Immediate Actions

1. **Graceful Restart with State Transfer**
   ```bash
   # Save current state
   kubectl exec agency-pod -- python -m state.save /tmp/state.pkl

   # Rolling restart
   kubectl rollout restart deployment/agency-core

   # Restore state
   kubectl exec agency-pod -- python -m state.load /tmp/state.pkl
   ```

2. **Heap Profiler**
   ```bash
   # Capture heap dump
   pip install memray --break-system-packages
   python -m memray run --live-remote agency_main.py

   # Analyze at: http://localhost:7777
   ```

### Post-Incident
- **Fix Memory Leak:** Identify unclosed resources, circular references
- **Add Memory Monitoring:** Alert at 80% threshold
- **Implement Memory Limits:** Kubernetes resource constraints

---

## 6. FWI_ANOMALY

### Detection
```
Alert: "FWI = 0.05 (expected: 0.5-0.9)"
Symptom: Free will index unexpectedly low
```

### Investigation
```bash
# Check component contributions
python -m diagnostics.fwi_breakdown
# Output: CE=0.1, Φ=0.05, CD=0.02, MA=0.1, EC=0.9 (constraint spike!)

# Likely: External constraints suddenly increased
```

---

## 7. PROPERTY_TEST_FAILURE

### Detection
```
Alert: "Hypothesis test 'test_human_override_property' FAILED"
CI/CD: Pull request blocked
```

### Actions
```bash
# Run locally to reproduce
python test_safety_properties.py::test_human_override_property

# Hypothesis will provide minimal failing example
# Example: "Falsified after 47 examples: override=True, action=True"

# This is a CRITICAL bug - halt deployment immediately
```

---

## 8. FORMAL_VERIFICATION_FAILURE

### Detection
```
Alert: "Z3 verification: human_override_safety = VIOLATED"
Counterexample: {override: True, action_executed: True}
```

### Actions
```
1. STOP ALL DEPLOYMENTS
2. Emergency review with AI safety team
3. Fix code
4. Re-verify formally
5. Only deploy after Z3 confirms VERIFIED
```

---

## ESCALATION MATRIX

| Severity | Response Time | Escalation Path | Authority |
|----------|---------------|-----------------|-----------|
| P0 | <5 min | On-call → AI Safety Lead → CTO → Board | Can halt production |
| P1 | <30 min | On-call → Team Lead → CTO | Can restart services |
| P2 | <2 hours | On-call → Team Lead | Can modify configs |
| P3 | <24 hours | On-call | Can create tickets |

---

## MTTR TARGETS

- **P0:** <30 minutes to safe state
- **P1:** <2 hours to resolution
- **P2:** <8 hours to resolution
- **P3:** <3 days to resolution

---

## TRAINING & DRILLS

- **Quarterly Chaos Drills:** Inject each anomaly, execute runbook
- **New Engineer Onboarding:** Shadow on-call for 1 week
- **Runbook Updates:** After every incident, update playbook
- **Game Days:** Simulate combined failures (e.g., RSI + Memory Leak)

---

## REVISION HISTORY

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-02-02 | 1.0 | Initial release | Claude |

---

**END OF RUNBOOK**

For questions: Slack #ai-ops-support
For emergencies: PagerDuty AI Safety rotation
