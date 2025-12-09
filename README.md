# FAIR MITRE ATT&CK Bayesian Risk Simulation Engine  
*A quantitative cyber risk model integrating MITRE ATT&CK, dependency-aware control strengths, and a hybrid Bayesian network + Monte Carlo attacker simulator.*

## Overview
This repository implements an advanced cyber risk quantification engine that merges:

- FAIR-style frequency × magnitude modeling  
- MITRE ATT&CK technique-level behavioral modeling  
- Bayesian networks for estimating attacker success and detection probabilities  
- Monte Carlo attacker-path simulation incorporating retries, fallbacks, adaptability, and detection  
- Control dependency logic (group health, “requires” relationships, influence weighting)  
- Technique relevance mapping for attacker capability modeling  

The engine is designed to support both interactive analysis and automated integration into larger systems.

## Directory Structure
```
.
├── build_mitigation_influence_template.py
├── build_technique_relevance_template.py
├── cyber_incident_pymc.py
├── mitre_control_strength_dashboard.py
├── mitigation_control_strengths.csv
├── technique_relevance.csv
└── README.md
```

## Core Capabilities
These capabilities come from the original README and remain fully valid:

- Dependency-aware mitigation strength computation  
- Group health modeling  
- Influence weighting  
- Technique relevance extraction  
- Tactic-level fallback simulation  
- Monte Carlo loss modeling  
- Poisson frequency modeling  
- Lognormal loss severity modeling  
- Flexible CSV-driven configuration  

**Nothing in the new Bayesian model removes or replaces these features.**

---

# Bayesian Network Enhancements (New)

## 1. Technique-Level Bayesian Network (BN)

A fully dynamic Bayesian network is constructed when technique-level inputs are available.

### BN Nodes
- λ (lambda) — Annual attack attempt frequency  
- Technique success nodes — One per `(tactic, technique)` pair  
- Technique detection nodes — One per technique  
- Tactic nodes — Deterministic OR of technique successes  
- Chain success node — AND of all relevant tactic nodes  

Technique priors derive from effective mitigation strengths with dependency logic, group health, and influence weighting.

### Tactic Filtering
Tactics with zero relevant techniques are automatically excluded from the BN chain.

---

## 2. Hybrid BN + Attacker Simulation (Default Mode)

When technique inputs exist, the model uses a hybrid approach:

### BN Provides
- Posterior λ  
- Posterior technique success probabilities  
- Posterior technique detection probabilities  

### Monte Carlo Engine Handles
- Technique selection  
- Retries  
- Adaptability gating  
- Detection increments  
- Fallback behavior  
- Final chain success  

### Corrected Fallback Behavior
- Triggered only under explicit conditions  
- Moves attacker backward exactly one tactic  
- Technique selection is re-randomized (may repeat previous technique)

### Corrected Adaptability Behavior
Adaptability no longer modifies success probability.  
It controls **only** whether retries are allowed.

---

## 3. Detection Modeling

### BN-Derived Detection Probabilities
If available, detection probability is drawn from technique BN posterior nodes.

### Detection Increment Logic
Per attempt:
```
detect_prob += rng.uniform(DETECT_INC_MIN, DETECT_INC_MAX)
```

### Detection Outcome
If `rng.random() < detect_prob`, the chain terminates immediately.

---

## 4. Automatic Mode Selection

### Technique Mode (BN Hybrid) — Default  
Activated whenever technique inputs can be constructed.

### Tactic Mode (Legacy Fallback)  
Used only when:
- technique relevance file missing  
- mitigation dashboard unavailable  
- no technique priors can be formed  

---

## 5. Attacker Path Logging (Optional)

`_simulate_attacker_path(record_path=True)` returns:
```
(success: bool, path_log: List[Dict])
```

Each log entry includes:
- tactic  
- technique  
- BN-derived success probability  
- BN-derived detection probability  
- retry count  
- fallback occurrence  
- per-step outcome  

---

## 6. Control Dependency and Group Health Integration

All dependency logic from the dashboard now influences BN priors:

- Supporting-control influence  
- Requires-dependencies  
- Group health sampling  
- Influence weighting  
- Cycle detection  
- Defense-in-depth aggregation  

---

## 7. Updated Outputs

### Primary Outputs
- Annual loss distribution  
- Annual incident counts  
- Posterior λ distribution  
- Posterior chain success distribution  

### Optional Outputs
- Technique BN posterior dumps  
- Attacker path logs (JSON or CSV)

---

# Running the Model

Example command:
```bash
python cyber_incident_pymc.py \
    --technique-relevance technique_relevance.csv \
    --mitigation-strengths mitigation_control_strengths.csv
```

Disable stochastic adaptability:
```
--no-adapt-stochastic
```

Disable stochastic impact reduction:
```
--no-stochastic-impact
```

---

# Architecture Summary

### Dashboard
Processes all control dependencies and mitigation strengths.

### Bayesian Network
Constructs λ, technique success/detection nodes, tactic aggregators, and chain success.

### Attacker Path Simulation
Technique mode (default)  
Tactic mode (fallback only)

### Annual Loss Engine
Combines BN posterior with Monte Carlo progression simulation.

---

## Recommended Practices

1. Calibrate mitigation ranges regularly  
2. Use relevance templates to focus on specific actors  
3. Review dependency and group definitions to ensure realistic modeling  
4. Validate modeled AAL and SLE against internal data  
5. Version all CSV inputs to preserve analytic lineage  

---

FAIR–MITRE ATT&CK Quantitative Cyber Risk Framework

Copyright 2025 Joshua M. Connors

Licensed under the Apache License, Version 2.0.

This software incorporates public data from the MITRE ATT&CK® framework.
