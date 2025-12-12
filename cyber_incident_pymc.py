#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 Joshua M. Connors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# FAIR–MITRE Bayesian Cyber Incident Model
# ---------------------------------------------------------------------------
# Quantifies cyber risk using Bayesian inference (PyMC) and FAIR-style loss
# distributions, informed by MITRE ATT&CK control strengths and relevance.
# ---------------------------------------------------------------------------
"""
=====================================================================================
Cyber Incident Risk Model with MITRE ATT&CK + FAIR (Subset-Aware, Impact-Reduced)
=====================================================================================
PURPOSE
-------------------------------------------------------------------------------------
This script quantifies cyber risk by combining:
  • MITRE ATT&CK tactic-level defensive control strengths (from the dashboard)
  • A Bayesian model (PyMC) for attack attempt frequency and per-stage success
  • FAIR-style per-incident losses (with heavy-tailed legal/reputation components)

ENHANCEMENTS IN THIS VERSION
-------------------------------------------------------------------------------------
1) Subset-aware modeling:
   - If available, calls mitre_control_strength_dashboard.get_mitre_tactic_strengths()
     to obtain an ordered subset of tactics and aggregated control strengths.
   - The model builds priors and simulates progression ONLY across those tactics.

2) Dashboard fallback:
   - If the dashboard is unavailable or fails, the script reverts to the legacy
     12-tactic SME map embedded in this file.

3) Impact-side reductions:
   - Incorporates two special mitigations provided by the dashboard:
       "Data Backup" (reduces Productivity & ResponseContainment losses)
       "Encrypt Sensitive Information" (reduces RegulatoryLegal & ReputationCompetitive)
   - Default behavior samples their strengths per posterior draw from their [min,max] range.
   - Toggleable via top-level variable and CLI flag.

CODE STYLE
-------------------------------------------------------------------------------------
- Highly commented for maintainability and clarity.
- No breaking changes to outputs except being subset-aware automatically.
=====================================================================================
"""

import os
import sys
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

# Optional import for technique-level simulation inputs
try:
    from mitre_control_strength_dashboard import get_technique_simulation_inputs
except Exception:
    get_technique_simulation_inputs = None

# ---------- Optional dependency (PyMC) ----------
try:
    import pymc as pm
    HAVE_PYMC = True
except Exception:
    HAVE_PYMC = False

# ---------- Dashboard integration ----------
try:
    from mitre_control_strength_dashboard import get_mitre_tactic_strengths
    HAVE_MITRE_ANALYZER = True
except Exception as e:
    print(f"⚠️ MITRE analyzer not available: {e}")
    HAVE_MITRE_ANALYZER = False

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

# =============================================================================
# Variables: GLOBAL CONFIG — priors, runtime, plotting
# =============================================================================
# Frequency prior (attempts/year), elicited via 90% CI → lognormal
# These are parameters that are similar to TEF in FAIR
# (These are fairly broad defaults; adjust as needed.)
CI_MIN_FREQ = 4
CI_MAX_FREQ = 24
Z_90 = 1.645

# =============================================================================
# Variables: Threat capability (higher = stronger attacker)
# =============================================================================
# Threat capability modifies per-stage success probabilities during simulation.
THREAT_CAPABILITY_STOCHASTIC = True
THREAT_CAPABILITY_RANGE = (0.4, 0.90)   # higher = more capable attacker

# =============================================================================
# Variables: PyMC sampling controls
# =============================================================================
# These control the Markov Chain Monte Carlo (MCMC) sampling process.
N_SAMPLES = 4000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 42

# Posterior predictive Monte Carlo (per posterior draw)
N_SIM_PER_DRAW = 1000  # number of attack attempts simulated per posterior draw

# =============================================================================
# Variables: Attacker progression controls
# =============================================================================
# These control the per-attempt simulation of stagewise progression.
MAX_RETRIES_PER_STAGE = 3
RETRY_PENALTY = 0.90
FALLBACK_PROB = 0.25
DETECT_BASE = 0.01
DETECT_INC_MIN = 0.01
DETECT_INC_MAX = 0.05
MAX_FALLBACKS_PER_CHAIN = 3

# Visualization Option: plot monetary values in millions
PLOT_IN_MILLIONS = True

# =============================================================================
# Variables: Adaptability (stochastic per retry) — logistic update mode
# =============================================================================
# Adaptability controls how quickly an attacker learns from failed attempts.
ADAPTABILITY_STOCHASTIC = True
ADAPTABILITY_RANGE = (0.3, 0.7)        # higher = faster learning on retries
ADAPTABILITY_MODE = "logistic"         # "logistic" (recommended) or "linear" (legacy)
ADAPTABILITY_EFFECT_SCALE = 1.0        # multiplier for linear mode; 1.0 = default

# =============================================================================
# Variables: Optional observed data for Poisson conditioning (keep None by default)
# =============================================================================
# If provided, these condition the posterior on observed incident counts.
observed_total_incidents = None
observed_years = None

# =============================================================================
# BN chain semantics configuration
# =============================================================================
# Controls how the BN computes tactic and chain success.
#
#   "legacy_mean_product"
#       p_tactic = mean(p_tech)
#       p_chain  = product(p_tactic)
#
#   "retry_detect_aware" (default)
#       p_tactic approximates simulator retry + detection behavior
#
# This preserves legacy behavior while enabling FAIR-aligned semantics.
BN_CHAIN_SEMANTICS = "retry_detect_aware"

# =============================================================================
# Technique statistics export behavior
# =============================================================================
EXPORT_ALL_TECHNIQUES = True

# =============================================================================
# MITRE STAGES (canonical fallback)
# =============================================================================
MITRE_STAGES = [
    "Initial Access","Execution","Persistence","Privilege Escalation","Defense Evasion",
    "Credential Access","Discovery","Lateral Movement","Collection","Command And Control",
    "Exfiltration","Impact",
]

# =============================================================================
# Variables: SME fallback (tactic → control block range in [0..0.95])
# =============================================================================
_SME_STAGE_CONTROL_MAP_FALLBACK = {
    "Initial Access": (0.20, 0.50),
    "Execution": (0.20, 0.50),
    "Persistence": (0.20, 0.55),
    "Privilege Escalation": (0.25, 0.55),
    "Defense Evasion": (0.25, 0.55),
    "Credential Access": (0.20, 0.50),
    "Discovery": (0.20, 0.55),
    "Lateral Movement": (0.20, 0.50),
    "Collection": (0.20, 0.50),
    "Command And Control": (0.20, 0.55),
    "Exfiltration": (0.20, 0.50),
    "Impact": (0.20, 0.50),
}

# =============================================================================
# Variables: FAIR TAXONOMY — per-incident losses (lognormal bodies + Pareto tails)
# =============================================================================
loss_categories = ["Productivity", "ResponseContainment", "RegulatoryLegal", "ReputationCompetitive"]

# Lognormal parameters from 5th and 95th percentiles (per category)
loss_q5_q95 = {
    "Productivity": (1_000, 200_000),
    "ResponseContainment": (10_000, 1_000_000),
    "RegulatoryLegal": (0, 3_000_000),
    "ReputationCompetitive": (0, 5_000_000),
}

Z_90 = 1.645  # reused

def _lognormal_from_q5_q95(q5: float, q95: float):
    q5, q95 = max(q5, 1.0), max(q95, q5 * 1.0001)
    ln5, ln95 = np.log(q5), np.log(q95)
    sigma = (ln95 - ln5) / (2.0 * Z_90)
    mu = 0.5 * (ln5 + ln95)
    return mu, sigma

cat_mu = np.zeros(len(loss_categories))
cat_sigma = np.zeros(len(loss_categories))
for i, cat in enumerate(loss_categories):
    mu, sg = _lognormal_from_q5_q95(*loss_q5_q95[cat])
    cat_mu[i], cat_sigma[i] = mu, sg

# =============================================================================
# Variables: Pareto tails for legal & reputation categories
# =============================================================================
pareto_defaults = {
    "RegulatoryLegal":       {"xm": 50_000.0, "alpha": 3.5},
    "ReputationCompetitive": {"xm": 100_000.0, "alpha": 2.75},
}

# =============================================================================
# Variables: Impact reduction (from dashboard) — toggle + multipliers
# =============================================================================
# The dashboard returns strengths in PERCENT (0–100) for:
#   "Data Backup" and "Encrypt Sensitive Information"
# We convert to [0..1] and scale category losses accordingly.
BACKUP_IMPACT_MULT  = 0.60   # scales Productivity & ResponseContainment
ENCRYPT_IMPACT_MULT = 0.50   # scales RegulatoryLegal & ReputationCompetitive

# Toggle: when True (default) sample per posterior draw from [min,max];
# when False, use mean strength (deterministic).
STOCHASTIC_IMPACT_REDUCTION = True

# =============================================================================
# Output directory (daily)
# =============================================================================
def make_output_dir(prefix="output"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(base_dir, f"{prefix}_{date_str}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")
    return out_dir

OUTPUT_DIR = make_output_dir()
LAST_CATEGORY_LOSSES = None  # populated after simulation

# =============================================================================
# Helpers: dashboard integration, priors, formatting, etc.
# =============================================================================
def _load_from_dashboard_or_fallback(dataset_path: str, csv_path: str):
    """
    Returns a 4-tuple:
      (tactics_included: List[str],
       stage_control_map: Dict[str, (float lo, float hi)],  # control block fractions [0..0.95]
       impact_reduction_controls: Dict[str, Dict[str,float]],  # Data Backup / Encrypt Sensitive Information
       mode_str: str)  # "Filtered" | "Full" | "Fallback"
    """
    if not HAVE_MITRE_ANALYZER:
        print("⚠️ MITRE dashboard unavailable — reverting to internal SME fallback control ranges.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

    try:
        detail_df, summary_df, control_strength_map, relevance_metadata, impact_controls = get_mitre_tactic_strengths(
            dataset_path=dataset_path,
            csv_path=csv_path,
            seed=42,
            build_figure=False,
            use_relevance=True,                # dashboard decides based on file presence
            relevance_file="technique_relevance.csv",
            quiet=True,
        )

        if not summary_df.empty and control_strength_map:
            tactics_included = list(summary_df["Tactic"])
            stage_map = {}
            for t in tactics_included:
                row = control_strength_map.get(t, {})
                lo = float(row.get("min_strength", 30.0)) / 100.0
                hi = float(row.get("max_strength", 70.0)) / 100.0
                lo = max(0.0, min(0.95, lo))
                hi = max(0.0, min(0.95, hi))
                if lo > hi:
                    lo, hi = hi, lo
                stage_map[t] = (lo, hi)

            mode = relevance_metadata.get("mode", "Full")
            print(f"✅ Loaded control strengths from MITRE ATT&CK dataset ({mode} mode).")
            print(f"🧩 Included tactics ({len(tactics_included)}): {', '.join(tactics_included)}")

            # Light log of impact-control means (percent values)
            if impact_controls:
                for k, v in impact_controls.items():
                    ms = float(v.get("mean_strength", 0.0))
                    print(f"   • Impact reduction available: {k} — mean {ms:.1f}%")

            return tactics_included, stage_map, impact_controls, mode

        print("⚠️ Dashboard returned no tactic summary — using SME fallback map.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

    except Exception as e:
        print(f"⚠️ MITRE dataset load failed in dashboard: {e}. Using SME fallback.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

def _print_stage_control_map(stage_map, tactics_included):
    """Diagnostic print + CSV export of tactic control strength ranges (subset-aware)."""
    print("\n--- Tactic Control Strength Parameters Used ---")
    print(f"{'Tactic':<25} {'MinStrength':>12} {'MaxStrength':>12}")
    for t in tactics_included:
        lo, hi = stage_map.get(t, (0.0, 0.0))
        print(f"{t:<25} {lo*100:>11.1f}% {hi*100:>11.1f}%")
    print("------------------------------------------------")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(OUTPUT_DIR, f"tactic_control_strengths_{ts}.csv")
    pd.DataFrame([
        {"Tactic": t, "MinStrength": stage_map[t][0] * 100, "MaxStrength": stage_map[t][1] * 100}
        for t in tactics_included
    ]).to_csv(csv_path, index=False)
    print(f"✅ Saved control strength parameters → {csv_path}")

def _success_interval_from_control(block_lo: float, block_hi: float):
    """
    Convert control block interval (fraction) to a susceptibility interval.

    Here:
      - block_lo, block_hi are control strengths (block probabilities) in [0..0.95]
      - we compute an interval for attacker SUCCESS after controls but
        before Threat Capability is applied:
          success = 1 - block
    """
    block_lo = max(0.0, min(0.95, block_lo))
    block_hi = max(0.0, min(0.95, block_hi))
    lo_succ = 1.0 - block_hi
    hi_succ = 1.0 - block_lo
    if lo_succ > hi_succ:
        lo_succ, hi_succ = hi_succ, lo_succ
    return lo_succ, hi_succ

def _beta_from_interval(lo: float, hi: float, strength: float = 200.0):
    mu = 0.5 * (lo + hi)
    k = max(2.0, float(strength))
    a = max(1e-3, mu * k)
    b = max(1e-3, (1 - mu) * k)
    return a, b

def _fmt_money(x: float, millions: bool = None) -> str:
    if millions is None:
        millions = PLOT_IN_MILLIONS
    if millions:
        return f"${x/1_000_000:,.2f}M"
    return f"${x:,.0f}"

# =============================================================================
# Posterior & simulation core (subset-aware)
# =============================================================================
def _build_beta_priors_from_stage_map(stage_map, tactics_included):
    """Return Beta(a,b) parameters per included tactic; prints preview capability."""
    rng = np.random.default_rng(RANDOM_SEED)

    # ---- FAIR-aligned preview ------------------------------------------------
    # Legacy adaptation preview removed (ADAPTATION_* no longer used).
    # We now preview Threat Capability, which is applied later during simulation.
    try:
        cap_stoch = THREAT_CAPABILITY_STOCHASTIC
        cap_range = THREAT_CAPABILITY_RANGE
    except NameError:
        # Safe defaults if globals were renamed elsewhere
        cap_stoch = True
        cap_range = (0.5, 1.0)
    print(f"Using threat capability range {cap_range} (stochastic={cap_stoch})")

    alphas, betas = [], []
    for t in tactics_included:
        blo, bhi = stage_map[t]
        # Keep resistance (control strengths) exactly as provided by MITRE;
        # threat capability is applied later when simulating attempts.
        slo, shi = _success_interval_from_control(blo, bhi)
        a, b = _beta_from_interval(slo, shi, strength=50.0)
        alphas.append(a); betas.append(b)
    return np.array(alphas), np.array(betas)

def _simulate_attacker_path(sim_struct, rng, tc=None, tech_posteriors=None, record_path=False):
    """
    Unified attacker path simulation.

    If sim_struct is a NumPy array:
        - Stage-level (tactic) simulation

    If sim_struct is a dict with techniques:
        - Technique-level simulation with p_min/p_max per technique
        - Adaptability gates retries only
        - Threat capability (tc) modifies technique success probability
    """
    # At top of function, before mode split:
    path_log = [] if record_path else None

    # ------------------------------------------------------------
    # Stage-level (tactic) simulation
    # ------------------------------------------------------------
    if isinstance(sim_struct, np.ndarray):
        success_probs = sim_struct
        n_stages = len(success_probs)
        if n_stages == 0:
            return False

        # Actor-level adaptability for retry gating
        if ADAPTABILITY_STOCHASTIC:
            actor_adapt = float(rng.uniform(*ADAPTABILITY_RANGE))
        else:
            actor_adapt = float(np.mean(ADAPTABILITY_RANGE))

        i = 0
        fallback_count = 0

        while 0 <= i < n_stages:
            p_stage = float(success_probs[i])

            detect_prob = DETECT_BASE
            if record_path:
                path_log[-1]["success_prob"] = p_stage
                path_log[-1]["detect_prob"] = detect_prob
            retries_left = MAX_RETRIES_PER_STAGE
            stage_completed = False

            while retries_left > 0:
                # Attempt
                if rng.random() < p_stage:
                    if record_path:
                        path_log[-1]["result"] = "success"
                    i += 1
                    stage_completed = True
                    break

                # Detection increment
                delta = float(rng.uniform(DETECT_INC_MIN, DETECT_INC_MAX))
                detect_prob = min(1.0, detect_prob + delta)

                if rng.random() < detect_prob:
                    if record_path:
                        path_log[-1]["result"] = "detected"
                    return (False, path_log) if record_path else False

                retries_left -= 1
                if retries_left <= 0:
                    break

                # Adaptability gates retries
                if rng.random() >= actor_adapt:
                    break

            if stage_completed:
                continue

            # Fallback
            if (
                rng.random() < FALLBACK_PROB
                and fallback_count < MAX_FALLBACKS_PER_CHAIN
                and i > 0
            ):
                fallback_count += 1
                i = max(0, i - 1)
                continue
            else:
                return False

        return (i >= n_stages, path_log) if record_path else (i >= n_stages)

    # ------------------------------------------------------------
    # Technique-level simulation
    # ------------------------------------------------------------
    tactics = sim_struct.get("tactics", [])
    techniques_by_tactic = sim_struct.get("techniques_by_tactic", {})
    technique_priors = sim_struct.get("technique_priors", {})

    n_tactics = len(tactics)
    path_log = [] if record_path else None
    if n_tactics == 0:
        return False

    # Actor-level adaptability for retry gating
    if ADAPTABILITY_STOCHASTIC:
        actor_adapt = float(rng.uniform(*ADAPTABILITY_RANGE))
    else:
        actor_adapt = float(np.mean(ADAPTABILITY_RANGE))

    current_tactic_index = 0
    fallback_count = 0

    while 0 <= current_tactic_index < n_tactics:
        tactic_name = tactics[current_tactic_index]
        tech_list = techniques_by_tactic.get(tactic_name, [])

        if not tech_list:
            current_tactic_index += 1
            continue

        # Random technique selection
        tech_id = rng.choice(tech_list)
        if record_path:
            path_log.append({
                "tactic": tactic_name,
                "technique": tech_id,
                "success_prob": None,   # fill in later
                "detect_prob": None,    # fill in later
                "result": None,         # success, fail, detected
                "retries": 0,
                "fallback": False
            })
        # Technique priors for that tactic
        priors = technique_priors.get(tactic_name, {}).get(tech_id)

        # --------------------------------------------------------
        # SUCCESS probability (susceptibility after controls)
        #   - priors["p_min"], priors["p_max"] are success bounds in [0..1]
        #   - tech_posteriors, if available, give BN posterior samples for
        #     the same success probability (before capability).
        # --------------------------------------------------------
        if priors is not None:
            succ_min = float(priors.get("p_min", 0.0))
            succ_max = float(priors.get("p_max", 1.0))

            if tech_posteriors is not None:
                key_succ = (tactic_name, tech_id, "success")
                post_arr = tech_posteriors.get(key_succ)
                if post_arr is not None and post_arr.size > 0:
                    idx = rng.integers(0, post_arr.size)
                    # Posterior sample is already a success probability in [0,1]
                    p_stage = float(np.clip(post_arr[idx], 0.0, 1.0))
                else:
                    # Fall back to prior interval
                    p_stage = float(np.clip(rng.uniform(succ_min, succ_max), 0.0, 1.0))
            else:
                p_stage = float(np.clip(rng.uniform(succ_min, succ_max), 0.0, 1.0))
        else:
            # No prior info for this technique; use neutral success before capability
            p_stage = 0.5

        # Threat Capability modifies technique-level success probability
        # in a FAIR-consistent way:
        #   vulnerability v = TC * Susceptibility
        # This guarantees:
        #   - v <= tc for all techniques
        #   - stronger controls (lower p_stage) produce lower v
        if tc is not None:
            p_stage = float(np.clip(tc * p_stage, 0.0, 1.0))

        # --------------------------------------------------------
        # DETECTION probability (BN posterior if available)
        # --------------------------------------------------------
        detect_prob = DETECT_BASE
        if tech_posteriors is not None:
            key_det = (tactic_name, tech_id, "detect")
            det_arr = tech_posteriors.get(key_det)
            if det_arr is not None and det_arr.size > 0:
                didx = rng.integers(0, det_arr.size)
                detect_prob = float(np.clip(det_arr[didx], 0.0, 1.0))

        retries_left = MAX_RETRIES_PER_STAGE
        stage_completed = False

        if record_path:
            path_log[-1]["success_prob"] = p_stage
            path_log[-1]["detect_prob"] = detect_prob

        while retries_left > 0:
            if rng.random() < p_stage:
                if record_path:
                    path_log[-1]["result"] = "success"
                current_tactic_index += 1
                stage_completed = True
                break

            delta = float(rng.uniform(DETECT_INC_MIN, DETECT_INC_MAX))
            detect_prob = min(1.0, detect_prob + delta)

            if rng.random() < detect_prob:
                if record_path:
                    path_log[-1]["result"] = "detected"
                return (False, path_log) if record_path else False

            retries_left -= 1
            if record_path:
                path_log[-1]["retries"] += 1
            if retries_left <= 0:
                break

            # Adaptability gates retries
            if rng.random() >= actor_adapt:
                break

        if stage_completed:
            continue

        # Fallback logic for technique mode
        if (
            fallback_count < MAX_FALLBACKS_PER_CHAIN
            and current_tactic_index > 0
            and rng.random() < FALLBACK_PROB
        ):
            fallback_count += 1
            if record_path:
                path_log[-1]["fallback"] = True
            current_tactic_index -= 1
            continue
        else:
            if record_path and path_log and path_log[-1]["result"] is None:
                path_log[-1]["result"] = "fail"
            return (False, path_log) if record_path else False

    result = current_tactic_index >= n_tactics
    if record_path:
        return result, path_log
    return result

# Determine whether technique-level simulation inputs are available
if get_technique_simulation_inputs is not None:
    technique_sim_struct = get_technique_simulation_inputs()
else:
    technique_sim_struct = None

technique_inputs_available = (
    isinstance(technique_sim_struct, dict)
    and "tactics" in technique_sim_struct
    and "techniques_by_tactic" in technique_sim_struct
    and "technique_priors" in technique_sim_struct
)

def _simulate_attacker_path_tactics(success_probs, rng):
    """
    Simulate stage-by-stage attacker progression with retries, detection checks,
    adaptability adjustments, and fallbacks.

    This version is identical to the original except that the detection
    probability increase per retry is no longer static. Instead it is sampled
    from DETECT_INC_MIN to DETECT_INC_MAX once per retry attempt.
    """
    n_stages = len(success_probs)
    if n_stages == 0:
        return False

    # Sample an actor-level adaptability value once for this path.
    if ADAPTABILITY_STOCHASTIC:
        actor_adapt = float(rng.uniform(*ADAPTABILITY_RANGE))
    else:
        actor_adapt = float(np.mean(ADAPTABILITY_RANGE))

    i = 0
    fallback_count = 0

    while 0 <= i < n_stages:
        p_stage = float(success_probs[i])

        detect_prob = DETECT_BASE
        retries_left = MAX_RETRIES_PER_STAGE

        stage_completed = False

        while retries_left > 0:
            # Attempt
            if rng.random() < p_stage:
                i += 1
                stage_completed = True
                break

            # Failure: detection increment sampled from range
            inc = float(rng.uniform(DETECT_INC_MIN, DETECT_INC_MAX))
            detect_prob = min(1.0, detect_prob + inc)

            # Detection check
            if rng.random() < detect_prob:
                return False

            # Retry consumed
            retries_left -= 1
            if retries_left <= 0:
                break

            # Adaptability gate: only determines if retry is allowed
            if rng.random() >= actor_adapt:
                break

        if stage_completed:
            continue

        # No success and not detected — fallback or failure
        if (
            rng.random() < FALLBACK_PROB
            and fallback_count < MAX_FALLBACKS_PER_CHAIN
            and i > 0
        ):
            fallback_count += 1
            i = max(0, i - 1)
            continue
        else:
            return False

    return i >= n_stages

if technique_inputs_available:
    simulate_fn = _simulate_attacker_path
else:
    simulate_fn = _simulate_attacker_path_tactics

def _extract_technique_posteriors_from_trace(trace, technique_sim_struct):
    """
    Extract posterior draws for each technique node from a BN trace.

    Returns:
        dict[(tactic_name, tech_id, kind)] -> 1D np.ndarray of posterior samples in [0,1],
        where kind is "success" or "detect".
    """
    if technique_sim_struct is None:
        return None

    tactics = technique_sim_struct.get("tactics", [])
    technique_priors = technique_sim_struct.get("technique_priors", {})
    if not tactics or not technique_priors:
        return None

    tech_posteriors = {}
    posterior = trace.posterior

    for tactic in tactics:
        priors_for_tactic = technique_priors.get(tactic, {})
        for tech_id in priors_for_tactic.keys():
            # Success node
            succ_name = (
                f"p_tech_{tactic.replace(' ', '_')}_"
                f"{tech_id.replace('.', '_')}"
            )
            if succ_name in posterior:
                arr = posterior[succ_name].values
                tech_posteriors[(tactic, tech_id, "success")] = np.asarray(
                    arr, dtype=float
                ).reshape(-1)

            # Detection node (if present)
            det_name = (
                f"p_detect_{tactic.replace(' ', '_')}_"
                f"{tech_id.replace('.', '_')}"
            )
            if det_name in posterior:
                darr = posterior[det_name].values
                tech_posteriors[(tactic, tech_id, "detect")] = np.asarray(
                    darr, dtype=float
                ).reshape(-1)

    if not tech_posteriors:
        return None
    return tech_posteriors

def _sample_posterior_lambda_and_success(alphas: np.ndarray, betas: np.ndarray, n_stages: int):
    """
    Build and sample PyMC model for λ and per-stage success (subset-aware).
    Returns (lambda_draws, success_chain_draws, succ_mat_or_None, technique_posteriors_or_None).
    """
    if not HAVE_PYMC:
        # Prior-only fallback (keeps script runnable without PyMC)
        rng = np.random.default_rng(RANDOM_SEED)
        mu_l = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sig_l = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lam = rng.lognormal(mean=mu_l, sigma=sig_l, size=N_SAMPLES)
        succ_mat = rng.beta(alphas, betas, size=(N_SAMPLES, n_stages))
        succ_chain = np.prod(succ_mat, axis=1)
        return lam, succ_chain, None, None

    # Select default model path. BN is default whenever technique inputs exist.
    use_bn_model = False
    if technique_inputs_available:
        use_bn_model = True

    if use_bn_model:
        model = build_technique_bn_model(
            technique_sim_struct,
            alphas,
            betas,
            observed_total_incidents,
            observed_years,
        )
    else:
        with pm.Model() as model:
            # Lognormal prior for attempt frequency (λ)
            mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
            sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
            lambda_rate = pm.Lognormal("lambda_rate", mu=mu_lambda, sigma=sigma_lambda)

            # Per-stage success probabilities (Beta priors) over the included tactics
            success_probs = pm.Beta("success_probs", alpha=alphas, beta=betas, shape=n_stages)

            if (observed_total_incidents is not None) and (observed_years is not None):
                pm.Poisson("obs_incidents", mu=lambda_rate * observed_years, observed=observed_total_incidents)
                print(f"Conditioning on observed data: {observed_total_incidents} incidents over {observed_years} years.")
            else:
                print("No observed incident data provided — running fully prior-driven.")

    # SHARED SAMPLING CALL
    with model:
        trace = pm.sample(
            draws=N_SAMPLES,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )

    # ============================================================
    # BN MODE POSTERIOR EXTRACTION
    # ============================================================
    if use_bn_model:

        # Validate required BN nodes exist
        if "lambda_rate" not in trace.posterior:
            raise RuntimeError("BN posterior missing lambda_rate node.")

        if "p_chain_success" not in trace.posterior:
            raise RuntimeError("BN posterior missing p_chain_success node.")

        # Extract lambda draws
        lambda_draws = trace.posterior["lambda_rate"].values.flatten()

        # Extract chain success draws
        chain_draws = trace.posterior["p_chain_success"].values.flatten()

        # BN mode does NOT use stage success matrices
        succ_mat = None

        # Validate shapes
        if len(lambda_draws) != len(chain_draws):
            raise RuntimeError(
                "BN posterior mismatch: lambda_draws and chain_draws lengths differ."
            )

        # Extract technique-level posterior samples for hybrid simulation
        technique_posteriors = _extract_technique_posteriors_from_trace(
            trace,
            technique_sim_struct,
        )

        return lambda_draws, chain_draws, succ_mat, technique_posteriors

    # ============================================================
    # TACTIC/STAGE MODEL POSTERIOR EXTRACTION (fallback)
    # ============================================================
    lambda_draws = np.asarray(trace.posterior["lambda_rate"]).reshape(-1)
    succ_mat = np.asarray(trace.posterior["success_probs"]).reshape(-1, n_stages)

    # Compute chain success for stage model
    succ_chain_draws = np.prod(succ_mat, axis=1)

    # Validate alignment
    if len(lambda_draws) != len(succ_chain_draws):
        raise RuntimeError(
            "Stage posterior mismatch: lambda_draws and succ_chain_draws lengths differ."
        )

    return lambda_draws, succ_chain_draws, succ_mat, None

def build_technique_bn_model(
    technique_sim_struct,
    alphas,
    betas,
    observed_total_incidents=None,
    observed_years=None,
    prior_strength=50.0,
):
    """
    Construct a TRUE Bayesian Network model using PyMC, where:

      • λ (attempt frequency) is a stochastic BN node.
      • Techniques are explicit probability nodes derived from mitigation strengths.
      • Tactics are deterministic aggregators of techniques, NOT independent priors.
      • Chain success is a deterministic product of tactics.
      • Incident likelihood is Poisson(λ * years * chain_success).

    This does NOT replace the existing PyMC model. It is an alternate model
    used ONLY when technique-level simulation is active and explicitly selected.
    """

    import pymc as pm
    # ------------------------------------------------------------
    # Approximate per-technique probability of "win before detect"
    # across multiple retries.
    #
    # This mirrors _simulate_attacker_path semantics in a BN-safe way.
    # ------------------------------------------------------------
    def _bn_tech_win_probability(p_succ, p_detect_base, p_detect_inc_mean, n_tries):
        survive = 1.0
        win = 0.0
        n = int(max(1, n_tries))

        for k in range(n):
            d_k = pm.math.clip(
                p_detect_base + k * p_detect_inc_mean,
                0.0,
                1.0
            )

            win = win + survive * p_succ
            survive = survive * (1.0 - p_succ) * (1.0 - d_k)

        return pm.math.clip(win, 0.0, 1.0)

    # ------------------------------------------------------------
    # Extract structures from technique_sim_struct
    # ------------------------------------------------------------
    tactics = technique_sim_struct.get("tactics", [])
    technique_priors = technique_sim_struct.get("technique_priors", {})

    if not tactics or not technique_priors:
        raise ValueError("Technique BN requested but technique_sim_struct is empty.")

    # ------------------------------------------------------------
    # Begin BN construction
    # ------------------------------------------------------------
    with pm.Model() as model:

        # ========================================================
        # 1. Frequency node λ  (attack attempt rate)
        # ========================================================
        mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)

        lambda_rate = pm.Lognormal(
            "lambda_rate",
            mu=mu_lambda,
            sigma=sigma_lambda
        )

        # ========================================================
        # 2. Technique-level BN nodes
        # ========================================================
        tech_nodes = {}

        for tactic in tactics:
            priors_for_tactic = technique_priors.get(tactic, {})

            for tech_id, pri in priors_for_tactic.items():

                p_min = float(pri["p_min"])
                p_max = float(pri["p_max"])

                # Convert [p_min, p_max] into a Beta prior suitable for a BN.
                # We keep the center at the midpoint, but cap the overall
                # concentration so that alpha+beta cannot explode when the
                # interval is extremely narrow.
                center = 0.5 * (p_min + p_max)
                # Keep center inside (0,1) for numerical stability
                center = min(max(center, 1e-6), 1.0 - 1e-6)

                width = abs(p_max - p_min)
                # Treat any interval narrower than 0.05 as "tight" for k scaling
                eff_width = max(width, 0.05)

                # Base concentration proportional to inverse width, but capped
                k_raw = prior_strength / eff_width
                # Clamp k between 2 and 200 to avoid extreme shapes
                k = float(min(max(k_raw, 2.0), 200.0))

                alpha = max(1.0, center * k)
                beta = max(1.0, (1.0 - center) * k)

                # Name node safely
                node_name = (
                    f"p_tech_{tactic.replace(' ', '_')}_"
                    f"{tech_id.replace('.', '_')}"
                )

                tech_nodes[(tactic, tech_id)] = pm.Beta(
                    node_name,
                    alpha=alpha,
                    beta=beta
                )

        # --------------------------------------------------------
        # 2b. Technique-level DETECTION nodes
        # --------------------------------------------------------
        detect_nodes = {}

        for tactic in tactics:
            priors_for_tactic = technique_priors.get(tactic, {})
            for tech_id, pri in priors_for_tactic.items():
                p_det_min = float(pri.get("p_detect_min", 0.0))
                p_det_max = float(pri.get("p_detect_max", 1.0))

                det_center = 0.5 * (p_det_min + p_det_max)
                det_center = min(max(det_center, 1e-6), 1.0 - 1e-6)

                det_width = abs(p_det_max - p_det_min)
                eff_width = max(det_width, 0.05)

                k_raw = prior_strength / eff_width
                k = float(min(max(k_raw, 2.0), 200.0))

                det_alpha = max(1.0, det_center * k)
                det_beta  = max(1.0, (1.0 - det_center) * k)

                node_name = (
                    f"p_detect_{tactic.replace(' ', '_')}_"
                    f"{tech_id.replace('.', '_')}"
                )

                detect_nodes[(tactic, tech_id)] = pm.Beta(
                    node_name,
                    alpha=det_alpha,
                    beta=det_beta
                )

        # ========================================================
        # 3. Tactic-level deterministic nodes
        #
        # NOTE: Only tactics containing ≥ 1 techniques are included.
        #       Tactics with zero relevant techniques ARE IGNORED.
        # ========================================================
        tactic_nodes = {}

        for tactic in tactics:

            # All technique nodes belonging to this tactic
            techs_here = [
                tech_nodes[(t, tech_id)]
                for (t, tech_id) in tech_nodes.keys()
                if t == tactic
            ]

            # If tactic has no techniques, ignore it entirely.
            if not techs_here:
                continue

            if BN_CHAIN_SEMANTICS == "legacy_mean_product":
                # Legacy behavior: mean susceptibility across techniques
                if len(techs_here) == 1:
                    p_tactic = techs_here[0]
                else:
                    p_tactic = pm.math.mean(pm.math.stack(techs_here))
            else:
                # Retry + detection aware tactic win probability
                det_inc_mean = 0.5 * (float(DETECT_INC_MIN) + float(DETECT_INC_MAX))
                p_det_base = float(DETECT_BASE)

                tech_ids_here = [
                    tech_id for (t, tech_id) in tech_nodes.keys()
                    if t == tactic
                ]

                wins = []
                for tech_id in tech_ids_here:
                    p_node = tech_nodes[(tactic, tech_id)]
                    d_node = detect_nodes.get((tactic, tech_id), p_det_base)

                    wins.append(
                        _bn_tech_win_probability(
                            p_node,
                            d_node,
                            det_inc_mean,
                            MAX_RETRIES_PER_STAGE
                        )
                    )

                # Uniform technique selection
                p_tactic = pm.math.mean(pm.math.stack(wins))

            tactic_nodes[tactic] = pm.Deterministic(
                f"p_tactic_{tactic.replace(' ', '_')}",
                p_tactic
            )

        # ========================================================
        # 4. Chain Success Node
        #
        # Product of the tactics that actually have tactic_nodes.
        # ========================================================
        ordered_tactics = [t for t in tactics if t in tactic_nodes]

        if not ordered_tactics:
            raise ValueError(
                "Technique BN could not build chain: no tactics with techniques."
            )

        chain_success = pm.Deterministic(
            "p_chain_success",
            pm.math.prod([tactic_nodes[t] for t in ordered_tactics])
        )

        # ========================================================
        # 5. Observed incidents (optional evidence)
        # ========================================================
        if (observed_total_incidents is not None) and (observed_years is not None):
            pm.Poisson(
                "obs_incidents",
                mu=lambda_rate * observed_years * chain_success,
                observed=observed_total_incidents
            )

    return model

def _simulate_annual_losses(lambda_draws, succ_chain_draws, succ_mat,
                            alphas, betas,
                            tactics_included,
                            impact_reduction_controls=None,
                            severity_median=500_000.0,
                            severity_gsd=2.0,
                            rng_seed=1234,
                            technique_posteriors=None):
    """
    Posterior predictive per-draw Monte Carlo.
      - Draw attempts ~ Poisson(λ)
      - For each attempt, simulate stage progression over *tactics_included*
      - On success, draw per-category losses and apply impact reductions if provided.
    Returns:
      losses (N,), successes (N,).
    Also populates LAST_CATEGORY_LOSSES with per-category annual totals.
    """
    global LAST_CATEGORY_LOSSES

    rng = np.random.default_rng(rng_seed)
    mu = math.log(max(1e-9, severity_median))
    sigma = math.log(max(1.000001, severity_gsd))

    n = len(lambda_draws)
    losses = np.zeros(n, dtype=float)
    successes = np.zeros(n, dtype=int)

    prod_losses = np.zeros(n, dtype=float)
    resp_losses = np.zeros(n, dtype=float)
    reg_losses  = np.zeros(n, dtype=float)
    rep_losses  = np.zeros(n, dtype=float)

    attempts_per_draw = np.zeros(n, dtype=int)

    # Technique statistics (aggregated across all posterior draws)
    technique_attempts = {}
    technique_successes = {}

    # Pre-extract ranges (converted to [0..1]) for the two impact controls
    backup_lo = backup_hi = encrypt_lo = encrypt_hi = 0.0
    backup_mean = encrypt_mean = 0.0
    if impact_reduction_controls:
        backup_ctrl = impact_reduction_controls.get("Data Backup", {}) or {}
        encrypt_ctrl = impact_reduction_controls.get("Encrypt Sensitive Information", {}) or {}

        backup_lo   = float(backup_ctrl.get("min_strength", 0.0)) / 100.0
        backup_hi   = float(backup_ctrl.get("max_strength", 0.0)) / 100.0
        backup_mean = float(backup_ctrl.get("mean_strength", 0.0)) / 100.0

        encrypt_lo   = float(encrypt_ctrl.get("min_strength", 0.0)) / 100.0
        encrypt_hi   = float(encrypt_ctrl.get("max_strength", 0.0)) / 100.0
        encrypt_mean = float(encrypt_ctrl.get("mean_strength", 0.0)) / 100.0

    for idx, lam in enumerate(lambda_draws):
        attempts = rng.poisson(lam=lam)
        attempts_per_draw[idx] = max(0, attempts)
        succ_count = 0
        prod_acc = resp_acc = reg_acc = rep_acc = 0.0
        total_loss = 0.0

        # Draw baseline Threat Capability for this posterior draw (higher = stronger attacker)
        if THREAT_CAPABILITY_STOCHASTIC:
            tc = float(rng.uniform(*THREAT_CAPABILITY_RANGE))
        else:
            tc = float(np.mean(THREAT_CAPABILITY_RANGE))

        # Sample or fix impact reductions once per posterior draw
        if STOCHASTIC_IMPACT_REDUCTION:
            backup_s  = rng.uniform(backup_lo, backup_hi) if backup_hi > backup_lo else backup_lo
            encrypt_s = rng.uniform(encrypt_lo, encrypt_hi) if encrypt_hi > encrypt_lo else encrypt_lo
        else:
            backup_s, encrypt_s = backup_mean, encrypt_mean

        # Each Poisson draw is a number of full attacker attempts in that year
        for _ in range(max(0, attempts)):
            # Technique-level mode - use technique_sim_struct and apply tc in the path
            if technique_inputs_available and technique_sim_struct:
                chain_success, path = _simulate_attacker_path(
                    technique_sim_struct,
                    rng,
                    tc=tc,
                    tech_posteriors=technique_posteriors,
                    record_path=True,
                )

                # Aggregate per technique stats from this path
                if path is not None:
                    for step in path:
                        tech_key = (step["tactic"], step["technique"])
                        technique_attempts[tech_key] = technique_attempts.get(tech_key, 0) + 1
                        if step.get("result") == "success":
                            technique_successes[tech_key] = technique_successes.get(tech_key, 0) + 1

            else:
                # Tactic-level mode - use posterior or prior stage susceptibility
                if succ_mat is not None:
                    # succ_mat[idx] is the sampled success probability after controls
                    stage_success_probs = succ_mat[idx].astype(float)
                else:
                    # Draw success probability after controls from the Beta priors
                    stage_success_probs = rng.beta(alphas, betas).astype(float)

                # Apply Threat Capability as a ceiling:
                #   vulnerability v_stage = tc * susceptibility_stage
                # This preserves the BN posterior over susceptibility and ensures:
                #   - v_stage <= tc for every stage
                #   - stronger controls (lower susceptibility) reduce v_stage
                stage_success_probs = np.clip(tc * stage_success_probs, 0.0, 1.0)

                chain_success = _simulate_attacker_path_tactics(stage_success_probs, rng)

                # BUGFIX: second call here was redundant and overwrote the first result.
                # chain_success = _simulate_attacker_path_tactics(stage_success_probs, rng)

            if not chain_success:
                continue

            # Draw per-category losses (bounded lognormal body plus optional Pareto tail)
            prod = resp = reg = rep = 0.0
            for j, cat in enumerate(loss_categories):
                mu_j, sigma_j = cat_mu[j], cat_sigma[j]
                base_draw = float(rng.lognormal(mean=mu_j, sigma=sigma_j))
                # Cap the lognormal body around the upper tail for numerical stability
                lognorm_cap = math.exp(mu_j + 3.09 * sigma_j)
                base_draw = min(base_draw, lognorm_cap)

                if cat == "RegulatoryLegal":
                    reg = base_draw
                    # Occasional heavy legal tail
                    if rng.random() < 0.025:
                        xm = pareto_defaults[cat]["xm"]
                        alpha = pareto_defaults[cat]["alpha"]
                        u = rng.uniform(0.001, 0.999)
                        tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                        tail_cap  = xm * (0.95) ** (-1.0 / alpha)
                        reg = max(reg, min(tail_draw, tail_cap))

                elif cat == "ReputationCompetitive":
                    rep = base_draw
                    # Occasional heavy reputation tail
                    if rng.random() < 0.015:
                        xm = pareto_defaults[cat]["xm"]
                        alpha = pareto_defaults[cat]["alpha"]
                        u = rng.uniform(0.001, 0.999)
                        tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                        tail_cap  = xm * (0.95) ** (-1.0 / alpha)
                        rep = max(rep, min(tail_draw, tail_cap))

                elif cat == "Productivity":
                    prod = base_draw

                elif cat == "ResponseContainment":
                    resp = base_draw

            # Apply impact reductions
            if backup_s > 0.0:
                scale = max(0.0, 1.0 - backup_s * BACKUP_IMPACT_MULT)
                prod *= scale
                resp *= scale
            if encrypt_s > 0.0:
                scale = max(0.0, 1.0 - encrypt_s * ENCRYPT_IMPACT_MULT)
                reg *= scale
                rep *= scale

            # Accumulate into per-draw totals
            prod_acc += prod
            resp_acc += resp
            reg_acc  += reg
            rep_acc  += rep
            total_loss += prod + resp + reg + rep
            succ_count += 1

        losses[idx] = total_loss
        successes[idx] = succ_count
        prod_losses[idx] = prod_acc
        resp_losses[idx] = resp_acc
        reg_losses[idx]  = reg_acc
        rep_losses[idx]  = rep_acc

    # Summarize top techniques by number of attempts (technique mode only)
    if technique_attempts:
        technique_rows = []
        for (tactic, tech), attempts in technique_attempts.items():
            succ = technique_successes.get((tactic, tech), 0)
            failures = attempts - succ
            rate = succ / attempts if attempts > 0 else 0.0
            technique_rows.append((tactic, tech, attempts, succ, failures, rate))

        # Sort by attempts descending
        technique_rows.sort(key=lambda r: r[2], reverse=True)
        top_ten = technique_rows[:10]

        print("\nTop 10 techniques by attempts (technique mode only):")
        for tactic, tech, attempts, succ, failures, rate in top_ten:
            print(
                f"{tactic} / {tech}: attempts={attempts}, successes={succ}, failures={failures}, success_rate={rate:.1%}"
            )

        # Save to CSV in the same output directory
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        label = "all" if EXPORT_ALL_TECHNIQUES else "top10"
        csv_path = os.path.join(
            OUTPUT_DIR,
            f"techniques_{label}_{ts}.csv"
        )
        df = pd.DataFrame(
            [
                {
                    "Tactic": tactic,
                    "Technique": tech,
                    "Attempts": attempts,
                    "Successes": succ,
                    "Failures": failures,
                    "SuccessRate": rate,
                }
                for tactic, tech, attempts, succ, failures, rate in (
                    technique_rows if EXPORT_ALL_TECHNIQUES else top_ten
                )
            ]
        )
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved top technique statistics -> {csv_path}")

    LAST_CATEGORY_LOSSES = {
        "Productivity": prod_losses,
        "ResponseContainment": resp_losses,
        "RegulatoryLegal": reg_losses,
        "ReputationCompetitive": rep_losses,
    }
    return losses, successes, attempts_per_draw

# =============================================================================
# Console output, viz, exports
# =============================================================================
def _print_aal_summary(losses: np.ndarray, successes: np.ndarray):
    aal_mean   = float(np.mean(losses))
    aal_median = float(np.median(losses))
    lo, hi     = np.quantile(losses, [0.025, 0.975])
    mean_succ  = float(np.mean(successes))
    succ_lo, succ_hi = np.quantile(successes, [0.025, 0.975])
    pct_zero   = float(np.mean(successes == 0) * 100.0)

    print("\nAAL posterior predictive summary (with severity & tails):")
    print(f"Mean AAL: {_fmt_money(aal_mean)}")
    print(f"Median AAL: {_fmt_money(aal_median)}")
    print(f"AAL 95% credible interval (annualized total loss): {_fmt_money(lo)} – {_fmt_money(hi)}")
    print(f"Mean successful incidents / year: {mean_succ:.2f}")
    print(f"95% credible interval (incidents / year): {succ_lo:.2f} – {succ_hi:.2f}")

    # Mean loss per successful incident (SLE)
    valid = successes > 0
    if np.any(valid):
        per_event_losses = np.divide(losses[valid], successes[valid],
                                     out=np.zeros_like(losses[valid]),
                                     where=successes[valid] > 0)
        mean_loss_per_event = float(np.mean(per_event_losses))
        lo_event, hi_event  = np.quantile(per_event_losses, [0.025, 0.975])
        print(f"Mean loss per successful incident: {_fmt_money(mean_loss_per_event)}")
        print(f"95% credible interval (loss / incident): {_fmt_money(lo_event)} – {_fmt_money(hi_event)}")
    else:
        print("Mean loss per successful incident: (no successful incidents in simulation)")

    print(f"% years with zero successful incidents: {pct_zero:.1f}%")

    # Category breakdown
    print("\nCategory-level annual loss 95% credible intervals:")
    if LAST_CATEGORY_LOSSES is not None:
        for c in loss_categories:
            arr = LAST_CATEGORY_LOSSES.get(c, np.zeros_like(losses))
            lw, up = np.quantile(arr, [0.025, 0.975])
            med    = float(np.median(arr))
            pct_of_med = (med / aal_median) * 100.0 if aal_median > 0 else 0.0
            print(f"  {c:<24} {_fmt_money(lw)} – {_fmt_money(up)} "
                  f"(median {_fmt_money(med)}, ~{pct_of_med:.1f}% of median AAL)")
    else:
        print("  (Per-category breakdown unavailable.)")

def _annotate_percentiles(ax, samples, money=False):
    pcts = [50, 90, 95, 99]
    vals = np.percentile(samples, pcts)
    ymin, ymax = ax.get_ylim()
    ytext = ymax * 0.95
    for i, (p, v) in enumerate(zip(pcts, vals)):
        ax.axvline(v, linestyle="--", linewidth=1.0)
        label = _fmt_money(v) if money else (f"{v:.3f}" if v < 10 else f"{v:,.2f}")
        y_offset = (i % 2) * 0.05 * (ymax - ymin)
        ax.text(v, ytext - y_offset, f"P{p}={label}",
                rotation=0, va="bottom", ha="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))

def _render_2x2_and_log_ale(losses: np.ndarray,
                            lambda_draws: np.ndarray,
                            success_chain_draws: np.ndarray,
                            successes: np.ndarray,
                            attempts_per_draw: np.ndarray,
                            show: bool = True):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rate_sim = successes.astype(float)   # successes per year (exposure = 1 year)
    # Implied per-attempt success probability from simulation
    attempts_safe = np.where(attempts_per_draw > 0, attempts_per_draw, 1)
    implied_success_prob = successes.astype(float) / attempts_safe.astype(float)
    # Use analytic posterior success metrics for the dashboard,
    # consistent with the original MCMC model:
    #   succ_chain_sim = success_chain_draws (end-to-end success probability)
    #   succ_per_year  = lambda_draws * success_chain_draws (expected successful incidents/year)
    succ_chain_sim = success_chain_draws.astype(float)
    succ_per_year = lambda_draws * success_chain_draws

    def _auto_clip(data, low=0.001, high=0.991):
        if len(data) == 0:
            return data
        low_v, high_v = np.percentile(data, [low * 100, high * 100])
        return data[(data >= low_v) & (data <= high_v)]

    lambda_plot = _auto_clip(lambda_draws)
    succ_chain_plot = succ_chain_sim
    succ_per_year_plot = succ_per_year
    losses_plot = _auto_clip(losses)

    def _millions(x, pos): return f"${x/1e6:,.1f}M"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0,0]
    ax.hist(lambda_plot, bins=60, edgecolor="black")
    ax.set_title("Posterior λ (incidents/year)")
    ax.set_xlabel("λ"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, lambda_plot, money=False)

    ax = axs[0,1]
    ax.hist(implied_success_prob, bins=60, alpha=0.9, edgecolor="black")
    ax.set_title("Posterior Success Probability (end-to-end)")
    ax.set_xlabel("Success prob"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, implied_success_prob, money=False)

    ax = axs[1,0]
    # Plot histogram in COUNT space
    counts, bin_edges, _ = ax.hist(
        rate_sim,
        bins=60,
        alpha=0.9,
        edgecolor="black"
    )
    # Compute correct bin width for scaling KDE to counts
    bin_width = bin_edges[1] - bin_edges[0]
    # KDE evaluated over the histogram range
    if bin_edges[-1] > bin_edges[0]:
        kde = gaussian_kde(rate_sim)
        xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
        ys = kde(xs) * len(rate_sim) * bin_width
        ax.plot(xs, ys, color="red", linewidth=1.2)
    else:
        # Degenerate case: all values identical (e.g., mostly zero incidents)
        # Skip KDE overlay to avoid matplotlib zero-width error
        pass
    ax.set_title("Successful Incidents / Year (posterior)")
    ax.set_xlabel("Incidents/year")
    ax.set_ylabel("Count")
    _annotate_percentiles(ax, rate_sim, money=False)


    ax = axs[1,1]
    ax.hist(losses_plot, bins=60, edgecolor="black")
    ax.set_title("Annual Loss (posterior predictive)")
    ax.set_xlabel("Annual loss"); ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    _annotate_percentiles(ax, losses_plot, money=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(top=0.90)
    dash_path = os.path.join(OUTPUT_DIR, f"dashboard_2x2_{ts}.png")
    fig.savefig(dash_path, dpi=150)
    print(f"✅ Saved 2×2 dashboard → {dash_path}")

    # Log-scale ALE histogram
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bins = np.logspace(np.log10(1e3), np.log10(max(1e5, max(1.0, losses_plot.max()))), 60)
    ax2.hist(losses_plot, bins=bins, edgecolor="black")
    ax2.set_xscale('log')
    ax2.set_title("Annualized Loss (Log Scale)")
    ax2.set_xlabel("Annual loss (log)"); ax2.set_ylabel("Count")
    ax2.xaxis.set_major_formatter(FuncFormatter(_millions))
    _annotate_percentiles(ax2, losses_plot, money=True)
    fig2.tight_layout()
    ale_path = os.path.join(OUTPUT_DIR, f"ale_log_chart_{ts}.png")
    fig2.savefig(ale_path, dpi=150)
    print(f"✅ Saved ALE chart → {ale_path}")

    # Loss Exceedance Curve (LEC)
    sorted_losses = np.sort(losses_plot)
    exceed_probs = 1.0 - np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    exceed_probs_percent = exceed_probs * 100

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(sorted_losses, exceed_probs_percent, lw=2, color="orange")
    ax3.set_xscale('log')
    ax3.set_xlabel("Annual Loss")
    ax3.set_ylabel("Exceedance Probability (%)")
    ax3.set_title("Loss Exceedance Curve (Annual Loss)")
    ax3.grid(True, which="both", ls="--", lw=0.5)
    ax3.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    pcts = [50, 90, 95, 99]
    vals = np.percentile(sorted_losses, pcts)
    for p, v in zip(pcts, vals):
        prob = 100 * (1 - p / 100.0)
        ax3.axvline(v, ls="--", lw=0.8, color="gray")
        y_text = min(100, prob + 5)
        ax3.text(v, y_text, f"P{p}\n${v:,.0f}",
                 rotation=90, va="bottom", ha="left", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))

    lec_path = os.path.join(OUTPUT_DIR, f"loss_exceedance_curve_{ts}.png")
    fig3.tight_layout()
    fig3.savefig(lec_path, dpi=150)
    print(f"✅ Saved Loss Exceedance Curve → {lec_path}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not display figures: {e}")

def _save_results_csvs(losses: np.ndarray, successes: np.ndarray,
                       lambda_draws: np.ndarray, success_chain_draws: np.ndarray):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_results_{ts}.csv")
    pd.DataFrame({
        "lambda": lambda_draws,
        "p_success_chain": success_chain_draws,
        "annual_loss": losses,
        "successful_incidents": successes
    }).to_csv(results_csv, index=False)

    aal_mean   = float(np.mean(losses))
    aal_median = float(np.median(losses))
    aal_lo, aal_hi = np.quantile(losses, [0.025, 0.975])

    mean_succ  = float(np.mean(successes))
    succ_lo, succ_hi = np.quantile(successes, [0.025, 0.975])
    pct_zero   = float(np.mean(successes == 0) * 100.0)

    valid = successes > 0
    if np.any(valid):
        per_event_losses = np.divide(losses[valid], successes[valid],
                                     out=np.zeros_like(losses[valid]),
                                     where=successes[valid] > 0)
        mean_loss_per_event = float(np.mean(per_event_losses))
        lo_event, hi_event  = np.quantile(per_event_losses, [0.025, 0.975])
    else:
        mean_loss_per_event, lo_event, hi_event = 0.0, 0.0, 0.0

    summary_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_summary_{ts}.csv")
    pd.DataFrame([{
        "Mean_AAL": aal_mean,
        "Median_AAL": aal_median,
        "AAL_95_Lower": aal_lo,
        "AAL_95_Upper": aal_hi,
        "Mean_Incidents": mean_succ,
        "Zero_Incident_Years_%": pct_zero,
        "n": int(losses.size),
        "Incidents_95_Lower": succ_lo,
        "Incidents_95_Upper": succ_hi,
        "Mean_Loss_Per_Incident": mean_loss_per_event,
        "Loss_Per_Incident_95_Lower": lo_event,
        "Loss_Per_Incident_95_Upper": hi_event,
        "Mean_AAL_Check_MeanInc_x_MeanLossPerIncident": mean_succ * mean_loss_per_event
    }]).to_csv(summary_csv, index=False)

    print(f"✅ Detailed results exported → {results_csv}")
    print(f"✅ Summary statistics exported → {summary_csv}")

# =============================================================================
# CLI + main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Cyber incident model with MITRE-informed controls + PyMC + FAIR.")
    p.add_argument("--dataset", default="enterprise-attack.json", help="MITRE ATT&CK STIX bundle path.")
    p.add_argument("--csv", default="mitigation_control_strengths.csv", help="Mitigation strengths CSV path.")
    p.add_argument("--no-adapt-stochastic", action="store_true",
                   help="Disable stochastic adaptation (use fixed factor)")
    p.add_argument("--no-plot", action="store_true", help="Save figures but do not open GUI windows.")
    p.add_argument("--print-control-strengths", action="store_true",
                   help="Print the per-tactic control strength parameters used (for diagnostics).")
    p.add_argument("--no-stochastic-impact", action="store_true",
                   help="Disable stochastic impact reduction (use mean instead).")
    p.add_argument(
        "--export-all-techniques",
        action="store_true",
        help="Export statistics for all in-scope techniques instead of only top 10."
    )
    return p.parse_args()

def main():
    global ADAPTABILITY_STOCHASTIC, STOCHASTIC_IMPACT_REDUCTION
    global EXPORT_ALL_TECHNIQUES

    args = parse_args()
    if args.no_adapt_stochastic:
        ADAPTABILITY_STOCHASTIC = False
    if args.no_stochastic_impact:
        STOCHASTIC_IMPACT_REDUCTION = False
    if args.export_all_techniques:
        EXPORT_ALL_TECHNIQUES = True

    # Load tactic subset & ranges from dashboard or fallback to SME map
    tactics_included, stage_map, impact_controls, mode = _load_from_dashboard_or_fallback(args.dataset, args.csv)

    if args.print_control_strengths:
        _print_stage_control_map(stage_map, tactics_included)

    # Build Beta priors for per-stage success over the included tactics
    alphas, betas = _build_beta_priors_from_stage_map(stage_map, tactics_included)

    # Sample posterior (λ and end-to-end success probability)
    lambda_draws, success_chain_draws, succ_mat, technique_posteriors = _sample_posterior_lambda_and_success(
        alphas, betas, n_stages=len(tactics_included)
    )


    # Posterior predictive simulation (annual losses & incident counts)
    losses, successes, attempts_per_draw = _simulate_annual_losses(
        lambda_draws=lambda_draws,
        succ_chain_draws=success_chain_draws,
        succ_mat=succ_mat,
        alphas=alphas,
        betas=betas,
        tactics_included=tactics_included,
        impact_reduction_controls=impact_controls,
        severity_median=500_000.0,
        severity_gsd=2.0,
        rng_seed=RANDOM_SEED + 1,
        technique_posteriors=technique_posteriors,
    )

    _print_aal_summary(losses, successes)
    _save_results_csvs(losses, successes, lambda_draws, success_chain_draws)
    _render_2x2_and_log_ale(
        losses,
        lambda_draws,
        success_chain_draws,
        successes,
        attempts_per_draw,
        show=(not args.no_plot),
    )

# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        sys.exit(2)