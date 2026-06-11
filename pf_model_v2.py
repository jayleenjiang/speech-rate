# -*- coding: utf-8 -*-
"""
Particle Filter Model of Distal Speech Rate Effects (Vectorized)
================================================================
Vectorized version for speed. Particles stored as parallel numpy arrays.
"""

import numpy as np
from collections import Counter
from scipy.stats import norm
import matplotlib.pyplot as plt
import time as timer

# =========================================================================
# 1. SETUP
# =========================================================================
STATES = ["START", "lei", "sure", "or", "time", "STOP"]
EMIT_STATES = ["lei", "sure", "or", "time"]
N_STATES = len(STATES)
IDX = {s: k for k, s in enumerate(STATES)}
STATE_NAME = {k: s for k, s in enumerate(STATES)}

BASE_DURATIONS = {"lei": 20, "sure": 20, "or": 10, "time": 20}
DURATION_STD_FACTOR = 0.20
MU = {"lei": -1.0, "sure": 0.15, "or": 0.05, "time": 1.0}
SIGMA = {s: 0.3 for s in EMIT_STATES}

# =========================================================================
# 2. TRANSITIONS
# =========================================================================
def make_speaker_transitions():
    T = np.zeros((N_STATES, N_STATES))
    T[IDX["START"], IDX["lei"]]  = 1.0
    T[IDX["lei"],   IDX["sure"]] = 1.0
    T[IDX["sure"],  IDX["or"]]   = 1.0
    T[IDX["or"],    IDX["time"]] = 1.0
    T[IDX["time"],  IDX["STOP"]] = 1.0
    T[IDX["STOP"],  IDX["STOP"]] = 1.0
    return T

def make_listener_transitions():
    T = np.zeros((N_STATES, N_STATES))
    def set_row(from_s, targets):
        row = np.zeros(N_STATES)
        for to_s, w in targets.items(): row[IDX[to_s]] = w
        leftover = max(0.0, 1.0 - row.sum())
        mask = np.ones(N_STATES, dtype=bool)
        for to_s in targets: mask[IDX[to_s]] = False
        mask[IDX["START"]] = False
        if from_s == "STOP": mask[:] = False; mask[IDX["STOP"]] = True
        if mask.sum() > 0 and leftover > 0: row[mask] = leftover / mask.sum()
        return row / row.sum()
    T[IDX["START"]] = set_row("START", {"lei": 0.97})
    T[IDX["lei"]]   = set_row("lei",   {"sure": 0.94})
    T[IDX["sure"]]  = set_row("sure",  {"or": 0.55, "time": 0.40})
    T[IDX["or"]]    = set_row("or",    {"time": 0.95})
    T[IDX["time"]]  = set_row("time",  {"STOP": 0.96})
    T[IDX["STOP"]]  = set_row("STOP",  {"STOP": 1.0})
    return T

SPEAKER_TRANS = make_speaker_transitions()
LISTENER_TRANS = make_listener_transitions()

# Precompute "go" distributions: for each state, the distribution over
# next states given that we DO transition (excluding self-loop).
LISTENER_GO = np.zeros((N_STATES, N_STATES))
for si in range(N_STATES):
    row = LISTENER_TRANS[si].copy()
    row[si] = 0.0
    s = row.sum()
    if s > 0: row /= s
    LISTENER_GO[si] = row

# Precompute per-state duration params as arrays for vectorized access
# Index by state_idx; non-emit states get dummy values
_BASE_DUR_ARR = np.zeros(N_STATES)
_DUR_STD_ARR = np.zeros(N_STATES)
for s in EMIT_STATES:
    _BASE_DUR_ARR[IDX[s]] = BASE_DURATIONS[s]
    _DUR_STD_ARR[IDX[s]] = BASE_DURATIONS[s] * DURATION_STD_FACTOR

_MU_ARR = np.full(N_STATES, np.nan)
_SIGMA_ARR = np.full(N_STATES, np.nan)
for s in EMIT_STATES:
    _MU_ARR[IDX[s]] = MU[s]
    _SIGMA_ARR[IDX[s]] = SIGMA[s]

# =========================================================================
# 3. VECTORIZED HAZARD
# =========================================================================
def hazard_stay_prob_vec(dwell_arr, state_arr, rate):
    """
    Vectorized: compute P(stay) for arrays of (dwell_time, state_idx).
    """
    N = len(dwell_arr)
    p_stay = np.zeros(N)
    
    mu_dur = _BASE_DUR_ARR[state_arr]
    sigma_dur = _DUR_STD_ARR[state_arr]
    
    # Only compute for emitting states
    emit_mask = sigma_dur > 0
    
    d_can = dwell_arr[emit_mask] * rate
    d_prev_can = (dwell_arr[emit_mask] - 1) * rate
    mu_d = mu_dur[emit_mask]
    sig_d = sigma_dur[emit_mask]
    
    surv_now = 1.0 - norm.cdf(d_can, loc=mu_d, scale=sig_d)
    surv_prev = 1.0 - norm.cdf(d_prev_can, loc=mu_d, scale=sig_d)
    
    # Safe division: avoid 0/0 when both survival values are tiny
    safe_prev = np.where(surv_prev > 1e-12, surv_prev, 1.0)
    ratio = np.where(surv_prev > 1e-12, surv_now / safe_prev, 0.0)
    p_stay[emit_mask] = np.clip(ratio, 0.0, 1.0)
    
    return p_stay

# =========================================================================
# 4. SPEAKER
# =========================================================================
def sigmoid_kappa(t, dur, steep):
    return 1.0 / (1.0 + np.exp(steep * (t - dur)))

def generate_utterance(rate, steepness=0.5, max_frames=1000, rng=None):
    if rng is None: rng = np.random.default_rng()
    s = "START"; states = []; emissions = []; dwell = 0
    for _ in range(max_frames):
        if s in ("START", "STOP"):
            probs = SPEAKER_TRANS[IDX[s]]
        else:
            eff = BASE_DURATIONS[s] / rate
            k = sigmoid_kappa(dwell, eff, steepness)
            row = SPEAKER_TRANS[IDX[s]].copy()
            row = (1-k)*row; row[IDX[s]] += k; probs = row/row.sum()
        sn = rng.choice(STATES, p=probs)
        if sn == "STOP": break
        dwell = dwell+1 if sn == s else 1
        s = sn; states.append(s)
        if s in EMIT_STATES:
            emissions.append(rng.normal(loc=MU[s], scale=SIGMA[s]))
    return states, np.array(emissions).reshape(-1,1)

# =========================================================================
# 5. VECTORIZED PARTICLE FILTER
# =========================================================================
def precompute_emission_logp(audio):
    T = len(audio)
    logp = np.full((T, N_STATES), -np.inf)
    for s in EMIT_STATES:
        logp[:, IDX[s]] = norm.logpdf(audio[:, 0], loc=MU[s], scale=SIGMA[s])
    return logp

def particle_filter_decode(audio, rate, num_particles=300,
                           resample_threshold=0.5, rng=None):
    if rng is None: rng = np.random.default_rng()
    
    T_len = len(audio)
    log_emit = precompute_emission_logp(audio)
    
    # Initialize: sample from START row
    start_probs = LISTENER_TRANS[IDX["START"]]
    states = rng.choice(N_STATES, size=num_particles, p=start_probs)
    dwells = np.ones(num_particles, dtype=np.float64)
    log_weights = log_emit[0, states].copy()
    
    state_dist = np.zeros((T_len, N_STATES))
    surprise = np.zeros(T_len)
    
    for t in range(T_len):
        # --- Record posterior ---
        max_lw = np.max(log_weights)
        if np.isfinite(max_lw):
            ws = np.exp(log_weights - max_lw)
            ws /= ws.sum()
        else:
            ws = np.full(num_particles, 1.0/num_particles)
        
        # Accumulate state distribution
        np.add.at(state_dist[t], states, ws)
        
        if t == T_len - 1:
            break
        
        # --- Propagate ---
        # 1. Compute P(stay) for each particle
        p_stay = hazard_stay_prob_vec(dwells, states, rate)
        
        # 2. Decide stay or go
        coin = rng.random(num_particles)
        staying = coin < p_stay
        
        # 3. For particles that go, sample from LISTENER_GO[current_state]
        going_mask = ~staying
        next_states = states.copy()
        
        if going_mask.any():
            go_indices = np.where(going_mask)[0]
            for idx in go_indices:
                go_probs = LISTENER_GO[states[idx]]
                if go_probs.sum() > 0:
                    next_states[idx] = rng.choice(N_STATES, p=go_probs)
        
        # 4. Surprise signal: weighted fraction that transitioned
        transition_mask = next_states != states
        surprise[t+1] = np.sum(ws[transition_mask])
        
        # 5. Update dwell
        dwells = np.where(next_states == states, dwells + 1, 1.0)
        states = next_states
        
        # 6. Update weights
        log_weights += log_emit[t+1, states]
        
        # --- Resample ---
        max_lw = np.max(log_weights)
        if np.isfinite(max_lw):
            ws = np.exp(log_weights - max_lw)
            ws /= ws.sum()
        else:
            ws = np.full(num_particles, 1.0/num_particles)
            log_weights[:] = np.log(1.0/num_particles)
        
        ess = 1.0 / np.sum(ws**2)
        if ess < num_particles * resample_threshold:
            # Systematic resample
            positions = (np.arange(num_particles) + rng.random()) / num_particles
            cumsum = np.cumsum(ws)
            indices = np.searchsorted(cumsum, positions)
            indices = np.clip(indices, 0, num_particles - 1)
            
            states = states[indices]
            dwells = dwells[indices]
            log_weights = np.full(num_particles, np.log(1.0/num_particles))
    
    # MAP path
    map_indices = np.argmax(state_dist, axis=1)
    map_path = [STATE_NAME[idx] for idx in map_indices]
    
    return map_path, state_dist, surprise

# =========================================================================
# 6. HELPERS
# =========================================================================
def compress(seq):
    return [s for i, s in enumerate(seq) if i == 0 or s != seq[i-1]]

def has_or(path):
    return "or" in compress(path)

# =========================================================================
# 7. EXPERIMENTS
# =========================================================================
def run_behavioral_experiment(n_trials=500, rate_slow=1.1, rate_fast=1.7,
                               num_particles=200, seed=42):
    rng_gen = np.random.default_rng(seed)
    rng_slow = np.random.default_rng(seed+1)
    rng_fast = np.random.default_rng(seed+2)
    
    slow_or, slow_tot, fast_or, fast_tot = 0, 0, 0, 0
    
    print(f"Running {n_trials} trials (particles={num_particles})...")
    t0 = timer.time()
    
    for k in range(n_trials):
        if k % 100 == 0 and k > 0:
            print(f"  Trial {k}/{n_trials}  ({timer.time()-t0:.1f}s)")
        
        _, audio = generate_utterance(rate=1.5, steepness=0.5, rng=rng_gen)
        if len(audio) < 5: continue
        
        p_s, _, _ = particle_filter_decode(audio, rate_slow, num_particles, rng=rng_slow)
        if "time" in p_s:
            slow_tot += 1
            if has_or(p_s): slow_or += 1
        
        p_f, _, _ = particle_filter_decode(audio, rate_fast, num_particles, rng=rng_fast)
        if "time" in p_f:
            fast_tot += 1
            if has_or(p_f): fast_or += 1
    
    elapsed = timer.time() - t0
    print(f"Done in {elapsed:.1f}s\n")
    
    pct_s = 100*slow_or/slow_tot if slow_tot > 0 else 0
    pct_f = 100*fast_or/fast_tot if fast_tot > 0 else 0
    
    print(f"  Slow (r={rate_slow}): 'or' detected {slow_or}/{slow_tot} = {pct_s:.1f}%")
    print(f"  Fast (r={rate_fast}): 'or' detected {fast_or}/{fast_tot} = {pct_f:.1f}%")
    print(f"  Gap: {pct_f - pct_s:.1f} pp")
    print(f"  Human reference: Fast=77.1%, Slow=13.5%, Gap=63.6 pp")
    
    return {"slow_pct": pct_s, "fast_pct": pct_f, "gap": pct_f - pct_s}


def run_single_trial_analysis(seed=123):
    rng_gen = np.random.default_rng(seed)
    
    for _ in range(100):
        states_true, audio = generate_utterance(rate=1.5, steepness=0.5, rng=rng_gen)
        if 35 < len(audio) < 65: break
    
    T_len = len(audio)
    print(f"Utterance: {T_len} frames, true path: {compress(states_true)}")
    
    or_onset = None
    for t, s in enumerate(states_true):
        if s == "or": or_onset = t; break
    print(f"True 'or' onset: frame {or_onset}")
    
    num_p = 500
    path_slow, dist_slow, surp_slow = particle_filter_decode(
        audio, rate=1.1, num_particles=num_p, rng=np.random.default_rng(42))
    path_fast, dist_fast, surp_fast = particle_filter_decode(
        audio, rate=1.7, num_particles=num_p, rng=np.random.default_rng(42))
    
    print(f"Slow MAP: {compress(path_slow)}")
    print(f"Fast MAP: {compress(path_fast)}")
    
    # ---- PLOT ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [3, 3, 2]})
    
    emit_idx = [IDX[s] for s in EMIT_STATES]
    
    for ax, dist, label, path in [
        (axes[0], dist_slow, 'Slow Listener (rate=1.1) — expects long segments', path_slow),
        (axes[1], dist_fast, 'Fast Listener (rate=1.7) — expects short segments', path_fast)
    ]:
        ax.imshow(dist[:, emit_idx].T, aspect='auto', origin='lower',
                  cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        map_plot = []
        for idx in np.argmax(dist, axis=1):
            map_plot.append(emit_idx.index(idx) if idx in emit_idx else np.nan)
        ax.plot(map_plot, 'r-', marker='.', ms=3, label='MAP path')
        if or_onset is not None:
            ax.axvline(or_onset, color='white', ls='--', alpha=0.7, label="True 'or' onset")
        ax.set_yticks(range(len(EMIT_STATES)))
        ax.set_yticklabels(EMIT_STATES)
        ax.set_ylabel('State')
        ax.set_title(label)
        ax.legend(loc='upper left', fontsize=8)
    
    ax = axes[2]
    frames = np.arange(T_len)
    ax.plot(frames, surp_slow, 'b-', lw=2, label='Slow (rate=1.1)')
    ax.plot(frames, surp_fast, 'r-', lw=2, label='Fast (rate=1.7)')
    if or_onset is not None:
        ax.axvline(or_onset, color='gray', ls='--', alpha=0.7, label="True 'or' onset")
        # Shade a window around "or" onset to indicate N1 time window
        ax.axvspan(or_onset, or_onset+5, alpha=0.15, color='orange', label='~N1 window')
    ax.set_ylabel('Segmentation Surprise\n(weighted transition fraction)')
    ax.set_xlabel('Time (frames)')
    ax.set_title('Neural Proxy: Segmentation Surprise Signal (→ predicts N1)')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('/home/claude/pf_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: pf_analysis.png")
    plt.close()


def run_parameter_sensitivity(n_trials=150, num_particles=150):
    """Sweep duration_std_factor and rate pairs."""
    global DURATION_STD_FACTOR, _DUR_STD_ARR
    
    std_factors = [0.10, 0.15, 0.20, 0.25, 0.30]
    rate_pairs = [(1.0, 2.0), (1.0, 2.5), (0.8, 2.0)]
    
    print(f"{'std_factor':>10} {'r_slow':>7} {'r_fast':>7} {'%slow':>7} {'%fast':>7} {'gap':>7}")
    print("-" * 50)
    
    results = []
    
    for sf in std_factors:
        DURATION_STD_FACTOR = sf
        for s in EMIT_STATES:
            _DUR_STD_ARR[IDX[s]] = BASE_DURATIONS[s] * sf
        
        for (rs, rf) in rate_pairs:
            rng_g = np.random.default_rng(42)
            rng_s = np.random.default_rng(43)
            rng_f = np.random.default_rng(44)
            
            so, st, fo, ft = 0, 0, 0, 0
            for _ in range(n_trials):
                _, a = generate_utterance(rate=1.5, steepness=0.5, rng=rng_g)
                if len(a) < 5: continue
                
                ps, _, _ = particle_filter_decode(a, rs, num_particles, rng=rng_s)
                if "time" in ps:
                    st += 1
                    if has_or(ps): so += 1
                
                pf, _, _ = particle_filter_decode(a, rf, num_particles, rng=rng_f)
                if "time" in pf:
                    ft += 1
                    if has_or(pf): fo += 1
            
            ps_pct = 100*so/st if st > 0 else 0
            pf_pct = 100*fo/ft if ft > 0 else 0
            gap = pf_pct - ps_pct
            print(f"{sf:>10.2f} {rs:>7.1f} {rf:>7.1f} {ps_pct:>6.1f}% {pf_pct:>6.1f}% {gap:>6.1f}%")
            results.append((sf, rs, rf, ps_pct, pf_pct, gap))
    
    # Reset
    DURATION_STD_FACTOR = 0.20
    for s in EMIT_STATES:
        _DUR_STD_ARR[IDX[s]] = BASE_DURATIONS[s] * 0.20
    
    return results


# =========================================================================
# 8. MAIN
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PARTICLE FILTER MODEL — DISTAL SPEECH RATE EFFECTS")
    print("=" * 60)
    
    print("\n--- Experiment 1: Single Trial Analysis ---")
    run_single_trial_analysis(seed=123)
    
    print("\n--- Experiment 2: Behavioral Replication ---")
    run_behavioral_experiment(n_trials=500, num_particles=200, seed=42)
    
    print("\n--- Experiment 3: Parameter Sensitivity ---")
    run_parameter_sensitivity(n_trials=150, num_particles=150)
