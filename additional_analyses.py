"""
================================================================================
Additional Analyses for ODCL / AOP Paper (Revised Submission)
================================================================================
Analyses included:
  PART 1 : Alveolar state analysis — ACP=3 (fixed, independent of AOP)
  PART 2 : Alveolar state analysis — AOP=ACP (same value per unit, no hysteresis)
  PART 3 : ODCL PEEP sensitivity analysis — AOP=ACP condition
  PART 4 : Crossover plots — AOP=ACP condition (2×2 panel + representative)
  PART 5 : SP range sensitivity analysis — ACP=3 and AOP=ACP (overlay plots)
  PART 6 : SD estimation for representative crossover plot (AOP=12, ACP=3)

Model notes:
  - AOP >= ACP constraint applied (physiologically necessary)
  - TOP >= TCP constraint applied (physiologically necessary)
  - AOP >= TOP constraint REMOVED (revised model)
  - N_RUNS = 50 for all Monte Carlo analyses
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count
import copy
import warnings
warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
OUT = '/sessions/eager-intelligent-pasteur/mnt/AOPODCLrevise/'

# ── Global constants ──────────────────────────────────────────────────────────
N_COMP   = 30
N_ALV    = 1000
TOTAL    = N_COMP * N_ALV
MAX_SP   = 14.5
DP       = 15.0
TOP_MEAN = 20.0;  TOP_SD = 4.0
TCP_MEAN =  2.0;  TCP_SD = 1.0
AOP_SD   =  4.0
N_RUNS   = 50

# ── Utility functions ─────────────────────────────────────────────────────────
def make_sp(max_sp):
    vals = np.linspace(0, max_sp, N_COMP)
    return vals, vals[:, np.newaxis]

def g(m, s, sp):
    """Generate (N_COMP, N_ALV) parameter array."""
    return np.random.normal(m, s, sp.shape[:1] + (N_ALV,))

def vol_fn(v0, h, p):
    return np.maximum(0, v0 * (1 - np.exp(-(np.maximum(0, p) * np.log(2)) / h)))

def costa(comps_list):
    arr      = np.array(comps_list)          # (n_peep, N_COMP)
    best_idx = np.argmax(arr, axis=0)
    best_c   = np.max(arr,  axis=0)
    results  = []
    for i in range(len(comps_list)):
        diff  = best_c - arr[i]
        valid = best_c > 1e-9
        tot   = best_c[valid].sum()
        ov = np.where((i < best_idx) & valid, diff, 0).sum() / tot * 100 if tot > 0 else 0
        co = np.where((i > best_idx) & valid, diff, 0).sum() / tot * 100 if tot > 0 else 0
        results.append({'overdistention': ov, 'collapse': co})
    return results

def find_odcl(peeps, ca):
    if not ca or len(ca) < 2: return np.nan
    co  = np.array([a['collapse']       for a in ca])
    ov  = np.array([a['overdistention'] for a in ca])
    d   = co - ov
    idx = np.where(np.diff(np.sign(d)))[0]
    if len(idx) > 0:
        i = idx[0]
        x1, c1, o1 = peeps[i],   co[i],   ov[i]
        x2, c2, o2 = peeps[i+1], co[i+1], ov[i+1]
        den = (c2 - o2) - (c1 - o1)
        if abs(den) > 1e-6:
            p = (x1 * (c2 - o2) - x2 * (c1 - o1)) / den
            if min(x1, x2) <= p <= max(x1, x2): return p
    return np.nan

# ── Core simulation: build model arrays ───────────────────────────────────────
def build_model_acp3(aop_mean, max_sp, seed=None):
    """ACP=3 fixed, independent of AOP."""
    if seed is not None: np.random.seed(seed)
    sp_vals, sp = make_sp(max_sp)
    v0       = np.full((N_COMP, N_ALV), 2.5 / TOTAL)
    aops_raw = np.random.normal(aop_mean, AOP_SD,  (N_COMP, N_ALV))
    acps_raw = np.random.normal(3.0,      1.0,     (N_COMP, N_ALV))
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD,  (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD,  (N_COMP, N_ALV))
    h        = np.random.normal(4.9,      0.1,     (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]: arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)   # AOP >= ACP
    tops = np.maximum(tops_raw, tcps_raw)   # TOP >= TCP  (AOP>=TOP removed)
    h[h <= 0.1] = 0.1
    return sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h

def build_model_aop_eq_acp(aop_mean, max_sp, seed=None):
    """ACP = AOP per unit (no hysteresis)."""
    if seed is not None: np.random.seed(seed)
    sp_vals, sp = make_sp(max_sp)
    v0       = np.full((N_COMP, N_ALV), 2.5 / TOTAL)
    aops_raw = np.random.normal(aop_mean, AOP_SD,  (N_COMP, N_ALV))
    acps_raw = aops_raw.copy()              # ACP = AOP per unit
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD,  (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD,  (N_COMP, N_ALV))
    h        = np.random.normal(4.9,      0.1,     (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]: arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)
    tops = np.maximum(tops_raw, tcps_raw)
    h[h <= 0.1] = 0.1
    return sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h

def step_fn(peep, dp_val, aw, alv, sp, aops, acps_raw, tops, tcps_raw):
    ti   = peep + dp_val - sp
    aw2  = aw | (ti >= aops)
    alv2 = (alv | (ti >= tops)) & aw2
    te   = peep - sp
    aw3  = alv2 & (te >= acps_raw)
    alv3 = alv2 & (te >= tcps_raw)
    can  = aw3 & alv3
    trap = alv2 & ~(te >= acps_raw)
    return aw3, can | trap, ti, te

def stabilize(sp, aops, acps_raw, tops, tcps_raw, n=15):
    aw  = (60 - sp) >= aops
    alv = aw & ((60 - sp) >= tops)
    for _ in range(n):
        aw, alv, _, _ = step_fn(24, DP, aw, alv, sp, aops, acps_raw, tops, tcps_raw)
    return aw, alv

def run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean):
    """Run decremental PEEP trial; return (comp_unc, comp_corr) per PEEP."""
    aw, alv = stabilize(sp, aops, acps_raw, tops, tcps_raw)
    for _ in range(5):
        aw, alv, _, _ = step_fn(24, DP, aw, alv, sp, aops, acps_raw, tops, tcps_raw)

    PEEP_LEVELS = np.arange(24, 3, -2)
    comp_unc = []; comp_corr = []
    aw_c, alv_c = aw.copy(), alv.copy()

    for peep in PEEP_LEVELS:
        naw, nalv, ti, _ = step_fn(peep, DP, aw_c, alv_c, sp, aops, acps_raw, tops, tcps_raw)
        te_exp  = peep - sp
        can_m   = nalv & (te_exp >= acps_raw)
        trap_m  = nalv & ~(te_exp >= acps_raw)
        vt_c = (np.sum(vol_fn(v0, h, ti)    * nalv,   axis=1)
              - np.sum(vol_fn(v0, h, te_exp) * can_m,  axis=1)
              - np.sum(vol_fn(v0, h, acps_raw - sp) * trap_m, axis=1))
        comp_unc.append(vt_c / DP * 1000)
        pip = peep + DP
        ep  = np.maximum(peep, aop_mean + sp_vals)
        dpe = pip - ep
        comp_corr.append(np.where(dpe > 1e-9, vt_c / dpe * 1000, 0))
        aw_c, alv_c = naw, nalv

    return PEEP_LEVELS, comp_unc, comp_corr

def get_odcl(peep_levels, comp_unc, comp_corr):
    pl = list(peep_levels)
    return find_odcl(pl, costa(comp_unc)), find_odcl(pl, costa(comp_corr))

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Alveolar state analysis — ACP=3
# ─────────────────────────────────────────────────────────────────────────────
def part1_alveolar_state_acp3():
    print("\n" + "="*70)
    print("PART 1: Alveolar state analysis  (AOP=12, ACP=3)")
    print("="*70)
    np.random.seed(42)
    AOP_MEAN = 12.0; ACP_MEAN = 3.0; ACP_SD = 1.0

    sp_vals, sp = make_sp(MAX_SP)
    v0       = np.full((N_COMP, N_ALV), 2.5 / TOTAL)
    aops_raw = np.random.normal(AOP_MEAN, AOP_SD, (N_COMP, N_ALV))
    acps_raw = np.random.normal(ACP_MEAN, ACP_SD, (N_COMP, N_ALV))
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD, (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD, (N_COMP, N_ALV))
    h        = np.random.normal(4.9, 0.1,         (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]: arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)
    tops = np.maximum(tops_raw, tcps_raw)
    h[h <= 0.1] = 0.1

    def step(peep, dp_val, aw, alv):
        return step_fn(peep, dp_val, aw, alv, sp, aops, acps_raw, tops, tcps_raw)

    aw, alv = stabilize(sp, aops, acps_raw, tops, tcps_raw)
    for _ in range(5): aw, alv, _, _ = step(24, DP, aw, alv)

    PEEP_LEVELS = np.arange(24, 3, -2)
    state_list = []; aw_c, alv_c = aw.copy(), alv.copy()
    for peep in PEEP_LEVELS:
        prev_aw = aw_c.copy()
        naw, nalv, ti, te = step(peep, DP, aw_c, alv_c)
        state_list.append(dict(peep=peep,
            open=int((nalv & naw).sum()), trap=int((nalv & ~naw).sum()),
            closed=int((~nalv).sum()),    aop=int((~prev_aw & (ti >= aops)).sum())))
        aw_c, alv_c = naw, nalv

    peeps_arr  = np.array([d['peep']   for d in state_list])
    open_arr   = np.array([d['open']   for d in state_list])
    trap_arr   = np.array([d['trap']   for d in state_list])
    closed_arr = np.array([d['closed'] for d in state_list])
    aop_arr    = np.array([d['aop']    for d in state_list])

    # Compartment-level at PEEP=12
    aw_h, alv_h = aw.copy(), alv.copy()
    for _ in range(5): aw_h, alv_h, _, _ = step(24, DP, aw_h, alv_h)
    for p in [22, 20, 18, 16, 14]: aw_h, alv_h, _, _ = step(p, DP, aw_h, alv_h)
    prev12 = aw_h.copy()
    naw12, nalv12, ti12, _ = step(12, DP, aw_h, alv_h)
    open_c   = np.array([(nalv12[c] &  naw12[c]).sum() for c in range(N_COMP)])
    trap_c   = np.array([(nalv12[c] & ~naw12[c]).sum() for c in range(N_COMP)])
    closed_c = np.array([(~nalv12[c]).sum()             for c in range(N_COMP)])
    aop_c    = np.array([(~prev12[c] & (ti12[c] >= aops[c])).sum() for c in range(N_COMP)])

    # Fig 1a: Parameter distributions
    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5))
    fig1.suptitle(
        f'Parameter Distributions  (AOP={AOP_MEAN}±{AOP_SD}, ACP={ACP_MEAN}±{ACP_SD} cmH₂O)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, TCP={TCP_MEAN}±{TCP_SD}  [N={N_COMP}×{N_ALV}={TOTAL:,} units]',
        fontsize=11, fontweight='bold')
    for ax, (name, fullname, arr, color) in zip(axes1, [
        ('AOP','Airway Opening Pressure', aops,     '#c0392b'),
        ('ACP','Airway Closing Pressure', acps_raw, '#2980b9'),
        ('TOP','Total Opening Pressure',  tops,     '#27ae60'),
        ('TCP','Total Closing Pressure',  tcps_raw, '#d68910')]):
        flat = arr.flatten()
        ax.hist(flat, bins=80, color=color, alpha=0.75, density=True)
        ax.axvline(flat.mean(), color='k', lw=2, ls='--')
        ax.set_xlabel('cmH₂O', fontsize=11); ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name}\n({fullname})', fontsize=11, fontweight='bold')
        ax.text(0.97, 0.95, f'mean={flat.mean():.2f}\nSD={flat.std():.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
        ax.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    fig1.savefig(OUT + 'Fig_distributions_acp3.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 1b: Alveolar states
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(
        f'Alveolar States During Decremental PEEP Trial  (ACP=3 cmH₂O, independent)\n'
        f'AOP={AOP_MEAN}±{AOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
        fontsize=12, fontweight='bold')
    ax = axes2[0]
    ax.stackplot(peeps_arr, open_arr/TOTAL*100, trap_arr/TOTAL*100, closed_arr/TOTAL*100,
                 labels=['Aerated','Trapped','Collapsed'],
                 colors=['#2ecc71','#f39c12','#e74c3c'], alpha=0.85)
    ax.set_xlim(peeps_arr.max(), peeps_arr.min()); ax.set_ylim(0, 100)
    ax.set_xlabel('PEEP (cmH₂O)', fontsize=12); ax.set_ylabel('Fraction (%)', fontsize=12)
    ax.set_title('Alveolar State Composition', fontsize=12)
    ax.legend(loc='upper left', fontsize=10); ax.grid(axis='y', ls='--', alpha=0.4)
    for d in state_list:
        if d['peep'] in [20, 12, 4]:
            ax.annotate(f"PEEP={d['peep']}\nAer:{d['open']/TOTAL*100:.0f}%\n"
                        f"Trap:{d['trap']/TOTAL*100:.0f}%\nColl:{d['closed']/TOTAL*100:.0f}%",
                        xy=(d['peep'], 50), ha='center', fontsize=8.5,
                        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax2r = axes2[1]; width = 0.7
    ax2r.bar(peeps_arr-width/2, closed_arr/TOTAL*100, width=width, color='#e74c3c', alpha=0.75, label='Collapsed')
    ax2r.bar(peeps_arr+width/2, trap_arr/TOTAL*100,   width=width, color='#f39c12', alpha=0.75, label='Trapped')
    ax_r = ax2r.twinx()
    ax_r.plot(peeps_arr, aop_arr/TOTAL*100, 'bs-', lw=2.5, ms=9, label='AOP manifested (%)')
    ax_r.set_ylabel('AOP manifested (%)', fontsize=11, color='steelblue')
    ax_r.tick_params(axis='y', labelcolor='steelblue')
    ax2r.set_xlabel('PEEP (cmH₂O)', fontsize=12); ax2r.set_ylabel('Fraction (%)', fontsize=12)
    ax2r.set_title('Collapsed/Trapped & AOP Manifestation', fontsize=12)
    ax2r.set_xlim(peeps_arr.max()+1, peeps_arr.min()-1); ax2r.grid(axis='y', ls='--', alpha=0.4)
    l1, lb1 = ax2r.get_legend_handles_labels(); l2, lb2 = ax_r.get_legend_handles_labels()
    ax2r.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=10)
    plt.tight_layout()
    fig2.savefig(OUT + 'Fig_alveolar_states_acp3.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 1c: Compartment-level at PEEP=12
    sp_threshold = 12 - ACP_MEAN
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 6))
    fig3.suptitle(
        f'Compartment-Level States at PEEP=12 cmH₂O  (ACP=3 cmH₂O)\n'
        f'Closure: SP > PEEP−ACP_mean = {sp_threshold:.0f} cmH₂O',
        fontsize=11, fontweight='bold')
    for ax, (data, title, color) in zip(axes3, [
        (open_c,   'Aerated',             '#27ae60'),
        (trap_c,   'Trapped',             '#e67e22'),
        (closed_c, 'Collapsed',           '#c0392b'),
        (aop_c,    'AOP Manifested',      '#2980b9')]):
        ax.bar(sp_vals, data, width=MAX_SP/N_COMP*0.85, color=color, alpha=0.8, edgecolor='none')
        ax.axvline(sp_threshold, color='darkred', lw=2, ls='--', label=f'SP={sp_threshold:.0f}')
        ax.set_xlabel('SP (cmH₂O)', fontsize=11); ax.set_ylabel('Units / compartment', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, MAX_SP+0.5); ax.set_ylim(0, N_ALV*1.1)
        ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=0.4)
        ax.text(0.97, 0.97, f'Total: {int(data.sum()):,}\n({data.sum()/TOTAL*100:.1f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    plt.tight_layout()
    fig3.savefig(OUT + 'Fig_compartment_acp3.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_distributions_acp3.png")
    print("  Saved: Fig_alveolar_states_acp3.png")
    print("  Saved: Fig_compartment_acp3.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Alveolar state analysis — AOP=ACP
# ─────────────────────────────────────────────────────────────────────────────
def part2_alveolar_state_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 2: Alveolar state analysis  (AOP=ACP=12, per unit)")
    print("="*70)
    np.random.seed(42)
    AOP_MEAN = 12.0

    sp_vals, sp = make_sp(MAX_SP)
    v0       = np.full((N_COMP, N_ALV), 2.5 / TOTAL)
    aops_raw = np.random.normal(AOP_MEAN, AOP_SD, (N_COMP, N_ALV))
    acps_raw = aops_raw.copy()
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD, (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD, (N_COMP, N_ALV))
    h        = np.random.normal(4.9, 0.1,         (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]: arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)
    tops = np.maximum(tops_raw, tcps_raw)
    h[h <= 0.1] = 0.1

    def step(peep, dp_val, aw, alv):
        return step_fn(peep, dp_val, aw, alv, sp, aops, acps_raw, tops, tcps_raw)

    aw, alv = stabilize(sp, aops, acps_raw, tops, tcps_raw)
    for _ in range(5): aw, alv, _, _ = step(24, DP, aw, alv)

    PEEP_LEVELS = np.arange(24, 3, -2)
    state_list = []; aw_c, alv_c = aw.copy(), alv.copy()
    for peep in PEEP_LEVELS:
        prev_aw = aw_c.copy()
        naw, nalv, ti, _ = step(peep, DP, aw_c, alv_c)
        state_list.append(dict(peep=peep,
            open=int((nalv & naw).sum()), trap=int((nalv & ~naw).sum()),
            closed=int((~nalv).sum()),    aop=int((~prev_aw & (ti >= aops)).sum())))
        aw_c, alv_c = naw, nalv

    peeps_arr  = np.array([d['peep']   for d in state_list])
    open_arr   = np.array([d['open']   for d in state_list])
    trap_arr   = np.array([d['trap']   for d in state_list])
    closed_arr = np.array([d['closed'] for d in state_list])
    aop_arr    = np.array([d['aop']    for d in state_list])

    # Compartment-level at PEEP=12
    aw_h, alv_h = aw.copy(), alv.copy()
    for _ in range(5): aw_h, alv_h, _, _ = step(24, DP, aw_h, alv_h)
    for p in [22, 20, 18, 16, 14]: aw_h, alv_h, _, _ = step(p, DP, aw_h, alv_h)
    prev12 = aw_h.copy()
    naw12, nalv12, ti12, _ = step(12, DP, aw_h, alv_h)
    open_c   = np.array([(nalv12[c] &  naw12[c]).sum() for c in range(N_COMP)])
    trap_c   = np.array([(nalv12[c] & ~naw12[c]).sum() for c in range(N_COMP)])
    closed_c = np.array([(~nalv12[c]).sum()             for c in range(N_COMP)])
    aop_c    = np.array([(~prev12[c] & (ti12[c] >= aops[c])).sum() for c in range(N_COMP)])

    # Fig 2a: Alveolar states
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(
        f'Alveolar States During Decremental PEEP Trial  (AOP=ACP per unit)\n'
        f'AOP=ACP={AOP_MEAN}±{AOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
        fontsize=12, fontweight='bold')
    ax = axes2[0]
    ax.stackplot(peeps_arr, open_arr/TOTAL*100, trap_arr/TOTAL*100, closed_arr/TOTAL*100,
                 labels=['Aerated','Trapped','Collapsed'],
                 colors=['#2ecc71','#f39c12','#e74c3c'], alpha=0.85)
    ax.set_xlim(peeps_arr.max(), peeps_arr.min()); ax.set_ylim(0, 100)
    ax.set_xlabel('PEEP (cmH₂O)', fontsize=12); ax.set_ylabel('Fraction (%)', fontsize=12)
    ax.set_title('Alveolar State Composition', fontsize=12)
    ax.legend(loc='upper left', fontsize=10); ax.grid(axis='y', ls='--', alpha=0.4)
    for d in state_list:
        if d['peep'] in [20, 12, 4]:
            ax.annotate(f"PEEP={d['peep']}\nAer:{d['open']/TOTAL*100:.0f}%\n"
                        f"Trap:{d['trap']/TOTAL*100:.0f}%\nColl:{d['closed']/TOTAL*100:.0f}%",
                        xy=(d['peep'], 50), ha='center', fontsize=8.5,
                        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax2r = axes2[1]; width = 0.7
    ax2r.bar(peeps_arr-width/2, closed_arr/TOTAL*100, width=width, color='#e74c3c', alpha=0.75, label='Collapsed')
    ax2r.bar(peeps_arr+width/2, trap_arr/TOTAL*100,   width=width, color='#f39c12', alpha=0.75, label='Trapped')
    ax_r = ax2r.twinx()
    ax_r.plot(peeps_arr, aop_arr/TOTAL*100, 'bs-', lw=2.5, ms=9, label='AOP manifested (%)')
    ax_r.set_ylabel('AOP manifested (%)', fontsize=11, color='steelblue')
    ax_r.tick_params(axis='y', labelcolor='steelblue')
    ax2r.set_xlabel('PEEP (cmH₂O)', fontsize=12); ax2r.set_ylabel('Fraction (%)', fontsize=12)
    ax2r.set_title('Collapsed/Trapped & AOP Manifestation', fontsize=12)
    ax2r.set_xlim(peeps_arr.max()+1, peeps_arr.min()-1); ax2r.grid(axis='y', ls='--', alpha=0.4)
    l1, lb1 = ax2r.get_legend_handles_labels(); l2, lb2 = ax_r.get_legend_handles_labels()
    ax2r.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=10)
    plt.tight_layout()
    fig2.savefig(OUT + 'Fig_alveolar_states_aop_eq_acp.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 2b: Compartment-level at PEEP=12
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 6))
    fig3.suptitle(
        f'Compartment-Level States at PEEP=12 cmH₂O  (AOP=ACP per unit)\n'
        f'Closure: SP > PEEP−AOP_mean = {12-AOP_MEAN:.0f} cmH₂O',
        fontsize=11, fontweight='bold')
    for ax, (data, title, color) in zip(axes3, [
        (open_c,   'Aerated',    '#27ae60'),
        (trap_c,   'Trapped',    '#e67e22'),
        (closed_c, 'Collapsed',  '#c0392b'),
        (aop_c,    'AOP Manifested', '#2980b9')]):
        ax.bar(sp_vals, data, width=MAX_SP/N_COMP*0.85, color=color, alpha=0.8, edgecolor='none')
        ax.axvline(12-AOP_MEAN, color='darkred', lw=2, ls='--', label=f'SP={12-AOP_MEAN:.0f}')
        ax.set_xlabel('SP (cmH₂O)', fontsize=11); ax.set_ylabel('Units / compartment', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, MAX_SP+0.5); ax.set_ylim(0, N_ALV*1.1)
        ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=0.4)
        ax.text(0.97, 0.97, f'Total: {int(data.sum()):,}\n({data.sum()/TOTAL*100:.1f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    plt.tight_layout()
    fig3.savefig(OUT + 'Fig_compartment_aop_eq_acp.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_alveolar_states_aop_eq_acp.png")
    print("  Saved: Fig_compartment_aop_eq_acp.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: ODCL sensitivity analysis — AOP=ACP
# ─────────────────────────────────────────────────────────────────────────────
def _run_odcl_aop_eq_acp(args):
    aop_mean, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, MAX_SP)
    peep_levels, comp_unc, comp_corr = run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl(peep_levels, comp_unc, comp_corr)

def part3_odcl_sensitivity_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 3: ODCL sensitivity analysis  (AOP=ACP, 4–16 cmH₂O)")
    print("="*70)
    AOP_LEVELS = np.arange(4, 17, 2)
    unc_m, unc_s, cor_m, cor_s = [], [], [], []

    for aop_val in AOP_LEVELS:
        args = [(aop_val, seed) for seed in range(N_RUNS)]
        with Pool(min(cpu_count(), 8)) as pool:
            raw = pool.map(_run_odcl_aop_eq_acp, args)
        uncs  = [r[0] for r in raw if not np.isnan(r[0])]
        corrs = [r[1] for r in raw if not np.isnan(r[1])]
        unc_m.append(np.mean(uncs)); unc_s.append(np.std(uncs))
        cor_m.append(np.mean(corrs)); cor_s.append(np.std(corrs))
        print(f"  AOP=ACP={aop_val:2d}: Uncorr={np.mean(uncs):.2f}±{np.std(uncs):.2f}  "
              f"Corr={np.mean(corrs):.2f}±{np.std(corrs):.2f}")

    aop_x = list(AOP_LEVELS)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(aop_x, unc_m, yerr=unc_s, fmt='rs-', lw=2.5, ms=9, capsize=6,
                label='Uncorrected ODCL PEEP')
    ax.errorbar(aop_x, cor_m, yerr=cor_s, fmt='bo-', lw=2.5, ms=9, capsize=6,
                label='AOP-Corrected ODCL PEEP')
    ax.fill_between(aop_x, np.array(unc_m)-np.array(unc_s),
                    np.array(unc_m)+np.array(unc_s), alpha=0.12, color='red')
    ax.fill_between(aop_x, np.array(cor_m)-np.array(cor_s),
                    np.array(cor_m)+np.array(cor_s), alpha=0.12, color='blue')
    ax.set_xlabel('AOP = ACP (cmH₂O)', fontsize=13)
    ax.set_ylabel('ODCL PEEP (cmH₂O)', fontsize=13)
    ax.set_title(
        f'ODCL PEEP Sensitivity: AOP=ACP (same value per unit)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=12); ax.set_xticks(aop_x)
    ax.set_ylim(0, 25); ax.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT + 'Fig_sensitivity_aop_eq_acp.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_sensitivity_aop_eq_acp.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Crossover plots — AOP=ACP (2×2 + representative)
# ─────────────────────────────────────────────────────────────────────────────
def _run_crossover_aop_eq_acp(aop_mean, seed=42):
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, MAX_SP)
    peep_levels, comp_unc, comp_corr = run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    ca_u = costa(comp_unc);  odcl_u = find_odcl(list(peep_levels), ca_u)
    ca_c = costa(comp_corr); odcl_c = find_odcl(list(peep_levels), ca_c)
    return peep_levels, ca_u, ca_c, odcl_u, odcl_c

def part4_crossover_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 4: Crossover plots  (AOP=ACP)")
    print("="*70)

    # 2×2 panel
    AOP_CASES = [4, 8, 12, 16]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Crossover Plots: Uncorrected vs AOP-Corrected  (AOP=ACP per unit)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
        fontsize=13, fontweight='bold')

    for ax, aop_val in zip(axes.flatten(), AOP_CASES):
        peeps, ca_u, ca_c, odcl_u, odcl_c = _run_crossover_aop_eq_acp(aop_val)
        pl = list(peeps)
        co_u=[a['collapse'] for a in ca_u]; ov_u=[a['overdistention'] for a in ca_u]
        co_c=[a['collapse'] for a in ca_c]; ov_c=[a['overdistention'] for a in ca_c]
        ax.plot(pl, co_u, 'v-',  color='#e74c3c', lw=2, ms=7, label='Collapse (Uncorr)')
        ax.plot(pl, ov_u, 'o-',  color='#3498db', lw=2, ms=7, label='Overdist (Uncorr)')
        ax.plot(pl, co_c, 'v--', color='#c0392b', lw=2, ms=7, alpha=0.7, label='Collapse (AOP-Corr)')
        ax.plot(pl, ov_c, 'o--', color='#2980b9', lw=2, ms=7, alpha=0.7, label='Overdist (AOP-Corr)')
        if not np.isnan(odcl_u): ax.axvline(odcl_u, color='#e74c3c', lw=2.5,
                                             label=f'ODCL Uncorr={odcl_u:.1f}')
        if not np.isnan(odcl_c): ax.axvline(odcl_c, color='#2980b9', lw=2.5, ls='--',
                                             label=f'ODCL AOP-Corr={odcl_c:.1f}')
        ax.set_xlim(max(pl)+1, min(pl)-1); ax.set_ylim(-2, 102)
        ax.set_xlabel('PEEP (cmH₂O)', fontsize=11); ax.set_ylabel('%', fontsize=10)
        ax.set_title(f'AOP=ACP={aop_val} cmH₂O', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper left'); ax.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT + 'Fig_crossover_aop_eq_acp.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Representative (AOP=ACP=12)
    peeps, ca_u, ca_c, odcl_u, odcl_c = _run_crossover_aop_eq_acp(12)
    pl = list(peeps)
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.plot(pl, [a['collapse'] for a in ca_u],       'v-',  color='#e74c3c', lw=2.5, ms=9, label='Collapse (Uncorrected)')
    ax2.plot(pl, [a['overdistention'] for a in ca_u], 'o-',  color='#3498db', lw=2.5, ms=9, label='Overdistention (Uncorrected)')
    ax2.plot(pl, [a['collapse'] for a in ca_c],       'v--', color='#c0392b', lw=2.5, ms=9, alpha=0.75, label='Collapse (AOP-Corrected)')
    ax2.plot(pl, [a['overdistention'] for a in ca_c], 'o--', color='#2980b9', lw=2.5, ms=9, alpha=0.75, label='Overdistention (AOP-Corrected)')
    if not np.isnan(odcl_u): ax2.axvline(odcl_u, color='#e74c3c', lw=3,
                                          label=f'ODCL (Uncorrected) = {odcl_u:.1f} cmH₂O')
    if not np.isnan(odcl_c): ax2.axvline(odcl_c, color='#2980b9', lw=3, ls='--',
                                          label=f'ODCL (AOP-Corrected) = {odcl_c:.1f} cmH₂O')
    ax2.set_xlim(max(pl)+1, min(pl)-1); ax2.set_ylim(-2, 102)
    ax2.set_xlabel('PEEP (cmH₂O)', fontsize=13); ax2.set_ylabel('%', fontsize=12)
    ax2.set_title(f'Representative Crossover  (AOP=ACP=12 cmH₂O)\n'
                  f'TOP={TOP_MEAN}±{TOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left'); ax2.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    fig2.savefig(OUT + 'Fig_crossover_aop_eq_acp_representative.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_crossover_aop_eq_acp.png")
    print("  Saved: Fig_crossover_aop_eq_acp_representative.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: SP range sensitivity — ACP=3 and AOP=ACP (overlay)
# ─────────────────────────────────────────────────────────────────────────────
SP_CONDITIONS = [
    {"label": "SP = 0–14.5 cmH₂O", "max_sp": 14.5, "color": "#2ca02c"},
    {"label": "SP = 0–7 cmH₂O",    "max_sp":  7.0,  "color": "#ff7f0e"},
    {"label": "SP = 0 cmH₂O",      "max_sp":  0.0,  "color": "#9467bd"},
]

def _run_sp_acp3(args):
    aop_mean, max_sp, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_acp3(aop_mean, max_sp)
    peep_levels, comp_unc, comp_corr = run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl(peep_levels, comp_unc, comp_corr)

def _run_sp_aop_eq_acp(args):
    aop_mean, max_sp, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, max_sp)
    peep_levels, comp_unc, comp_corr = run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl(peep_levels, comp_unc, comp_corr)

def _collect_sp(run_func, label):
    AOP_LEVELS = np.arange(4, 17, 2)
    print(f"\n  {label}")
    data = {c['label']: {'unc_m':[],'unc_s':[],'cor_m':[],'cor_s':[]} for c in SP_CONDITIONS}
    for cond in SP_CONDITIONS:
        print(f"    SP: {cond['label']}")
        for aop_val in AOP_LEVELS:
            args = [(aop_val, cond['max_sp'], seed) for seed in range(N_RUNS)]
            with Pool(min(cpu_count(), 8)) as pool:
                raw = pool.map(run_func, args)
            uncs  = [r[0] for r in raw if not np.isnan(r[0])]
            corrs = [r[1] for r in raw if not np.isnan(r[1])]
            data[cond['label']]['unc_m'].append(np.mean(uncs))
            data[cond['label']]['unc_s'].append(np.std(uncs))
            data[cond['label']]['cor_m'].append(np.mean(corrs))
            data[cond['label']]['cor_s'].append(np.std(corrs))
            print(f"      AOP={aop_val:2d}: Uncorr={np.mean(uncs):.2f}±{np.std(uncs):.2f}  "
                  f"Corr={np.mean(corrs):.2f}±{np.std(corrs):.2f}")
    return data

def _plot_overlay(data, title, fname):
    AOP_LEVELS = np.arange(4, 17, 2); aop_x = list(AOP_LEVELS)
    fig, ax = plt.subplots(figsize=(11, 7))
    for cond in SP_CONDITIONS:
        d = data[cond['label']]; color = cond['color']
        um, us = np.array(d['unc_m']), np.array(d['unc_s'])
        cm, cs = np.array(d['cor_m']), np.array(d['cor_s'])
        ax.errorbar(aop_x, um, yerr=us, fmt='o-',  color=color, lw=2.5, ms=8, capsize=5,
                    label=f"{cond['label']}  [Uncorrected]")
        ax.fill_between(aop_x, um-us, um+us, alpha=0.10, color=color)
        ax.errorbar(aop_x, cm, yerr=cs, fmt='s--', color=color, lw=2.5, ms=8, capsize=5,
                    alpha=0.75, label=f"{cond['label']}  [AOP-Corrected]")
        ax.fill_between(aop_x, cm-cs, cm+cs, alpha=0.07, color=color)
    legend_extra = [Line2D([0],[0],color='k',lw=2.5,ls='-', label='─  Uncorrected'),
                    Line2D([0],[0],color='k',lw=2.5,ls='--',label='--  AOP-Corrected')]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles+legend_extra, labels+['─  Uncorrected','--  AOP-Corrected'],
              fontsize=9, loc='upper left', ncol=2)
    ax.set_xlabel('AOP Mean (cmH₂O)', fontsize=13)
    ax.set_ylabel('ODCL PEEP (cmH₂O)', fontsize=13)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(aop_x); ax.set_ylim(0, 25); ax.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT + fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

def part5_sp_sensitivity():
    print("\n" + "="*70)
    print("PART 5: SP range sensitivity analysis")
    print("="*70)
    data_acp3    = _collect_sp(_run_sp_acp3,       "ACP=3 (fixed)")
    data_aop_acp = _collect_sp(_run_sp_aop_eq_acp, "AOP=ACP (per unit)")

    _plot_overlay(data_acp3,
        f'SP Sensitivity  (ACP=3 cmH₂O, fixed)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
        'Fig_SP_sensitivity_overlay_acp3.png')

    _plot_overlay(data_aop_acp,
        f'SP Sensitivity  (AOP=ACP per unit)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
        'Fig_SP_sensitivity_overlay_aop_eq_acp.png')

# ─────────────────────────────────────────────────────────────────────────────
# PART 6: SD estimation for representative crossover (AOP=12, ACP=3)
# ─────────────────────────────────────────────────────────────────────────────
def _run_crossover_acp3_sd(_):
    np.random.seed(None)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_acp3(12.0, MAX_SP)
    peep_levels, comp_unc, comp_corr = run_peep_trial(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, 12.0)
    return get_odcl(peep_levels, comp_unc, comp_corr)

def part6_crossover_sd():
    print("\n" + "="*70)
    print("PART 6: SD estimation — representative crossover (AOP=12, ACP=3)")
    print("="*70)
    with Pool(min(cpu_count(), 8)) as pool:
        raw = pool.map(_run_crossover_acp3_sd, range(N_RUNS))
    uncs  = [r[0] for r in raw if not np.isnan(r[0])]
    corrs = [r[1] for r in raw if not np.isnan(r[1])]
    print(f"  N = {N_RUNS} runs  (AOP=12, ACP=3, SP=0–{MAX_SP}, DP={DP})")
    print(f"  Uncorrected ODCL PEEP : {np.mean(uncs):.2f} ± {np.std(uncs):.2f} cmH₂O"
          f"  (range {np.min(uncs):.2f}–{np.max(uncs):.2f})")
    print(f"  AOP-Corrected ODCL PEEP: {np.mean(corrs):.2f} ± {np.std(corrs):.2f} cmH₂O"
          f"  (range {np.min(corrs):.2f}–{np.max(corrs):.2f})")
    print(f"  Difference (Corr−Uncorr): {np.mean(corrs)-np.mean(uncs):.2f} cmH₂O")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  Additional Analyses — ODCL / AOP Paper")
    print(f"  N_RUNS = {N_RUNS} per data point")
    print("=" * 70)

    part1_alveolar_state_acp3()
    part2_alveolar_state_aop_eq_acp()
    part3_odcl_sensitivity_aop_eq_acp()
    part4_crossover_aop_eq_acp()
    part5_sp_sensitivity()
    part6_crossover_sd()

    print("\n" + "=" * 70)
    print("  All analyses complete.")
    print("=" * 70)
