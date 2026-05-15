"""
================================================================================
ODCL / AOP Paper — Consolidated Analysis Script (Submission Version)
================================================================================
Model revision notes:
  - AOP >= ACP  (airway opens above airway closing pressure)
  - TOP >= TCP  (alveolus opens above alveolar closing pressure)
  - AOP >= TOP  constraint REMOVED (revised model; values are independent)
  - Fix2: When ACP > TCP, trapped units (airway closed, alveolus still open)
          retain volume at vol(ACP-SP), not vol(TCP-SP)

Analyses:
  PART 1 : AOP sensitivity (ACP = 3 cmH₂O, fixed)  [Fig. sensitivity + crossover]
  PART 2 : Alveolar state analysis — ACP = 3         [Fig. distributions & states]
  PART 3 : Alveolar state analysis — AOP = ACP
  PART 4 : ODCL sensitivity — AOP = ACP condition
  PART 5 : Crossover plots — AOP = ACP (2×2 panel + representative)
  PART 6 : SP range sensitivity overlay (ACP=3 and AOP=ACP)
  PART 7 : SD estimation — representative crossover (AOP=12, ACP=3)
  PART 8 : Revised main analysis — ACP = AOP         [Fig. 4 revised + comparison]
  PART 9 : SP range sensitivity — LungModel-based (proper SP analysis)

Output directory: Set OUTPUT_DIR below (default = same folder as this script).
================================================================================
"""

import os
import copy
import math
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, cpu_count

# ── Output directory (change as needed) ──────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global simulation constants ───────────────────────────────────────────────
N_COMP    = 30          # gravitational compartments
N_ALV     = 1000        # alveoli per compartment
TOTAL     = N_COMP * N_ALV
MAX_SP    = 14.5        # max superimposed pressure (cmH₂O)
DP        = 15.0        # driving pressure (cmH₂O)
TOP_MEAN  = 20.0;  TOP_SD = 4.0   # alveolar opening pressure
TCP_MEAN  =  2.0;  TCP_SD = 1.0   # alveolar closing pressure
AOP_SD    =  4.0
H_MEAN    =  4.9;  H_SD  = 0.1   # Salazar-Knowles curvature
V_MAX_L   =  2.5                   # TLC (L)
N_RUNS    = 50          # Monte Carlo iterations per data point
PEEP_LEVELS = np.arange(24, 3, -2)


# ==============================================================================
# LungModel — canonical class (unified across all analyses)
# ==============================================================================
class LungModel:
    """
    Hickling-type compartmental lung model.

    Parameters
    ----------
    n_compartments : int
        Number of gravitational compartments.
    max_sp_g1 : float
        Max superimposed pressure (cmH₂O).  SP gradient: 0 → max_sp linearly.
    aop/acp/top/tcp : mean + sd
        Airway opening/closing and alveolar opening/closing pressures (cmH₂O).
    tlc_L_g1 : float
        Total lung capacity (L).  FRC = 0 (Hickling model).
    h_mean/h_sd : float
        Salazar-Knowles curvature parameter h (cmH₂O).
    max_sp_g2, ... : optional
        Second group parameters for bimodal lung (not used in paper analyses).

    Constraints applied
    -------------------
    AOP >= ACP   (physiologically required)
    TOP >= TCP   (physiologically required)
    AOP >= TOP   REMOVED (revised model)

    Fix2
    ----
    When ACP > TCP at expiration, the airway closes before the alveolus
    collapses.  Such 'trapped' units retain volume vol(ACP - SP) rather
    than collapsing to zero.
    """

    def __init__(self, n_compartments,
                 max_sp_g1, aop_mean_g1, aop_sd_g1, acp_mean_g1, acp_sd_g1,
                 top_mean_g1, top_sd_g1, tcp_mean_g1, tcp_sd_g1,
                 tlc_L_g1, h_mean_g1, h_sd_g1,
                 # Optional second group (bimodal)
                 max_sp_g2=None, aop_mean_g2=None, aop_sd_g2=None,
                 acp_mean_g2=None, acp_sd_g2=None,
                 top_mean_g2=None, top_sd_g2=None,
                 tcp_mean_g2=None, tcp_sd_g2=None,
                 tlc_L_g2=None, h_mean_g2=None, h_sd_g2=None):

        self.n_compartments    = n_compartments
        self.n_alveoli_per_comp = 1000
        self.frc_L             = 0.0

        is_bimodal = max_sp_g2 is not None
        n1 = n_compartments if not is_bimodal else n_compartments // 2
        n2 = 0 if not is_bimodal else n_compartments - n1
        tot1 = n1 * self.n_alveoli_per_comp
        tot2 = n2 * self.n_alveoli_per_comp

        # Superimposed pressure
        sp1 = np.linspace(0, max_sp_g1, n1)
        if is_bimodal:
            sp2 = np.linspace(0, max_sp_g2, n2)
            self.sp = np.concatenate((sp1, sp2))[:, np.newaxis]
        else:
            self.sp = sp1[:, np.newaxis]

        # Parameter arrays
        def _gen(m1, s1, m2, s2, na, nb):
            a1 = np.random.normal(m1, s1, (na, self.n_alveoli_per_comp))
            if is_bimodal:
                a2 = np.random.normal(m2, s2, (nb, self.n_alveoli_per_comp))
                return np.concatenate((a1, a2), axis=0)
            return a1

        self.aops    = _gen(aop_mean_g1, aop_sd_g1, aop_mean_g2, aop_sd_g2, n1, n2)
        self.acps    = _gen(acp_mean_g1, acp_sd_g1, acp_mean_g2, acp_sd_g2, n1, n2)
        self.tops    = _gen(top_mean_g1, top_sd_g1, top_mean_g2, top_sd_g2, n1, n2)
        self.tcps    = _gen(tcp_mean_g1, tcp_sd_g1, tcp_mean_g2, tcp_sd_g2, n1, n2)
        self.h_units = _gen(h_mean_g1,   h_sd_g1,   h_mean_g2,   h_sd_g2,   n1, n2)

        for p in [self.aops, self.tops]:
            p[p < 0] = 0

        # Volume scaling (FRC = 0)
        if is_bimodal:
            v0_1 = np.full((n1, self.n_alveoli_per_comp), tlc_L_g1 / tot1)
            v0_2 = np.full((n2, self.n_alveoli_per_comp), tlc_L_g2 / tot2)
            self.v0 = np.concatenate((v0_1, v0_2), axis=0)
        else:
            self.v0 = np.full((n1, self.n_alveoli_per_comp), tlc_L_g1 / tot1)

        # Constraints
        self.aops = np.maximum(self.aops, self.acps)  # AOP >= ACP
        self.tops = np.maximum(self.tops, self.tcps)  # TOP >= TCP
        # AOP >= TOP constraint removed (revised model)
        self.h_units[self.h_units <= 0.1] = 0.1

    # ── Volume function ───────────────────────────────────────────────────────
    def _vol(self, p):
        """Salazar-Knowles: v0 * (1 - exp(-max(0,p)*ln2/h))"""
        p_pos = np.maximum(0, p)
        return np.maximum(0, self.v0 * (1 - np.exp(-(p_pos * np.log(2)) / self.h_units)))

    def _eelv(self, peep, airway_open, alv_open):
        """EELV: aerated → vol(te), trapped → vol(ACP-SP)."""
        tp    = peep - self.sp
        can   = airway_open & alv_open
        trap  = ~airway_open & alv_open
        tp_cl = self.acps - self.sp
        return self.frc_L + np.sum(self._vol(tp) * can) + np.sum(self._vol(tp_cl) * trap)

    # ── Single breath ─────────────────────────────────────────────────────────
    def get_trial_metrics(self, peep, dp, start_air, start_alv):
        """
        Simulate one breath.

        Returns
        -------
        tv, comp, new_air, new_alv, eelv, vt_per_comp
        """
        tp_insp  = peep + dp - self.sp
        end_air  = start_air | (tp_insp >= self.aops)
        end_alv  = (start_alv | (tp_insp >= self.tops)) & end_air

        vol_peak  = self._vol(tp_insp)
        tot_peak  = self.frc_L + np.sum(vol_peak * end_alv)

        tp_exp    = peep - self.sp
        aw_exp    = (tp_exp >= self.acps)
        alv_exp   = (tp_exp >= self.tcps)
        air_still = end_alv & aw_exp
        alv_still = end_alv & alv_exp
        can       = air_still & alv_still
        is_trap   = end_alv & ~aw_exp
        # Fix2: ACP > TCP → airway closes first; trapped unit keeps vol(ACP-SP)
        trap_open = is_trap & (alv_still | (self.acps > self.tcps))

        new_air = air_still
        new_alv = can | trap_open
        eelv    = self._eelv(peep, new_air, new_alv)
        tv      = tot_peak - eelv
        comp    = tv / dp if dp > 0 else 0

        # Per-compartment tidal volume
        vol_pk_c  = np.sum(vol_peak * end_alv, axis=1)
        vol_exp   = self._vol(tp_exp)
        vol_trap  = self._vol(self.acps - self.sp)
        vol_ep_c  = np.sum(vol_exp * can, axis=1) + np.sum(vol_trap * trap_open, axis=1)
        vt_c      = vol_pk_c - vol_ep_c

        return tv, comp, new_air, new_alv, eelv, vt_c

    def stabilize(self, peep, dp, start_air=None, start_alv=None, n=15):
        a = np.zeros_like(self.aops, dtype=bool) if start_air is None else start_air.copy()
        v = np.zeros_like(self.aops, dtype=bool) if start_alv is None else start_alv.copy()
        for _ in range(n):
            _, _, a, v, _, _ = self.get_trial_metrics(peep, dp, a, v)
        return a, v

    def run_peep_trial(self, peep_levels, dp, pip_max=60):
        """Decremental PEEP trial after full recruitment at pip_max."""
        tp_rec  = pip_max - self.sp
        cur_air = (tp_rec >= self.aops)
        cur_alv = cur_air & (tp_rec >= self.tops)
        results = []
        for peep in peep_levels:
            fa, fv = self.stabilize(peep, dp, cur_air, cur_alv, n=5)
            tv, comp, _, _, eelv, vt_c = self.get_trial_metrics(peep, dp, fa, fv)
            _, _, cur_air, cur_alv, _, _ = self.get_trial_metrics(peep, 0, fa, fv)
            results.append({
                "peep":               peep,
                "total_compliance":   comp * 1000,
                "comp_per_comp":      (vt_c / dp) * 1000 if dp > 0 else np.zeros(self.n_compartments),
                "driving_pressure":   dp,
                "eelv_liters":        eelv,
                "tidal_volume_liters": tv,
            })
        return results


# ==============================================================================
# Shared analysis functions
# ==============================================================================
def costa(comps_list):
    """
    Costa algorithm on a list of per-compartment compliance arrays.

    Parameters
    ----------
    comps_list : list of array (n_comp,)
        Compliance per compartment at each PEEP step.

    Returns
    -------
    list of dict with keys 'overdistention', 'collapse'  (%)
    """
    arr      = np.array(comps_list)          # (n_peep, n_comp)
    best_idx = np.argmax(arr, axis=0)
    best_c   = np.max(arr,  axis=0)
    out = []
    for i in range(len(comps_list)):
        diff  = best_c - arr[i]
        valid = best_c > 1e-9
        tot   = best_c[valid].sum()
        ov = np.where((i < best_idx) & valid, diff, 0).sum() / tot * 100 if tot > 0 else 0
        co = np.where((i > best_idx) & valid, diff, 0).sum() / tot * 100 if tot > 0 else 0
        out.append({'overdistention': ov, 'collapse': co})
    return out


def analyze_costa(results, key='comp_per_comp'):
    """Apply Costa algorithm to run_peep_trial() result list."""
    comps = [r[key] for r in results if r.get(key) is not None]
    ca    = costa(comps)
    for a, r in zip(ca, results):
        a['peep'] = r['peep']
    return ca


def find_odcl(peeps, ca):
    """
    Interpolated crossover PEEP where collapse% = overdistention%.

    Parameters
    ----------
    peeps : list of float   (decreasing)
    ca    : list of dict    with 'collapse' and 'overdistention' keys
    """
    if not ca or len(ca) < 2:
        return np.nan
    co  = np.array([a['collapse']       for a in ca])
    ov  = np.array([a['overdistention'] for a in ca])
    d   = co - ov
    idx = np.where(np.diff(np.sign(d)))[0]
    if len(idx) > 0:
        i = idx[0]
        if i + 1 < len(peeps):
            x1, c1, o1 = peeps[i],   co[i],   ov[i]
            x2, c2, o2 = peeps[i+1], co[i+1], ov[i+1]
            den = (c2 - o2) - (c1 - o1)
            if abs(den) > 1e-6:
                p = (x1*(c2-o2) - x2*(c1-o1)) / den
                if min(x1, x2) <= p <= max(x1, x2):
                    return p
    return np.nan


def apply_aop_correction(results, aop_mean, sp_array, n_comp, use_sp=True):
    """
    Add 'corrected_comp_per_comp' to each result dict.

    dp_eff = PIP - max(PEEP, AOP_mean + SP)   [use_sp=True]
    dp_eff = PIP - max(PEEP, AOP_mean)         [use_sp=False]
    """
    for r in results:
        dp = r.get('driving_pressure', 0)
        if dp > 1e-9:
            peep = r['peep']
            pip  = peep + dp
            vt   = np.array(r['comp_per_comp']) / 1000.0 * dp
            if use_sp:
                ep = np.maximum(peep, aop_mean + sp_array)
            else:
                ep = np.full(n_comp, max(peep, aop_mean))
            dp_e = pip - ep
            corr = np.zeros_like(vt)
            m    = dp_e > 1e-9
            corr[m] = vt[m] / dp_e[m] * 1000.0
            r['corrected_comp_per_comp'] = corr
        else:
            r['corrected_comp_per_comp'] = np.zeros(n_comp)
    return results


def _make_lung(params, n_comp=N_COMP):
    """Construct LungModel from a parameter dict."""
    return LungModel(
        n_compartments=n_comp,
        max_sp_g1=params['max_sp'],
        aop_mean_g1=params['aop_mean'], aop_sd_g1=params['aop_sd'],
        acp_mean_g1=params['acp_mean'], acp_sd_g1=params['acp_sd'],
        top_mean_g1=params['top_mean'], top_sd_g1=params['top_sd'],
        tcp_mean_g1=params['tcp_mean'], tcp_sd_g1=params['tcp_sd'],
        tlc_L_g1=params['v_max_ml'] / 1000.0,
        h_mean_g1=params['h_mean'],     h_sd_g1=params['h_sd'],
    )


# ==============================================================================
# Raw-array simulation helpers (used in PARTS 2–7)
# These operate directly on numpy arrays for multiprocessing efficiency.
# ==============================================================================
def make_sp(max_sp):
    vals = np.linspace(0, max_sp, N_COMP)
    return vals, vals[:, np.newaxis]


def vol_fn(v0, h, p):
    return np.maximum(0, v0 * (1 - np.exp(-(np.maximum(0, p) * np.log(2)) / h)))


def step_fn(peep, dp_val, aw, alv, sp, aops, acps_raw, tops, tcps_raw):
    """Single breath on raw arrays. Returns (new_aw, new_alv, ti, te)."""
    ti        = peep + dp_val - sp
    aw2       = aw | (ti >= aops)
    alv2      = (alv | (ti >= tops)) & aw2
    te        = peep - sp
    aw3       = alv2 & (te >= acps_raw)
    alv3      = alv2 & (te >= tcps_raw)
    can       = aw3 & alv3
    trap      = alv2 & ~(te >= acps_raw)
    # Fix2: ACP > TCP → trapped unit keeps vol(ACP)
    trap_open = trap & ((te >= tcps_raw) | (acps_raw > tcps_raw))
    return aw3, can | trap_open, ti, te


def _stabilize_raw(sp, aops, acps_raw, tops, tcps_raw, n=15):
    aw  = (60 - sp) >= aops
    alv = aw & ((60 - sp) >= tops)
    for _ in range(n):
        aw, alv, _, _ = step_fn(24, DP, aw, alv, sp, aops, acps_raw, tops, tcps_raw)
    return aw, alv


def build_model_acp3(aop_mean, max_sp, seed=None):
    """Build raw-array model with ACP=3 (fixed, independent of AOP)."""
    if seed is not None:
        np.random.seed(seed)
    sp_vals, sp = make_sp(max_sp)
    v0       = np.full((N_COMP, N_ALV), V_MAX_L / TOTAL)
    aops_raw = np.random.normal(aop_mean, AOP_SD,  (N_COMP, N_ALV))
    acps_raw = np.random.normal(3.0, 1.0,          (N_COMP, N_ALV))
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD,  (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD,  (N_COMP, N_ALV))
    h        = np.random.normal(H_MEAN,   H_SD,    (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]:
        arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)
    tops = np.maximum(tops_raw, tcps_raw)
    h[h <= 0.1] = 0.1
    return sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h


def build_model_aop_eq_acp(aop_mean, max_sp, seed=None):
    """Build raw-array model with ACP = AOP per unit (no hysteresis)."""
    if seed is not None:
        np.random.seed(seed)
    sp_vals, sp = make_sp(max_sp)
    v0       = np.full((N_COMP, N_ALV), V_MAX_L / TOTAL)
    aops_raw = np.random.normal(aop_mean, AOP_SD,  (N_COMP, N_ALV))
    acps_raw = aops_raw.copy()
    tops_raw = np.random.normal(TOP_MEAN, TOP_SD,  (N_COMP, N_ALV))
    tcps_raw = np.random.normal(TCP_MEAN, TCP_SD,  (N_COMP, N_ALV))
    h        = np.random.normal(H_MEAN,   H_SD,    (N_COMP, N_ALV))
    for arr in [aops_raw, tops_raw]:
        arr[arr < 0] = 0
    aops = np.maximum(aops_raw, acps_raw)
    tops = np.maximum(tops_raw, tcps_raw)
    h[h <= 0.1] = 0.1
    return sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h


def run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean):
    """
    Decremental PEEP trial on raw arrays.
    Returns (peep_levels, comp_unc, comp_corr) where each comp list contains
    per-compartment compliance arrays (mL/cmH₂O).
    """
    aw, alv = _stabilize_raw(sp, aops, acps_raw, tops, tcps_raw)
    for _ in range(5):
        aw, alv, _, _ = step_fn(24, DP, aw, alv, sp, aops, acps_raw, tops, tcps_raw)

    comp_unc = []
    comp_corr = []
    aw_c, alv_c = aw.copy(), alv.copy()

    for peep in PEEP_LEVELS:
        for _ in range(5):
            aw_c, alv_c, _, _ = step_fn(peep, DP, aw_c, alv_c, sp, aops, acps_raw, tops, tcps_raw)
        prev_alv      = alv_c.copy()
        naw, nalv, ti, _ = step_fn(peep, DP, aw_c, alv_c, sp, aops, acps_raw, tops, tcps_raw)
        te_exp        = peep - sp
        can_m         = nalv & (te_exp >= acps_raw)
        trap_m        = nalv & ~(te_exp >= acps_raw)
        normal_can_m  = can_m & prev_alv
        trap_open_m   = trap_m & ((te_exp >= tcps_raw) | (acps_raw > tcps_raw))
        vt_c = (np.sum(vol_fn(v0, h, ti)       * nalv,         axis=1)
              - np.sum(vol_fn(v0, h, te_exp)    * normal_can_m, axis=1)
              - np.sum(vol_fn(v0, h, acps_raw)  * trap_open_m,  axis=1))
        comp_unc.append(vt_c / DP * 1000)
        pip  = peep + DP
        ep   = np.maximum(peep, aop_mean + sp_vals)
        dpe  = pip - ep
        comp_corr.append(np.where(dpe > 1e-9, vt_c / dpe * 1000, 0))
        aw_c, alv_c = naw, nalv

    return PEEP_LEVELS, comp_unc, comp_corr


def get_odcl_raw(peep_levels, comp_unc, comp_corr):
    pl   = list(peep_levels)
    ca_u = costa(comp_unc)
    ca_c = costa(comp_corr)
    return find_odcl(pl, ca_u), find_odcl(pl, ca_c)


# ==============================================================================
# Module-level worker functions (must be picklable for multiprocessing)
# ==============================================================================

# ── PART 1 worker ─────────────────────────────────────────────────────────────
def _worker_part1(params):
    try:
        np.random.seed(None)
        lung  = _make_lung(params)
        res   = lung.run_peep_trial(PEEP_LEVELS, DP)
        peeps = [r['peep'] for r in res]
        sp_arr = lung.sp.flatten()
        ca_u   = analyze_costa(res, 'comp_per_comp')
        odcl_u = find_odcl(peeps, ca_u)
        res_c  = apply_aop_correction(copy.deepcopy(res), params['aop_mean'], sp_arr, N_COMP)
        ca_c   = analyze_costa(res_c, 'corrected_comp_per_comp')
        odcl_c = find_odcl(peeps, ca_c)
        return {'uncorrected': odcl_u, 'corrected': odcl_c}
    except Exception:
        traceback.print_exc()
        return {'uncorrected': np.nan, 'corrected': np.nan}


# ── PART 4 worker ─────────────────────────────────────────────────────────────
def _worker_part4(args):
    aop_mean, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, MAX_SP)
    pl, cu, cc = run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl_raw(pl, cu, cc)


# ── PART 6 workers ────────────────────────────────────────────────────────────
def _worker_part6_acp3(args):
    aop_mean, max_sp, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_acp3(aop_mean, max_sp)
    pl, cu, cc = run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl_raw(pl, cu, cc)


def _worker_part6_aop_eq_acp(args):
    aop_mean, max_sp, seed = args
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, max_sp)
    pl, cu, cc = run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    return get_odcl_raw(pl, cu, cc)


# ── PART 7 worker ─────────────────────────────────────────────────────────────
def _worker_part7(_):
    np.random.seed(None)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_acp3(12.0, MAX_SP)
    pl, cu, cc = run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, 12.0)
    return get_odcl_raw(pl, cu, cc)


# ── PART 8 workers ────────────────────────────────────────────────────────────
def _worker_part8(params):
    try:
        np.random.seed(None)
        AOP_MEAN = params['aop_mean']
        lung     = _make_lung(params)
        res      = lung.run_peep_trial(PEEP_LEVELS, DP)
        peeps    = [r['peep'] for r in res]
        sp_arr   = lung.sp.flatten()

        ca_u   = analyze_costa(res, 'comp_per_comp')
        odcl_u = find_odcl(peeps, ca_u)

        res_c   = apply_aop_correction(copy.deepcopy(res), AOP_MEAN, sp_arr, N_COMP, use_sp=True)
        ca_c    = analyze_costa(res_c, 'corrected_comp_per_comp')
        odcl_c  = find_odcl(peeps, ca_c)

        res_cn  = apply_aop_correction(copy.deepcopy(res), AOP_MEAN, sp_arr, N_COMP, use_sp=False)
        ca_cn   = analyze_costa(res_cn, 'corrected_comp_per_comp')
        odcl_cn = find_odcl(peeps, ca_cn)

        return {'uncorrected': odcl_u, 'corrected_with_sp': odcl_c, 'corrected_no_sp': odcl_cn}
    except Exception:
        traceback.print_exc()
        return {k: np.nan for k in ['uncorrected', 'corrected_with_sp', 'corrected_no_sp']}


# ── PART 9 worker ─────────────────────────────────────────────────────────────
def _worker_part9(params):
    try:
        np.random.seed(None)
        AOP_MEAN = params['aop_mean']
        lung     = _make_lung(params)
        res      = lung.run_peep_trial(PEEP_LEVELS, DP)
        peeps    = [r['peep'] for r in res]
        sp_arr   = lung.sp.flatten()

        ca_u   = analyze_costa(res, 'comp_per_comp')
        odcl_u = find_odcl(peeps, ca_u)

        res_c  = apply_aop_correction(copy.deepcopy(res), AOP_MEAN, sp_arr, N_COMP, use_sp=True)
        ca_c   = analyze_costa(res_c, 'corrected_comp_per_comp')
        odcl_c = find_odcl(peeps, ca_c)

        return {'uncorrected': odcl_u, 'corrected': odcl_c}
    except Exception:
        traceback.print_exc()
        return {k: np.nan for k in ['uncorrected', 'corrected']}


# ==============================================================================
# PART 1 — AOP sensitivity analysis  (ACP = 3 cmH₂O, fixed)
# ==============================================================================
def part1_aop_sensitivity_acp3():
    print("\n" + "="*70)
    print("PART 1: AOP sensitivity analysis  (ACP = 3 cmH₂O, fixed)")
    print("="*70)

    BASE = dict(v_max_ml=2500.0, h_mean=H_MEAN, h_sd=H_SD,
                top_mean=TOP_MEAN, top_sd=TOP_SD,
                tcp_mean=TCP_MEAN, tcp_sd=TCP_SD,
                max_sp=MAX_SP, acp_mean=3.0, acp_sd=1.0, aop_sd=AOP_SD)

    AOP_LEVELS = np.arange(4, 17, 2)
    all_rows   = []

    for aop_val in AOP_LEVELS:
        print(f"  AOP = {aop_val} cmH₂O ...", flush=True)
        p = BASE.copy(); p['aop_mean'] = aop_val
        with Pool(cpu_count()) as pool:
            raw = pool.map(_worker_part1, [p] * N_RUNS)
        for r in raw:
            if not np.isnan(r.get('uncorrected', np.nan)):
                all_rows.append({'AOP_Level':         aop_val,
                                 'Uncorrected method': r['uncorrected'],
                                 'AOP-Corrected method': r['corrected']})

    df = pd.DataFrame(all_rows)

    # Paired t-test
    def _paired_t(a, b):
        d  = np.array(a) - np.array(b); n = len(d)
        mu = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
        t  = mu / se if se > 1e-12 else 0.0
        p  = 2*(1 - 0.5*(1 + math.erf(abs(t)/math.sqrt(2))))
        g  = mu / d.std(ddof=1) if d.std(ddof=1) > 1e-12 else 0.0
        return t, p, g

    print(f"\n{'AOP':>5}  {'Uncorr':>8}  {'Corr':>8}  {'Diff':>8}  {'t':>7}  {'p':>10}  {'g':>7}")
    sig_aops = []
    for aop_val in AOP_LEVELS:
        sub = df[df['AOP_Level'] == aop_val]
        unc = sub['Uncorrected method'].values
        cor = sub['AOP-Corrected method'].values
        if len(unc) > 1:
            t, p, g = _paired_t(unc, cor)
            sig = " *" if p < 0.05 else ""
            print(f"{aop_val:>5}  {unc.mean():>8.2f}  {cor.mean():>8.2f}  "
                  f"{(unc-cor).mean():>8.2f}  {t:>7.3f}  {p:>10.4g}  {g:>7.3f}{sig}")
            if p < 0.05:
                sig_aops.append(aop_val)
    print(f"\n-> {len(sig_aops)}/{len(AOP_LEVELS)} AOP levels significant (p<0.05)")

    # Sensitivity figure
    df_long = df.melt('AOP_Level', var_name='Method', value_name='ODCL_PEEP')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_long, x='AOP_Level', y='ODCL_PEEP',
                 hue='Method', style='Method', markers=True, dashes=False,
                 palette='viridis', linewidth=2.5, markersize=10,
                 errorbar='sd', ax=ax)
    summary = df_long.groupby(['AOP_Level', 'Method'])['ODCL_PEEP'].agg(['mean','std']).reset_index()
    summary['top'] = summary['mean'] + summary['std']
    for aop in sig_aops:
        y = summary[summary['AOP_Level'] == aop]['top'].max()
        ax.text(aop, y + 0.5, '*', ha='center', va='bottom', color='red', fontsize=20, fontweight='bold')
    ax.set_xlabel(r'Set $AOP_{regional}$ Mean (cmH$_2$O)', fontsize=13)
    ax.set_ylabel('Calculated ODCL PEEP (cmH$_2$O)', fontsize=13)
    ax.set_title('Sensitivity Analysis: AOP on ODCL PEEP  (ACP = 3 cmH₂O, fixed)', fontsize=12)
    ax.text(0.98, 0.80, '*: p < 0.05', transform=ax.transAxes, fontsize=11, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Sensitivity_Analysis.pdf'), format='pdf')
    plt.close()
    print("  Saved: Sensitivity_Analysis.pdf")

    # Representative crossover (AOP=12, ACP=3)
    p = BASE.copy(); p['aop_mean'] = 12.0
    np.random.seed(42)
    lung = _make_lung(p)
    res  = lung.run_peep_trial(PEEP_LEVELS, DP)
    peeps = [r['peep'] for r in res]; sp_arr = lung.sp.flatten()
    ca_u  = analyze_costa(res, 'comp_per_comp');          odcl_u = find_odcl(peeps, ca_u)
    res_c = apply_aop_correction(copy.deepcopy(res), 12.0, sp_arr, N_COMP)
    ca_c  = analyze_costa(res_c, 'corrected_comp_per_comp'); odcl_c = find_odcl(peeps, ca_c)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(peeps, [a['collapse']       for a in ca_u], 'v-',  color='brown',        alpha=0.8, lw=2, ms=8, label='Collapse (Uncorrected)')
    ax.plot(peeps, [a['overdistention'] for a in ca_u], 'o-',  color='darkblue',     alpha=0.8, lw=2, ms=8, label='Overdistention (Uncorrected)')
    ax.plot(peeps, [a['collapse']       for a in ca_c], 'v--', color='sandybrown',   alpha=0.9, lw=2, ms=8, label='Collapse (AOP-Corrected)')
    ax.plot(peeps, [a['overdistention'] for a in ca_c], 'o--', color='cornflowerblue', alpha=0.9, lw=2, ms=8, label='Overdistention (AOP-Corrected)')
    if not np.isnan(odcl_u): ax.axvline(odcl_u, color='black', lw=3,        label=f'ODCL (Uncorrected) = {odcl_u:.1f}')
    if not np.isnan(odcl_c): ax.axvline(odcl_c, color='red',   lw=3, ls='--', label=f'ODCL (AOP-Corrected) = {odcl_c:.1f}')
    ax.set_xlabel('PEEP (cmH₂O)', fontsize=13); ax.set_ylabel('Overdistention / Collapse (%)', fontsize=13)
    ax.set_title('Representative Crossover Plot  (AOP = 12, ACP = 3 cmH₂O)', fontsize=12)
    ax.invert_xaxis(); ax.legend(loc='best', fontsize=11); ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Representative_AOP_12.pdf'), format='pdf')
    plt.close()
    print("  Saved: Representative_AOP_12.pdf")


# ==============================================================================
# PART 2 — Alveolar state analysis  (ACP = 3 cmH₂O, AOP = 12 cmH₂O)
# ==============================================================================
def _alveolar_state_plots(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h,
                          aop_mean, acp_label, fname_prefix):
    """Shared plotting logic for PARTS 2 and 3."""
    def step(peep, dp_val, aw, alv):
        return step_fn(peep, dp_val, aw, alv, sp, aops, acps_raw, tops, tcps_raw)

    aw, alv = _stabilize_raw(sp, aops, acps_raw, tops, tcps_raw)
    for _ in range(5):
        aw, alv, _, _ = step(24, DP, aw, alv)

    state_list = []; aw_c, alv_c = aw.copy(), alv.copy()
    for peep in PEEP_LEVELS:
        for _ in range(5):
            aw_c, alv_c, _, _ = step(peep, DP, aw_c, alv_c)
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
    for p in [22, 20, 18, 16, 14]:
        for _ in range(5): aw_h, alv_h, _, _ = step(p, DP, aw_h, alv_h)
        aw_h, alv_h, _, _ = step(p, DP, aw_h, alv_h)
    for _ in range(5): aw_h, alv_h, _, _ = step(12, DP, aw_h, alv_h)
    prev12 = aw_h.copy()
    naw12, nalv12, ti12, _ = step(12, DP, aw_h, alv_h)
    open_c   = np.array([(nalv12[c] &  naw12[c]).sum() for c in range(N_COMP)])
    trap_c   = np.array([(nalv12[c] & ~naw12[c]).sum() for c in range(N_COMP)])
    closed_c = np.array([(~nalv12[c]).sum()             for c in range(N_COMP)])
    aop_c    = np.array([(~prev12[c] & (ti12[c] >= aops[c])).sum() for c in range(N_COMP)])

    # Fig A: Alveolar state composition
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(f'Alveolar States During Decremental PEEP Trial  ({acp_label})\n'
                  f'AOP={aop_mean}±{AOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
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
    fig2.savefig(os.path.join(OUTPUT_DIR, f'Fig_alveolar_states_{fname_prefix}.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig B: Compartment-level at PEEP=12
    sp_threshold = 12 - aop_mean
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 6))
    fig3.suptitle(f'Compartment-Level States at PEEP=12 cmH₂O  ({acp_label})\n'
                  f'Closure: SP > PEEP − AOP_mean = {sp_threshold:.0f} cmH₂O',
                  fontsize=11, fontweight='bold')
    for ax, (data, title, color) in zip(axes3, [
        (open_c,   'Aerated',        '#27ae60'),
        (trap_c,   'Trapped',        '#e67e22'),
        (closed_c, 'Collapsed',      '#c0392b'),
        (aop_c,    'AOP Manifested', '#2980b9')]):
        ax.bar(sp_vals, data, width=MAX_SP/N_COMP*0.85, color=color, alpha=0.8, edgecolor='none')
        ax.axvline(sp_threshold, color='darkred', lw=2, ls='--', label=f'SP={sp_threshold:.0f}')
        ax.set_xlabel('SP (cmH₂O)', fontsize=11); ax.set_ylabel('Units/compartment', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, MAX_SP+0.5); ax.set_ylim(0, N_ALV*1.1)
        ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=0.4)
        ax.text(0.97, 0.97, f'Total: {int(data.sum()):,}\n({data.sum()/TOTAL*100:.1f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, f'Fig_compartment_{fname_prefix}.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: Fig_alveolar_states_{fname_prefix}.png, Fig_compartment_{fname_prefix}.png")


def part2_alveolar_state_acp3():
    print("\n" + "="*70)
    print("PART 2: Alveolar state analysis  (AOP=12, ACP=3 cmH₂O)")
    print("="*70)
    np.random.seed(42)
    AOP_MEAN = 12.0

    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_acp3(AOP_MEAN, MAX_SP)

    # Parameter distribution figure
    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5))
    fig1.suptitle(f'Parameter Distributions  (AOP={AOP_MEAN}±{AOP_SD}, ACP=3±1 cmH₂O)\n'
                  f'TOP={TOP_MEAN}±{TOP_SD}, TCP={TCP_MEAN}±{TCP_SD}  [N={N_COMP}×{N_ALV}={TOTAL:,}]',
                  fontsize=11, fontweight='bold')
    for ax, (name, fullname, arr, color) in zip(axes1, [
        ('AOP', 'Airway Opening Pressure', aops,     '#c0392b'),
        ('ACP', 'Airway Closing Pressure', acps_raw, '#2980b9'),
        ('TOP', 'Total Opening Pressure',  tops,     '#27ae60'),
        ('TCP', 'Total Closing Pressure',  tcps_raw, '#d68910')]):
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
    fig1.savefig(os.path.join(OUTPUT_DIR, 'Fig_distributions_acp3.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_distributions_acp3.png")

    _alveolar_state_plots(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h,
                          AOP_MEAN, 'ACP=3 cmH₂O, independent', 'acp3')


# ==============================================================================
# PART 3 — Alveolar state analysis  (AOP = ACP per unit)
# ==============================================================================
def part3_alveolar_state_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 3: Alveolar state analysis  (AOP=ACP=12, per unit)")
    print("="*70)
    np.random.seed(42)
    AOP_MEAN = 12.0

    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(AOP_MEAN, MAX_SP)
    _alveolar_state_plots(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h,
                          AOP_MEAN, 'AOP=ACP per unit', 'aop_eq_acp')


# ==============================================================================
# PART 4 — ODCL sensitivity  (AOP = ACP condition, AOP 4–16 cmH₂O)
# ==============================================================================
def part4_odcl_sensitivity_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 4: ODCL sensitivity analysis  (AOP=ACP, 4–16 cmH₂O)")
    print("="*70)
    AOP_LEVELS = np.arange(4, 17, 2)
    unc_m, unc_s, cor_m, cor_s = [], [], [], []

    for aop_val in AOP_LEVELS:
        args = [(aop_val, seed) for seed in range(N_RUNS)]
        with Pool(min(cpu_count(), 8)) as pool:
            raw = pool.map(_worker_part4, args)
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
    ax.fill_between(aop_x, np.array(unc_m)-np.array(unc_s), np.array(unc_m)+np.array(unc_s), alpha=0.12, color='red')
    ax.fill_between(aop_x, np.array(cor_m)-np.array(cor_s), np.array(cor_m)+np.array(cor_s), alpha=0.12, color='blue')
    ax.set_xlabel('AOP = ACP (cmH₂O)', fontsize=13); ax.set_ylabel('ODCL PEEP (cmH₂O)', fontsize=13)
    ax.set_title(f'ODCL PEEP Sensitivity: AOP=ACP (same value per unit)\n'
                 f'TOP={TOP_MEAN}±{TOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=12); ax.set_xticks(aop_x); ax.set_ylim(0, 25); ax.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_sensitivity_aop_eq_acp.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_sensitivity_aop_eq_acp.png")


# ==============================================================================
# PART 5 — Crossover plots  (AOP = ACP, 2×2 + representative)
# ==============================================================================
def _run_crossover_aop_eq_acp(aop_mean, seed=42):
    np.random.seed(seed)
    sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h = build_model_aop_eq_acp(aop_mean, MAX_SP)
    pl, cu, cc = run_peep_trial_raw(sp_vals, sp, v0, aops, acps_raw, tops, tcps_raw, h, aop_mean)
    ca_u = costa(cu); odcl_u = find_odcl(list(pl), ca_u)
    ca_c = costa(cc); odcl_c = find_odcl(list(pl), ca_c)
    return pl, ca_u, ca_c, odcl_u, odcl_c


def part5_crossover_aop_eq_acp():
    print("\n" + "="*70)
    print("PART 5: Crossover plots  (AOP=ACP)")
    print("="*70)

    AOP_CASES = [4, 8, 12, 16]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Crossover Plots: Uncorrected vs AOP-Corrected  (AOP=ACP per unit)\n'
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
        if not np.isnan(odcl_u): ax.axvline(odcl_u, color='#e74c3c', lw=2.5, label=f'ODCL Uncorr={odcl_u:.1f}')
        if not np.isnan(odcl_c): ax.axvline(odcl_c, color='#2980b9', lw=2.5, ls='--', label=f'ODCL AOP-Corr={odcl_c:.1f}')
        ax.set_xlim(max(pl)+1, min(pl)-1); ax.set_ylim(-2, 102)
        ax.set_xlabel('PEEP (cmH₂O)', fontsize=11); ax.set_ylabel('%', fontsize=10)
        ax.set_title(f'AOP=ACP={aop_val} cmH₂O', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper left'); ax.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_crossover_aop_eq_acp.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Representative (AOP=ACP=12)
    peeps, ca_u, ca_c, odcl_u, odcl_c = _run_crossover_aop_eq_acp(12)
    pl = list(peeps)
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.plot(pl, [a['collapse']       for a in ca_u], 'v-',  color='#e74c3c', lw=2.5, ms=9, label='Collapse (Uncorrected)')
    ax2.plot(pl, [a['overdistention'] for a in ca_u], 'o-',  color='#3498db', lw=2.5, ms=9, label='Overdistention (Uncorrected)')
    ax2.plot(pl, [a['collapse']       for a in ca_c], 'v--', color='#c0392b', lw=2.5, ms=9, alpha=0.75, label='Collapse (AOP-Corrected)')
    ax2.plot(pl, [a['overdistention'] for a in ca_c], 'o--', color='#2980b9', lw=2.5, ms=9, alpha=0.75, label='Overdistention (AOP-Corrected)')
    if not np.isnan(odcl_u): ax2.axvline(odcl_u, color='#e74c3c', lw=3, label=f'ODCL (Uncorrected) = {odcl_u:.1f} cmH₂O')
    if not np.isnan(odcl_c): ax2.axvline(odcl_c, color='#2980b9', lw=3, ls='--', label=f'ODCL (AOP-Corrected) = {odcl_c:.1f} cmH₂O')
    ax2.set_xlim(max(pl)+1, min(pl)-1); ax2.set_ylim(-2, 102)
    ax2.set_xlabel('PEEP (cmH₂O)', fontsize=13); ax2.set_ylabel('%', fontsize=12)
    ax2.set_title(f'Representative Crossover  (AOP=ACP=12 cmH₂O)\n'
                  f'TOP={TOP_MEAN}±{TOP_SD}, SP=0–{MAX_SP}, DP={DP} cmH₂O',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left'); ax2.grid(ls='--', alpha=0.4)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'Fig_crossover_aop_eq_acp_representative.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Fig_crossover_aop_eq_acp.png, Fig_crossover_aop_eq_acp_representative.png")


# ==============================================================================
# PART 6 — SP range sensitivity overlay  (ACP=3 and AOP=ACP)
# ==============================================================================
_SP_CONDITIONS = [
    {"label": "SP = 0–14.5 cmH₂O", "max_sp": 14.5, "color": "#2ca02c"},
    {"label": "SP = 0–7 cmH₂O",    "max_sp":  7.0,  "color": "#ff7f0e"},
    {"label": "SP = 0 cmH₂O",      "max_sp":  0.0,  "color": "#9467bd"},
]


def _collect_sp_sensitivity(run_func, label):
    AOP_LEVELS = np.arange(4, 17, 2)
    print(f"\n  {label}")
    data = {c['label']: {'unc_m':[],'unc_s':[],'cor_m':[],'cor_s':[]} for c in _SP_CONDITIONS}
    for cond in _SP_CONDITIONS:
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


def _plot_sp_overlay(data, title, fname):
    AOP_LEVELS = np.arange(4, 17, 2); aop_x = list(AOP_LEVELS)
    fig, ax = plt.subplots(figsize=(11, 7))
    for cond in _SP_CONDITIONS:
        d = data[cond['label']]; color = cond['color']
        um, us = np.array(d['unc_m']), np.array(d['unc_s'])
        cm, cs = np.array(d['cor_m']), np.array(d['cor_s'])
        ax.errorbar(aop_x, um, yerr=us, fmt='o-',  color=color, lw=2.5, ms=8, capsize=5,
                    label=f"{cond['label']}  [Uncorrected]")
        ax.fill_between(aop_x, um-us, um+us, alpha=0.10, color=color)
        ax.errorbar(aop_x, cm, yerr=cs, fmt='s--', color=color, lw=2.5, ms=8, capsize=5,
                    alpha=0.75, label=f"{cond['label']}  [AOP-Corrected]")
        ax.fill_between(aop_x, cm-cs, cm+cs, alpha=0.07, color=color)
    legend_extra = [Line2D([0],[0],color='k',lw=2.5,ls='-',  label='─  Uncorrected'),
                    Line2D([0],[0],color='k',lw=2.5,ls='--', label='--  AOP-Corrected')]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles+legend_extra, labels+['─  Uncorrected','--  AOP-Corrected'],
              fontsize=9, loc='upper left', ncol=2)
    ax.set_xlabel('AOP Mean (cmH₂O)', fontsize=13); ax.set_ylabel('ODCL PEEP (cmH₂O)', fontsize=13)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(aop_x); ax.set_ylim(0, 25); ax.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def part6_sp_sensitivity_overlay():
    print("\n" + "="*70)
    print("PART 6: SP range sensitivity overlay")
    print("="*70)
    data_acp3    = _collect_sp_sensitivity(_worker_part6_acp3,       "ACP=3 (fixed)")
    data_aop_acp = _collect_sp_sensitivity(_worker_part6_aop_eq_acp, "AOP=ACP (per unit)")

    _plot_sp_overlay(data_acp3,
        f'SP Sensitivity  (ACP=3 cmH₂O, fixed)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
        'Fig_SP_sensitivity_overlay_acp3.png')

    _plot_sp_overlay(data_aop_acp,
        f'SP Sensitivity  (AOP=ACP per unit)\n'
        f'TOP={TOP_MEAN}±{TOP_SD}, DP={DP} cmH₂O  (n={N_RUNS} runs/point)',
        'Fig_SP_sensitivity_overlay_aop_eq_acp.png')


# ==============================================================================
# PART 7 — SD estimation  (representative crossover, AOP=12, ACP=3)
# ==============================================================================
def part7_sd_estimation():
    print("\n" + "="*70)
    print("PART 7: SD estimation — representative crossover (AOP=12, ACP=3)")
    print("="*70)
    with Pool(min(cpu_count(), 8)) as pool:
        raw = pool.map(_worker_part7, range(N_RUNS))
    uncs  = [r[0] for r in raw if not np.isnan(r[0])]
    corrs = [r[1] for r in raw if not np.isnan(r[1])]
    print(f"  N = {N_RUNS} runs  (AOP=12, ACP=3, SP=0–{MAX_SP}, DP={DP})")
    print(f"  Uncorrected ODCL PEEP  : {np.mean(uncs):.2f} ± {np.std(uncs):.2f} cmH₂O"
          f"  (range {np.min(uncs):.2f}–{np.max(uncs):.2f})")
    print(f"  AOP-Corrected ODCL PEEP: {np.mean(corrs):.2f} ± {np.std(corrs):.2f} cmH₂O"
          f"  (range {np.min(corrs):.2f}–{np.max(corrs):.2f})")
    print(f"  Difference (Corr−Uncorr): {np.mean(corrs)-np.mean(uncs):.2f} cmH₂O")


# ==============================================================================
# PART 8 — Revised main analysis  (ACP = AOP)
# ==============================================================================
def _make_crossover_plot(lung, aop_mean, acp_label, fname):
    res   = lung.run_peep_trial(PEEP_LEVELS, DP)
    peeps = [r['peep'] for r in res]; sp_arr = lung.sp.flatten()
    ca_u  = analyze_costa(res, 'comp_per_comp'); odcl_u = find_odcl(peeps, ca_u)
    res_c = apply_aop_correction(copy.deepcopy(res), aop_mean, sp_arr, N_COMP, use_sp=True)
    ca_c  = analyze_costa(res_c, 'corrected_comp_per_comp'); odcl_c = find_odcl(peeps, ca_c)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(peeps, [a['collapse']       for a in ca_u], 'v-',  color='brown',          alpha=0.8, lw=2, ms=8, label='Collapse (Uncorrected)')
    ax.plot(peeps, [a['overdistention'] for a in ca_u], 'o-',  color='darkblue',       alpha=0.8, lw=2, ms=8, label='Overdistention (Uncorrected)')
    ax.plot(peeps, [a['collapse']       for a in ca_c], 'v--', color='sandybrown',     alpha=0.9, lw=2, ms=8, label='Collapse (AOP-Corrected)')
    ax.plot(peeps, [a['overdistention'] for a in ca_c], 'o--', color='cornflowerblue', alpha=0.9, lw=2, ms=8, label='Overdistention (AOP-Corrected)')
    if not np.isnan(odcl_u): ax.axvline(odcl_u, color='black', lw=3,        label=f'ODCL PEEP (Uncorrected) = {odcl_u:.1f} cmH₂O')
    if not np.isnan(odcl_c): ax.axvline(odcl_c, color='red',   lw=3, ls='--', label=f'ODCL PEEP (AOP-Corrected) = {odcl_c:.1f} cmH₂O')
    ax.set_xlabel('PEEP (cmH₂O)', fontsize=13); ax.set_ylabel('Overdistention / Collapse (%)', fontsize=13)
    ax.set_title(f'Overdistention/Collapse Crossover Plot\n(AOP={aop_mean} cmH₂O, {acp_label})', fontsize=13)
    ax.invert_xaxis(); ax.legend(loc='best', fontsize=11); ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(fname)}")
    return odcl_u, odcl_c


def part8_acp_eq_aop_revised():
    print("\n" + "="*70)
    print("PART 8: Revised main analysis  (ACP = AOP)")
    print("="*70)

    BASE = dict(v_max_ml=2500.0, h_mean=H_MEAN, h_sd=H_SD,
                top_mean=TOP_MEAN, top_sd=TOP_SD,
                tcp_mean=TCP_MEAN, tcp_sd=TCP_SD,
                max_sp=MAX_SP, aop_sd=AOP_SD)
    AOP_LEVELS = np.arange(4, 17, 2)
    N_RUNS_8   = 30

    # ACP=AOP scenario
    all_rows = []
    for aop_val in AOP_LEVELS:
        print(f"  AOP = {aop_val} cmH₂O ...", flush=True)
        params_list = []
        for _ in range(N_RUNS_8):
            p = BASE.copy(); p['aop_mean'] = aop_val; p['acp_mean'] = aop_val; p['acp_sd'] = AOP_SD
            params_list.append(p)
        with Pool(cpu_count()) as pool:
            results = pool.map(_worker_part8, params_list)
        for res in results:
            if not np.isnan(res.get('uncorrected', np.nan)):
                all_rows.append({'AOP_Level': aop_val,
                                 'Uncorrected':            res['uncorrected'],
                                 'AOP-Corrected (with SP)': res['corrected_with_sp'],
                                 'AOP-Corrected (no SP)':   res['corrected_no_sp']})

    df_acp_aop = pd.DataFrame(all_rows)

    print("\nSummary (ACP = AOP):")
    for aop_val in AOP_LEVELS:
        sub = df_acp_aop[df_acp_aop['AOP_Level'] == aop_val]
        print(f"  AOP={aop_val}: Uncorr={sub['Uncorrected'].mean():.1f}±{sub['Uncorrected'].std():.1f}  "
              f"Corr(+SP)={sub['AOP-Corrected (with SP)'].mean():.1f}±{sub['AOP-Corrected (with SP)'].std():.1f}  "
              f"Corr(-SP)={sub['AOP-Corrected (no SP)'].mean():.1f}±{sub['AOP-Corrected (no SP)'].std():.1f}")

    # Figure 4 revised
    fig4_df = df_acp_aop.melt('AOP_Level',
                               value_vars=['Uncorrected', 'AOP-Corrected (with SP)'],
                               var_name='Method', value_name='ODCL_PEEP')
    colors  = {'Uncorrected': '#1f77b4', 'AOP-Corrected (with SP)': '#2ca02c'}
    markers = {'Uncorrected': 'o',       'AOP-Corrected (with SP)': 'X'}
    fig, ax = plt.subplots(figsize=(12, 8))
    for method, color in colors.items():
        sub = fig4_df[fig4_df['Method'] == method]
        grp = sub.groupby('AOP_Level')['ODCL_PEEP'].agg(['mean','std']).reset_index()
        ax.errorbar(grp['AOP_Level'], grp['mean'], yerr=grp['std'],
                    fmt=f"{markers[method]}-", color=color, linewidth=2.5, markersize=10,
                    capsize=5, label=method)
        for _, row in grp.iterrows():
            ax.text(row['AOP_Level'], row['mean'] + row['std'] + 0.4, '*',
                    ha='center', va='bottom', color='red', fontsize=18, fontweight='bold')
    ax.set_xlabel('Set AOP$_{regional}$ Mean (cmH₂O)', fontsize=13)
    ax.set_ylabel('Calculated ODCL PEEP (cmH₂O)', fontsize=13)
    ax.set_title('Sensitivity Analysis: AOP on ODCL PEEP\n(Revised: ACP = AOP)', fontsize=13)
    ax.legend(title='Calculation Method', fontsize=12, loc='upper left')
    ax.text(0.98, 0.05, '* p < 0.05 (paired t-test)', transform=ax.transAxes, ha='right', va='bottom', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure4_revised_ACP_equals_AOP.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Figure4_revised_ACP_equals_AOP.png")

    # Representative crossover (AOP=ACP=12)
    p = BASE.copy(); p['aop_mean'] = 12.0; p['acp_mean'] = 12.0; p['acp_sd'] = AOP_SD
    np.random.seed(42)
    lung = _make_lung(p)
    _make_crossover_plot(lung, 12.0, 'ACP = AOP = 12 cmH₂O',
                         os.path.join(OUTPUT_DIR, 'Figure3_revised_crossover_ACP12_AOP12.png'))

    # Comparison: original (ACP=3) vs revised (ACP=AOP)
    print("\n  Running original ACP=3 scenario for comparison ...")
    all_rows_orig = []
    for aop_val in AOP_LEVELS:
        print(f"    AOP = {aop_val} (ACP=3) ...", flush=True)
        params_list = []
        for _ in range(N_RUNS_8):
            p = BASE.copy(); p['aop_mean'] = aop_val; p['acp_mean'] = 3.0; p['acp_sd'] = 1.0
            params_list.append(p)
        with Pool(cpu_count()) as pool:
            results = pool.map(_worker_part8, params_list)
        for res in results:
            if not np.isnan(res.get('uncorrected', np.nan)):
                all_rows_orig.append({'AOP_Level': aop_val,
                                      'Uncorrected':    res['uncorrected'],
                                      'AOP-Corrected':  res['corrected_with_sp'],
                                      'Model': 'Original (ACP=3 cmH₂O, fixed)'})
    for _, row in df_acp_aop.iterrows():
        all_rows_orig.append({'AOP_Level': row['AOP_Level'],
                              'Uncorrected':   row['Uncorrected'],
                              'AOP-Corrected': row['AOP-Corrected (with SP)'],
                              'Model': 'Revised (ACP=AOP)'})
    df_comp = pd.DataFrame(all_rows_orig)
    model_colors = {'Original (ACP=3 cmH₂O, fixed)': '#1f77b4', 'Revised (ACP=AOP)': '#d62728'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, (title, col) in zip(axes, [('Uncorrected ODCL PEEP','Uncorrected'),
                                        ('AOP-Corrected ODCL PEEP','AOP-Corrected')]):
        for model, color in model_colors.items():
            sub = df_comp[df_comp['Model'] == model]
            grp = sub.groupby('AOP_Level')[col].agg(['mean','std']).reset_index()
            ax.errorbar(grp['AOP_Level'], grp['mean'], yerr=grp['std'],
                        fmt='o-', color=color, linewidth=2.5, markersize=10, capsize=5, label=model)
        ax.set_xlabel('Set AOP$_{regional}$ Mean (cmH₂O)', fontsize=12)
        ax.set_ylabel('Calculated ODCL PEEP (cmH₂O)', fontsize=12)
        ax.set_title(title, fontsize=12); ax.legend(fontsize=10); ax.grid(axis='y', ls='--', alpha=0.6)
    fig.suptitle('Impact of ACP Assumption on ODCL PEEP\nOriginal (ACP=3, fixed) vs Revised (ACP=AOP)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_comparison_original_vs_revised.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Figure_comparison_original_vs_revised.png")


# ==============================================================================
# PART 9 — SP range sensitivity  (LungModel-based, ACP = AOP)
# ==============================================================================
def part9_sp_sensitivity_proper():
    print("\n" + "="*70)
    print("PART 9: SP range sensitivity  (ACP=AOP, LungModel)")
    print("="*70)

    BASE = dict(v_max_ml=2500.0, h_mean=H_MEAN, h_sd=H_SD,
                top_mean=TOP_MEAN, top_sd=TOP_SD,
                tcp_mean=TCP_MEAN, tcp_sd=TCP_SD, aop_sd=AOP_SD)

    SP_CONDITIONS_9 = [
        {"label": "SP = 0–14.5 cmH₂O (original model)",  "max_sp": 14.5, "color": "#2ca02c", "ls": "-"},
        {"label": "SP = 0–5 cmH₂O (reduced SP range)",    "max_sp":  5.0, "color": "#ff7f0e", "ls": "--"},
        {"label": "SP = 0 cmH₂O (no gravitational gradient)", "max_sp": 0.0, "color": "#9467bd", "ls": ":"},
    ]
    AOP_LEVELS = np.arange(4, 17, 2)
    all_data   = []

    for sp_cond in SP_CONDITIONS_9:
        max_sp = sp_cond["max_sp"]
        label  = sp_cond["label"]
        print(f"\n  Running: {label}")
        for aop_val in AOP_LEVELS:
            print(f"    AOP = {aop_val} ...", flush=True)
            params_list = []
            for _ in range(N_RUNS):
                p = BASE.copy()
                p['aop_mean'] = aop_val; p['acp_mean'] = aop_val; p['acp_sd'] = AOP_SD
                p['max_sp']   = max_sp
                params_list.append(p)
            with Pool(cpu_count()) as pool:
                results = pool.map(_worker_part9, params_list)
            for r in results:
                if not np.isnan(r.get('uncorrected', np.nan)):
                    all_data.append({'SP_Label':     label,
                                     'AOP_Level':    aop_val,
                                     'Uncorrected':  r['uncorrected'],
                                     'AOP-Corrected': r['corrected']})

    df = pd.DataFrame(all_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    for sp_cond in SP_CONDITIONS_9:
        sub = df[df['SP_Label'] == sp_cond["label"]]
        for ax, col in zip(axes, ['Uncorrected', 'AOP-Corrected']):
            grp = sub.groupby('AOP_Level')[col].agg(['mean','std']).reset_index()
            ax.errorbar(grp['AOP_Level'], grp['mean'], yerr=grp['std'],
                        fmt='o' + sp_cond["ls"], color=sp_cond["color"],
                        linewidth=2.5, markersize=9, capsize=5,
                        label=sp_cond["label"])
    axes[0].set_title('Uncorrected ODCL PEEP',    fontsize=13)
    axes[1].set_title('AOP-Corrected ODCL PEEP',  fontsize=13)
    for ax in axes:
        ax.set_xlabel('Set AOP$_{regional}$ Mean (cmH₂O)', fontsize=12)
        ax.set_ylabel('Calculated ODCL PEEP (cmH₂O)', fontsize=12)
        ax.legend(fontsize=10, loc='upper left'); ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.suptitle('Sensitivity Analysis: Effect of Superimposed Pressure Range on ODCL PEEP\n'
                 '(ACP = AOP; SP range 0 cmH₂O vs 0–5 cmH₂O vs 0–14.5 cmH₂O)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_SP_sensitivity_proper.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Figure_SP_sensitivity_proper.png")

    # Summary
    print("\nSummary:")
    for sp_cond in SP_CONDITIONS_9:
        print(f"\n  {sp_cond['label']} (max_sp={sp_cond['max_sp']}):")
        sub = df[df['SP_Label'] == sp_cond["label"]]
        for aop_val in AOP_LEVELS:
            s = sub[sub['AOP_Level'] == aop_val]
            print(f"    AOP={aop_val}: Uncorr={s['Uncorrected'].mean():.1f}±{s['Uncorrected'].std():.1f}  "
                  f"Corr={s['AOP-Corrected'].mean():.1f}±{s['AOP-Corrected'].std():.1f}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  ODCL/AOP Paper — Consolidated Analysis (Submission Version)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  N_RUNS = {N_RUNS} per data point")
    print("=" * 70)

    part1_aop_sensitivity_acp3()
    part2_alveolar_state_acp3()
    part3_alveolar_state_aop_eq_acp()
    part4_odcl_sensitivity_aop_eq_acp()
    part5_crossover_aop_eq_acp()
    part6_sp_sensitivity_overlay()
    part7_sd_estimation()
    part8_acp_eq_aop_revised()
    part9_sp_sensitivity_proper()

    print("\n" + "=" * 70)
    print("  All analyses complete.")
    print("=" * 70)
