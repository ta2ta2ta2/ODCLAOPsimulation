import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import core simulation logic from the original file
# We will duplicate the core logic here to ensure it runs standalone
class LungModel:
    def __init__(self, n_compartments, 
                 max_sp_g1, aop_mean_g1, aop_sd_g1, acp_mean_g1, acp_sd_g1,
                 top_mean_g1, top_sd_g1, tcp_mean_g1, tcp_sd_g1,
                 tlc_L_g1, h_mean_g1, h_sd_g1):
        
        self.n_compartments = n_compartments
        self.n_alveoli_per_comp = 1000
        self.frc_L = 0.0 
        
        n_comp_g1 = self.n_compartments
        n_comp_g2 = 0
        total_units_g1 = n_comp_g1 * self.n_alveoli_per_comp
        
        sp_g1 = np.linspace(0, max_sp_g1, n_comp_g1)
        self.sp = sp_g1[:, np.newaxis] 

        def _generate_params_array(mean1, sd1):
            return np.random.normal(mean1, sd1, (n_comp_g1, self.n_alveoli_per_comp))

        self.aops = _generate_params_array(aop_mean_g1, aop_sd_g1)
        self.acps = _generate_params_array(acp_mean_g1, acp_sd_g1)
        self.tops = _generate_params_array(top_mean_g1, top_sd_g1)
        self.tcps = _generate_params_array(tcp_mean_g1, tcp_sd_g1)
        self.h_units = _generate_params_array(h_mean_g1, h_sd_g1)
        
        for p_array in [self.aops, self.tops]: p_array[p_array < 0] = 0
        
        v0_unit_L_g1 = tlc_L_g1 / total_units_g1
        self.v0_unit_L_array = np.full((n_comp_g1, self.n_alveoli_per_comp), v0_unit_L_g1)

        self.aops = np.maximum(self.aops, self.acps)
        self.tops = np.maximum(self.tops, self.tcps)
        self.aops = np.maximum(self.aops, self.tops)
        self.h_units[self.h_units <= 0.1] = 0.1

    def _calculate_volume_liters(self, pressure):
        pressure_pos = np.maximum(0, pressure)
        exponent = - (pressure_pos * np.log(2)) / self.h_units
        volume = self.v0_unit_L_array * (1 - np.exp(exponent))
        return np.maximum(0, volume)

    def _calculate_volume_for_state(self, peep, airway_open, alveoli_open_or_trapped):
        tp_at_peep = peep - self.sp
        can_deflate = airway_open & alveoli_open_or_trapped
        is_trapped = ~airway_open & alveoli_open_or_trapped
        
        vol_deflating_units = self._calculate_volume_liters(tp_at_peep)
        tp_at_closure = self.acps - self.sp
        vol_trapped_units = self._calculate_volume_liters(tp_at_closure)
        
        recruited_vol = np.sum(vol_deflating_units * can_deflate) + np.sum(vol_trapped_units * is_trapped)
        return self.frc_L + recruited_vol

    def get_trial_metrics(self, peep, dp, start_airway_open, start_alveoli_open_or_trapped):
        tp_insp = peep + dp - self.sp
        end_insp_airway_open = start_airway_open | (tp_insp >= self.aops)
        newly_ready_alveoli = (tp_insp >= self.tops)
        alveoli_ready_to_open = start_alveoli_open_or_trapped | newly_ready_alveoli
        end_insp_alveoli_open = alveoli_ready_to_open & end_insp_airway_open

        vol_at_peak_units = self._calculate_volume_liters(tp_insp)
        total_vol_at_peak = self.frc_L + np.sum(vol_at_peak_units * end_insp_alveoli_open)

        tp_exp = peep - self.sp
        airways_would_be_open = (tp_exp >= self.acps)
        alveoli_would_be_open = (tp_exp >= self.tcps)
        
        airways_that_are_still_open = end_insp_alveoli_open & airways_would_be_open
        alveoli_that_are_still_open = end_insp_alveoli_open & alveoli_would_be_open
        
        can_deflate = airways_that_are_still_open & alveoli_that_are_still_open
        is_trapped = end_insp_alveoli_open & ~airways_would_be_open

        new_start_airway_open = airways_that_are_still_open
        new_start_alveoli_open_or_trapped = can_deflate | is_trapped
        
        eelv = self._calculate_volume_for_state(peep, new_start_airway_open, new_start_alveoli_open_or_trapped)
        tidal_volume = total_vol_at_peak - eelv
        total_compliance = tidal_volume / dp if dp > 0 else 0
        
        vol_at_peak_per_comp = np.sum(vol_at_peak_units * end_insp_alveoli_open, axis=1)
        vol_deflating_units = self._calculate_volume_liters(tp_exp)
        tp_at_closure = self.acps - self.sp
        vol_trapped_units = self._calculate_volume_liters(tp_at_closure)
        
        vol_at_peep_per_comp = (
            np.sum(vol_deflating_units * can_deflate, axis=1) + 
            np.sum(vol_trapped_units * is_trapped, axis=1)
        )
        
        vt_per_comp = vol_at_peak_per_comp - vol_at_peep_per_comp
        return tidal_volume, total_compliance, new_start_airway_open, new_start_alveoli_open_or_trapped, eelv, vt_per_comp

    def stabilize_lung_state(self, peep, dp, start_air=None, start_alv=None, num_breaths=15):
        s_air = np.zeros_like(self.aops, dtype=bool) if start_air is None else start_air.copy()
        s_alv = np.zeros_like(self.aops, dtype=bool) if start_alv is None else start_alv.copy()
        for _ in range(num_breaths):
            _, _, s_air, s_alv, _, _ = self.get_trial_metrics(peep, dp, s_air, s_alv)
        return s_air, s_alv

    def run_peep_trial(self, peep_levels, dp_or_vt, pip_max_recruitment=60):
        results = []
        tp_recruitment = pip_max_recruitment - self.sp
        current_airway_open = (tp_recruitment >= self.aops)
        current_alveoli_open = current_airway_open & (tp_recruitment >= self.tops)

        for peep in peep_levels:
            dp = dp_or_vt
            final_airway_open, final_alveoli_open = self.stabilize_lung_state(peep, dp, current_airway_open, current_alveoli_open, num_breaths=5)
            tidal_volume, final_comp, _, _, final_eelv, vt_per_comp = self.get_trial_metrics(peep, dp, final_airway_open, final_alveoli_open)
            _, _, current_airway_open, current_alveoli_open, _, _ = self.get_trial_metrics(peep, 0, final_airway_open, final_alveoli_open)

            results.append({
                "peep": peep, 
                "total_compliance": final_comp * 1000, 
                "comp_per_comp": (vt_per_comp / dp) * 1000 if dp > 0 else np.zeros(self.n_compartments), 
                "driving_pressure": dp, 
                "eelv_liters": final_eelv,
                "tidal_volume_liters": tidal_volume,
            })
        return results

def analyze_costa(peep_trial_results, key='comp_per_comp'):
    all_comps_per_comp = np.array([r.get(key) for r in peep_trial_results if r.get(key) is not None])
    if all_comps_per_comp.size == 0: return []
    
    best_comp_indices = np.argmax(all_comps_per_comp, axis=0)
    best_comps = np.max(all_comps_per_comp, axis=0)
    analysis = []
    
    for i, result in enumerate(peep_trial_results):
        current_comps = result.get(key)
        if current_comps is None: continue
        comp_diff = best_comps - current_comps
        overdistention_mask = i < best_comp_indices
        collapse_mask = i > best_comp_indices
        valid_best_comps = best_comps > 1e-9
        
        weighted_over = np.sum(np.where(overdistention_mask & valid_best_comps, comp_diff, 0))
        weighted_collapse = np.sum(np.where(collapse_mask & valid_best_comps, comp_diff, 0))
        
        total_best_comp_sum = np.sum(best_comps[valid_best_comps])
        total_overdistention = (weighted_over / total_best_comp_sum * 100) if total_best_comp_sum > 0 else 0
        total_collapse = (weighted_collapse / total_best_comp_sum * 100) if total_best_comp_sum > 0 else 0
        
        analysis.append({"peep": result["peep"], "overdistention": total_overdistention, "collapse": total_collapse})
        
    return analysis

def find_odcl_peep(peeps, costa_analysis):
    if not costa_analysis or len(costa_analysis) < 2: return np.nan
    collapses = np.array([a['collapse'] for a in costa_analysis])
    overdistentions = np.array([a['overdistention'] for a in costa_analysis])
    
    diff = collapses - overdistentions
    cross_indices = np.where(np.diff(np.sign(diff)))[0]
    
    if len(cross_indices) > 0:
        idx = cross_indices[0]
        if idx + 1 < len(peeps):
            x1, yc1, yh1 = peeps[idx], collapses[idx], overdistentions[idx]
            x2, yc2, yh2 = peeps[idx+1], collapses[idx+1], overdistentions[idx+1]
            denominator = (yc2 - yh2) - (yc1 - yh1)
            if abs(denominator) > 1e-6:
                costa_peep = (x1 * (yc2 - yh2) - x2 * (yc1 - yh1)) / denominator
                if min(x1, x2) <= costa_peep <= max(x1, x2): return costa_peep
    return np.nan

def run_single_targeted_trial_logic(params):
    AOP_MEAN = params['aop_mean']
    DP_OR_VT = 15.0
    N_COMPARTMENTS = 30
    PEEP_LEVELS_FULL = np.arange(24, 3, -2)

    v_max_L = params["v_max_ml"] / 1000.0
    
    lung_full = LungModel(
        n_compartments=N_COMPARTMENTS, 
        max_sp_g1=params["max_sp"], 
        aop_mean_g1=params['aop_mean'],
        aop_sd_g1=params['aop_sd'],
        acp_mean_g1=params['acp_mean'],
        acp_sd_g1=params['acp_sd'],
        top_mean_g1=params['top_mean'],
        top_sd_g1=params['top_sd'],
        tcp_mean_g1=params['tcp_mean'],
        tcp_sd_g1=params['tcp_sd'],
        tlc_L_g1=v_max_L,
        h_mean_g1=params['h_mean'],
        h_sd_g1=params['h_sd']
    )
    
    results_full = lung_full.run_peep_trial(PEEP_LEVELS_FULL, DP_OR_VT)
    peeps_full = [r['peep'] for r in results_full]
    
    costa_full_uncorrected = analyze_costa(results_full, key='comp_per_comp')
    odcl_full_uncorrected = find_odcl_peep(peeps_full, costa_full_uncorrected)
    
    for r in results_full:
        vt_per_comp = r['comp_per_comp'] * r['driving_pressure'] / 1000 if r['driving_pressure'] > 0 else np.zeros(N_COMPARTMENTS)
        vt_total = r['tidal_volume_liters']
        pip = r['peep'] + r['driving_pressure']
        peep = r['peep']
        dp_effective = pip - max(peep, AOP_MEAN)
        r['corrected_comp_per_comp'] = (vt_per_comp / dp_effective) * 1000 if dp_effective > 0 else np.zeros_like(vt_per_comp)
        
    costa_full_corrected = analyze_costa(results_full, key='corrected_comp_per_comp')
    odcl_full_corrected = find_odcl_peep(peeps_full, costa_full_corrected)

    return {'uncorrected': odcl_full_uncorrected, 'corrected': odcl_full_corrected}

def run_wrapper(params):
    try: 
        return run_single_targeted_trial_logic(params)
    except Exception:
        return {'uncorrected': np.nan, 'corrected': np.nan}

if __name__ == '__main__':
    BASE_PARAMS = { 
        "v_max_ml": 2500.0,
        "h_mean": 4.9,
        "h_sd": 0.1,
        "top_mean": 20.0,
        "top_sd": 4.0,
        "tcp_mean": 2.0,
        "tcp_sd": 1.0,
        "max_sp": 14.5,
        "acp_mean": 3.0, 
        "acp_sd": 1.0, 
        "aop_sd": 2.4,
        "aop_mean": 16.0 # Fixed high AOP to clearly see the correction effect
    }

    # Define ranges for each parameter to test
    SENSITIVITY_RANGES = {
        "v_max_ml": [1500.0, 2000.0, 3000.0, 4000.0],
        "h_mean": [2.0, 4.0, 6.0, 8.0],
        "h_sd": [0.5, 1.0, 2.0],
        "top_mean": [10.0, 15.0, 25.0, 30.0],
        "top_sd": [2.0, 6.0, 8.0],
        "tcp_mean": [0.0, 4.0, 6.0, 8.0],
        "tcp_sd": [0.5, 1.5, 2.0, 3.0],
        "max_sp": [5.0, 10.0, 20.0, 25.0],
        "acp_mean": [0.0, 6.0, 9.0],
        "acp_sd": [0.5, 2.0, 4.0],
        "aop_sd": [1.0, 4.0, 6.0]
    }

    N_RUNS = 20 # Runs per configuration to get stable mean
    results_summary = []

    print("Starting Comprehensive Sensitivity Analysis...")
    print("Base aop_mean is set to 16.0 cmH2O")
    
    # Also test base parameters
    tasks = [BASE_PARAMS for _ in range(N_RUNS)]
    with Pool(cpu_count()) as p:
        base_results = list(p.imap(run_wrapper, tasks))
    
    valid_base = [r for r in base_results if not np.isnan(r['uncorrected']) and not np.isnan(r['corrected'])]
    if valid_base:
        mu_uncorr = np.mean([r['uncorrected'] for r in valid_base])
        mu_corr = np.mean([r['corrected'] for r in valid_base])
        sd_uncorr = np.std([r['uncorrected'] for r in valid_base])
        sd_corr = np.std([r['corrected'] for r in valid_base])
        results_summary.append({
            'Parameter': 'Base',
            'Value': 'N/A',
            'Uncorrected_Mean': mu_uncorr,
            'Uncorrected_SD': sd_uncorr,
            'Corrected_Mean': mu_corr,
            'Corrected_SD': sd_corr,
            'Condition_Met': mu_corr < mu_uncorr
        })

    for param, values in SENSITIVITY_RANGES.items():
        print(f"Testing variation in {param}...")
        for val in values:
            test_params = BASE_PARAMS.copy()
            test_params[param] = val
            
            tasks = [test_params for _ in range(N_RUNS)]
            with Pool(cpu_count()) as p:
                res = list(p.imap(run_wrapper, tasks))
                
            valid_res = [r for r in res if not np.isnan(r['uncorrected']) and not np.isnan(r['corrected'])]
            if valid_res:
                mu_uncorr = np.mean([r['uncorrected'] for r in valid_res])
                mu_corr = np.mean([r['corrected'] for r in valid_res])
                sd_uncorr = np.std([r['uncorrected'] for r in valid_res])
                sd_corr = np.std([r['corrected'] for r in valid_res])
                results_summary.append({
                    'Parameter': param,
                    'Value': val,
                    'Uncorrected_Mean': mu_uncorr,
                    'Uncorrected_SD': sd_uncorr,
                    'Corrected_Mean': mu_corr,
                    'Corrected_SD': sd_corr,
                    'Condition_Met': mu_corr < mu_uncorr
                })

    # Save results to a CSV
    df = pd.DataFrame(results_summary)
    df.to_csv('comprehensive_sensitivity_results.csv', index=False)
    
    # Check if ALL cases met the condition
    total_cases = len(df)
    cases_met = df['Condition_Met'].sum()
    
    print("\n--- Summary Report ---")
    print(df.to_string())
    print("\n")
    if total_cases == cases_met:
        print("CONCLUSION: YES, the AOP-corrected ODCL PEEP is consistently SMALLER than the uncorrected across ALL tested parameter variations.")
    else:
        failed_cases = df[~df['Condition_Met']]
        print(f"CONCLUSION: NO, the condition failed in {total_cases - cases_met} cases.")
        print(failed_cases.to_string())
