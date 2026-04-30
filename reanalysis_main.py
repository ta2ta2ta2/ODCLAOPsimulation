import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import copy
import pingouin as pg

# ==============================================================================
# 肺モデルのコアロジック (V73シミュレーター版に完全置換)
# ==============================================================================
class LungModel:
    # === ▼ 修正 (V74) ▼ ===
    # V73シミュレーターのLungModel定義をそのまま移植
    # (FRC=0, ACP/TCPマイナス許容, 気道バグ修正済み)
    def __init__(self, n_compartments, 
                 # Group 1 (Right Lung) / or Single Model
                 max_sp_g1, aop_mean_g1, aop_sd_g1, acp_mean_g1, acp_sd_g1,
                 top_mean_g1, top_sd_g1, tcp_mean_g1, tcp_sd_g1,
                 tlc_L_g1, h_mean_g1, h_sd_g1,
                 # Group 2 (Left Lung) - Optional
                 max_sp_g2=None, aop_mean_g2=None, aop_sd_g2=None, acp_mean_g2=None, acp_sd_g2=None,
                 top_mean_g2=None, top_sd_g2=None, tcp_mean_g2=None, tcp_sd_g2=None,
                 tlc_L_g2=None, h_mean_g2=None, h_sd_g2=None):
        
        self.n_compartments = n_compartments
        self.n_alveoli_per_comp = 1000
        
        # FRCは 0 L (Hicklingモデル)
        self.frc_L = 0.0 
        
        is_bimodal = max_sp_g2 is not None

        if is_bimodal:
            # (解析コードではBimodalは使用しないが、互換性のため残す)
            n_comp_g1 = self.n_compartments // 2
            n_comp_g2 = self.n_compartments - n_comp_g1
            total_units_g1 = n_comp_g1 * self.n_alveoli_per_comp
            total_units_g2 = n_comp_g2 * self.n_alveoli_per_comp
        else:
            n_comp_g1 = self.n_compartments
            n_comp_g2 = 0
            total_units_g1 = n_comp_g1 * self.n_alveoli_per_comp
            total_units_g2 = 0

        # 1. Superimposed Pressure (SP)
        sp_g1 = np.linspace(0, max_sp_g1, n_comp_g1)
        if is_bimodal:
            sp_g2 = np.linspace(0, max_sp_g2, n_comp_g2)
            sp_flat_comp = np.concatenate((sp_g1, sp_g2))
        else:
            sp_flat_comp = sp_g1
        self.sp = sp_flat_comp[:, np.newaxis] 

        # 2. パラメータ配列 (AOP, ACP, TOP, TCP, h)
        def _generate_params_array(mean1, sd1, mean2, sd2):
            params_g1 = np.random.normal(mean1, sd1, (n_comp_g1, self.n_alveoli_per_comp))
            if is_bimodal:
                params_g2 = np.random.normal(mean2, sd2, (n_comp_g2, self.n_alveoli_per_comp))
                return np.concatenate((params_g1, params_g2), axis=0) 
            else:
                return params_g1 

        self.aops = _generate_params_array(aop_mean_g1, aop_sd_g1, aop_mean_g2, aop_sd_g2)
        self.acps = _generate_params_array(acp_mean_g1, acp_sd_g1, acp_mean_g2, acp_sd_g2)
        self.tops = _generate_params_array(top_mean_g1, top_sd_g1, top_mean_g2, top_sd_g2)
        self.tcps = _generate_params_array(tcp_mean_g1, tcp_sd_g1, tcp_mean_g2, tcp_sd_g2)
        self.h_units = _generate_params_array(h_mean_g1, h_sd_g1, h_mean_g2, h_sd_g2)
        
        # AOP/TOP (Opening pressures) must be >= 0. ACP/TCP (Closing pressures) can be negative.
        for p_array in [self.aops, self.tops]: p_array[p_array < 0] = 0
        
        # 3. TLC / V0 (FRCは 0 L)
        if is_bimodal:
            if tlc_L_g1 <= 0 or tlc_L_g2 <= 0:
                raise ValueError("TLCは0より大きい必要があります (G1 or G2)。")
            v0_total_L_g1 = tlc_L_g1
            v0_total_L_g2 = tlc_L_g2
            v0_unit_L_g1 = v0_total_L_g1 / total_units_g1 if total_units_g1 > 0 else 0
            v0_unit_L_g2 = v0_total_L_g2 / total_units_g2 if total_units_g2 > 0 else 0
            
            v0_units_g1_array = np.full((n_comp_g1, self.n_alveoli_per_comp), v0_unit_L_g1)
            v0_units_g2_array = np.full((n_comp_g2, self.n_alveoli_per_comp), v0_unit_L_g2)
            self.v0_unit_L_array = np.concatenate((v0_units_g1_array, v0_units_g2_array), axis=0)
        else:
            if tlc_L_g1 <= 0:
                raise ValueError("TLCは0より大きい必要があります。")
            v0_total_L_g1 = tlc_L_g1
            v0_unit_L_g1 = v0_total_L_g1 / total_units_g1
            self.v0_unit_L_array = np.full((n_comp_g1, self.n_alveoli_per_comp), v0_unit_L_g1)

        # 5. 制約適用
        self.aops = np.maximum(self.aops, self.acps)  # AOP >= ACP（気道は閉鎖圧以上で開く）
        self.tops = np.maximum(self.tops, self.tcps)  # TOP >= TCP（肺胞は虚脱圧以上で開く）
        # AOP >= TOP 制約は除去: AOP < TOP（気道が先に開き、肺胞は高圧で開く）を許容する
        self.h_units[self.h_units <= 0.1] = 0.1
    # === ▲ 修正 (V74) ▲ ===

    def _calculate_volume_liters(self, pressure):
        pressure_pos = np.maximum(0, pressure)
        # === ▼ 修正 (V74) ▼ ===
        # V73のロジックに統一
        exponent = - (pressure_pos * np.log(2)) / self.h_units
        volume = self.v0_unit_L_array * (1 - np.exp(exponent))
        # === ▲ 修正 (V74) ▲ ===
        return np.maximum(0, volume)

    def _calculate_volume_for_state(self, peep, airway_open, alveoli_open_or_trapped):
        tp_at_peep = peep - self.sp
        can_deflate = airway_open & alveoli_open_or_trapped
        is_trapped = ~airway_open & alveoli_open_or_trapped
        
        vol_deflating_units = self._calculate_volume_liters(tp_at_peep)
        tp_at_closure = self.acps - self.sp
        vol_trapped_units = self._calculate_volume_liters(tp_at_closure)
        
        recruited_vol = np.sum(vol_deflating_units * can_deflate) + np.sum(vol_trapped_units * is_trapped)
        total_vol = self.frc_L + recruited_vol # self.frc_L は 0.0
        return total_vol

    def get_trial_metrics(self, peep, dp, start_airway_open, start_alveoli_open_or_trapped):
        tp_insp = peep + dp - self.sp
        end_insp_airway_open = start_airway_open | (tp_insp >= self.aops)
        newly_ready_alveoli = (tp_insp >= self.tops)
        alveoli_ready_to_open = start_alveoli_open_or_trapped | newly_ready_alveoli
        end_insp_alveoli_open = alveoli_ready_to_open & end_insp_airway_open

        vol_at_peak_units = self._calculate_volume_liters(tp_insp)
        total_vol_at_peak = self.frc_L + np.sum(vol_at_peak_units * end_insp_alveoli_open) # self.frc_L は 0.0

        tp_exp = peep - self.sp
        airways_would_be_open = (tp_exp >= self.acps)
        alveoli_would_be_open = (tp_exp >= self.tcps)
        
        airways_that_are_still_open = end_insp_alveoli_open & airways_would_be_open
        alveoli_that_are_still_open = end_insp_alveoli_open & alveoli_would_be_open
        
        can_deflate = airways_that_are_still_open & alveoli_that_are_still_open
        is_trapped = end_insp_alveoli_open & ~airways_would_be_open

        # === ▼ 修正 (V74) ▼ ===
        # V72/V73で修正された「気道状態引き継ぎバグ」を修正
        # new_start_airway_open = can_deflate (旧コードのバグ)
        new_start_airway_open = airways_that_are_still_open
        # === ▲ 修正 (V74) ▲ ===
        
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

    def run_peep_trial(self, peep_levels, dp_or_vt, mode='pcv', pip_max_recruitment=60):
        results = []
        last_dp = 15.0 # VCVモード用のダミー (このスクリプトではPCVのみ使用)
        
        tp_recruitment = pip_max_recruitment - self.sp
        current_airway_open = (tp_recruitment >= self.aops)
        current_alveoli_open = current_airway_open & (tp_recruitment >= self.tops)

        for peep in peep_levels:
            if mode == 'vcv':
                target_vt_L = dp_or_vt
                dp_low, dp_high = 0.1, 50.0
                dp = last_dp

                for _ in range(10): # 10回の反復でDPを探索
                    s_air_stable, s_alv_stable = self.stabilize_lung_state(peep, dp, current_airway_open, current_alveoli_open, num_breaths=5)
                    vt_achieved, _, _, _, _, _ = self.get_trial_metrics(peep, dp, s_air_stable, s_alv_stable)

                    if vt_achieved < target_vt_L:
                        dp_low = dp
                    else:
                        dp_high = dp
                    
                    new_dp = (dp_low + dp_high) / 2.0
                    if abs(new_dp - dp) < 0.01:
                        dp = new_dp
                        break
                    dp = new_dp
                last_dp = dp
            else: # pcv mode
                dp = dp_or_vt
            
            # 1. このPEEPレベルでの安定状態を計算
            final_airway_open, final_alveoli_open = self.stabilize_lung_state(peep, dp, current_airway_open, current_alveoli_open, num_breaths=5)
            
            # 2. 指標を計算
            tidal_volume, final_comp, _, _, final_eelv, vt_per_comp = self.get_trial_metrics(peep, dp, final_airway_open, final_alveoli_open)
            
            # 3. PEEP=0への呼気状態を次のステップに引き継ぐ
            _, _, current_airway_open, current_alveoli_open, _, _ = self.get_trial_metrics(peep, 0, final_airway_open, final_alveoli_open) # This line is already correct, no change needed.

            results.append({
                "peep": peep, 
                "total_compliance": final_comp * 1000, 
                "comp_per_comp": (vt_per_comp / dp) * 1000 if dp > 0 else np.zeros(self.n_compartments), 
                "driving_pressure": dp, 
                "eelv_liters": final_eelv, # EELVも保存
                "tidal_volume_liters": tidal_volume
            })
        return results

# ==============================================================================
# 解析用コード (Hyperdistension -> Overdistention に修正)
# ==============================================================================
def analyze_costa(peep_trial_results, key='comp_per_comp'):
    all_comps_per_comp = np.array([r.get(key) for r in peep_trial_results if r.get(key) is not None]);
    if all_comps_per_comp.size == 0: return []
    
    best_comp_indices = np.argmax(all_comps_per_comp, axis=0); 
    best_comps = np.max(all_comps_per_comp, axis=0); 
    analysis = []
    
    for i, result in enumerate(peep_trial_results):
        current_comps = result.get(key);
        if current_comps is None: continue
        
        comp_diff = best_comps - current_comps; 
        overdistention_mask = i < best_comp_indices;
        collapse_mask = i > best_comp_indices; 
        valid_best_comps = best_comps > 1e-9
        
        weighted_over = np.sum(np.where(overdistention_mask & valid_best_comps, comp_diff, 0));
        weighted_collapse = np.sum(np.where(collapse_mask & valid_best_comps, comp_diff, 0))
        
        total_best_comp_sum = np.sum(best_comps[valid_best_comps]); 
        total_overdistention = (weighted_over / total_best_comp_sum * 100) if total_best_comp_sum > 0 else 0
        total_collapse = (weighted_collapse / total_best_comp_sum * 100) if total_best_comp_sum > 0 else 0
        
        analysis.append({"peep": result["peep"], "overdistention": total_overdistention, "collapse": total_collapse})
        
    return analysis

def find_odcl_peep(peeps, costa_analysis):
    if not costa_analysis or len(costa_analysis) < 2: return np.nan
    
    collapses = np.array([a['collapse'] for a in costa_analysis]); 
    overdistentions = np.array([a['overdistention'] for a in costa_analysis])
    
    diff = collapses - overdistentions;
    cross_indices = np.where(np.diff(np.sign(diff)))[0]
    
    if len(cross_indices) > 0:
        idx = cross_indices[0]
        if idx + 1 < len(peeps):
            x1, yc1, yh1 = peeps[idx], collapses[idx], overdistentions[idx];
            x2, yc2, yh2 = peeps[idx+1], collapses[idx+1], overdistentions[idx+1]
            
            denominator = (yc2 - yh2) - (yc1 - yh1)
            if abs(denominator) > 1e-6:
                costa_peep = (x1 * (yc2 - yh2) - x2 * (yc1 - yh1)) / denominator
                if min(x1, x2) <= costa_peep <= max(x1, x2): return costa_peep
    return np.nan

def apply_aop_correction(results, aop_mean, sp_array, n_compartments):
    """
    PEEP trialの結果にAOP補正を適用し、補正後のコンプライアンスを計算する。
    """
    for r in results:
        dp = r.get('driving_pressure', 0)
        if dp > 1e-9:
            peep = r['peep']
            pip = peep + dp
            # 元のcomp_per_compからvt_per_compを逆算
            vt_per_comp = (np.array(r['comp_per_comp']) / 1000.0) * dp
            
            # 区画ごとの実効PEEPと実効駆動圧を計算
            effective_peep_per_comp = np.maximum(peep, aop_mean + sp_array)
            dp_effective_per_comp = pip - effective_peep_per_comp
            
            # 補正後のコンプライアンスを計算
            corrected_comp_per_comp = np.zeros_like(vt_per_comp)
            mask = dp_effective_per_comp > 1e-9
            corrected_comp_per_comp[mask] = (vt_per_comp[mask] / dp_effective_per_comp[mask]) * 1000.0
            r['corrected_comp_per_comp'] = corrected_comp_per_comp
        else:
            r['corrected_comp_per_comp'] = np.zeros(n_compartments)
    return results

# ==============================================================================
# シミュレーション実行ロジック
# ==============================================================================
def run_single_targeted_trial_logic(params):
    AOP_MEAN = params['aop_mean']; DP_SETTING = 15.0; N_COMPARTMENTS = 30
    PEEP_LEVELS_FULL = np.arange(24, 3, -2) # PEEP 24, 22, ..., 4

    v_max_L = params["v_max_ml"] / 1000.0
    
    # === ▼ 修正 (V74) ▼ ===
    # V73の __init__ に合わせて引数を修正
    # (v_max_L -> tlc_L_g1, max_sp -> max_sp_g1 など)
    def _create_lung(params_dict, v_max_liters):
        return LungModel(
            n_compartments=N_COMPARTMENTS, 
            max_sp_g1=params_dict["max_sp"], 
            aop_mean_g1=params_dict['aop_mean'],
            aop_sd_g1=params_dict['aop_sd'],
            acp_mean_g1=params_dict['acp_mean'],
            acp_sd_g1=params_dict['acp_sd'],
            top_mean_g1=params_dict['top_mean'],
            top_sd_g1=params_dict['top_sd'],
            tcp_mean_g1=params_dict['tcp_mean'],
            tcp_sd_g1=params_dict['tcp_sd'],
            tlc_L_g1=v_max_liters, # v_max_L を tlc_L_g1 にマッピング
            h_mean_g1=params_dict['h_mean'],
            h_sd_g1=params_dict['h_sd']
        )
    
    lung_full = _create_lung(params, v_max_L)
    # === ▲ 修正 (V74) ▲ ===

    results_full = lung_full.run_peep_trial(PEEP_LEVELS_FULL, DP_SETTING, mode='pcv'); peeps_full = [r['peep'] for r in results_full]
    sp_array = lung_full.sp.flatten()
    
    # 1. Uncorrected (従来法)
    costa_full_uncorrected = analyze_costa(results_full, key='comp_per_comp')
    odcl_full_uncorrected = find_odcl_peep(peeps_full, costa_full_uncorrected)
    
    # 2. AOP補正法
    results_aop_corrected = apply_aop_correction(copy.deepcopy(results_full), AOP_MEAN, sp_array, N_COMPARTMENTS)
    costa_full_corrected = analyze_costa(results_aop_corrected, key='corrected_comp_per_comp')
    odcl_full_corrected = find_odcl_peep(peeps_full, costa_full_corrected)

    return {'full_uncorrected': odcl_full_uncorrected, 'full_corrected': odcl_full_corrected}

def run_and_analyze_single_iteration_wrapper(params):
    try: return run_single_targeted_trial_logic(params)
    except Exception:
        traceback.print_exc()
        return {k: np.nan for k in ['full_uncorrected', 'full_corrected']}

def create_unified_crossover_plot(params):
    print(f"\n--- AOP={params['aop_mean']}cmH2Oの統合クロスオーバープロット ---")
    AOP_MEAN = params['aop_mean']; DP_SETTING = 15.0; N_COMPARTMENTS = 30; PEEP_LEVELS_FULL = np.arange(24, 3, -2)
    
    v_max_L = params["v_max_ml"] / 1000.0

    # === ▼ 修正 (V74) ▼ ===
    # V73の __init__ に合わせて引数を修正
    def _create_lung(params_dict, v_max_liters):
        return LungModel(
            n_compartments=N_COMPARTMENTS, 
            max_sp_g1=params_dict["max_sp"], 
            aop_mean_g1=params_dict['aop_mean'],
            aop_sd_g1=params_dict['aop_sd'],
            acp_mean_g1=params_dict['acp_mean'],
            acp_sd_g1=params_dict['acp_sd'],
            top_mean_g1=params_dict['top_mean'],
            top_sd_g1=params_dict['top_sd'],
            tcp_mean_g1=params_dict['tcp_mean'],
            tcp_sd_g1=params_dict['tcp_sd'],
            tlc_L_g1=v_max_liters, # v_max_L を tlc_L_g1 にマッピング
            h_mean_g1=params_dict['h_mean'],
            h_sd_g1=params_dict['h_sd']
        )
    
    lung_full = _create_lung(params, v_max_L)
    # === ▲ 修正 (V74) ▲ ===

    results_full = lung_full.run_peep_trial(PEEP_LEVELS_FULL, DP_SETTING, mode='pcv'); peeps_full = [r['peep'] for r in results_full]
    sp_array = lung_full.sp.flatten()
    costa_full_uncorrected = analyze_costa(results_full, key='comp_per_comp'); odcl_full_uncorrected = find_odcl_peep(peeps_full, costa_full_uncorrected)
    
    # AOP補正（apply_aop_correction により 'corrected_comp_per_comp' キーが各結果に追加される）
    # 補正式: dp_eff = PIP − max(PEEP, AOP_mean + SP)  ← 論文記載と一致
    results_aop_corrected = apply_aop_correction(copy.deepcopy(results_full), AOP_MEAN, sp_array, N_COMPARTMENTS)
    costa_full_corrected = analyze_costa(results_aop_corrected, key='corrected_comp_per_comp'); odcl_full_corrected = find_odcl_peep(peeps_full, costa_full_corrected)
    
    print("\n--- ベンチマーク（C_best）のシフト分析 ---")
    all_uncorrected_comps = np.array([r['comp_per_comp'] for r in results_full])
    all_corrected_comps = np.array([r['corrected_comp_per_comp'] for r in results_aop_corrected])
    peeps_full_np = np.array(peeps_full)

    best_peep_indices_uncorr = np.argmax(all_uncorrected_comps, axis=0)
    best_peep_per_comp_uncorr = peeps_full_np[best_peep_indices_uncorr]
    
    best_peep_indices_corr = np.argmax(all_corrected_comps, axis=0)
    best_peep_per_comp_corr = peeps_full_np[best_peep_indices_corr]

    count_uncorr = np.sum(best_peep_per_comp_uncorr < AOP_MEAN)
    count_corr = np.sum(best_peep_per_comp_corr < AOP_MEAN)

    print(f"  AOP以下のPEEPで最大コンプライアンスを示す区画の数:")
    print(f"    - 従来法            : {count_uncorr} / {N_COMPARTMENTS} 区画")
    print(f"    - AOP補正法         : {count_corr} / {N_COMPARTMENTS} 区画")

    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    ax1.set_xlabel('PEEP (cmH$_2$O)', fontsize=12)
    ax1.set_ylabel('Overdistention / Collapse (%)', fontsize=12)
    
    ax1.plot(peeps_full, [a['collapse'] for a in costa_full_uncorrected], 'v-', color='brown', alpha=0.7, label='Collapse (Uncorrected method)')
    ax1.plot(peeps_full, [a['overdistention'] for a in costa_full_uncorrected], 'o-', color='darkblue', alpha=0.7, label='Overdistention (Uncorrected method)')
    ax1.plot(peeps_full, [a['collapse'] for a in costa_full_corrected], 'v--', color='sandybrown', alpha=0.9, label='Collapse (AOP-Corrected method)')
    ax1.plot(peeps_full, [a['overdistention'] for a in costa_full_corrected], 'o--', color='cornflowerblue', alpha=0.9, label='Overdistention (AOP-Corrected)')
    
    ax1.axvline(x=odcl_full_uncorrected, color='black', linestyle='-', linewidth=3, alpha=0.9, label=f'ODCL (Uncorrected) = {odcl_full_uncorrected:.1f}')
    ax1.axvline(x=odcl_full_corrected, color='red', linestyle='--', linewidth=3, alpha=0.9, label=f'ODCL (AOP-Corrected) = {odcl_full_corrected:.1f}')
    
    ax1.tick_params(axis='y')
    ax1.invert_xaxis()
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    
    ax1.legend(loc='best')
    
    fig.tight_layout()
    plt.savefig(f'Representative_AOP_{AOP_MEAN}.pdf', format='pdf')
    plt.show()

# --- メイン実行部 ---
if __name__ == '__main__':
    try:
        import seaborn
        from tqdm import tqdm
        import pandas
        import pingouin as pg
    except ImportError:
        print("必要なライブラリが不足しています。ターミナルで以下を実行してください:")
        print("pip install numpy scipy matplotlib seaborn tqdm pandas pingouin")
        exit()
        
    # ==============================================================================
    # 基本パラメータ設定
    # ==============================================================================
    BASE_PARAMS = { 
        "v_max_ml": 2500.0,    # 論文の V_o = 2.5L
        "h_mean": 4.9,         # 論文の h = 4.9
        "h_sd": 0.1,           # 論文は固定値のためSDは小さく設定
        
        "top_mean": 20.0,      # 論文の 0-40 の中間
        "top_sd": 4.0,         # 論文の [SD 4]
        "tcp_mean": 2.0,       # 論文の 0-4 の中間
        "tcp_sd": 1.0,         # 論文の [SD 1]
        "max_sp": 14.5,        # 論文の 0-14.5
        
        # --- AOP/ACP (ユーザー維持希望) パラメータ (元の値を流用) ---
        "acp_mean": 3.0, 
        "acp_sd": 1.0, 
        "aop_sd": 4.0 
    }

    print("\n" + "="*70 + "\n## AOP感度分析：異なるAOPレベルでODCL PEEP計算法を比較 ##\n" + "="*70)
    
    AOP_LEVELS_TO_TEST = np.arange(4, 17, 2) # 4から16まで2刻みでテスト
    N_RUNS = 50
    
    all_results_list = []
    subject_id_counter = 0
    
    for aop_val in AOP_LEVELS_TO_TEST:
        print(f"\n--- AOP = {aop_val} cmH2O のシミュレーションを開始 ---")
        current_params = BASE_PARAMS.copy()
        current_params['aop_mean'] = aop_val
        
        tasks = [current_params for _ in range(N_RUNS)]
        with Pool(cpu_count()) as p:
            results_for_aop = list(tqdm(p.imap(run_and_analyze_single_iteration_wrapper, tasks), total=N_RUNS))
        
        df_aop = pd.DataFrame(results_for_aop).dropna().reset_index(drop=True)
        df_aop['Subject'] = range(subject_id_counter, subject_id_counter + len(df_aop))
        subject_id_counter += len(df_aop)
        df_long_aop = df_aop.melt(id_vars=['Subject'], var_name='Method', value_name='ODCL_PEEP')
        df_long_aop['AOP_Level'] = aop_val
        all_results_list.append(df_long_aop)
        
    final_df = pd.concat(all_results_list, ignore_index=True)

    print("\n" + "="*70 + "\n## 感度分析 結果プロット ##\n" + "="*70)
    
    final_df['Method'] = final_df['Method'].replace({
        'full_uncorrected': 'Uncorrected method', 
        'full_corrected': 'AOP-Corrected method'
    })
    
    print("\n" + "="*70 + "\n## 感度分析 統計サマリー (Mean ± SD) ##\n" + "="*70)
    
    stats_summary = final_df.groupby(['AOP_Level', 'Method'])['ODCL_PEEP'].agg(['mean', 'std']).reset_index()
    
    for aop_level in AOP_LEVELS_TO_TEST:
        print(f"\n--- AOP = {aop_level} cmH2O ---")
        level_data = stats_summary[stats_summary['AOP_Level'] == aop_level]
        if level_data.empty:
            print("  データがありません。")
            continue
            
        for _, row in level_data.iterrows():
            print(f"  {row['Method']:<25}: {row['mean']:.2f} ± {row['std']:.2f} cmH2O")

    print("\n" + "="*70 + "\n## 各AOPレベルの代表ケース詳細解析 ##\n" + "="*70)
    representative_params = BASE_PARAMS.copy()
    representative_params['aop_mean'] = 12.0
    create_unified_crossover_plot(representative_params)

    print("\n" + "="*70 + "\n## 統計的詳細分析 (ANOVA & 多重比較) ##\n" + "="*70)
    
    # AOP_Levelをカテゴリ変数として扱う
    df_for_anova = final_df.copy()
    df_for_anova['AOP_Level'] = df_for_anova['AOP_Level'].astype('category')

    print("\n--- 1. 混合計画分散分析 (Mixed ANOVA) ---\n")
    print("ODCL PEEPに対する AOPレベル(between) と 計算手法(within) の影響を評価します。")
    
    # 混合計画分散分析の実行
    aov = pg.mixed_anova(data=df_for_anova, dv='ODCL_PEEP', between='AOP_Level', within='Method', subject='Subject')
    print(aov)

    print("\n[ANOVA結果の解釈]")
    p_interaction = aov.loc[aov['Source'] == 'Interaction', 'p-unc'].iloc[0]

    if p_interaction < 0.05:
        print("-> 交互作用 (AOP_Level と Method) が有意です (p < 0.05)。")
        print("   これは、AOPレベルによって、計算手法がODCL PEEPに与える影響が異なることを示唆します。")
        print("   例えば、AOPレベルが高くなるにつれて、手法間の差が大きくなる、などの現象が統計的に裏付けられます。")
    else:
        print("-> 交互作用は有意ではありません (p >= 0.05)。")

    print("\n--- 2. 多重比較 (Post-hoc test) ---\n")
    print("AOPレベルごとに、2つの計算手法間に有意な差があるかを対応のあるt検定で調べ、p値をBonferroni法で補正します。")
    
    # AOPレベルごとにグループ化してペアワイズ検定を実行する、より安定した方法に変更
    posthocs_list = []
    for aop_level, group_df in df_for_anova.groupby('AOP_Level', observed=True):
        # 各AOPレベル内でペアワイズt検定を実行
        # 'pairwise_ttests'は非推奨のため、'pairwise_tests'を使用する
        # 'pairwise_tests'は対応のある検定を自動で判断し、補正後のp値列名が'p-bonf'等になる
        ph = pg.pairwise_tests(data=group_df, dv='ODCL_PEEP', within='Method', subject='Subject', padjust='bonf')
        ph['AOP_Level'] = aop_level # 結果にAOPレベルを追加
        posthocs_list.append(ph)
    
    # 全ての結果を結合
    posthocs = pd.concat(posthocs_list, ignore_index=True)

    # 結果の表示（このシミュレーションでは全ての比較でp<0.05となるはずです）
    print("Post-hoc test results (selected columns):")
    print(posthocs[['AOP_Level', 'A', 'B', 'T', 'p-unc', 'hedges']])
    
    significant_pairs = posthocs[posthocs['p-unc'] < 0.05]
    if len(significant_pairs) == len(posthocs):
        print("\n[検定結果の解釈]: 全てのAOPレベルで、2つの計算手法間に統計的に有意な差が検出されました (p < 0.05)。")

    print("\n" + "="*70 + "\n## 感度分析 結果プロット (統計情報付き) ##\n" + "="*70)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.lineplot(
        data=final_df,
        x='AOP_Level',
        y='ODCL_PEEP',
        hue='Method',
        style='Method', 
        markers=True,   
        dashes=False,   
        palette='viridis',
        linewidth=2.5,
        markersize=10,
        errorbar='sd',
        ax=ax
    )
    
    # 有意差を示すアスタリスクを追加
    if not significant_pairs.empty:
        significant_aops = significant_pairs['AOP_Level'].unique()
        # 各AOPレベルでのエラーバーの上端を取得
        summary_for_plot = final_df.groupby(['AOP_Level', 'Method'])['ODCL_PEEP'].agg(['mean', 'std']).reset_index()
        summary_for_plot['error_top'] = summary_for_plot['mean'] + summary_for_plot['std']
        
        # アスタリスクが収まるようにY軸上限を計算するための変数
        max_y_limit = ax.get_ylim()[1]

        for aop in significant_aops:
            # アスタリスクをプロットするy座標を決定 (2つの手法のエラーバーの上端のうち、より高い方)
            y_pos_data = summary_for_plot[summary_for_plot['AOP_Level'] == aop]
            if not y_pos_data.empty:
                y_pos = y_pos_data['error_top'].max()
                ax.text(aop, y_pos + 0.5, '*', ha='center', va='bottom', color='red', fontsize=20, fontweight='bold')
                # アスタリスクの高さ分（+1.5程度）余裕を持たせる
                if (y_pos + 1.5) > max_y_limit:
                    max_y_limit = y_pos + 1.5
        
        ax.set_ylim(top=max_y_limit)

        # 右上に注釈を追加 (Bonferroni補正後) - 凡例の下に配置
        ax.text(0.98, 0.80, '*: p < 0.05', transform=ax.transAxes, fontsize=12, ha='right', va='top')

    ax.set_xlabel(r'Set $AOP_{regional}$ Mean (cmH$_2$O)', fontsize=14)
    ax.set_ylabel('Calculated ODCL PEEP (cmH$_2$O)', fontsize=14)
    ax.legend(title='Calculation Method', fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.5)
    fig.tight_layout()
    plt.savefig('Sensitivity_Analysis.pdf', format='pdf')
    plt.show()
