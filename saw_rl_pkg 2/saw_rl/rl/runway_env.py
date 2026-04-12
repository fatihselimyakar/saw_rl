"""
SAW Runway Sequencing RL Environment (Final Versiyon)
======================================================
Gymnasium-compatible environment.

Observation : Rolling window N_WINDOW uçak × N_FEATURES özellik
Action      : Discrete(N_WINDOW) — hangi uçağı sıradaki seçiyoruz
Reward      : Hibrit Göreceli Ödül
              reward = (gap_fcfs - gap_model) * reward_alpha     [pist boşluğu, FCFS'e göre]
                     - (model_delay / MAX_DELAY) * reward_beta   [bu uçağa yüklenen gecikme]
              Varsayılan: alpha=10, beta=0 → saf gap reward (kanıtlanmış baseline)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

from saw_rl.constants import CAT_MAP, get_sep

N_WINDOW   = 10
N_FEATURES = 10
MPS_K      = 3

R_CPS_VIOLATION = -10.0
MAX_DELAY_MIN   =  90.0

# Clip sınırları (VecNormalize kullansak da aşırı sapan verileri törpülemek için)
# index 7 (fcfs_rank) üst sınırı runtime'da instance-level set edilir (bkz. ADR-003)
_LO = np.array([-600.0,  0.0,    0.0, 100.0, -4000.0, 0.0, 0.0,   0.0,   0.0,   0.0], dtype=np.float32)
_HI = np.array([7200.0, 200.0, 45000.0, 600.0,  4000.0, 5.0, 1.0, 500.0, 180.0, 160.0], dtype=np.float32)


class RunwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, arrivals_csv, departures_csv,
                 n_window=N_WINDOW, mps_k=MPS_K, render_mode=None,
                 reward_alpha=10.0, reward_beta=0.0):
        super().__init__()
        self.n_window     = n_window
        self.mps_k        = mps_k
        self.render_mode  = render_mode
        self.reward_alpha = reward_alpha
        self.reward_beta  = reward_beta

        self._load_data(arrivals_csv, departures_csv)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_window * N_FEATURES,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_window)
        # reset() __init__ içinde çağrılmaz; SB3 eğitim öncesi reset() çağırır

    def _load_data(self, arr_csv, dep_csv):
        arr = pd.read_csv(arr_csv, parse_dates=['target_time', 'last_seen'])
        dep = pd.read_csv(dep_csv, parse_dates=['target_time', 'last_seen'])
        arr['phase_enc'] = 0
        dep['phase_enc'] = 1

        for df in [arr, dep]:
            df['category']      = df['category'].fillna('A3').apply(
                                      lambda c: c if c in CAT_MAP else 'A3')
            df['cat_enc']       = df['category'].map(CAT_MAP).fillna(3).astype(int)
            df['dist_km']       = df['dist_km'].fillna(50.0)
            df['gs_avg']        = df['gs_avg'].fillna(250.0)
            df['baro_rate_med'] = df['baro_rate_med'].fillna(-1000.0)
            df['hdg_diff']      = df['hdg_diff'].fillna(180.0)
            df['alt_last']      = df['alt_last'].fillna(10000.0)

        self.df = pd.concat([arr, dep], ignore_index=True)
        self.df = self.df.dropna(subset=['target_time'])
        self.df = self.df.sort_values('target_time').reset_index(drop=True)
        self.df['fcfs_rank'] = np.arange(len(self.df))
        self.n_total = len(self.df)

        self._ts     = np.array([t.timestamp() for t in self.df["target_time"]], dtype=np.float64)
        self._dist   = self.df['dist_km'].values.astype(np.float32)
        self._alt    = self.df['alt_last'].values.astype(np.float32)
        self._gs     = self.df['gs_avg'].values.astype(np.float32)
        self._rate   = self.df['baro_rate_med'].values.astype(np.float32)
        self._cat    = self.df['cat_enc'].values.astype(np.float32)
        self._phase  = self.df['phase_enc'].values.astype(np.float32)
        self._fcfs   = self.df['fcfs_rank'].values.astype(np.float32)
        self._hdg    = self.df['hdg_diff'].values.astype(np.float32)
        self._catstr = self.df['category'].values

        # ADR-003: global _HI'yi mutasyona uğratmak yerine instance-level kopya al
        self._obs_hi = np.array(_HI)
        self._obs_hi[7] = float(self.n_total)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done_mask     = np.zeros(self.n_total, dtype=bool)
        self.scheduled     = []
        self.last_ts       = self._ts[0]
        self.last_cat      = 'A3'
        self.current_step  = 0
        self.total_delay   = 0.0
        self.violations    = 0
        self._win_cache    = None
        return self._get_obs(), {}

    def step(self, action: int):
        window = self._get_window()

        if action >= len(window):
            return self._get_obs(), R_CPS_VIOLATION, False, False, {'invalid': True}

        # 1. Modelin Seçimi
        idx = int(window[action])
        earliest_ts = self._ts[idx]
        sep_sec     = get_sep(self.last_cat, self._catstr[idx])
        sched_ts    = max(earliest_ts, self.last_ts + sep_sec)
        
        gap_model_min = (sched_ts - self.last_ts) / 60.0
        model_delay_min = max(0.0, (sched_ts - earliest_ts) / 60.0) 

        # 2. FCFS'in Seçimi (Referans)
        valid_indices = []
        sched_pos = len(self.scheduled)
        for w_idx in window:
            if abs(sched_pos - int(self._fcfs[w_idx])) <= self.mps_k:
                valid_indices.append(w_idx)
        
        if not valid_indices:
            valid_indices = list(window)
            
        fcfs_idx = valid_indices[np.argmin([self._fcfs[i] for i in valid_indices])]
        earliest_ts_fcfs = self._ts[fcfs_idx]
        sep_sec_fcfs     = get_sep(self.last_cat, self._catstr[fcfs_idx])
        sched_ts_fcfs    = max(earliest_ts_fcfs, self.last_ts + sep_sec_fcfs)
        
        gap_fcfs_min = (sched_ts_fcfs - self.last_ts) / 60.0

        # 3. ÖDÜL: Hibrit Göreceli Ödül
        #   gap bileşeni   — FCFS'e göre pist boşluğunu minimize et (dominant sinyal)
        #   delay bileşeni — bu uçağa yüklenen gecikmeyi penalize et (FCFS karşılaştırması yok)
        #                    normalize: model_delay_min / MAX_DELAY_MIN → [0, 1]
        reward = (gap_fcfs_min - gap_model_min) * self.reward_alpha \
               - (model_delay_min / MAX_DELAY_MIN) * self.reward_beta

        # CPS kısıtı kontrolü
        cps_viol = abs(len(self.scheduled) - int(self._fcfs[idx])) > self.mps_k
        if cps_viol:
            reward += R_CPS_VIOLATION
            self.violations += 1

        # 4. State Güncellemesi
        self.done_mask[idx] = True
        self.scheduled.append(idx)
        self.last_ts       = sched_ts
        self.last_cat      = self._catstr[idx]
        self.total_delay  += model_delay_min
        self.current_step += 1
        self._win_cache    = None

        terminated = bool(len(self.scheduled) == self.n_total)
        truncated  = bool(model_delay_min > MAX_DELAY_MIN)

        info = {'delay_min': round(model_delay_min, 2), 'cps_viol': cps_viol,
                'total_delay': round(self.total_delay, 2)}
        
        if terminated or truncated:
            info['episode_summary'] = {
                'total_delay_min': round(self.total_delay, 2),
                'n_scheduled'    : len(self.scheduled),
                'violations'     : self.violations,
                'avg_delay_min'  : round(self.total_delay / max(len(self.scheduled), 1), 2),
            }
            
        return self._get_obs(), reward, terminated, truncated, info

    def _get_window(self):
        if self._win_cache is not None:
            return self._win_cache
        remaining = np.where(~self.done_mask)[0]
        if len(remaining) == 0:
            self._win_cache = []
            return []
        
        order = np.argsort(self._ts[remaining], kind='stable') 
        self._win_cache = remaining[order[:self.n_window]]
        return self._win_cache

    def _get_obs(self) -> np.ndarray:
        window = self._get_window()
        obs    = np.zeros((self.n_window, N_FEATURES), dtype=np.float32)
        now_ts = self.last_ts

        for i, idx in enumerate(window):
            current_sep = get_sep(self.last_cat, self._catstr[idx])
            raw = np.array([
                self._ts[idx] - now_ts,
                self._dist[idx],
                self._alt[idx],
                self._gs[idx],
                self._rate[idx],
                self._cat[idx],
                self._phase[idx],
                self._fcfs[idx],
                self._hdg[idx],
                current_sep,
            ], dtype=np.float32)
            obs[i] = np.clip(2.0 * (raw - _LO) / (self._obs_hi - _LO + 1e-8) - 1.0, -1.0, 1.0)

        return np.nan_to_num(obs.flatten(), nan=0.0, posinf=1.0, neginf=-1.0)

    def action_masks(self) -> np.ndarray:
        window    = self._get_window()
        mask      = np.zeros(self.n_window, dtype=bool)
        sched_pos = len(self.scheduled)
        for i, idx in enumerate(window):
            if abs(sched_pos - int(self._fcfs[idx])) <= self.mps_k:
                mask[i] = True
        if not mask.any():
            mask[:len(window)] = True
        return mask

    def render(self):
        if self.render_mode != "human":
            return
        window = self._get_window()
        t = datetime.fromtimestamp(self.last_ts)
        print(f"\n── Adım {self.current_step} | {t.strftime('%H:%M:%S')} | "
              f"Kalan: {self.n_total - len(self.scheduled)} | "
              f"Gecikme: {self.total_delay:.1f} dk")
        masks = self.action_masks()
        for i, idx in enumerate(window):
            eta_t = datetime.fromtimestamp(self._ts[idx])
            print(f"  [{i}] {self._catstr[idx]} "
                  f"{'ARR' if self._phase[idx]==0 else 'DEP'} "
                  f"ETA:{eta_t.strftime('%H:%M:%S')} "
                  f"dist:{self._dist[idx]:.0f}km "
                  f"{'✓' if masks[i] else '✗'}")