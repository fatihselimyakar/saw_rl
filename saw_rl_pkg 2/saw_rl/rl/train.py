"""
SAW Runway Sequencing — MaskablePPO Training
=============================================
Kullanım:
    python train.py --arr adsb2_rapor_saw_arrivals.csv \
                    --dep adsb2_rapor_saw_departures.csv \
                    --timesteps 500000

Gereksinimler:
    pip install gymnasium stable-baselines3 sb3-contrib tqdm rich
"""

import argparse
import os
import random
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from saw_rl.rl.runway_env import RunwayEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def run_fcfs_baseline(env: RunwayEnv) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        done = terminated or truncated
    summary = info.get('episode_summary', {})
    return {
        'total_reward'   : round(total_reward, 2),
        'total_delay_min': summary.get('total_delay_min', 0),
        'avg_delay_min'  : summary.get('avg_delay_min', 0),
        'violations'     : summary.get('violations', 0),
    }


class TrainingLogger(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_delays = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode_summary' in info:
                self.episode_delays.append(info['episode_summary'].get('total_delay_min', 0))
        if self.n_calls % self.log_interval == 0 and self.episode_delays:
            recent = self.episode_delays[-20:]
            print(f"  [Step {self.n_calls:>7d}] "
                  f"Ort gecikme: {np.mean(recent):.1f} dk | "
                  f"Min: {np.min(recent):.1f} dk")
        return True


def train(args):
    set_seed(args.seed)
    print("=" * 60)
    print("SAW Runway Sequencing — MaskablePPO")
    print("=" * 60)

    env = RunwayEnv(
        arrivals_csv=args.arr,
        departures_csv=args.dep,
        n_window=args.window,
        mps_k=args.mps_k,
    )

    print("[*] Environment kontrolü yapılıyor...")
    check_env(env, warn=True)
    print("    ✓ Environment geçerli.")

    print("\n[*] FCFS baseline hesaplanıyor...")
    fcfs = run_fcfs_baseline(env)
    print(f"    FCFS Toplam gecikme : {fcfs['total_delay_min']:.1f} dk")
    print(f"    FCFS Ort. gecikme   : {fcfs['avg_delay_min']:.2f} dk/uçak")
    print(f"    FCFS İhlaller       : {fcfs['violations']}")
    print(f"    FCFS Reward         : {fcfs['total_reward']:.1f}")

    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    print(f"\n[*] MaskablePPO modeli oluşturuluyor...")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=420,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=0,
        tensorboard_log=args.log_dir or None,
    )

    print(f"    Observation boyutu: {env.observation_space.shape}")
    print(f"    Action boyutu     : {env.action_space.n}")
    print(f"    Parametreler      : {sum(p.numel() for p in model.policy.parameters()):,}")

    print(f"\n[*] Eğitim başlıyor ({args.timesteps:,} timestep)...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=TrainingLogger(),
        progress_bar=False,
        reset_num_timesteps=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "saw_sequencer")
    model.save(save_path)
    print(f"\n[✓] Model kaydedildi: {save_path}.zip")

    print("\n[*] Eğitilmiş model değerlendiriliyor...")
    results = evaluate_model(model, args.arr, args.dep, args.window, args.mps_k)

    print(f"\n{'='*60}")
    print(f"  SONUÇLAR (10 episode ortalaması)")
    print(f"{'='*60}")
    print(f"  {'Metrik':<30} {'FCFS':>10} {'RL':>10} {'İyileşme':>10}")
    print(f"  {'-'*60}")
    rl_delay = results['avg_total_delay']
    rl_avg   = results['avg_delay_per_ac']
    imp_total = (fcfs['total_delay_min'] - rl_delay) / max(fcfs['total_delay_min'], 1) * 100
    imp_avg   = (fcfs['avg_delay_min'] - rl_avg) / max(fcfs['avg_delay_min'], 1) * 100
    print(f"  {'Toplam gecikme (dk)':<30} {fcfs['total_delay_min']:>10.1f} {rl_delay:>10.1f} {imp_total:>+9.1f}%")
    print(f"  {'Ort. gecikme/ucak (dk)':<30} {fcfs['avg_delay_min']:>10.2f} {rl_avg:>10.2f} {imp_avg:>+9.1f}%")
    print(f"  {'Ihlal sayisi':<30} {fcfs['violations']:>10d} {results['avg_violations']:>10.1f}")
    print(f"  {'Episode reward':<30} {fcfs['total_reward']:>10.1f} {results['avg_reward']:>10.1f}")


def evaluate_model(model, arr_csv, dep_csv, n_window, mps_k, n_episodes=10):
    env = RunwayEnv(arr_csv, dep_csv, n_window=n_window, mps_k=mps_k)
    env = ActionMasker(env, mask_fn)
    all_delays, all_rewards, all_violations = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False; ep_reward = 0.0
        while not done:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        summary = info.get('episode_summary', {})
        all_delays.append(summary.get('total_delay_min', 0))
        all_violations.append(summary.get('violations', 0))
        all_rewards.append(ep_reward)
    return {
        'n_episodes'      : n_episodes,
        'avg_total_delay' : round(np.mean(all_delays), 2),
        'avg_delay_per_ac': round(np.mean(all_delays) / max(env.unwrapped.n_total, 1), 3),
        'avg_violations'  : round(np.mean(all_violations), 2),
        'avg_reward'      : round(np.mean(all_rewards), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arr",       required=True)
    parser.add_argument("--dep",       required=True)
    parser.add_argument("--timesteps", type=int,   default=500_000)
    parser.add_argument("--window",    type=int,   default=10)
    parser.add_argument("--mps_k",     type=int,   default=3)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save_dir",  default="models")
    parser.add_argument("--log_dir",   default="logs")
    args = parser.parse_args()
    train(args)