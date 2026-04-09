"""
SAW Runway Sequencing — MaskablePPO Training (Final Versiyon)
=============================================================
- VecNormalize devrede (norm_reward KAPANMIŞ durumda)
- Sabit yüksek öğrenme oranı (3e-4) ve entropi (0.05) devrede
- EvalCallback ile eğitim sırasında "En İyi Model" otomatik yakalanır.
"""

import argparse
import os
import random
import numpy as np
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    print("SAW Runway Sequencing — MaskablePPO (Final)")
    print("=" * 60)

    # 1. Eğitim Ortamı
    base_env = RunwayEnv(
        arrivals_csv=args.arr,
        departures_csv=args.dep,
        n_window=args.window,
        mps_k=args.mps_k,
    )

    print("[*] Environment kontrolü yapılıyor...")
    check_env(base_env, warn=True)
    print("    ✓ Environment geçerli.")

    print("\n[*] FCFS baseline hesaplanıyor...")
    fcfs = run_fcfs_baseline(base_env)
    print(f"    FCFS Toplam gecikme : {fcfs['total_delay_min']:.1f} dk")

    env = ActionMasker(base_env, mask_fn)
    env = Monitor(env)
    
    # CRITICAL: norm_reward=False. Ödül normalize edilmez!
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 2. Test/Değerlendirme (Evaluation) Ortamı
    eval_env = RunwayEnv(args.arr, args.dep, n_window=args.window, mps_k=args.mps_k)
    eval_env = ActionMasker(eval_env, mask_fn)
    eval_env = Monitor(eval_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    # Eval ortamı da eğitimdeki VecNormalize istatistiklerini kullanmalı (training=False)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # En iyi modeli "models/best_model" klasörüne kaydedecek callback
    eval_callback = EvalCallback(
        eval_vec_env, 
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=args.log_dir, 
        eval_freq=20000,          # Her 20k adımda bir test et
        deterministic=True,       
        render=False
    )

    print(f"\n[*] MaskablePPO modeli oluşturuluyor...")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,       # Sabit LR
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,            # Yüksek Entropi (Modele Keşfetmesini Söyler)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        seed=args.seed,
        verbose=0,
        tensorboard_log=args.log_dir or None,
    )

    print(f"\n[*] Eğitim başlıyor ({args.timesteps:,} timestep)...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[TrainingLogger(), eval_callback],  # Her iki Callback de devrede
        progress_bar=False,
        reset_num_timesteps=True,
    )

    # Eğitimin son halini kaydet
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "saw_sequencer_final")
    model.save(save_path)
    # Normalizasyon değerlerini kaydet (Değerlendirme sırasında bu dosyaya ihtiyacın olacak)
    vec_env.save(os.path.join(args.save_dir, "vec_normalize.pkl"))
    print(f"\n[✓] Final Model ve Normalizasyon istatistikleri kaydedildi.")
    print(f"[✓] En iyi model '{os.path.join(args.save_dir, 'best_model')}' klasöründe!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arr",       required=True)
    parser.add_argument("--dep",       required=True)
    parser.add_argument("--timesteps", type=int,   default=2_000_000) # Timestep 2 milyona çıkarıldı
    parser.add_argument("--window",    type=int,   default=10)
    parser.add_argument("--mps_k",     type=int,   default=3)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save_dir",  default="models")
    parser.add_argument("--log_dir",   default="logs")
    args = parser.parse_args()
    train(args)