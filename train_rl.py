import os
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.env_ssim import KspaceEnv
from rl.reconstructor import build_reconstructor, ReconWrapper
from rl.fastmri_loader import SingleCoilKneeDataset
from stable_baselines3.common.utils import get_linear_fn
from cli_args import parse_train_rl_args


def mask_fn(env):
    return env.action_masks()

def make_env(ds, recon, budget, device, acs: int, start_with_acs: bool):
    def _thunk():
        env = KspaceEnv(ds, recon, budget=budget, device=device, acs=acs, start_with_acs=start_with_acs)
        # <<< wrap each single env here, not the VecEnv >>>
        env = ActionMasker(env, mask_fn)
        return env
    return _thunk

if __name__ == "__main__":
    args = parse_train_rl_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SingleCoilKneeDataset(args.train_list)

    # Reconstructor (frozen)
    recon_core = build_reconstructor("unet_large", base=64, use_se=True, p_drop=0.0).to(device)
    if args.recon and os.path.isfile(args.recon):
        ckpt = torch.load(args.recon, map_location="cpu")
        recon_core.load_state_dict(ckpt["state_dict"])
        print("Loaded recon ckpt:", args.recon)
    recon = ReconWrapper(recon_core)

    n_envs = 4
    env = DummyVecEnv([
        make_env(train_ds, recon, args.budget, device, args.acs, bool(args.start_with_acs))
        for _ in range(n_envs)
    ])
    env = VecMonitor(env)   # monitor for vector envs


    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=512,               # 512 * 8 envs = 4096 rollout
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=args.timesteps)

    model.save("checkpoints/ppo_maskable")
    print("Saved RL model to checkpoints/ppo_maskable.zip")
