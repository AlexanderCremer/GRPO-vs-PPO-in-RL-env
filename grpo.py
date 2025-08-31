import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from numpy import cos, sin, pi
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
# Add this import at the top of your script
import copy
import matplotlib.pyplot as plt
#import seaborn as sns
import tensorboard as tb


@dataclass
class Args:
    num_groups: int = 8
    '''number of groups to generate'''
    #best so far 0.2 (for G=10 at least)
    kl_coef: float = 0.2
    '''coefficient of the kl divergence penalty'''


    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "GRPO_gymnasium"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    # best so far 1e-2
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 500
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 6
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


def train(G, seed=1, env="CartPole-v1"):
    success = 0

    args = tyro.cli(Args)
    args.num_groups = G
    args.env_id = env

    # Optional: shorter run for Acrobot
    if env == "Acrobot-v1":
        args.total_timesteps = 200_000

    # These no longer depend on "iterations"
    args.batch_size = int(args.num_groups * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # args.num_iterations is unused with a while-loop; keep if you want, but it's not needed.

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.seed = seed

    if args.track:
        import wandb
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"{env}GRPO_G{args.num_groups}_{args.env_id}",
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Vector envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.capture_video, run_name) for _ in range(args.num_groups)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Singleton env
    singleton_env = gym.make(args.env_id)
    singleton_obs, _ = singleton_env.reset(seed=args.seed)

    global_step = 0
    start_time = time.time()

    # Initial obs for all groups
    initial_obs = []
    for i in range(args.num_groups):
        obs_i, _ = envs.envs[i].reset(seed=args.seed)
        initial_obs.append(obs_i)
    next_obs = torch.from_numpy(np.array(initial_obs, dtype=np.float32)).to(device)
    next_done = torch.zeros(args.num_groups, device=device)

    # If you want to keep a record of max return per rollout:
    max_return_per_rollout = []

    while global_step < args.total_timesteps:
        # ----- Anneal LR by progress (no iteration needed) -----
        if args.anneal_lr:
            progress = min(1.0, float(global_step) / max(1, args.total_timesteps))
            lrnow = (1.0 - progress) * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ----- Rollout storage -----
        obs = torch.zeros((args.num_steps, args.num_groups) + envs.single_observation_space.shape, device=device)
        actions = torch.zeros((args.num_steps, args.num_groups) + envs.single_action_space.shape, dtype=torch.long, device=device)
        logprobs = torch.zeros((args.num_steps, args.num_groups), device=device)
        logits = torch.zeros((args.num_steps, args.num_groups), dtype=torch.long, device=device)
        rewards = torch.zeros((args.num_steps, args.num_groups), dtype=torch.float32, device=device)
        dones = torch.zeros((args.num_steps, args.num_groups), device=device)
        group_steps = torch.ones(args.num_groups, dtype=torch.int32, device=device)

        finish_iteration = np.zeros(args.num_groups, dtype=bool)

        cumulative_rewards = torch.zeros(args.num_groups, device=device)
        starting_state = next_obs[0]

        # ----- Collect a rollout -----
        for step in range(args.num_steps):
            if finish_iteration.all():
                break

            # Only count active envs toward steps
            global_step += int((~finish_iteration).sum())

            obs[step, ~finish_iteration] = next_obs[~finish_iteration]
            dones[step, ~finish_iteration] = next_done[~finish_iteration]

            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)

            actions[step, ~finish_iteration] = action[~finish_iteration]
            logprobs[step, ~finish_iteration] = logprob[~finish_iteration].detach()
            logits[step, ~finish_iteration] = action[~finish_iteration]  # here "logits" is just the chosen action

            # Step envs on CPU, keep as numpy until batch convert
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # Minimal-copy conversions
            reward_tensor = torch.from_numpy(np.asarray(reward, dtype=np.float32)).to(device)
            rewards[step, ~finish_iteration] = reward_tensor[~finish_iteration]

            mask_active = torch.from_numpy((~finish_iteration).astype(np.float32)).to(device)
            cumulative_rewards += reward_tensor * mask_active

            next_done_np = np.logical_or(terminations, truncations)
            finish_iteration = np.logical_or(finish_iteration, next_done_np)

            next_obs = torch.from_numpy(next_obs_np).float().to(device)
            next_done = torch.from_numpy(next_done_np.astype(np.float32)).to(device)

            group_steps[~finish_iteration] += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ----- Compute per-group advantages (normalized) -----
        with torch.no_grad():
            std = torch.std(cumulative_rewards)
            if std == 0:
                advantages = torch.zeros_like(cumulative_rewards)
            else:
                advantages = (cumulative_rewards - torch.mean(cumulative_rewards)) / std

        # Track max return per rollout (no indexing by iteration)
        max_return_per_rollout.append(float(cumulative_rewards.max().item()))
        if bool((cumulative_rewards >= 200).any()):
            success += 1

        # ----- Flatten by group (keeping valid steps per group) -----
        b_obs, b_actions, b_logprobs = [], [], []
        for g in range(args.num_groups):
            valid = int(group_steps[g].item())
            b_obs.append(obs[:valid, g])
            b_actions.append(actions[:valid, g])
            b_logprobs.append(logprobs[:valid, g])

        # (If you keep indices, make sure they're ints, not tensors.)
        b_inds_per_group = [np.arange(int(group_steps[g].item())) for g in range(args.num_groups)]

        # ----- Policy update -----
        for _ in range(args.update_epochs):
            total_policy_loss = 0.0
            total_kl_penalty = 0.0

            for g in range(args.num_groups):
                mb_obs = b_obs[g]
                mb_actions = b_actions[g]
                with torch.no_grad():
                    mb_old_logprobs = b_logprobs[g]

                mb_advantage = advantages[g]

                _, new_logprobs, _ = agent.get_action(mb_obs, mb_actions)
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = log_ratio.exp()

                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantage
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Simple KL penalty from log-probs (avoid probs underflow)
                with torch.no_grad():
                    # reverse ratio to approximate KL(new||old) or KL(old||new); you had a custom term:
                    old_probs = mb_old_logprobs.exp()
                    new_probs = new_logprobs.exp()
                    prob_ratio = old_probs / new_probs
                    kl_elements = prob_ratio - torch.log(prob_ratio) - 1.0
                    kl_penalty = kl_elements.mean()

                total_policy_loss += policy_loss
                total_kl_penalty += kl_penalty

            final_policy_loss = (total_policy_loss + args.kl_coef * total_kl_penalty) / args.num_groups

            optimizer.zero_grad()
            final_policy_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # ----- Sync groups to a common state from singleton env -----
        with torch.no_grad():
            a_sync, _, _ = agent.get_action(starting_state.unsqueeze(0))
        # ensure int for single env
        obs_sync, _, s_term, s_trunc, _ = singleton_env.step(int(a_sync.item()))
        state_to_copy = copy.deepcopy(singleton_env.unwrapped.state)
        #print(state_to_copy)
        if s_term or s_trunc:
            singleton_env.reset()

        envs.reset()
        obs_list = []
        for i in range(args.num_groups):
            envs.envs[i].unwrapped.state = copy.deepcopy(state_to_copy)
            obs_i, _ = envs.envs[i].reset()
            obs_list.append(obs_i)
        next_obs = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)

        # ----- Evaluation (greedy) -----
        agent.eval()
        eval_env = gym.make(args.env_id)
        eval_rewards = []
        for _ in range(10):
            eo, _ = eval_env.reset()
            eo = torch.tensor(eo, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            total = 0.0
            while not done:
                with torch.no_grad():
                    act, _, _ = agent.get_action(eo)
                no, r, t, tr, _ = eval_env.step(int(act.item()))
                eo = torch.tensor(no, dtype=torch.float32, device=device).unsqueeze(0)
                total += float(r)
                done = t or tr
            eval_rewards.append(total)
        eval_env.close()
        agent.train()

        eval_mean_reward = float(np.mean(eval_rewards))

        # ----- Logging (by global_step, not iteration) -----
        writer.add_scalar("charts/kl_divergence", float(total_kl_penalty), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("reward/mean_reward_vs_steps", eval_mean_reward, global_step)
        writer.add_scalar("reward/mean_reward_vs_time", eval_mean_reward, time.time() - start_time)
        writer.add_scalar("losses/total_loss", float(final_policy_loss), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    return success


if __name__ == "__main__":
    #for i in [2,4]:
    a = train(10, seed=1, env="Acrobot-v1")
    print("Successes:", a)
