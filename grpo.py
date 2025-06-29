import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
# Add this import at the top of your script
import copy


@dataclass
class Args:
    num_groups: int = 8
    '''number of groups to generate'''
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
    wandb_project_name: str = "GRPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    # best so far 1e-2
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 200
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


def train(G, seed=1):
    success = 0

    args = tyro.cli(Args)
    args.num_groups = G
    args.batch_size = int(args.num_groups * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.num_steps
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    args.seed = seed
    if args.track:
        import wandb

        if wandb.run is not None:
            wandb.finish()

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,     #
            config=vars(args),
            name=f"GRPO_G{args.num_groups}_{args.env_id}",
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.capture_video, run_name) for i in range(args.num_groups)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    initial_obs, _ = envs.envs[0].reset(seed=args.seed)  # Get the initial state from the first environment

    # Sync the state across all environments by manually setting the state in each environment.
    for i in range(1, args.num_groups):
        # Manually reset the environment to match the first environment's initial state
        envs.envs[i].reset(seed=args.seed)

    # Replicate the initial state across all groups
    next_obs = np.stack([initial_obs] * args.num_groups)  # Replicate the initial state across all groups
    next_obs = torch.Tensor(next_obs).to(device)  # Convert to tensor for use
    next_done = torch.zeros(args.num_groups).to(device)  # Initialize done flags for all groups

    mean_reward = np.zeros(args.num_iterations)
    for iteration in range(1, args.num_iterations + 1):
        # Environment setup
        obs = torch.zeros((args.num_steps, args.num_groups) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_groups) + envs.single_action_space.shape, dtype=torch.long).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_groups)).to(device)
        logits = torch.zeros((args.num_steps, args.num_groups), dtype=torch.long).to(device)
        rewards = torch.zeros((args.num_steps, args.num_groups), dtype=torch.float64).to(device)
        dones = torch.zeros((args.num_steps, args.num_groups)).to(device)
        group_steps = torch.ones(args.num_groups, dtype=torch.int32).to(device)

        finish_iteration = np.array([False] * args.num_groups)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            #more iterations, less learning rate
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        cumulative_rewards = torch.zeros(args.num_groups).to(device)
        for step in range(0, args.num_steps):
            if finish_iteration.all():
                break
            global_step += np.count_nonzero(finish_iteration == 0)

            obs[step][~finish_iteration] = next_obs[~finish_iteration]
            dones[step][~finish_iteration] = next_done[~finish_iteration]

            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)

            actions[step][~finish_iteration] = action[~finish_iteration]
            logprobs[step][~finish_iteration] = logprob[~finish_iteration].detach()
            logits[step][~finish_iteration] = action[~finish_iteration]  # if logits really = action

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            reward_tensor = torch.tensor(reward).to(device).view(-1)
            rewards[step][~finish_iteration] = reward_tensor[~finish_iteration]

            mask = torch.tensor(~finish_iteration).to(device)
            cumulative_rewards += reward_tensor * mask.float()

            next_done_np = np.logical_or(terminations, truncations)
            finish_iteration = np.logical_or(finish_iteration, next_done_np)

            next_obs = torch.tensor(next_obs_np).to(device)
            next_done = torch.tensor(next_done_np).float().to(device)

            group_steps[~finish_iteration] += 1

            if "final_info" in infos:   #if the episode is done (multiple parallel episodes)
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        next_obs = next_obs[0].repeat(args.num_groups, *[1] * (next_obs.ndim - 1))  # Sync next_obs across groups
        next_done = next_done[0].repeat(args.num_groups, *[1] * (next_done.ndim - 1))  # Sync next_done across groups

        # bootstrap value if not done
        with torch.no_grad():
            advantages = torch.nan_to_num((cumulative_rewards - torch.mean(cumulative_rewards))/ torch.std(cumulative_rewards), nan=0.0)
        mean_reward[iteration-1] = torch.max(cumulative_rewards)
        if any(r >= 200 for r in cumulative_rewards):
            success += 1

        # flatten the batch but keeping the group structure
        b_obs = []
        b_actions = []
        b_logprobs = []
        for group in range(args.num_groups):
            valid_steps = group_steps[group]  # how many valid steps for this group
            b_obs.append(obs[:valid_steps, group])            # take only valid steps
            b_actions.append(actions[:valid_steps, group])
            b_logprobs.append(logprobs[:valid_steps, group])

        for epoch in range(args.update_epochs):
            total_policy_loss = 0
            total_kl_penalty = 0

            for group in range(args.num_groups):
                mb_obs = b_obs[group]
                mb_actions = b_actions[group]
                with torch.no_grad():
                    mb_old_logprobs = b_logprobs[group]
                mb_advantage = advantages[group]  # Use the single advantage per group
                _, new_logprobs, entropy = agent.get_action(mb_obs, mb_actions)
                log_ratio = new_logprobs - mb_old_logprobs
                ratio = log_ratio.exp()

                # Policy loss with clipping
                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantage
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # KL Divergence penalty
                new_probs = new_logprobs.exp()
                old_probs = mb_old_logprobs.exp()
                prob_ratio = old_probs / new_probs
                with torch.no_grad():
                    kl_elements = prob_ratio - torch.log(prob_ratio) - 1
                #print(kl_elements)
                kl_penalty = kl_elements.mean()

                # Entropy bonus

                # Accumulate losses across groups
                total_policy_loss += policy_loss
                total_kl_penalty += kl_penalty

            # Compute the mean loss over all groups
            final_policy_loss = (total_policy_loss + args.kl_coef * total_kl_penalty) / args.num_groups

            # Backpropagation
            optimizer.zero_grad()
            final_policy_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()




        eval_env = gym.make(args.env_id)

        # Evaluate using greedy actors
        eval_rewards = []

        # Deepcopy the agent so the evaluation doesn't interfere with training
        eval_agent = copy.deepcopy(agent)
        eval_agent.eval()

        for _ in range(10):  # Run 10 greedy evaluations
            eval_obs, _ = eval_env.reset()
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dim
            eval_done = False
            eval_total_reward = 0

            while not eval_done:
                with torch.no_grad():
                    eval_logits = eval_agent.actor(eval_obs)
                    eval_action = torch.argmax(eval_logits, dim=-1).item()

                eval_next_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                eval_obs = torch.tensor(eval_next_obs, dtype=torch.float32).to(device).unsqueeze(0)
                eval_total_reward += eval_reward
                eval_done = eval_terminated or eval_truncated

            eval_rewards.append(eval_total_reward)
        eval_mean_reward = np.average(eval_rewards)
        writer.add_scalar("evaluation/mean_greedy_reward", eval_mean_reward, iteration)

        # Optional: Close the eval environment after use
        eval_env.close()


        writer.add_scalar("charts/kl_divergence", total_kl_penalty.item(), global_step)

        # record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("reward/mean_reward", cumulative_rewards.mean().item(), global_step)
        writer.add_scalar("reward/max_reward", cumulative_rewards.max().item(), global_step)
        writer.add_scalar("losses/total_loss", final_policy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    return success

if __name__ == "__main__":
    #for i in [2,4]:
    a = train(4)
    print("Successes:", a)
