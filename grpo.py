
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard as tb


@dataclass
class Args:
    num_groups: int = 8
    '''number of groups to generate'''
    kl_coef: float = 0.01
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
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    # best so far 2.5e-2
    learning_rate: float = 2.5e-2
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

    #def get_value(self, x):
        #return self.critic(x)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()#, self.critic(x)


def train(G):
    success = 0

    args = tyro.cli(Args)
    args.num_groups = G
    # args.batch_size = int(args.num_envs * args.num_steps)
    args.total_timesteps = args.total_timesteps * args.num_groups
    args.batch_size = int(args.num_groups * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"GRPO_G{args.num_groups}",
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

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_groups) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_groups) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_groups)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_groups)).to(device)
    dones = torch.zeros((args.num_steps, args.num_groups)).to(device)
    #values = torch.zeros((args.num_steps, args.num_groups)).to(device)



    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Environment setup
    # Environment setup
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
        finish_iteration = np.array([False] * args.num_groups)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            #more iterations, less learning rate
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        for step in range(0, args.num_steps):
            if finish_iteration.all():
                break

            global_step += args.num_groups
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs) #select action
                #values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            #taking action in the environment
            """next_obs gives you the state of taking the action
            do we only take one action: meaning the state for all the groups is the same?
            do we take n amounts of actions per group therefore having to create a loop for each group?
                but then how do the rewards get calculated? Maybe its just the last reward or discounted sum of rewards """
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            #print(terminations)
            #print(reward)
            next_done = np.logical_or(terminations, truncations)
            finish_iteration = np.logical_or(finish_iteration, next_done)
            rewards[step] = torch.tensor(reward).to(device).view(-1)    #rewards are changed from np array into tensor form
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device) #rewards are changed from np array into tensor form

            if "final_info" in infos:   #if the episode is done (multiple parallel episodes)
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        """take a step in the environment and log data. maybe take n steps instead of 1
        this will be ri in advantages"""

        next_obs = next_obs[0].repeat(args.num_groups, *[1] * (next_obs.ndim - 1))  # Sync next_obs across groups
        next_done = next_done[0].repeat(args.num_groups, *[1] * (next_done.ndim - 1))  # Sync next_done across groups

        # bootstrap value if not done
        with torch.no_grad():
            #next_value = agent.get_value(next_obs).reshape(1, -1)

            cumulative_rewards = torch.zeros(args.num_groups).to(device)
            active_mask = torch.ones(args.num_groups, dtype=torch.bool).to(device)  #multiply by 1 if not finished and by 0 if finished
            #lastgaelam = 0
            for t in range(args.num_steps):
                cumulative_rewards += rewards[t] * active_mask

                # If a reward is 0, stop accumulating for that group
                active_mask &= (rewards[t] != 0)
            #print(cumulative_rewards - torch.mean(cumulative_rewards))
            advantages = torch.nan_to_num((cumulative_rewards - torch.mean(cumulative_rewards))/ torch.std(cumulative_rewards), nan=0.0)
            #print(advantages)
        #print(cumulative_rewards)
        mean_reward[iteration-1] = torch.max(cumulative_rewards)
        if any(r >= 200 for r in cumulative_rewards):
            success += 1

        #print(advantages)
        #returns = advantages + values

        # flatten the batch but keeping the group structure
        b_obs = obs.reshape((args.num_groups, -1) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(args.num_groups, -1)
        b_actions = actions.reshape((args.num_groups, -1) + envs.single_action_space.shape)
        #b_advantages = advantages.reshape(args.num_groups, -1)
        #print(b_logprobs[4])
        #b_returns = returns.reshape(-1)
        #b_values = values.reshape(-1)

        # Optimizing the policy network
        #b_inds = np.arange(args.batch_size)
        clipfracs = []
        b_inds_per_group = [np.arange(args.batch_size // args.num_groups) for _ in range(args.num_groups)]
        #print(b_inds_per_group)

        for epoch in range(args.update_epochs):
            total_policy_loss = 0
            total_kl_penalty = 0
            total_entropy_bonus = 0

            for group in range(args.num_groups):
                #np.random.shuffle(b_inds_per_group[group])

                mb_inds = b_inds_per_group[group]  # Use all indices at once
                mb_obs = b_obs[group, mb_inds]
                mb_actions = b_actions[group, mb_inds]
                mb_old_logprobs = b_logprobs[group, mb_inds]
                mb_advantage = advantages[group]  # Use the single advantage per group
                # Get new policy probabilities
                _, new_logprobs, entropy = agent.get_action(mb_obs, mb_actions)
                log_ratio = new_logprobs - mb_old_logprobs
                #print(mb_old_logprobs)
                ratio = log_ratio.exp()

                # Policy loss with clipping
                pg_loss1 = -ratio * mb_advantage
                pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantage
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # KL Divergence penalty
                kl = (mb_old_logprobs - new_logprobs).mean()
                kl_penalty = args.kl_coef * kl

                # Entropy bonus
                #entropy_bonus = args.ent_coef * entropy.mean()

                # Accumulate losses across groups
                total_policy_loss += policy_loss
                total_kl_penalty += kl_penalty
                #total_entropy_bonus += entropy_bonus

            # Compute the mean loss over all groups
            #final_policy_loss = (total_policy_loss + total_kl_penalty - total_entropy_bonus) / args.num_groups
            final_policy_loss = (total_policy_loss + total_kl_penalty) / args.num_groups
            #print(final_policy_loss)

            # Backpropagation
            optimizer.zero_grad()
            final_policy_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/mean_reward": cumulative_rewards.mean().item(),
            "charts/max_reward": cumulative_rewards.max().item(),
            "losses/policy_loss": final_policy_loss.item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }, step=global_step)

        '''writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("reward/mean_reward", cumulative_rewards.mean().item(), global_step)
        writer.add_scalar("reward/max_reward", cumulative_rewards.max().item(), global_step)
        writer.add_scalar("losses/policy_loss", final_policy_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)'''
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    smoothed_mean_rewards = np.convolve(mean_reward, np.ones(10)/10, mode='valid')
    sns.lineplot(x=np.arange(len(smoothed_mean_rewards)), y=smoothed_mean_rewards, label="max Reward", color="red")

    plt.xlabel("Iteration")
    plt.ylabel("Max Reward")
    plt.title("Max Reward Over Iterations")
    plt.legend()
    #plt.savefig(f"plots/reward_over_iterations_GRPO_g{args.num_groups}_max.png")
    #plt.show()
    envs.close()
    writer.close()
    wandb.finish()
    return success

if __name__ == "__main__":

    a = train(5)
    print("Successes:", a)
