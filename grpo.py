
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

@dataclass
class Args:
    num_groups: int = 5
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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
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
            name=run_name,
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



    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            #more iterations, less learning rate
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
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
            next_done = np.logical_or(terminations, truncations)
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
            advantages = torch.zeros_like(rewards).to(device)
            #lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    #nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    #nextvalues = values[t + 1]
                #delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                #print(rewards[t])
                #print(torch.mean(rewards[t]))
                #print(torch.std(rewards[t]))
                advantages[t] = torch.nan_to_num((rewards[t] - torch.mean(rewards[t])) / torch.std(rewards[t]), nan=0.0)     #note that I am calculating advantages for each group

            #print(advantages)
            #returns = advantages + values

        # flatten the batch but keeping the group structure
        b_obs = obs.reshape((args.num_groups, -1) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(args.num_groups, -1)
        b_actions = actions.reshape((args.num_groups, -1) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(args.num_groups, -1)

        #b_returns = returns.reshape(-1)
        #b_values = values.reshape(-1)

        # Optimizing the policy and value network
        #b_inds = np.arange(args.batch_size)
        clipfracs = []
        b_inds_per_group = [np.arange(args.batch_size // args.num_groups) for _ in range(args.num_groups)]
        total_loss = 0
        for group_idx in range(args.num_groups):
            np.random.shuffle(b_inds_per_group[group_idx])  # Shuffle per group
            for start in range(0, args.batch_size // args.num_groups, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds_per_group[group_idx][start:end]  # Sample from this group only

                _, newlogprob, entropy = agent.get_action(
                    b_obs[group_idx, mb_inds],
                    b_actions[group_idx, mb_inds].long()
                )
                logratio = newlogprob - b_logprobs[group_idx, mb_inds]
                #print(logratio)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

                #why are we normalizing the advantages?
                mb_advantages = b_advantages[group_idx, mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                #print(ratio)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                #print("pg_loss1", pg_loss1)
                #print("pg_loss2", pg_loss2)
                total_loss += torch.max(pg_loss1, pg_loss2).mean()



                #entropy_loss = entropy.mean()
                #loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        loss = total_loss/args.num_groups
        '''optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y'''
        # Instead of stopping training when KL > target_kl, we apply group-wise KL penalty
        kl_penalties = []

        for group_idx in range(args.num_groups):
            group_kl = ((b_logprobs[group_idx] - newlogprob[group_idx]).mean()).item()
            kl_penalties.append(group_kl)

        # Compute KL penalty for all groups
        mean_kl_penalty = np.mean(kl_penalties)

        # Add KL regularization to loss
        #loss += args.kl_coef * mean_kl_penalty  # args.kl_coef is a new hyperparameter
        print("loss", loss)
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        #writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", total_loss.item(), global_step)
        #writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        #writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
