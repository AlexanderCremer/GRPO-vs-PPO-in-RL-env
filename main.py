import grpo
import acrobot_ppo
from trash import acrobot_grpo

modules = {grpo: ["Acrobot-v1", "Cartpole-v1"]}

# Loop through each module and call its train function
for module in modules.keys():
    if module == grpo:
        for env_name in modules[module]:
            print(f"Running train() from {module.__name__} for environment {env_name}")
            for group_size in [2, 4, 10]:
                for episode in range(1,11):
                    print(f"Episode {episode}: Running train({group_size}) from {module.__name__}")
                    module.train(group_size, seed=episode, env=env_name)
    if module in [acrobot_ppo]:
        for episode in range(1,6):
            print(f"Episode {episode}: Running train() from {module.__name__}")
            module.train(seed=episode)
    else:
        for episode in range(1,11):
            print(f"Episode {episode}: Running train() from {module.__name__}")
            module.train(seed=episode)
