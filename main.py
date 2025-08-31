import grpo
import acrobot_ppo
import ppo
import shimmy_catch_grpo
import shimmy_catch_ppo
import MinAtar_grpo
import MinAtar_ppo
from trash import acrobot_grpo

modules = {MinAtar_ppo: None}

# Loop through each module and call its train function
for module in modules.keys():
    if module == grpo:
        for env_name in modules[module]:
            print(f"Running train() from {module.__name__} for environment {env_name}")
            for group_size in [10]:
                for episode in range(1,11):
                    print(f"Episode {episode}: Running train({group_size}) from {module.__name__}")
                    module.train(group_size, seed=episode, env=env_name)
    if module in [acrobot_ppo, ppo, shimmy_catch_ppo, MinAtar_ppo]:
        for episode in range(1, 11):
            print(f"Episode {episode}: Running train() from {module.__name__}")
            module.train(seed=episode)
    if module == shimmy_catch_grpo or module == MinAtar_grpo:
        for group_size in [2, 4, 10]:
            for episode in range(1,11):
                print(f"Episode {episode}: Running train({group_size}) from {module.__name__}")
                module.train(group_size, seed=episode)
    else:
        print(f"ERROR - No matching module found for {module.__name__}")
