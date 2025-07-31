import grpo
import acrobot_ppo
from trash import acrobot_grpo

modules = [acrobot_grpo]

# Loop through each module and call its train function
for module in modules:
    if module in [grpo, acrobot_grpo]:
        for group_size in [2, 4, 10]:
            for episode in range(1,11):
                print(f"Episode {episode}: Running train({group_size}) from {module.__name__}")
                module.train(group_size, seed=episode)
    if module in [acrobot_ppo]:
        for episode in range(1,6):
            print(f"Episode {episode}: Running train() from {module.__name__}")
            module.train(seed=episode)
    else:
        for episode in range(1,11):
            print(f"Episode {episode}: Running train() from {module.__name__}")
            module.train(seed=episode)
