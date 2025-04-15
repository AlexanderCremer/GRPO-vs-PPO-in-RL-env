import grpo
import matplotlib.pyplot as plt
import seaborn as sns

# Setting Seaborn style
sns.set(style="whitegrid")  # You can also use 'darkgrid', 'ticks', etc.

all_successes = []
for i in range(1, 11):
    success = grpo.train(i)
    all_successes.append(success)
    print(i)

print("All successes:", all_successes)

# Create the plot
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 11), y=all_successes, marker="o", label="Successes", color="blue")

# Customizing the plot
plt.xlabel("Group Size", fontsize=12)
plt.ylabel("Number of Successes", fontsize=12)
plt.title("Successes Over Different Group Sizes", fontsize=14)
plt.legend(title="Successes", loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)

# Saving the plot as a high-quality PNG
plt.savefig(f"plots/successes_over_group_size.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
