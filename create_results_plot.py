import matplotlib.pyplot as plt
from pathlib import Path

# Create images folder if it doesn't exist
Path("images").mkdir(exist_ok=True)

epochs = [1, 2, 3]

# Actual results from the final training run
train_accuracy_5 = [72.40, 86.44, 88.16]
train_accuracy_7 = [63.61, 85.03, 87.00]

train_loss_5 = [0.7269, 0.3832, 0.3356]
train_loss_7 = [0.9180, 0.4221, 0.3642]

test_loss_5 = [0.4206, 0.3511, 0.3257]
test_loss_7 = [0.4653, 0.3894, 0.3762]

test_accuracy_5 = [85.03, 87.21, 88.48]
test_accuracy_7 = [83.36, 86.00, 86.17]


# --------------------------------------------------
# 1. Final Test Accuracy Comparison
# --------------------------------------------------

models = ["AlexNet-5Conv", "AlexNet-7Conv"]
final_accuracy = [88.48, 86.17]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, final_accuracy)

plt.ylabel("Test Accuracy (%)")
plt.title("FashionMNIST Test Accuracy Comparison")
plt.ylim(0, 100)

for bar, value in zip(bars, final_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "images/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 2. Training Accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accuracy_5, marker="o", label="AlexNet-5Conv")
plt.plot(epochs, train_accuracy_7, marker="o", label="AlexNet-7Conv")

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy Comparison")
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "images/training_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 3. Training Loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss_5, marker="o", label="AlexNet-5Conv")
plt.plot(epochs, train_loss_7, marker="o", label="AlexNet-7Conv")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "images/training_loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 4. Test Loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(epochs, test_loss_5, marker="o", label="AlexNet-5Conv")
plt.plot(epochs, test_loss_7, marker="o", label="AlexNet-7Conv")

plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison")
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "images/test_loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("All graphs generated successfully!")
print("Saved to: images/")