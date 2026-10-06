import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


# =========================
# Configuration
# =========================

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# =========================
# Data Preparation
# =========================

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# AlexNet - 5 Convolutional Layers
# =========================

class AlexNet5Conv(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =========================
# AlexNet - 7 Convolutional Layers
# =========================

class AlexNet7Conv(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =========================
# Training and Evaluation
# =========================

def train_test_loop(model, model_name, writer):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(NUM_EPOCHS):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total_train += labels.size(0)

            correct_train += (
                predictions == labels
            ).sum().item()

        train_acc = correct_train / total_train

        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # Evaluation
        # -------------------------

        model.eval()

        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                test_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                total_test += labels.size(0)

                correct_test += (
                    predictions == labels
                ).sum().item()

        test_acc = correct_test / total_test

        avg_test_loss = test_loss / len(test_loader)

        # -------------------------
        # Store History
        # -------------------------

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)

        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(test_acc)

        # -------------------------
        # TensorBoard
        # -------------------------

        writer.add_scalar(
            f"{model_name}/Train_Loss",
            avg_train_loss,
            epoch + 1
        )

        writer.add_scalar(
            f"{model_name}/Train_Accuracy",
            train_acc,
            epoch + 1
        )

        writer.add_scalar(
            f"{model_name}/Test_Loss",
            avg_test_loss,
            epoch + 1
        )

        writer.add_scalar(
            f"{model_name}/Test_Accuracy",
            test_acc,
            epoch + 1
        )

        print(
            f"[{model_name}] "
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return history


# =========================
# Main
# =========================

if __name__ == "__main__":

    writer = SummaryWriter(
        log_dir="./runs/alexnet_experiment"
    )

    print("\n==== Training AlexNet-5Conv ====")

    net5 = AlexNet5Conv(num_classes=10)

    history_5conv = train_test_loop(
        net5,
        "AlexNet_5Conv",
        writer
    )

    print("\n==== Training AlexNet-7Conv ====")

    net7 = AlexNet7Conv(num_classes=10)

    history_7conv = train_test_loop(
        net7,
        "AlexNet_7Conv",
        writer
    )

    writer.close()

    print("\nTraining complete!")
    print("TensorBoard logs saved to ./runs/alexnet_experiment")

    print("\nFinal Results")
    print("-" * 60)

    print(
        f"AlexNet-5Conv | "
        f"Train Acc: {history_5conv['train_acc'][-1] * 100:.2f}% | "
        f"Test Acc: {history_5conv['test_acc'][-1] * 100:.2f}% | "
        f"Test Loss: {history_5conv['test_loss'][-1]:.4f}"
    )

    print(
        f"AlexNet-7Conv | "
        f"Train Acc: {history_7conv['train_acc'][-1] * 100:.2f}% | "
        f"Test Acc: {history_7conv['test_acc'][-1] * 100:.2f}% | "
        f"Test Loss: {history_7conv['test_loss'][-1]:.4f}"
    )