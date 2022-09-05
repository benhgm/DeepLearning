import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
STORAGE_PATH = "feedforwardnet_lr001.pth"
TRAINING_RECORD = "training_record_lr001.txt"

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input):
        flattened_data = self.flatten(input)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
    
        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update model weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

    return loss


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    with open(TRAINING_RECORD, 'w') as f:
        f.write(f'Training model: {STORAGE_PATH}\nLearning Rate: {LEARNING_RATE}\n')
    for i in range(epochs):
        with open(TRAINING_RECORD, 'a') as f:
            f.write(f"Epoch: {i + 1}\n")

        print(f"Epoch: {i + 1}")
        loss = train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------------------")

        with open(TRAINING_RECORD, 'a') as f:
            f.write(f"Loss: {loss}\n")
            f.write("-----------------------\n")
    
    print("Training is done.")


if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the training set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    
    # build model
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        params=feed_forward_net.parameters(),
        lr=LEARNING_RATE
    )
    
    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), STORAGE_PATH)
    print("Model trained and stored at feedforwardnet.pth")