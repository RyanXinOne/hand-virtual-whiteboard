from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FingertipDetector
from dataset import Hagrid3IndexFingertipDataset
from utils import device


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


train_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='train'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
test_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='test'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)

model = FingertipDetector().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_loop():
    model.train()
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop():
    model.eval()
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n")


for e in range(EPOCHS):
    print(f"Epoch {e+1}\n-------------------------------")
    train_loop()
    test_loop()
