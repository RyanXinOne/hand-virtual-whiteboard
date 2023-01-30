import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FingertipDetector
from dataset import Hagrid3IndexFingertipDataset
from utils import device


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
PRETRAINED_WEIGHTS = ""
PRETRAINED_EPOCHS = 0
CHECKPOINT_DIR = "checkpoints/fingertip"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='train'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
test_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='test'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)

model = FingertipDetector().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if PRETRAINED_WEIGHTS:
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
else:
    # initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


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
            print(f"loss: {loss:>6f}  [{current:>5d}/{size:>5d}]")


def test_loop():
    model.eval()
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>6f} \n")
    return test_loss


for e in range(PRETRAINED_EPOCHS, EPOCHS):
    e += 1
    print(f"Epoch {e}\n-------------------------------")
    train_loop()
    test_loss = test_loop()
    # save checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"fingertip_model_ckpt{e}_loss{test_loss:>6f}.pt"))
