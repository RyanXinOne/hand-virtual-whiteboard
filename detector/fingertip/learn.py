import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fingertip.dataset import Hagrid3IndexFingertipDataset
from fingertip.model import load_model


EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
PRETRAINED_WEIGHTS = ""
PRETRAINED_EPOCHS = 0
CHECKPOINT_DIR = "checkpoints/fingertip"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='train'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
test_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='test'), batch_size=BATCH_SIZE, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)

model = load_model(weights_path=PRETRAINED_WEIGHTS)
loss_fn = nn.MSELoss()
def optim(): return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train_loop():
    model.train()
    batch_num = len(train_dataloader)
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>6f} | {batch+1}/{batch_num}[{current}/{size}]")


def test_loop():
    model.eval()
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>6f}")
    return test_loss


optimizer = optim()
for e in range(PRETRAINED_EPOCHS, EPOCHS):
    e += 1
    # decrease learning rate
    if e in (6, 11, 16):
        LEARNING_RATE /= 5
        optimizer = optim()
        print(f'Decreasing learning rate to {LEARNING_RATE}')
    print(f"Epoch {e}/{EPOCHS}\n-------------------------------")
    train_loop()
    test_loss = test_loop()
    # save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fingertip_model_ckpt{e}_loss{test_loss:>6f}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to '{ckpt_path}'")
    print()
