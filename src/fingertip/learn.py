import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fingertip.dataset import Hagrid3IndexFingertipDataset
from fingertip.model import load_model


EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PRETRAINED_WEIGHTS = ""
PRETRAINED_EPOCHS = 0
CHECKPOINT_DIR = "checkpoints/fingertip"
N_CPU = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
test_dataloader = DataLoader(Hagrid3IndexFingertipDataset(dataset='test'), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)

model = load_model(weights_path=PRETRAINED_WEIGHTS, device=DEVICE)
loss_fn = nn.MSELoss()
def optim(): return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train_loop():
    model.train()
    pbar = tqdm(train_dataloader, desc="Training")
    for X, y in pbar:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f'{loss.item():.6f}')


def test_loop():
    model.eval()
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader, desc="Testing"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>6f}")
    return test_loss


if __name__ == "__main__":
    optimizer = optim()
    for e in range(PRETRAINED_EPOCHS, EPOCHS):
        e += 1
        print(f"Epoch {e}/{EPOCHS}\n-------------------------------")
        train_loop()
        test_loss = test_loop()
        # save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fingertip_model_ckpt{e}_loss{test_loss:>6f}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to '{ckpt_path}'")
        print()
