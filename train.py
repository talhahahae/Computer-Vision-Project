import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import MSELoss
import numpy as np
import os

def train_model(model, dataloader, device, epochs=20, lr=1e-3, log_writer=None, model_idx=0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Model {model_idx+1} | Epoch {epoch+1}"):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        if log_writer:
            log_writer.add_scalar(f"Loss/Model_{model_idx+1}", epoch_loss, epoch)
        print(f"Model {model_idx+1} | Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    return model
