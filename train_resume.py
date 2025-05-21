import torch
import os
from tqdm import tqdm

def train_model(model, train_loader, device, epochs, model_idx, log_writer=None):
    checkpoint_path = f"checkpoints/model_{model_idx}.pth"
    state_path = f"checkpoints/state_{model_idx}.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    start_epoch = 0

    # Resume from checkpoint if available
    if os.path.exists(state_path):
        print(f"🔁 Resuming training for model_{model_idx}...")
        checkpoint = torch.load(state_path)
        model.load_state_dict(torch.load(checkpoint_path))
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f" Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Model {model_idx+1} | Epoch {epoch+1}"):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Model {model_idx}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        if log_writer:
            log_writer.add_scalar(f"Train/Loss_Model_{model_idx}", avg_loss, epoch)

        # Save model and state after each epoch
        torch.save(model.state_dict(), checkpoint_path)
        torch.save({
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict()
        }, state_path)

    return model
