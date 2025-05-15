from torch.utils.tensorboard import SummaryWriter
import os

def get_tensorboard_writer(log_dir="runs", tag="ensemble_run"):
    run_path = os.path.join(log_dir, tag)
    return SummaryWriter(log_dir=run_path)
