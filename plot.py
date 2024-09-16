import os
import re
import matplotlib.pyplot as plt
from datetime import datetime


def parse_log(log_content):
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    for line in log_content.split('\n'):
        if 'Step' in line and 'Loss:' in line:
            match = re.search(r'Step (\d+), Loss: ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                train_steps.append(step)
                train_losses.append(loss)
        elif 'Validation loss:' in line:
            match = re.search(r'Validation loss: ([\d.]+)', line)
            if match:
                loss = float(match.group(1))
                val_steps.append(train_steps[-1])  # Use the last training step
                val_losses.append(loss)

    return train_steps, train_losses, val_steps, val_losses


def plot_losses(train_steps, train_losses, val_steps, val_losses, folder = None):
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', marker='o', linestyle='-', markersize=4)

    # Plot validation loss
    plt.plot(val_steps, val_losses, label='Validation Loss', color='red', marker='s', linestyle='-', markersize=6)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Set y-axis to start from 0
    plt.ylim(bottom=0)

    plt.tight_layout()
    fpath = 'loss_plot.png' if folder is None else os.path.join(folder, 'loss_plot.png')
    plt.savefig(fpath)
    print(f"Plot has been saved as '{fpath}'")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    # Your log content goes here
    cwd = os.path.dirname(os.path.abspath(__file__))

    filename = 'expirements/rope/training_20240916_091140.log'
    file_path = os.path.join(cwd, filename)
    file_folder = os.path.dirname(file_path)
    with open(file_path, encoding='utf-8') as fd:
        log_content = fd.read()

    train_steps, train_losses, val_steps, val_losses = parse_log(log_content)
    plot_losses(train_steps, train_losses, val_steps, val_losses, file_folder)

