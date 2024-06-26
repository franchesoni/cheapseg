import numpy as np
import json
import matplotlib.pyplot as plt

def main(scalars_path, out_path='scalars.png', log=False):
    with open(scalars_path, 'r') as f:
        lines = f.readlines()

    has_mIoU = ['mIoU' in l for l in lines]
    train_lines = [json.loads(lines[i]) for i in range(len(lines)) if not has_mIoU[i]]
    val_lines = [json.loads(lines[i]) for i in range(len(lines)) if has_mIoU[i]]

    train_steps = [l['step'] for l in train_lines]
    val_steps = [l['step'] for l in val_lines]
    train_losses = [l['loss'] for l in train_lines]
    val_mious = [l['mIoU'] for l in val_lines]
    train_lrs = [l['lr'] for l in train_lines]

    plt.figure()
    fig, ax1 = plt.subplots()

    if log:
        train_losses = np.log(train_losses)
    ax1.plot(train_steps, train_losses, 'b-', label='train loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Train Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Customize the primary y-axis to show original train losses values
    if log:
        original_ticks = np.linspace(min(train_losses), max(train_losses), num=10)
        original_labels = np.round(np.exp(original_ticks), 2)
        ax1.set_yticks(original_ticks)
        ax1.set_yticklabels(original_labels)
        ax1.set_ylabel('Train Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(val_steps, val_mious, 'r-', label='val mIoU')
    ax2.set_ylabel('Val mIoU', color='r')
    ax2.tick_params('y', colors='r')

    ax3 = ax1.twinx()
    ax3.plot(train_steps, train_lrs, 'g-', label='learning rate')
    ax3.set_ylabel('', color='g')
    ax3.set_yticks([])

    fig.legend()
    plt.savefig(out_path)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)