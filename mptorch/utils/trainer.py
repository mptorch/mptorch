import torch
import math
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# utility function to see if execution can be done on a CUDA-enabled GPU
def try_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"


# utility function to evaluate the accuracy
def evaluate_accuracy(net, iter, device=try_gpu()):
    net.eval()
    num_correct, num_total = 0.0, 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(iter):
            X, y = X.to(device), y.to(device)
            _, predicted = torch.max(net(X), dim=1)
            correct = (predicted == y).sum()
            num_correct += correct
            num_total += y.shape[0]
    return float(num_correct) / num_total


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def trainer(
    net,
    train_loader,
    test_loader,
    num_epochs,
    lr,
    batch_size,
    loss=nn.CrossEntropyLoss(reduction="mean"),
    optimizer=None,
    device=try_gpu(),
    scheduler=None,
    plot_file=None,
    init_scale=None,
    log_wandb=False,
):
    # 1. move the model to the appropriate device for training
    net.to(device)
    train_acc_list = []
    test_acc_list = []

    # 2. set up the optimizer and updater
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if device == "cpu" or init_scale is None:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
    # 3. the training loop
    for epoch in range(num_epochs):
        # use a tqdm progress bar to see how training progresses
        tq = tqdm(total=len(train_loader) * batch_size)
        if scheduler is not None:
            lr = scheduler.get_last_lr()[-1]
        tq.set_description(f"Epoch {epoch}, lr {get_lr(optimizer):.3f}")
        net.train()
        train_acc, train_loss, train_size = 0.0, 0.0, 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # compute the output
            y_hat = net(X)
            # compute loss and perform back-propagation
            l = loss(y_hat, y)

            # scale gradient and record loss
            if scaler is None:
                l.backward()
            else:
                scaler.scale(l).backward()

            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            tq.update(batch_size)
            with torch.no_grad():
                train_loss += l.data.item() * X.shape[0]
                _, pred = torch.max(y_hat, dim=1)
            train_acc += (pred == y).sum().item()
            train_size += X.shape[0]

            tq.set_postfix(
                train_acc="{:.5f}".format(train_acc / train_size),
                train_loss="{:.5f}".format(train_loss / train_size),
            )
        tq.close()

        test_acc = evaluate_accuracy(net, test_loader, device)
        if scheduler is not None:
            scheduler.step()
        train_acc_list.append(train_acc / train_size)
        test_acc_list.append(test_acc)

        if log_wandb:  # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss / train_size,
                "train_acc": train_acc / train_size,
                "test_acc": test_acc,
                "learning_rate": get_lr(optimizer)
            })

        if scaler is None:
            print(
                f"loss {train_loss / train_size:.3f}, train acc {train_acc / train_size:.3f}, test acc {test_acc:.3f}"
            )
        else:
            print(
                f"loss {train_loss / train_size:.3f}, train acc {train_acc / train_size:.3f}, test acc {test_acc:.3f}, scale {scaler.get_scale():.3E}"
            )

    if plot_file is not None:
        plt.plot(range(num_epochs), train_acc_list, label="train_acc")
        plt.plot(range(num_epochs), test_acc_list, label="test_acc")
        plt.legend()
        plt.savefig(plot_file)
