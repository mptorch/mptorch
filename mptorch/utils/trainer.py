import torch
import math
from torch import nn
from tqdm import tqdm

# utility function to see if execution can be done on a CUDA-enabled GPU
def try_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"


# function addapted from the torch.nn.init code
def get_fan_in_and_fan_out(tensor):
    dims = tensor.dim()
    num_ifmaps = tensor.size(1)
    num_ofmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_ifmaps * receptive_field_size
    fan_out = num_ofmaps * receptive_field_size
    return fan_in, fan_out


# function to perform the scaling of the learning rate using the
# strategy described in Sec. 2.5 of the BinaryConnect paper
def lr_scaling(net, lr, optimizer="Adam"):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            fan_in, fan_out = get_fan_in_and_fan_out(m.weight)
            lr_scale = math.sqrt(1.5 / float(fan_in + fan_out))
            if optimizer != "Adam":  # SGD or Momentum approaches
                lr_scale *= lr_scale
            param_list.append({"params": m.weight, "lr": lr / lr_scale})
            param_list.append({"params": m.bias})
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            param_list.append({"params": m.parameters()})
    return param_list


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


def init_weights(m, mode="Glorot"):
    if mode == "Glorot":
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    else:  # Kaiming
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


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
    loss_scaling=1,
    optimizer=None,
    device=try_gpu(),
    scheduler=None,
):
    # 1. initialize the weights
    net.apply(init_weights)

    # 2. move the model to the appropriate device for training
    net.to(device)

    # 3. set up the optimizer and updater
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 4. the training loop
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
            # compute the output
            y_hat = net(X)
            l = loss(y_hat, y)
            # do loss scaling (if specified)
            l = l * loss_scaling
            # compute gradient and record loss
            optimizer.zero_grad()
            l.backward()
            # restore FP32 parameter values for update
            # (in case updates are done in FP32)
            for p in list(net.parameters()):
                if hasattr(p, "org"):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(net.parameters()):
                if hasattr(p, "org"):
                    p.org.copy_(p.data)

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
        print(
            f"loss {train_loss / train_size:.3f}, train acc {train_acc / train_size:.3f}, test acc {test_acc:.3f}"
        )
