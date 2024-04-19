import torch
from torch import nn


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return torch.sum(cmp)


def evaluate_accuracy(net, data_iter, rnn=False):
    net.eval()  # 设置为评估模式
    # 正确预测的数量，总预测的数量
    device = torch.device("cuda:0")
    metric = torch.zeros(2, device=device)
    with torch.no_grad():
        for X, y in data_iter:
            if rnn:
                X = X.permute(1, 0, 2)
                net.reset_state()
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


def cal_sparsity(net):
    cnt, numel = 0, 0
    for module in net.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            cnt += torch.sum(module.weight == 0)
            numel += module.weight.numel()

    return cnt / numel


def train_net_val(net, loss, trainer, data_iter, epochs, path, scheduler=None):
    train_iter, val_iter, test_iter = data_iter

    val_acc_best = 0
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数
        if scheduler:
            scheduler.step()

        val_acc = evaluate_accuracy(net, val_iter)
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(net.state_dict(), path)

        print("epoch: %d    val_acc: %.2f%%" % (epoch + 1, val_acc * 100))

    if val_acc < val_acc_best:
        net.load_state_dict(torch.load(path))

    return cal_sparsity(net), 1 - evaluate_accuracy(net, test_iter)


def train_net(net, loss, trainer, data_iter, epochs, path, scheduler=None):
    train_iter, test_iter = data_iter

    test_acc_best = 0
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            trainer.zero_grad()  # 清除了优化器中的grad
            l.backward()  # 通过进行反向传播来计算梯度
            trainer.step()  # 通过调用优化器来更新模型参数
        if scheduler:
            scheduler.step()

        test_acc = evaluate_accuracy(net, test_iter)
        if test_acc > test_acc_best:
            test_acc_best = test_acc
            torch.save(net.state_dict(), path)

        print("epoch: %d    test_acc: %.2f%%" % (epoch + 1, test_acc * 100))

    return cal_sparsity(net), test_acc_best
