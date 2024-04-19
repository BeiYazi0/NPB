import torch
from torch import nn

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.optim import lr_scheduler

from dataload import load_cifar_10, load_cifar_100, load_tiny_imagenet
from network import ResNet20, Vgg, ResNet18
from train import train_net
from pruner import NPBPruner


def set_axes(axes, xlabel, ylabel, xlim, ylim, legend, xscale, yscale):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, title=None,
         ylim=None, xscale='linear', yscale='linear', xticklabels=None,
         fmts=('^-r', '-b', '-g', '-m', '-C1', '-C5'), figsize=(4.5, 4), axes=None, twins=False, ylim2=None):
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    if twins:
        ax2 = axes.twinx()
        ax2.set_ylim(ylim2)
        ax2.set_ylabel(ylabel[1])
    i = 0
    ax = axes
    f = []
    for x, y, fmt in zip(X, Y, fmts):
        if twins and (i > 0):
            ax = ax2
        if len(x):
            h, = ax.plot(x, y, fmt)
        else:
            h, = ax.plot(y, fmt)
        f.append(h)
        i += 1
    if title:
        axes.set_title(title)
    set_axes(axes, xlabel, ylabel[0], xlim, ylim, legend, xscale, yscale)
    if xticklabels:
        axes.set_xticks(X[0])
        axes.set_xticklabels(xticklabels, rotation=60, fontsize=12)


class Tester:
    @staticmethod
    def compute_npb(file="res/compute_path_nodes"):
        sparsity = [0.6838, 0.9, 0.9684, 0.99, 0.9944, 0.9968, 0.9982, 0.999]

        T, num_classes, prune_batch_size = 100, 100, 1

        paths_his, nodes_his = [], []
        for final_s in sparsity:
            net = ResNet20().cuda(torch.device("cuda:0"))
            resnet_pruner = NPBPruner(final_s, max_param_per_kernel=9)
            paths, nodes = resnet_pruner.prune(net, T, (num_classes * 10, 3, 32, 32), prune_batch_size, metric=True)
            paths_his.append(paths)
            nodes_his.append(nodes)
            print(paths, nodes)

        plot(torch.arange(len(sparsity)), [torch.tensor(paths_his), torch.tensor(nodes_his)],
             legend=['paths', 'nodes'], fmts=['-C7', '-r'],
             xticklabels=[str(round(s * 100, 2)) for s in sparsity],
             xlabel='Sparsity (%)', ylabel=['Remain ratio'], figsize=(10, 8))
        # plt.savefig(file)

    @staticmethod
    def test_masked_resnet18_npb(final_s, path="models/res18"):
        epoch, batch_size, lr = 100, 128, 0.01
        T, num_classes, prune_batch_size = 100, 200, 1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_tiny_imagenet(batch_size)

        net = ResNet18().cuda(torch.device("cuda:0"))
        resnet_pruner = NPBPruner(final_s, max_param_per_kernel=6, beta=2, chunk_size=16)
        resnet_pruner.prune(net, T, (num_classes * 10, 3, 64, 64), prune_batch_size, metric=True)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[30, 60, 80], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_resnet18_muti_npb(path="models/resnet18", file="res/ResNet-18_syn_wide"):
        print("test_masked_resnet18_muti")
        sparsity = [0.6838, 0.9, 0.9684, 0.99]

        sparsity_ratios, test_accs = [], []
        for final_s in sparsity:
            sparsity_ratio, test_acc = Tester.test_masked_resnet18_npb(final_s, path)
            sparsity_ratios.append(sparsity_ratio)
            test_accs.append(test_acc)

        # plot(torch.arange(len(sparsity)), torch.tensor(test_accs) * 100,  # ylim=[66, 75],
        #      xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="res/ResNet-18 (Tiny-ImageNet)",
        #      xticklabels=[str(round(s * 100, 2)) for s in sparsity], figsize=(8, 8))
        # plt.savefig(file)

    @staticmethod
    def test_masked_vgg19_npb(final_s, path="models/vgg16"):
        epoch, batch_size, lr = 160, 128, 0.1
        T, num_classes, prune_batch_size = 100, 100, 1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_100(batch_size)

        net = Vgg(num_classes=num_classes).cuda(torch.device("cuda:0"))
        vgg_pruner = NPBPruner(final_s, max_param_per_kernel=16)
        vgg_pruner.prune(net, T, (num_classes * 10, 3, 32, 32), prune_batch_size, metric=True)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[60, 120], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_vgg19_muti_npb(path="models/vgg", file="res/VGG-19 (CIFAR-100)"):
        print("test_masked_vgg19_muti")
        sparsity = [0.6838, 0.9, 0.9684, 0.99]

        sparsity_ratios, test_accs = [], []
        for final_s in sparsity:
            sparsity_ratio, test_acc = Tester.test_masked_vgg19_npb(final_s, path)
            sparsity_ratios.append(sparsity_ratio)
            test_accs.append(test_acc)

        # plot(torch.arange(len(sparsity)), torch.tensor(test_accs) * 100,  # ylim=[66, 75],
        #      xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="VGG-19 (CIFAR-100)",
        #      xticklabels=[str(round(s * 100, 2)) for s in sparsity], figsize=(8, 8))
        # plt.savefig(file)

    @staticmethod
    def test_masked_resnet20_npb(final_s, path="models/res"):
        epoch, batch_size, lr = 160, 128, 0.1
        T, num_classes, prune_batch_size = 100, 10, 1

        loss = nn.CrossEntropyLoss().cuda(torch.device("cuda:0"))

        data_iter = load_cifar_10(batch_size)

        net = ResNet20().cuda(torch.device("cuda:0"))
        resnet_pruner = NPBPruner(final_s, max_param_per_kernel=9)
        resnet_pruner.prune(net, T, (num_classes * 10, 3, 32, 32), prune_batch_size, metric=True)

        trainer = torch.optim.SGD(net.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[60, 120], gamma=0.1)
        sparsity_ratio, test_acc = train_net(net, loss, trainer, data_iter, epoch, path, scheduler)
        print("true sparsity: %f%%    test_acc: %.2f%%" % (sparsity_ratio * 100, test_acc * 100))
        return sparsity_ratio, test_acc

    @staticmethod
    def test_masked_resnet20_muti_npb(path="models/res", file="res/ResNet-20 (CIFAR-10)"):
        sparsity = [0.6838, 0.9, 0.9684, 0.99]

        test_accs = []
        for final_s in sparsity:
            _, test_acc = Tester.test_masked_resnet20_npb(final_s, path)
            test_accs.append(test_acc)

        # plot(torch.arange(len(sparsity)), [torch.tensor(test_accs) * 100],
        #      xlabel='Sparsity (%)', ylabel=['Test Accuracy (%)'], title="ResNet-20 (CIFAR-10)",
        #      xticklabels=[str(round(s * 100, 2)) for s in sparsity], figsize=(6, 5))
        # plt.savefig(file)
