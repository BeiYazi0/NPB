import copy
import math
import time
import cvxpy as cp

import numpy as np
import torch
from torch import nn

from dataload import load_prune_data


def compute_path_nodes(net, masks, data_iter):
    net = copy.deepcopy(net)

    data = next(iter(data_iter))
    input_dim = list(data.shape)
    input_dim[0] = 1
    X = torch.ones(input_dim).double()

    parameters = {}
    with torch.no_grad():
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.weight.data.copy_(masks[name])
                parameters[name] = module.weight

    net.eval()
    net.cpu().double()
    y = net(X)
    term = torch.sum(y)
    term.backward()

    paths = term.item()
    print(torch.log10(term))
    nodes = torch.sum(y > 0)
    cnt = y.shape[1]

    with torch.no_grad():
        for weight in parameters.values():
            if weight.dim() == 4:
                p = torch.sum(weight.grad, dim=[0, 2, 3])
            else:
                p = torch.sum(weight.grad, dim=0)
            nodes += torch.sum(p > 0)
            cnt += p.shape[0]
            weight.data.fill_(1)
        y = net(X)

    print(torch.log(nodes))
    del net
    return (paths / torch.sum(y)).item(), (nodes / cnt).item()


class NPBPruner:
    def __init__(self, final_s, alpha=0.01, beta=1,
                 max_param_per_kernel=None, min_param_to_node=None,
                 chunk_size=32, is_scale_weight=False, scale_weight=None,
                 node_constraint=False):
        self.final_s = final_s
        self.weight_num = 0
        self.masks = {}
        self.parameters = {}
        self.intermediate_inputs = {}

        self.alpha = alpha
        self.beta = beta
        self.chunk_size = chunk_size
        self.is_scale_weight = is_scale_weight
        self.scale_weight = scale_weight
        self.node_constraint = node_constraint
        self.max_param_per_kernel = max_param_per_kernel  # 2D 核最多保留的连接数
        self.min_param_to_node = min_param_to_node

    def init_parameters(self):
        for _, weight in self.parameters.items():
            weight.data.fill_(1.)

    def get_layer_wise_sparsity(self):
        # print('initialize by ERK')
        density = 1 - self.final_s
        erk_power_scale = 1

        total_params = self.weight_num
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(mask.shape) / np.prod(mask.shape)
                                              ) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        sparsity_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            if name in dense_layers:
                sparsity_dict[name] = 0
            else:
                probability_one = epsilon * raw_probabilities[name]
                sparsity_dict[name] = 1 - probability_one
            # print(
            #     f"layer: {name}, shape: {mask.shape}, sparsity: {sparsity_dict[name]}"
            # )
            total_nonzero += (1 - sparsity_dict[name]) * mask.numel()
        # print(f"Overall sparsity {1 - total_nonzero / total_params}")
        return sparsity_dict
        #     mask.data.copy_((torch.rand(mask.shape) < density_dict[name]).float().data.cuda())

        #     total_nonzero += density_dict[name] * mask.numel()
        # print(f"Overall sparsity {total_nonzero / total_params}")

    def check_layer_ineff_param(self, net, input_shape, layer_name):  # 修改 mask 后重新检测有效连接数
        X = torch.ones(input_shape).double()
        y = net(X)
        term = torch.sum(y)
        term.backward()

        with torch.no_grad():
            mask = self.masks[layer_name]
            eff_mask = torch.where(self.parameters[layer_name].grad.data != 0, 1, 0).to(mask.device) * mask
            ineff_mask = mask - eff_mask  # 失活连接
            n_ineff = ineff_mask.sum().item()
        return n_ineff, eff_mask

    def fine_tune_mask(self, net, input_shape):
        """将无效的连接断开，在当前层添加新的连接，并将剩余的无效连接数送至下一层以继续添加连接，一般在稀疏度（>=0.9684）较高时会起效。
        """
        X = torch.ones(input_shape).double()

        with torch.no_grad():  # 将复制的 net 中的权重 requires_grad 设定为 True
            for name, weight in self.parameters.items():
                weight.detach_()
                weight.requires_grad = True

        net = net.cpu().double()
        net.eval()
        y = net(X)
        term = torch.sum(y)
        term.backward()

        # 统计有效的连接
        eff_masks = {}
        for name, param in self.parameters.items():  # 乘上 mask，为 0 的连接 grad 也应该为 0
            eff_masks[name] = torch.where(param.grad.data != 0, 1, 0) * self.masks[name].to(torch.device("cpu"))

        n_ineff_after = 0
        for name, mask in self.masks.items():
            if len(mask.shape) in [4]:
                c_out, c_in, k, w = mask.shape
                eff_mask = eff_masks[name]
                ineff_mask = mask - eff_mask.to(mask.device)  # 计算当前权重中失效的连接数
                n_ineff = ineff_mask.sum().item() + n_ineff_after  # 目前的总失效数（除去给与其它权重后重新激活的连接）
                if n_ineff > 1:
                    print(f'Adding ones to mask of layer {name}')
                    if k == 1:  # 残差连接
                        new_mask = eff_mask.view(c_out, c_in)
                        tmp = eff_mask.sum(dim=0).view(-1)  # 输入节点的有效连接数
                        idx = torch.argsort(tmp, descending=True)
                        count = 0
                        while n_ineff > 0:  # 优先补全连接了更多有效输出通道的输入通道
                            curr = new_mask[:, idx[count]].sum()
                            need = c_out - curr
                            n_ineff = n_ineff - need
                            new_mask[:, idx[count]].copy_(torch.ones_like(new_mask[:, idx[count]]))
                            count += 1
                    else:
                        new_mask = eff_mask.view(-1, k, w)
                        tmp = eff_mask.sum(dim=(2, 3)).view(-1)  # sum over kernels and
                        idx = torch.argsort(tmp, descending=True)  # sort in desceding order
                        count = 0
                        while n_ineff > 0:  # 优先补全具有更多有效连接的某对输出通道的输入通道
                            curr = new_mask[idx[count]].sum()
                            need = k * w - curr
                            n_ineff = n_ineff - need
                            new_mask[idx[count]].copy_(torch.ones_like(new_mask[idx[count]]))
                            count += 1
                    # 应用新 mask
                    new_mask = new_mask.view(c_out, c_in, k, w)
                    mask.data.copy_(new_mask)
                    with torch.no_grad():
                        self.parameters[name].copy_(mask)  # apply mask
                    # 再次确认是否存在无效连接
                    n_ineff_after, eff_mask = self.check_layer_ineff_param(net, input_shape, name)
                    mask.data.copy_(eff_mask)  # copy new mask
                    with torch.no_grad():
                        self.parameters[name].copy_(mask)  # apply mask
            elif len(mask.shape) in [2]:  # 设计的网络只要一层全连接，因此直接计算失效数即可，不补
                f_out, f_in = mask.shape
                eff_mask = eff_masks[name]
                ineff_mask = mask - eff_mask.to(mask.device)
                n_ineff = ineff_mask.sum().item() + n_ineff_after
                print(f'number of ineff params in {name} is {n_ineff}')

    @staticmethod
    def optimize_layerwise(mask, inp, sparsity, alpha=0.7,
                           beta=0.001, max_param_per_kernel=None,
                           min_param_to_node=None,
                           init_weight=None,
                           node_constraint=False):
        start_time = time.time()
        # print('Optimizing layerwise sparse pattern')
        is_conv = False

        # Params in layer
        n_params = int(math.ceil((1 - sparsity) * mask.numel()))  # 保留连接数

        #  P_in 表示输入节点的路径数量
        if len(mask.shape) == 4:
            C_out, C_in, kernel_size, kernel_size = mask.shape
            min_param_per_kernel = int(math.ceil(n_params / (C_in * C_out)))
            if max_param_per_kernel is None:
                max_param_per_kernel = kernel_size * kernel_size
            # Ensure enough params to assign to valid the sparsity
            elif max_param_per_kernel < min_param_per_kernel:
                max_param_per_kernel = min_param_per_kernel
            else:  # it's oke
                pass

            if min_param_to_node is None:  # 每个 node 的最小连接数
                min_param_to_node = 1
            # Ensure the valid of eff node constraint
            elif min_param_to_node > min_param_per_kernel:
                min_param_to_node = min_param_per_kernel
            else:  # it's oke
                pass

            P_in = torch.sum(inp, dim=(1, 2)).cpu().numpy()
            is_conv = True
        else:
            C_out, C_in = mask.shape
            kernel_size = 1
            max_param_per_kernel = kernel_size
            min_param_to_node = 1
            # P_in = torch.sum(inp, dim=)
            if len(inp.shape) != 1:
                P_in = torch.sum(inp, dim=1).cpu().numpy()
            else:
                P_in = inp.cpu().numpy()
            if len(P_in.shape) != 1 and P_in.shape[0] != C_out:
                raise ValueError('Wrong input dimension')

        # Mask variable
        M = cp.Variable((C_in, C_out), integer=True)

        scaled_M = None
        if init_weight is not None:
            if is_conv:
                mag_orders = init_weight.transpose(1, 0).view(C_in, C_out, -1).abs().argsort(dim=-1,
                                                                                             descending=True).cpu().numpy()
                init_weight = torch.sum(init_weight, dim=(2, 3)).transpose(1, 0).cpu().numpy()
            else:
                init_weight = init_weight.transpose(1, 0).cpu().numpy()
            init_weight = np.abs(init_weight)
            # scaled_M = cp.multiply(M, init_weight)

        # Sun
        sum_in = cp.sum(M, axis=1)
        sum_out = cp.sum(M, axis=0)
        # sum_in = cp.sum(M, axis=1) * P_in
        # sum_out = cp.sum(cp.diag(P_in)@M, axis=0)

        # If eff_node_in is small which means there is a large number of input effective node
        inv_eff_node_in = cp.sum(cp.pos(min_param_to_node - sum_in))
        inv_eff_node_out = cp.sum(cp.pos(min_param_to_node - sum_out))

        # OPtimize nodes
        max_nodes = C_in + C_out
        A = max_nodes - (inv_eff_node_in + inv_eff_node_out)
        # A = A / max_nodes   # Scale to 1

        # Optimize paths
        # B = (cp.sum(P_in @ M)) / cp.sum(P_in)   # Downscale with input nodes' values
        min_out_node = int(n_params / (C_out * max_param_per_kernel))
        remainder = n_params - min_out_node * (C_out * max_param_per_kernel)
        try:
            max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel)) + \
                       remainder * np.sort(P_in)[-(min_out_node + 1)]
        except:
            max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel))

        if scaled_M is not None:
            B = (cp.sum(P_in @ scaled_M))
            # B = (cp.sum(P_in @ scaled_M)) / np.sum(P_in)
        else:
            B = (cp.sum(P_in @ M)) / max_path
            A = A / max_nodes
        # C = (cp.sum(P_in @ M)) / max_path
        # Regulaziration
        Reg = (n_params - cp.sum(cp.pos(1 - M))) / n_params  # maximize number of edges
        # Reg = 0

        # Constraint the total activated params

        constraint = [cp.sum(M) <= n_params, M <= max_param_per_kernel, M >= 0]

        if node_constraint:
            constraint.append(
                cp.max(cp.sum(M, axis=0)) <= int(C_in * max_param_per_kernel ** 2 * (1 - sparsity))
            )
            constraint.append(
                cp.max(cp.sum(M, axis=1)) <= int(C_out * max_param_per_kernel ** 2 * (1 - sparsity))
            )
        # Objective function
        # alpha = 0.7
        obj = cp.Maximize(alpha * A + (1 - alpha) * B + beta * Reg)

        # Init problem
        prob = cp.Problem(obj, constraint)

        # Solving
        prob.solve()
        # prob.value

        if is_conv:
            a = torch.tensor(M.value, dtype=torch.int16)
            mat = []
            for i in range(C_out):
                row = []
                for j in range(C_in):
                    try:
                        r = np.zeros(kernel_size ** 2)
                        if init_weight is not None:
                            one_idxs = mag_orders[j, i][:a[j, i]]
                            r[one_idxs] = 1
                        else:
                            r[:a[j, i]] = 1
                            np.random.shuffle(r)
                        row.append(r.reshape(kernel_size, kernel_size))
                    except:
                        print(r)
                        print(a[j, i])
                mat.append(row)
            mat = np.array(mat)
            mask.data.copy_(torch.tensor(mat))
        else:
            mask.data.copy_(torch.tensor(M.value).transpose(1, 0))

        actual_sparsity = 1 - mask.sum().item() / mask.numel()
        end_time = time.time()
        # print(f'Pruning time is {end_time - start_time}')

        return mask

    def optimization(self, data_iter, net, device):
        data = next(iter(data_iter))
        input_dim = list(data.shape)
        input_dim[0] = 1
        input_ = torch.ones(input_dim).double().to(device)

        layer_id = 0
        estimate_time = 0
        is_resnet20 = False
        if net.__class__.__name__ == 'ResNet' and input_dim[2] == 32:
            is_resnet20 = True

        saved_params = {}
        i = 0
        for name in self.parameters.keys():
            if self.is_scale_weight:
                saved_params[name] = self.scale_weight[i]
            else:
                saved_params[name] = None
            i += 1

        net.eval()
        self.init_parameters()
        sparsity_dict = self.get_layer_wise_sparsity()
        for name, mask in self.masks.items():
            # 获取中间输入
            net(input_)
            prev = self.intermediate_inputs[name].detach().requires_grad_(False)
            if mask.dim() == 4:  # 卷积层
                if layer_id == 0:  # Input layer
                    c_out, c_in, kernel_size, _ = mask.shape
                    start_time = time.time()
                    new_mask = self.optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name],
                                                       alpha=self.alpha, beta=self.beta,
                                                       max_param_per_kernel=None,
                                                       init_weight=saved_params[name])
                    mask.copy_(new_mask)
                    estimate_time = estimate_time + time.time() - start_time
                else:
                    c_out, c_in, kernel_size, _ = mask.shape
                    # 较大的卷积核，每次对 chunk_size 个输出通道进行优化
                    if (c_out * c_in > 128 * 128) or (is_resnet20 and c_out * c_in > 64 * 32):  # Using Chunking
                        n_chunks = int(c_out / self.chunk_size)
                        new_mask = copy.deepcopy(mask)
                        # chunked_masks = []
                        for idx in range(n_chunks):
                            start_time = time.time()
                            start_c_out = idx * self.chunk_size
                            end_c_out = (idx + 1) * self.chunk_size
                            chunked_mask = copy.deepcopy(new_mask[start_c_out:end_c_out, :, :, :])
                            chunked_sparsity = sparsity_dict[name]
                            if self.is_scale_weight:
                                chunked_init_weight = saved_params[name][start_c_out:end_c_out, :, :, :]
                            else:
                                chunked_init_weight = None
                            if kernel_size == 1:  # 残差连接
                                chunked_mask = self.optimize_layerwise(chunked_mask, prev[0],
                                                                       sparsity=sparsity_dict[name],
                                                                       alpha=self.alpha,
                                                                       init_weight=chunked_init_weight,
                                                                       node_constraint=self.node_constraint)
                            else:
                                chunked_mask = self.optimize_layerwise(chunked_mask, prev[0], sparsity=chunked_sparsity,
                                                                       alpha=self.alpha, beta=self.beta,
                                                                       max_param_per_kernel=self.max_param_per_kernel,
                                                                       min_param_to_node=self.min_param_to_node,
                                                                       init_weight=chunked_init_weight,
                                                                       node_constraint=self.node_constraint)
                            mask[start_c_out:end_c_out, :, :, :].copy_(chunked_mask)
                            end_time = time.time()
                        estimate_time = estimate_time + end_time - start_time + 10  # 这里时间算的不对吧

                    else:  # small size
                        start_time = time.time()
                        if kernel_size == 1:  # 残差连接
                            # pass
                            mask.copy_(self.optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name],
                                                               alpha=self.alpha, init_weight=saved_params[name],
                                                               node_constraint=self.node_constraint))
                        else:
                            # pass
                            mask.copy_(self.optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name],
                                                               alpha=self.alpha, beta=self.beta,
                                                               max_param_per_kernel=self.max_param_per_kernel,
                                                               min_param_to_node=self.min_param_to_node,
                                                               init_weight=saved_params[name],
                                                               node_constraint=self.node_constraint))
                        end_time = time.time()
                        estimate_time = estimate_time + end_time - start_time
                layer_id += 1
                if kernel_size == 1:  # 残差连接
                    if self.max_param_per_kernel > 5:
                        self.max_param_per_kernel -= 2

            elif mask.dim() == 2:  # Linear layer
                start_time = time.time()
                f_out, f_in = mask.shape
                if f_out * f_in > 512 * 10:  # 较大的全连接层，每次对 chunk_size 个输出通道进行优化
                    n_chunks = int(f_out / 10)
                    new_mask = copy.deepcopy(mask)
                    for idx in range(n_chunks):
                        start_f_out = idx * 10
                        end_f_out = (idx + 1) * 10
                        # print(f'Consider C_out from {start_f_out} to {end_f_out}')
                        chunked_mask = copy.deepcopy(new_mask[start_f_out:end_f_out, :])
                        chunked_sparsity = sparsity_dict[name]
                        if self.is_scale_weight:
                            chunked_init_weight = saved_params[name][start_f_out:end_f_out, :]
                        else:
                            chunked_init_weight = None
                        chunked_mask = self.optimize_layerwise(chunked_mask, prev[0], sparsity=chunked_sparsity,
                                                               alpha=self.alpha, beta=0,
                                                               init_weight=chunked_init_weight)

                        mask[start_f_out:end_f_out, :].copy_(chunked_mask)
                else:
                    print(prev.shape)
                    mask.copy_(self.optimize_layerwise(mask, prev[0], sparsity=sparsity_dict[name],
                                                       alpha=self.alpha, beta=0,
                                                       init_weight=saved_params[name]))
                layer_id += 1
                actual_sparsity = 1 - mask.sum().item() / mask.numel()
                end_time = time.time()
                estimate_time = estimate_time + end_time - start_time
                # print(f'Desired sparsity is {sparsity_dict[name]} and optimizer finds sparsity is {actual_sparsity}')

            else:
                # print(f'Ignore layer {name}')
                continue

            # apply_mask
            self.parameters[name].data.copy_(mask)
            if net.__class__.__name__ == 'VGG':  # vgg 从第二个块开始，在每个块最后一个卷积层开始优化前  max_param_per_kernel
                if layer_id in [3, 7, 11]:
                    print(True, c_in, c_out)
                    self.max_param_per_kernel -= 2
            # try:
            #     if net.__class__.__name__ == 'VGG':
            #         print(True)
            #         if layer_id in [1, 4, 9, 14]:  # 每个 max_pool 之前执行次操作
            #             if layer_id != 1:
            #                 self.max_param_per_kernel -= 2
            #             layer_id += 1  # 越过 max_pool
            # except:
            #     print('Done Pruning!')

        self.fine_tune_mask(net, input_dim)
        # count_ineff_param(cloned_net, input_shape)

    @staticmethod
    def apply_mask(module, mask):
        mask.require_grad = False
        nn.init.kaiming_normal_(module.weight)
        module.weight.data *= mask
        module.weight.register_hook(lambda grad: grad * mask)  # 冻结梯度

    def prune(self, net, T, size, batch_size, metric=False):
        def register_forward_hook(module, name):  # hook 每层的输入
            def hook_fn(module, input, output):
                self.intermediate_inputs[name] = input[0].data.cpu()
            module.register_forward_hook(hook_fn)

        device = next(net.parameters()).device

        net_copy = copy.deepcopy(net)
        net_copy.double()
        for name, module in net_copy.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.weight.requires_grad = False
                self.masks[name] = torch.ones_like(module.weight, device=device)
                self.parameters[name] = module.weight
                self.weight_num += module.weight.numel()
                register_forward_hook(module, name)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        data_iter = load_prune_data(size, batch_size, device)
        self.optimization(data_iter, net_copy, device)

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                self.apply_mask(module, self.masks[name].float())

        del net_copy
        if metric:
            return compute_path_nodes(net, self.masks, data_iter)