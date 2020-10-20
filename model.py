import torch
import torch.nn as nn

import FrEIA.FrEIA.framework as Ff
import FrEIA.FrEIA.modules as Fm


def one_hot(labels, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1, 1), value=1.)
    return out


class cINN(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_in()

    def build_in(self):
        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLu(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, 28, 28)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))
        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(
                Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {'subnet_constructor': subnet, 'clamp': 1.0}, conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l):
        z = self.cinn(x, c=one_hot(l))
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)
