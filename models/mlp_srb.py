import torch
import torch.nn as nn

class SR_BLOCK(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SR_BLOCK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
        )

    def features(self, x):
        x = self.linear(x)
        return x

    def forward(self, x, pen=False):
        x = self.features(x)
        return x

class MLP_SRB(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=256, block_division=10):
        super(MLP_SRB, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.block_division = block_division
        self.hidden_dim = hidden_dim
        self.l1_blocks = nn.ModuleList([SR_BLOCK(self.in_dim, int(hidden_dim/self.block_division)) for l in range(self.block_division)])
        self.l2_blocks = nn.ModuleList([SR_BLOCK(self.hidden_dim, int(hidden_dim/self.block_division)) for l in range(self.block_division)])
        self.last = nn.Linear(hidden_dim, out_dim) 

    def features(self, x):
        # layer 1
        x_out = self.l1_blocks[0](x.view(-1,self.in_dim))
        for l in range(1,self.block_division):
            x_out = torch.cat([x_out, self.l1_blocks[l](x.view(-1,self.in_dim))], dim=1)

        return [x_out]

        # # layer 2
        # x_out2 = self.l2_blocks[0](x_out)
        # for l in range(1,self.block_division):
        #     x_out2 = torch.cat([x_out2, self.l2_blocks[l](x_out)], dim=1)

        # return [x_out, x_out2]

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x, pen=False, div=False):
        x = self.features(x)
        if pen and div:
            return x, self.hidden_dim, self.block_division
        elif pen:
            return x[-1]
        else:
            x = self.logits(x[-1])
            return x

    def penultimate(self, x):
        x = self.features(x)
        return x

    def adaptive_forward(self, x):
        min_val = torch.pow(self.l1_blocks[0](x.view(-1,self.in_dim)), 2).sum().detach().cpu()
        min_ind = 0
        for l in range(1,self.block_division):
            val = torch.pow(self.l1_blocks[l](x.view(-1,self.in_dim)), 2).sum().detach().cpu()
            if val > min_val:
                min_ind = l
                min_val = val

        x_out = self.l1_blocks[0](x.view(-1,self.in_dim))
        for l in range(1,self.block_division):
            if l == min_ind:
                x_out = torch.cat([x_out, self.l1_blocks[l](x.view(-1,self.in_dim))], dim=1)
            else:
                x_out = torch.cat([x_out, self.l1_blocks[l](x.view(-1,self.in_dim)).detach()], dim=1)

        x = self.logits(x_out)
        return x

def MLP100(out_dim):
    return MLP_SRB(hidden_dim=100)

def MLP200(out_dim):
    return MLP_SRB(hidden_dim=200)

def MLP300(out_dim):
    return MLP_SRB(hidden_dim=300)

def MLP400(out_dim):
    return MLP_SRB(hidden_dim=400)

def MLP500(out_dim):
    return MLP_SRB(hidden_dim=500)

def MLP600(out_dim):
    return MLP_SRB(hidden_dim=600)

def MLP800(out_dim):
    return MLP_SRB(hidden_dim=800)


def MLP1000(out_dim):
    return MLP_SRB(hidden_dim=1000)


def MLP2000(out_dim):
    return MLP_SRB(hidden_dim=2000)


def MLP5000(out_dim):
    return MLP_SRB(hidden_dim=5000)