import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        print(x, x.device)
        return x

model = MyModel()
model = nn.DataParallel(model,device_ids=[0,1,2,3])
model.cuda()

batch_size = 64
x = torch.arange(21 * 8).float().view(21, -1)
x.cuda()
out = model(x)
