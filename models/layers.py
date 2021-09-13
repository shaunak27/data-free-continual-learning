import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.width = (kernel_size-1)//2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.width, self.width, self.width, self.width), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


class CosineScaling(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Add BN and linear layers
        Note: just for MLP at the moment. need bn2d compatible for cnn
        """
        super(CosineScaling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.end_linear = nn.Linear(in_features, out_features)
        self.scale_linear = nn.Linear(in_features, 1, bias=False)
        self.fc_w = nn.Parameter(self.end_linear.weight)
        self.cossin_layer = CenterCosineSimilarity
        self.bn_scale = nn.BatchNorm1d(1) 

    def forward(self, x):

        scale = self.scale_linear(x)
        scale = self.bn_scale(scale)
        scale = torch.exp(scale)
        torch.transpose(scale,0,1)
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.fc_w)
        w_norm_transposed = torch.transpose(w_norm, 0, 1)
        x_cos = torch.mm(x_norm, w_norm_transposed)
        x_scaled = scale * x_cos
        return x_scaled


class CenterCosineSimilarity(nn.Module):
    def __init__(self, feat_dim, num_centers, eps=1e-8):
        super(CenterCosineSimilarity, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        self.eps = eps

    def forward(self, feat):
        norm_f = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_normalized = torch.div(feat, norm_f)
        norm_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        center_normalized = torch.div(self.centers, norm_c)
        return torch.mm(feat_normalized, center_normalized.t())

class CosineSimilarity(nn.Module):
    def __init__(self, feat_dim, num_centers):
        super(CosineSimilarity, self).__init__()
        self.in_features = feat_dim
        self.out_features = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, feat_dim))
        torch.nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))  # The init is the same as nn.Linear

    def forward(self, feat):
        feat_normalized = F.normalize(feat)
        center_normalized = F.normalize(self.centers)

        return torch.mm(feat_normalized, center_normalized.t())

    def extra_repr(self):
        return 'feat_dim={feat_dim}, num_center={num_center}'.format(
            feat_dim=self.centers.size(1), num_center=self.centers.size(0))
