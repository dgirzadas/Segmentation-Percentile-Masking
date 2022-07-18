from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = initialize_device()


class RegionConv2d(nn.Module):
    """
    A replacement class for the actual region convolution.
    It simulates convolving only specific regions by convolving everything, but masking areas
    that are not meant to be convolved. Then the masked output is summed with the input for a residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(RegionConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)

    def forward(self, input, mask):
        x = input + self.conv(input) * mask
        return x


class PercentileMask(nn.Module):
    """
    A class that defines the percentile masking layer of the network.
    This layer generates the binary mask from the model prediction matrix, by determining the threshold of each class,
    based on parameter q and applying it on the prediction matrix.
    """
    def __init__(self):
        super(PercentileMask, self).__init__()

    def forward(self, input, class_qlims):
        xmax, xmax_ids = torch.max(input, dim=3)

        qlims = torch.zeros(xmax.shape).to(DEVICE)
        for i in range(qlims.shape[0]):
            qlims[i] = class_qlims[i, xmax_ids[i]]
        x = 1 - binarize(xmax, threshold=qlims).transpose(1, 2)

        return x.unsqueeze(1).type(torch.IntTensor).to(DEVICE)


class CombineRegionConv(nn.Module):
    """
    A class that defines the intermediate representation refinement layer.
    This layer applies convolutions with different dilations and combines the outputs of these convolutions into
    one output matrix. This matrix is then masked and combined with input for region convolution simulation.
    """
    def __init__(self, in_channels=128, out_channels=32):
        super(CombineRegionConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=8, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=16, dilation=8)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask=None):
        if mask == None:
            mask = torch.ones_like(x[0]).to(DEVICE)

        x1 = self.bn1(F.relu(self.conv1(x)))
        x2 = self.bn2(F.relu(self.conv2(x)))
        x3 = self.bn3(F.relu(self.conv3(x)))
        x4 = self.bn4(F.relu(self.conv4(x)))

        convs = torch.cat([x1, x2, x3, x4], 1)
        x = x + convs * mask

        return x


class RegionNet_Stem(nn.Module):
    """
    A class that defines the stem of the network, consisting of four different convolutions.
    the output of the stem is the same shape as input, but with 128 channels instead of 3
    """
    def __init__(self):
        super(RegionNet_Stem, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, dilation=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7, padding=3, dilation=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=9, padding=4, dilation=1)

    def forward(self, x):
        #         x = F.relu(self.stem(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class RegionNet_Stage1(nn.Module):
    """
    A class that defines the first stage of the network.
    This stage consists of two multiple-dilation 'refinement' layers and a classification branch.
    The forward pass of this layer returns the internal representation after the two refinement layers and the
    classification predictions of the first classification branch.
    """
    def __init__(self, stem_channels, n_classes):
        super(RegionNet_Stage1, self).__init__()

        # Stage 1:
        self.s1_conv1 = CombineRegionConv()
        self.s1_conv2 = CombineRegionConv()

        # Classifier 1:
        self.class1_conv1 = nn.Conv2d(stem_channels, 64, kernel_size=9, padding=4)
        self.bn_class1 = nn.BatchNorm2d(64)
        self.class1_conv2 = nn.Conv2d(64, 50, kernel_size=7, padding=3)
        self.bn_class2 = nn.BatchNorm2d(50)
        self.class1_conv3 = nn.Conv2d(50, n_classes, kernel_size=5, padding=2)
        self.bn_class3 = nn.BatchNorm2d(n_classes)

    def forward(self, x):
        # Stage 1:
        x = self.s1_conv1(x)
        x = self.s1_conv2(x)

        # Classification 1:
        preds1 = F.relu(self.class1_conv1(x))
        preds1 = self.bn_class1(preds1)
        preds1 = F.relu(self.class1_conv2(preds1))
        preds1 = self.bn_class2(preds1)
        preds1 = F.relu(self.class1_conv3(preds1))
        preds1 = self.bn_class3(preds1)

        return x, preds1

class RegionNet_Stage2(nn.Module):
    """
    A class that defines the second stage of the network.
    Just like the first stage, this stage consists of two multiple-dilation 'refinement' layers and a
    classification branch. The forward pass of this layer returns the internal representation after the
    two refinement layers and the classification predictions of the second classification branch.
    """
    def __init__(self, stem_channels, n_classes):
        super(RegionNet_Stage2, self).__init__()

        # Stage 2:
        self.s2_conv1 = CombineRegionConv()
        self.s2_conv2 = CombineRegionConv()

        # Classifier 2:
        self.class2_conv1 = nn.Conv2d(stem_channels, 64, kernel_size=9, padding=4)
        self.bn_class1 = nn.BatchNorm2d(64)
        self.class2_conv2 = nn.Conv2d(64, 50, kernel_size=7, padding=3)
        self.bn_class2 = nn.BatchNorm2d(50)
        self.class2_conv3 = nn.Conv2d(50, n_classes, kernel_size=5, padding=2)
        self.bn_class3 = nn.BatchNorm2d(n_classes)

    def forward(self, x, mask1):
        # Stage 2:
        x = self.s2_conv1(x, mask1)
        x = self.s2_conv2(x, mask1)

        # Classification 2:
        preds2 = F.relu(self.class2_conv1(x))
        preds2 = self.bn_class1(preds2)
        preds2 = F.relu(self.class2_conv2(preds2))
        preds2 = self.bn_class2(preds2)
        preds2 = F.relu(self.class2_conv3(preds2))
        preds2 = self.bn_class3(preds2)

        return x, preds2

class RegionNet_Stage3(nn.Module):
    """
    A class that defines the third stage of the network.
    Just like the other two, this stage consists of two multiple-dilation 'refinement' layers and a
    classification branch. The forward pass of this layer returns only the classification predictions,
    as this is the final stage of the model.
    """
    def __init__(self, stem_channels, n_classes):
        super(RegionNet_Stage3, self).__init__()

        # Stage 3:
        self.s3_conv1 = CombineRegionConv()
        self.s3_conv2 = CombineRegionConv()

        # Classifier 3:
        self.class3_conv1 = nn.Conv2d(stem_channels, 64, kernel_size=9, padding=4)
        self.bn_class1 = nn.BatchNorm2d(64)
        self.class3_conv2 = nn.Conv2d(64, 50, kernel_size=7, padding=3)
        self.bn_class2 = nn.BatchNorm2d(50)
        self.class3_conv3 = nn.Conv2d(50, n_classes, kernel_size=5, padding=2)
        self.bn_class3 = nn.BatchNorm2d(n_classes)

    def forward(self, x, mask2):
        # Stage 3:
        x = self.s3_conv1(x, mask2)
        x = self.s3_conv2(x, mask2)

        # Classification 2:
        preds3 = F.relu(self.class3_conv1(x))
        preds3 = self.bn_class1(preds3)
        preds3 = F.relu(self.class3_conv2(preds3))
        preds3 = self.bn_class2(preds3)
        preds3 = F.relu(self.class3_conv3(preds3))
        preds3 = self.bn_class3(preds3)

        return preds3


class RegionNet(nn.Module):
    """
    A class that defines the entire architecture of the model.
    It combines all the blocks pre-defined above and forms the whole staged architecture.
    Besides the classifications and refinements, the forward pass implements the mask creation and application.
    """
    def __init__(self, n_classes):
        super(RegionNet, self).__init__()

        # Stem:
        self.stem = RegionNet_Stem()

        self.stem_channels = 128

        # Stage 1:
        self.s1 = RegionNet_Stage1(self.stem_channels, n_classes)

        # Stage 2:
        self.s2 = RegionNet_Stage2(self.stem_channels, n_classes)

        # Stage 3:
        self.s3 = RegionNet_Stage3(self.stem_channels, n_classes)

        # Mask creation
        self.q_mask = PercentileMask()

    def forward(self, x, class_qs=torch.tensor([0.5]).repeat(10).to(DEVICE), apply_mask=True):
        # Stem:
        x = self.stem(x)

        # Stage 1:

        if apply_mask:
            x, preds1 = self.s1(x)
            qlims = get_percentiles(probscale(preds1), q=class_qs)
            mask1 = self.q_mask(probscale(preds1).transpose(1, 3), qlims)

        else:
            x, preds1 = self.s1(x)
            mask1 = torch.ones(x.shape).to(DEVICE)

        # Stage 2:

        if apply_mask:
            x, preds2 = self.s2(x, mask1)
            preds2 = preds2 * mask1
            qlims = get_percentiles(probscale(preds2), q=class_qs)
            mask2 = self.q_mask(probscale(preds2).transpose(1, 3), qlims)

        else:
            x, preds2 = self.s2(x, mask1)
            mask2 = torch.ones(x.shape).to(DEVICE)

        # Stage 3:

        if apply_mask:
            preds3 = self.s3(x, mask2)
            preds3 = preds3 * mask1 * mask2
        else:
            preds3 = self.s3(x, mask2)

        return preds1, mask1, preds2, mask2, preds3

    def predict_image(self, x, class_qs, apply_mask=True):
        preds1, mask1, preds2, mask2, preds3 = self.forward(x, class_qs, apply_mask)
        pred = preds1 * (1 - mask1) + preds2 * mask1 * (1 - mask2) + preds3 * mask1 * mask2

        return probscale(pred)
