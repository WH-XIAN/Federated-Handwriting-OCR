'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, norm='BatchNorm', reduction=4):
        super(SeModule, self).__init__()
        self.norm = norm
        self.se1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if norm == 'BatchNorm':
            self.bn1 = nn.BatchNorm2d(in_size // reduction)
        else:
            self.bn1 = nn.LayerNorm(in_size // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.se2 = nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False)
        if norm == 'BatchNorm':
            self.bn2 = nn.BatchNorm2d(in_size)
        else:
            self.bn2 = nn.LayerNorm(in_size)
        self.hs = hsigmoid()

    def forward(self, x):
        out = self.se1(x)
        if self.norm == 'BatchNorm':
            out = self.bn1(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn1(out)
            out = out.permute(0, 3, 1, 2)
        out = self.se2(self.relu(out))
        if self.norm == 'BatchNorm':
            out = self.bn2(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn2(out)
            out = out.permute(0, 3, 1, 2)
        out = self.hs(out)
        return x * out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, norm):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.norm = norm

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        if norm == 'BatchNorm':
            self.bn1 = nn.BatchNorm2d(expand_size)
        else:
            self.bn1 = nn.LayerNorm(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        if norm == 'BatchNorm':
            self.bn2 = nn.BatchNorm2d(expand_size)
        else:
            self.bn2 = nn.LayerNorm(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        if norm == 'BatchNorm':
            self.bn3 = nn.BatchNorm2d(out_size)
        else:
            self.bn3 = nn.LayerNorm(out_size)

        self.shortcut = nn.Sequential()
        self.bn4 = None
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
            if norm == 'BatchNorm':
                self.bn4 = nn.BatchNorm2d(out_size)
            else:
                self.bn4 = nn.LayerNorm(out_size)
                
    def forward(self, x):
        out = self.conv1(x)
        if self.norm == 'BatchNorm':
            out = self.bn1(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn1(out)
            out = out.permute(0, 3, 1, 2)
        out = self.nolinear1(out)
        out = self.conv2(out)
        if self.norm == 'BatchNorm':
            out = self.bn2(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn2(out)
            out = out.permute(0, 3, 1, 2)
        out = self.nolinear2(out)
        out = self.conv3(out)
        if self.norm == 'BatchNorm':
            out = self.bn3(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn3(out)
            out = out.permute(0, 3, 1, 2)
        if self.se != None:
            out = self.se(out)
        # out = out + self.shortcut(x) if self.stride==1 else out
        if self.stride == 1:
            y = self.shortcut(x)
            if self.bn4 != None:
                if self.norm == 'BatchNorm':
                    y = self.bn4(y)
                else:
                    y = y.permute(0, 2, 3, 1)
                    y = self.bn4(y)
                    y = y.permute(0, 3, 1, 2)
            out = out + y
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, norm='BatchNorm', num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        if norm == 'BatchNorm':
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.bn1 = nn.LayerNorm(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1, norm),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2, norm),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1, norm),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40, norm), (2, 1), norm),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40, norm), 1, norm),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40, norm), 1, norm),
            Block(3, 40, 240, 80, hswish(), None, (2, 1), norm),
            Block(3, 80, 200, 80, hswish(), None, 1, norm),
            Block(3, 80, 184, 80, hswish(), None, 1, norm),
            Block(3, 80, 184, 80, hswish(), None, 1, norm),
            Block(3, 80, 480, 112, hswish(), SeModule(112, norm), 1, norm),
            Block(3, 112, 672, 112, hswish(), SeModule(112, norm), 1, norm), # Block(3, 112, 672, 160, hswish(), SeModule(160), （2， 1）),
            Block(5, 112, 672, 160, hswish(), SeModule(160, norm), 1, norm),
            Block(5, 160, 672, 160, hswish(), SeModule(160, norm), (2, 1), norm),
            Block(5, 160, 960, 160, hswish(), SeModule(160, norm), 1, norm),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        if norm == 'BatchNorm':
            self.bn2 = nn.BatchNorm2d(960)
        else:
            self.bn2 = nn.LayerNorm(960)
        self.hs2 = hswish()
        # self.linear3 = nn.Linear(960, 1280)
        # self.bn3 = nn.BatchNorm1d(1280)
        # self.hs3 = hswish()
        # self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        if self.norm == 'BatchNorm':
            out = self.bn1(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn1(out)
            out = out.permute(0, 3, 1, 2)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        if self.norm == 'BatchNorm':
            out = self.bn2(out)
        else:
            out = out.permute(0, 2, 3, 1)
            out = self.bn2(out)
            out = out.permute(0, 3, 1, 2)
        out = self.hs2(out)
        # out = F.avg_pool2d(out, 7)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, norm='BatchNorm', num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        # 灰度图 input channel 为 1 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), (2, 1)),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), (2, 1)),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.hs2 = hswish()
        # 我不做分类 后买呢不需要
        # self.linear3 = nn.Linear(576, 1280)
        # self.bn3 = nn.BatchNorm1d(1280)
        # self.hs3 = hswish()
        # self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # out = self.hs1(self.bn1(self.conv1(x)))
        # out = self.bneck(out)
        
        # print('out.shape', out.shape)
        # out = self.hs2(self.bn2(self.conv2(out)))
        
        return x + x


class MobileNetV3_Large_Revised(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            # Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            # Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, (2, 1)),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1), 
            Block(3, 112, 672, 160, hswish(), SeModule(160), (2,1)),
            # Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            # Block(5, 160, 672, 160, hswish(), SeModule(160), (2, 1)),
            # Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        return out


def get_net():
    net = MobileNetV3_Small()
    # net = net.eval()
    return net
    # x = torch.randn(1,1,32,320)
    # y = net(x)
    # print(y.size())
    
def get_large_net(norm='BatchNorm'):
    net = MobileNetV3_Large(norm)
    # net = net.eval()
    return net

def get_revised_large_net():
    net = MobileNetV3_Large_Revised()
    return net
