import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import softmax
from collections import OrderedDict
import numpy as np
from PIL import Image
import math

def resizeNormalize(img,imgH=32):
    scale = img.size[1]*1.0 / imgH
    w     = img.size[0] / scale
    w     = int(w)
    img   = img.resize((320,imgH),Image.BILINEAR)
    w,h   = img.size
    img = (np.array(img)/255.0-0.5)/0.5
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 1)
    return img

def strLabelConverter(res,alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)

def conv3x3(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        """
        self.module_list = []
        filter_list = [32,32, 64, 64]
        for idx, num_filter in enumerate(filter_list):
            if idx == 0:
                self.module_list.append(self._make_layer(block, num_filter, layers[idx]))
            else:
                self.module_list.append(self._make_layer(block, num_filter, layers[idx], stride=2))
        """

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 1))
        
        #self.avgpool = nn.AvgPool2d(3, stride=1)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def load_weights(self,path):
        
        trainWeights = torch.load(path,map_location=lambda storage, loc: storage)
        ### add here
        if 'state_dict' in trainWeights.keys():
            trainWeights = trainWeights['state_dict']
        else:
            trainWeights = trainWeights
            
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.','') # remove `module.`
            modelWeights[name] = v      
        self.load_state_dict(modelWeights)
        self.cuda()
        self.eval()
    
    def predict(self,image):
        pred = F.softmax(self.forward(image), dim=-1)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_resnet():
    model = ResNet(BasicBlock, [2, 2, 2, 2],2)
    model.apply(weights_init)
    return model
    

if __name__ == '__main__':
    pass
    # model = get_resnet()
    # model.apply(weights_init)
    # input = torch.randn(32, 1, 32, 320)
    # output = model(input)
    # pdb.set_trace()

    # flops, params = profile(model, (input,))
    # print('flops: ', flops, 'params: ', params)
