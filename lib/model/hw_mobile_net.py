from torch import nn
import torch, math

# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

import torch.nn.functional as F

import sys, os

import lib.config.alphabets_new as alphabets_raw #
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import softmax
from . import resnet, mobilenetv3


def strLabelConverter(res, alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
        print('raw', raw)
    return ''.join(raw)


def resizeNormalize(img, imgH=32):
    # scale = img.size[1]*1.0 / imgH
    # w     = img.size[0] / scale
    # w     = int(w)
    # img   = img.resize((w,imgH),Image.BILINEAR)
    # w,h   = img.size
    img = (np.array(img)/255.0-0.588)/0.193
    return img


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=False)
        )


class ConvLNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        super(ConvLNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False)
        self.ln = nn.LayerNorm(out_channel)
        self.relu = nn.ReLU6(inplace=False)
    
    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1)
        out.contiguous()
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)
        out.contiguous()
        out = self.relu(out)
        return out


class FusedInvertedResidual(nn.Module):
    def __init__(self, in_channel, expand_ratio):
        super(FusedInvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = True

        layers = []
        layers.extend([
            # 3x3 
            ConvBNReLU(in_channel, hidden_channel),
            # 1x1 
            ConvBNReLU(hidden_channel, in_channel, kernel_size=1)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout, max_len=80, gpu_idx=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ts = max_len
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe
        # self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print('position embedding shape ', self.pe.shape)
        x = x + self.pe[:, :self.ts].to(x.device)
        return self.dropout(x)


class CamNet(nn.Module):
    def __init__(self, num_classes=1000, norm='BatchNorm', gpu_idx=0, img_w=320 ,alpha=1.0, round_nearest=8):
        super(CamNet, self).__init__()

        # combine feature layers
        # self.features = resnet.get_resnet() #  plan1: nn.Sequential(*features)
        # self.one_by_one = ConvBNReLU(512, 256, kernel_size=1)
        self.norm = norm
        self.features = mobilenetv3.get_large_net(norm)
        if norm == 'BatchNorm':
            self.one_by_one = ConvBNReLU(960, 256, kernel_size=1)
        else:
            self.one_by_one = ConvLNReLU(960, 256, kernel_size=1)
        # self.features = mobilenetv3.get_revised_large_net()
        
        # self.one_by_one = ConvBNReLU(256, 256, kernel_size=1)
        
        # building classifier
        assert img_w % 32 == 0
        time_step = int(img_w / 4)
        
        self.pos_embedding = PositionalEncoding(256, 0.1, time_step, gpu_idx)

        self.attn =  SelfAttention(256) 

        # class_dim = number of words 
        self.trans_layer = nn.Linear(256, num_classes + 1)

    def forward(self, x):
        # 1, 1, h, w  32 320 
        b, c, h, w = x.shape # channel ==> concat blocks
        # print('x shape ====*****' ,x.shape)

        x = self.features(x) # convnext
        # print('features ===>', x.shape)
        x = self.one_by_one(x)
        # print('x before pooling shape ===>', x.shape)

        # print('x  before perm ===>', x.shape)
        x = x[:, : , 0, :] # 32 256 40
        x = x.permute(0, 2, 1) # b 40 256
        
        # print('x before self attention layer &&&&&&&&&&&>', x.shape)

        x = self.pos_embedding(x)
        
        # x += self.pos_embedding[:, :]
        # x = self.dropout(x)
        
        # b 40 256
        
        x = self.attn(x) # 
        
        x = self.trans_layer(x) # dim ===>  dict size
        # print('hidden dim shape ===>', x.shape)
        x = F.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)
        # print('x de final shape ', x.shape)
        return x

    def load_weights(self, path, use_gpu=True):
        
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
        if use_gpu:
            self.cuda()
        self.eval()

    def predict(self,image):
        image = resizeNormalize(image,32)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image   = image.cuda()
        else:
            image   = image.cpu()
        image       = image.view(1,1, *image.size())
        image       = Variable(image)
        preds       = self(image)
        print('preds shape is ', preds.shape)
        # get probs
        print(softmax(preds, dim=-1).shape)
        values, prob = softmax(preds, dim=-1).max(2)
        preds_idx = (prob > 0).nonzero()
        sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].detach()
        # get preds
        _, preds    = preds.max(2)
        preds       = preds.transpose(1, 0).contiguous().view(-1)
        raw         = strLabelConverter(preds, alphabets_raw.alphabet_cn)
        print(raw, sent_prob)

        return raw, sent_prob


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, num_classes=1000, norm='BatchNorm', gpu_idx=0, img_w=320 , nh=256):
        super(CRNN, self).__init__()

        self.norm = norm
        self.features = mobilenetv3.get_large_net(norm)
        if norm == 'BatchNorm':
            self.one_by_one = ConvBNReLU(960, 256, kernel_size=1)
        else:
            self.one_by_one = ConvLNReLU(960, 256, kernel_size=1)
        
        assert img_w % 32 == 0
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, nh, nh),
            BidirectionalLSTM(nh, nh, num_classes))
    
    def forward(self, x):
        out = self.features(x)
        out = self.one_by_one(out)
        out = out[:, : , 0, :] # 32 256 40
        out = out.permute(2, 0, 1) # 40 b 256
        out = F.log_softmax(self.rnn(out), dim=2)
        return out

    def load_weights(self, path, use_gpu=True):
        
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
        if use_gpu:
            self.cuda()
        self.eval()

    def predict(self,image):
        image = resizeNormalize(image,32)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image   = image.cuda()
        else:
            image   = image.cpu()
        image       = image.view(1,1, *image.size())
        image       = Variable(image)
        preds       = self(image)
        print('preds shape is ', preds.shape)
        # get probs
        print(softmax(preds, dim=-1).shape)
        values, prob = softmax(preds, dim=-1).max(2)
        preds_idx = (prob > 0).nonzero()
        sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].detach()
        # get preds
        _, preds    = preds.max(2)
        preds       = preds.transpose(1, 0).contiguous().view(-1)
        raw         = strLabelConverter(preds, alphabets_raw.alphabet_cn)
        # print(raw, sent_prob)

        return raw, sent_prob


class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.attention_head_size = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        # self.relative_position_bias = nn.Parameter(torch.randn(1, 400, 256))
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def transpose_for_scores(self, x):
        # plan 1
        # new_x_shape = x.size()[:-1] + (self.heads, self.attention_head_size)

        # y = x.view(*new_x_shape) # b, t, heads, dim_per_head
        # b, t, heads, dim_per_head = x.shape
        
        new_x_shape = x.size()[:-1] + (self.heads, self.attention_head_size)
        b, t, heads, dim_per_head = new_x_shape
        y = x.view( b, t, heads, dim_per_head) 
        
        heads_part = x.chunk(4, -1)
        new_parts = []
        for part in heads_part:
            new_parts.append(part.unsqueeze(-2))
        x = torch.cat(new_parts, 2)
        # print('x====y ', x == y)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots) # b h n 

        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        
        out = out.permute(0, 2, 1, 3)
        o1, o2, o3, o4 = out[:, :, 0, :], out[:, :, 1, :], out[:, :, 2, :], out[:, :, 3, :]
        out = torch.cat((o1, o2, o3, o4 ), dim=2)

        return self.to_out(out)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_model(config):
    if config.MODEL.TYPE == 'RNN':
        model = CRNN(config.MODEL.NUM_CLASSES, config.MODEL.NORM, config.GPUID, config.MODEL.IMAGE_SIZE.W, config.MODEL.NUM_HIDDEN)
        # model.apply(weights_init)
    else:
        model = CamNet(config.MODEL.NUM_CLASSES, config.MODEL.NORM, config.GPUID, config.MODEL.IMAGE_SIZE.W)
        # model.apply(weights_init)

    return model


if __name__ == '__main__':
    # input = torch.rand((1, 320, 256))
    # sa = SelfAttention(256)
    # sa(input)
    # net = CamNet()
    # input = torch.rand((1, 1, 32, 320))
    # ret = net(input)
    
    len_classes = len(alphabets_raw.alphabet_cn)
    model = CamNet(len_classes, 0)
    # path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Ch&En/crnn/2022-03-16-17-48/checkpoints/checkpoint_14_acc_0.9382.pth'
    path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Ch&En/attn_mobile_320_finetune/2022-03-31-22-51/checkpoints/checkpoint_0_acc_0.7515.pth'
    # path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Ch&En/attn_mobile_320_finetune/2022-03-31-22-51/checkpoints/checkpoint_2_acc_0.7511.pth'
    path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Ch&En/attn_mobile_320_finetune/2022-04-01-16-34/checkpoints/checkpoint_0_acc_0.7490.pth'
    
    model.load_weights(path)
    

    import cv2
    img_2_padd = cv2.imread('/mnt/Data02/junxi/ai_ocr_app_server/jc.jpg')  
    print('img_2_padd', img_2_padd.shape)
    from PIL import Image
    img_2_padd = Image.fromarray(img_2_padd.astype('uint8'))
    img = img_2_padd.convert('L')

    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    if width < 320:
        resized_img = img.resize([width, 32], Image.ANTIALIAS)
        bg = Image.new('L', (320, 32))
        bg.paste(resized_img)
       
    else:
        bg = img.resize([320, 32], Image.ANTIALIAS)

    model.predict(bg)
    
    





