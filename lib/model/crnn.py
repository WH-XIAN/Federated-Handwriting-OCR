import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import softmax
from collections import OrderedDict
from thop import profile
import numpy as np
from PIL import Image
import pdb

def resizeNormalize(img,imgH=32):
    scale = img.size[1]*1.0 / imgH
    w     = img.size[0] / scale
    w     = int(w)
    img   = img.resize((w,imgH),Image.BILINEAR)
    w,h   = img.size
    img = (np.array(img)/255.0-0.5)/0.5
    return img

def strLabelConverter(res,alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)

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
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, alphabet=None):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.alphabet = alphabet
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn 
        # add pooling
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        '''
        # test rot
        conv_180 = torch.rot90(conv, 2, [1, 2])
        #conv[0, :, :] = conv_1.clone()
        #pdb.set_trace()
        a = torch.randint(0, 2, [1, 64]).T
        a = a.float()
        conv = a.unsqueeze(1)*conv + (1-a).unsqueeze(1)*conv_180
        '''
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)

        return output

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
        raw         = strLabelConverter(preds,self.alphabet)

        return raw, sent_prob

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(config):

    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)

    return model

if __name__ == '__main__':
    model = CRNN(32, 1, 7000, 256)
    model.apply(weights_init)
    input = torch.randn(32, 1, 32, 320)
    pdb.set_trace()
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)