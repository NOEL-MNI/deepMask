import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_3d=nn.BatchNorm3d

def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = BN_3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.outChans = outChans
        self.conv1 = nn.Conv3d(2, outChans, kernel_size=5, padding=2)
        self.bn1 = BN_3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to self.outChans channels
        rpt = x.repeat(1, self.outChans // 2, 1, 1, 1)
        out = self.relu1(torch.add(out, rpt))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = BN_3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = BN_3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.outChans = outChans
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = BN_3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 4 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // self.outChans, self.outChans) # view: reshape, numel: number of elements
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    def __init__(self, n_filters=16, outChans=2, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(n_filters, elu)
        self.down_tr32 = DownTransition(n_filters, 1, elu)
        self.down_tr64 = DownTransition(n_filters*2, 2, elu)
        self.down_tr128 = DownTransition(n_filters*4, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(n_filters*8, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(n_filters*16, n_filters*16, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(n_filters*16, n_filters*8, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(n_filters*8, n_filters*4, 1, elu)
        self.up_tr32 = UpTransition(n_filters*4, n_filters*2, 1, elu)
        self.out_tr = OutputTransition(n_filters*2, outChans, elu, nll)

    # the network topology as described in the diagram
    # in the VNet paper
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


def build_model(args):
    # do not change these variable for inference
    model = VNet(n_filters=6, outChans=2, elu=True, nll=True)
    model = nn.parallel.DataParallel(model, device_ids=args.device_ids)

    if os.path.isfile(args.inference):
        print("=> loading checkpoint '{}'".format(args.inference))
        if args.cuda:
            checkpoint = torch.load(args.inference)
        else:
            checkpoint = torch.load(args.inference, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    else:
        sys.exit("=> no checkpoint found at '{}'".format(args.inference))

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.cuda:
        print('moving the model to GPU')
        model = model.cuda()

    return model