# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_model.inception3dv3.ipynb (unless otherwise specified).

__all__ = ['inception_learner', 'BasicConv3d', 'PaddedMaxPool3d', 'calc_same_padding', 'Inception3dV3',
           'Inception3dV3Outputs', 'Inception3dA', 'Inception3dB', 'Inception3dC', 'Inception3dD', 'Inception3dE']

# Cell
from collections import namedtuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List
import numpy as np
from torchsummary import summary
from fastai.basics import *

# Cell

@delegates(Learner.__init__)
def inception_learner(dls, loss_func=None, y_range=None, config=None, n_out=None, **kwargs):
    "Build Inception3dV3 learner"

    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = Inception3dV3(num_classes=n_out, init_weights=True)
    learn = Learner(dls, model, loss_func=loss_func, **kwargs)
    return learn

# Cell

class BasicConv3d(nn.Module):
    "Module for Conv3d-BN-relu, with the option for tensorflow-style `same` padding"
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias:bool=False,
        same_padding:bool=False,
        **kwargs: Any
    ) -> None:
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=bias, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)
        self.same_padding = same_padding
        self.kernel = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.padding_size = None

    def forward(self, x: Tensor) -> Tensor:
        if self.same_padding:
            if self.padding_size == None:
                self.padding_size = calc_same_padding(x.shape, self.kernel, self.stride)
            x = F.pad(x, self.padding_size)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class PaddedMaxPool3d(nn.Module):
    "Module for MaxPool3d with optional tensorflow-style `same` padding"

    def __init__(self, same_padding:bool=False, **kwargs: Any) -> None:
        super(PaddedMaxPool3d, self).__init__()
        self.pool = nn.MaxPool3d(**kwargs)
        self.same_padding = same_padding
        self.kernel = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.padding_size = None

    def forward(self, x:Tensor) -> Tensor:
        if self.same_padding:
            if self.padding_size == None:
                self.padding_size = calc_same_padding(x.shape, self.kernel, self.stride)
            x = F.pad(x, self.padding_size)
        x = self.pool(x)
        return x

# Cell

def calc_same_padding(inshape:tuple, kernel:tuple, strides:tuple) -> Tuple[int, int, int]:
    """
    Calculate layer sizes similarly to tensorflow padding='same' for 3d data.
    [left, right, top, bot, front, back] is the order for F.pad.
    Has some kind of performance penalty.
    """
    _, _, in_d, in_h, in_w = inshape
    krl_d, krl_h, krl_w = kernel
    str_d, str_h, str_w = strides

    out_d = np.ceil(float(in_d) / float(str_d))
    out_h = np.ceil(float(in_h) / float(str_h))
    out_w = np.ceil(float(in_w) / float(str_w))

    # depth padding
    if (in_d % str_d == 0):
        pad_along_d = max(krl_d - str_d, 0)
    else:
        pad_along_d = max(krl_d - (in_d % str_d), 0)

    # width padding
    if (in_w % str_w == 0):
        pad_along_w = max(krl_w - str_w, 0)
    else:
        pad_along_w = max(krl_w - (in_w % str_w), 0)

    # height padding
    if (in_h % str_h == 0):
        pad_along_h = max(krl_h - str_h, 0)
    else:
        pad_along_h = max(krl_h - (in_h % str_h), 0)

    pad_front = pad_along_d // 2
    pad_back = pad_along_d - pad_front
    pad_left = pad_along_w // 2
    pad_right = pad_along_w - pad_left
    pad_top = pad_along_h // 2
    pad_bot = pad_along_h - pad_top
    return (pad_left, pad_right, pad_top, pad_bot, pad_front, pad_back)

# Cell

Inception3dV3Outputs = namedtuple('Inception3dV3Outputs', ['logits'])
Inception3dV3Outputs.__annotations__ = {'logits': torch.Tensor}

class Inception3dV3(nn.Module):
    "InceptionV3 for volumetric data with dimensions of 1x105x40x40"

    def __init__(self,
                 num_classes:int=1,
                 inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
                 init_weights: Optional[bool] = None
                ) -> None:
        super(Inception3dV3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv3d, Inception3dA, Inception3dB, Inception3dC, Inception3dD,
                                Inception3dE]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
              'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
              ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 6

        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]

        self.Conv3d_1a_3x2x2 = conv_block(1, 32, kernel_size=(3,2,2), stride=(2,2,2)) # valid pad
        self.Conv3d_2a_3x2x2 = conv_block(32, 32, kernel_size=(3,2,2), stride=(1,1,1)) # valid pad
        self.Conv3d_2b_3x2x2 = conv_block(32, 64, kernel_size=(3,2,2), stride=(1,1,1),
                                          same_padding=True) # same pad
        self.maxpool1 = PaddedMaxPool3d(kernel_size=(2,1,1), stride=(1,1,1), same_padding=True)

        self.Conv3d_3b_1x1x1 = conv_block(64, 80, kernel_size=(1,1,1), stride=(1,1,1),
                                           same_padding=True) # same pad
        self.Conv3d_4a_3x2x2 = conv_block(80, 192, kernel_size=(3,2,2), stride=(1,1,1),
                                          same_padding=True) # same pad
        self.maxpool2 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), same_padding=True)

        # Inception layers
        self.Mixed_5b = inception_a(192, [64,64,96,32])
        self.Mixed_5c = inception_a(256, [64,64,96,64])
        self.Mixed_5d = inception_a(288, [64,64,96,64])

        self.Mixed_6a = inception_b(288, [384,96,96,64])

        self.Mixed_6b = inception_c(768, [192,192,192,192],128)
        self.Mixed_6c = inception_c(768, [192,192,192,192],160)
        self.Mixed_6d = inception_c(768, [192,192,192,192],160)
        self.Mixed_6e = inception_c(768, [192,192,192,192],192)

        self.Mixed_7a = inception_d(768,  [192,320])

        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad(): m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x:Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 1 x 105 x 40 x 40
        x = self.Conv3d_1a_3x2x2(x)
        # N x 32 x 52 x 20 x 20
        x = self.Conv3d_2a_3x2x2(x)
        # N x 32 x 50 x 19 x 19
        x = self.Conv3d_2b_3x2x2(x)
        # N x 64 x 50 x 19 x 19
        x = self.maxpool1(x) # Same padding
        # N x 64 x 50 x 19 x 19
        x = self.Conv3d_3b_1x1x1(x)
        # N x 80 x 50 x 19 x 19
        x = self.Conv3d_4a_3x2x2(x)
        # N x 192 x 50 x 19 x 19
        #x = F.pad(x, calc_same_padding(x.shape, (2,2,2), (2,2,2))) # maxpool paddings need to have stride 1 to work
        x = self.maxpool2(x)
        # TODO calc input size
        x = self.Mixed_5b(x)
        # TODO calc input size
        x = self.Mixed_5c(x)
        # TODO calc input size
        x = self.Mixed_5d(x)
        # TODO calc input size
        x = self.Mixed_6a(x)
        # TODO calc input size
        x = self.Mixed_6b(x)
        # TODO calc input size
        x = self.Mixed_6c(x)
        # TODO calc input size
        x = self.Mixed_6d(x)
        # TODO calc input size
        x = self.Mixed_6e(x)
        # TODO calc input size

        # TODO calc input size
        x = self.Mixed_7a(x)
        # TODO calc input size
        x = self.Mixed_7b(x)
        # TODO calc input size
        x = self.Mixed_7c(x)
        # TODO calc input size
        # Adaptive average pooling
        x = self.avgpool(x)
        # TODO calc input size
        x = self.dropout(x)
        # TODO calc input size
        x = torch.flatten(x, 1)
        # TODO calc input size
        x = self.fc(x)
        # TODO calc input size
        return x

    @torch.jit.unused
    def eager_outputs(self,
                      x:Tensor,
                      #aux:Optional[Tensor]
                     ) -> Inception3dV3Outputs:
        return x
        #if self.training and self.aux_logits:
        #    return Inception3dV3Outputs(x, aux)
        #else: return x #type: ignore[return-value]

    def forward(self, x: Tensor) -> Inception3dV3Outputs:
        x = self._forward(x)
        #aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            #if not aux_defined:
            #    warnings.warn("Scripted Inception3dV3 always results Inception3dV3 Tuple")
            return x#Inception3dV3Outputs(x)#, aux)
        else:
            return self.eager_outputs(x)#, aux)

# Cell

class Inception3dA(nn.Module):
    "First Inception block"
    def __init__(
        self,
        in_channels:int,
        outshapes:int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception3dA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d

        # Ayreys blocks are on the opposite order compared to torchvision
        self.branch_a_1 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(1,1,1), same_padding=True)
        self.branch_a_2 = conv_block(in_channels, outshapes[3], kernel_size=(1,1,1), stride=(1,1,1),
                                     same_padding=True) # Same pad

        # Second bit
        self.branch_b_1 = conv_block(in_channels, 64, kernel_size=(1,1,1), stride=(1,1,1),
                                     same_padding=True) # Same pad
        self.branch_b_2 = conv_block(64, 96, kernel_size=(3,2,2), stride=(1,1,1),
                                     same_padding=True) # Same pad
        self.branch_b_3 = conv_block(96, outshapes[2], kernel_size=(3,2,2), stride=(1,1,1),
                                     same_padding=True) # Same pad

        # Third bit
        self.branch_c_1 = conv_block(in_channels, 48, kernel_size=(1,1,1), stride=(1,1,1),
                                     same_padding=True) # Same pad
        self.branch_c_2 = conv_block(48, outshapes[1], kernel_size=(4,3,3), stride=(1,1,1),
                                     same_padding=True) # Same pad

        # Fourth bit
        self.branch_d_1 = conv_block(in_channels, outshapes[0], kernel_size=(1,1,1), stride=(1,1,1),
                                     same_padding=True) # Same pad

    def _forward(self, x:Tensor) -> List[Tensor]:
        #branch_a = F.pad(x, calc_same_padding(x.shape, (2,2,2), (2,2,2)))
        branch_a = self.branch_a_1(x)
        branch_a = self.branch_a_2(branch_a)

        branch_b = self.branch_b_1(x)
        branch_b = self.branch_b_2(branch_b)
        branch_b = self.branch_b_3(branch_b)

        branch_c = self.branch_c_1(x)
        branch_c = self.branch_c_2(branch_c)

        branch_d = self.branch_d_1(x)

        outputs = [branch_a, branch_b, branch_c, branch_d]
        return outputs

    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# Cell

class Inception3dB(nn.Module):
    "Inception_block 2"
    def __init__(
        self,
        in_channels:int,
        outshapes:List[int],
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception3dB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d

        self.branch_a_1 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)
                                         )
        self.branch_b_1 = conv_block(in_channels, outshapes[3], kernel_size=(1,1,1), stride=(1,1,1),
                                     same_padding=True) # Same pad
        self.branch_b_2 = conv_block(64,outshapes[2], kernel_size=(2,2,2), stride=(1,1,1),
                                     same_padding=True) # Same pad
        self.branch_b_3 = conv_block(96,outshapes[1], kernel_size=(2,2,2), stride=(2,2,2)) # Valid pad

        self.branch_c_1 = conv_block(in_channels, outshapes[0], kernel_size=(2,2,2), stride=(2,2,2)) # Valid pad

    def _forward(self, x:Tensor) -> List[Tensor]:
        branch_a = self.branch_a_1(x)

        branch_b = self.branch_b_1(x)
        branch_b = self.branch_b_2(branch_b)
        branch_b = self.branch_b_3(branch_b)

        branch_c = self.branch_c_1(x)

        outputs = [branch_a, branch_b, branch_c]
        return outputs

    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# Cell

class Inception3dC(nn.Module):
    "Inception block 3"
    def __init__(
        self,
        in_channels:int,
        outshapes:List[int],
        pool:int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception3dC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d

        self.branch_a_1 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(1,1,1), same_padding=True)
        self.branch_a_2 = conv_block(in_channels, outshapes[3], kernel_size=(1,1,1), stride=(1,1,1), same_padding=True) # same pad

        # all same pad
        self.branch_b_1 = conv_block(in_channels, pool, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b_2 = conv_block(pool, pool, kernel_size=(6,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b_3 = conv_block(pool, pool, kernel_size=(1,5,1), stride=(1,1,1), same_padding=True)
        self.branch_b_4 = conv_block(pool, pool, kernel_size=(1,1,5), stride=(1,1,1), same_padding=True)
        self.branch_b_5 = conv_block(pool, outshapes[2], kernel_size=(6,1,1), stride=(1,1,1), same_padding=True)

        self.branch_c_1 = conv_block(in_channels, pool, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_c_2 = conv_block(pool, pool, kernel_size=(6,1,1), stride=(1,1,1), same_padding=True)
        self.branch_c_3 = conv_block(pool, pool, kernel_size=(1,1,5), stride=(1,1,1), same_padding=True)
        self.branch_c_4 = conv_block(pool, outshapes[1], kernel_size=(1,5,1), stride=(1,1,1), same_padding=True)

        self.branch_d_1 = conv_block(in_channels, outshapes[0], kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)


    def _forward(self, x:Tensor) -> List[Tensor]:
        branch_a = self.branch_a_1(x)
        branch_a = self.branch_a_2(branch_a)

        branch_b = self.branch_b_1(x)
        branch_b = self.branch_b_2(branch_b)
        branch_b = self.branch_b_3(branch_b)
        branch_b = self.branch_b_4(branch_b)
        branch_b = self.branch_b_5(branch_b)

        branch_c = self.branch_c_1(x)
        branch_c = self.branch_c_2(branch_c)
        branch_c = self.branch_c_3(branch_c)
        branch_c = self.branch_c_4(branch_c)

        branch_d = self.branch_d_1(x)

        return [branch_a, branch_b, branch_c, branch_d]

    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# Cell

class Inception3dD(nn.Module):
    "Inception block 4"
    def __init__(
        self,
        in_channels:int,
        outshapes:List[int]=None,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception3dD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d

        self.branch_a_1 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), same_padding=True)

        # all same pad
        self.branch_b_1 = conv_block(in_channels, outshapes[1], kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b_2 = conv_block(outshapes[1], outshapes[1], kernel_size=(6,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b_3 = conv_block(outshapes[1], outshapes[1], kernel_size=(1,5,1), stride=(1,1,1), same_padding=True)
        self.branch_b_4 = conv_block(outshapes[1], outshapes[1], kernel_size=(1,1,5), stride=(1,1,1), same_padding=True)
        self.branch_b_5 = conv_block(outshapes[1], outshapes[1], kernel_size=(2,2,2), stride=(2,2,2), same_padding=True)

        self.branch_c_1 = conv_block(in_channels, outshapes[1], kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_c_2 = conv_block(outshapes[1], outshapes[0], kernel_size=(2,2,2), stride=(2,2,2), same_padding=True)

    def _forward(self, x:Tensor) -> List[Tensor]:
        branch_a = self.branch_a_1(x)

        branch_b = self.branch_b_1(x)
        branch_b = self.branch_b_2(branch_b)
        branch_b = self.branch_b_3(branch_b)
        branch_b = self.branch_b_4(branch_b)
        branch_b = self.branch_b_5(branch_b)

        branch_c = self.branch_c_1(x)
        branch_c = self.branch_c_2(branch_c)
        return [branch_a, branch_b, branch_c]

    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# Cell

class Inception3dE(nn.Module):
    "Inception block 5"
    def __init__(
        self,
        in_channels:int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception3dE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d

        self.branch_a_1 = PaddedMaxPool3d(kernel_size=(2,2,2), stride=(1,1,1), same_padding=True)
        self.branch_a_2 = conv_block(in_channels, 192, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)

        self.branch_b_1 = conv_block(in_channels, 448, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b_2 = conv_block(448, 384, kernel_size=(2,2,2), stride=(1,1,1), same_padding=True)
        self.branch_b1_1 = conv_block(384, 256, kernel_size=(3,1,1), stride=(1,1,1), same_padding=True)
        self.branch_b2_1 = conv_block(384, 256, kernel_size=(1,1,2), stride=(1,1,1), same_padding=True)
        self.branch_b3_1 = conv_block(384, 256, kernel_size=(1,2,1), stride=(1,1,1), same_padding=True)

        self.branch_c_1 = conv_block(in_channels, 384, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)
        self.branch_c1_1 = conv_block(384, 256, kernel_size=(3,1,1), stride=(1,1,1), same_padding=True)
        self.branch_c2_1 = conv_block(384, 256, kernel_size=(1,1,2), stride=(1,1,1), same_padding=True)
        self.branch_c3_1 = conv_block(384, 256, kernel_size=(1,2,1), stride=(1,1,1), same_padding=True)

        self.branch_d_1 = conv_block(in_channels, 320, kernel_size=(1,1,1), stride=(1,1,1), same_padding=True)

    def _forward(self, x:Tensor) -> List[Tensor]:
        branch_a = self.branch_a_1(x)
        branch_a = self.branch_a_2(branch_a)

        branch_b = self.branch_b_1(x)
        branch_b = self.branch_b_2(branch_b)
        branch_b1 = self.branch_b1_1(branch_b)
        branch_b2 = self.branch_b2_1(branch_b)
        branch_b3 = self.branch_b3_1(branch_b)

        branch_c = self.branch_c_1(x)
        branch_c1 = self.branch_c1_1(branch_c)
        branch_c2 = self.branch_c2_1(branch_c)
        branch_c3 = self.branch_c3_1(branch_c)

        branch_d = self.branch_d_1(x)
        return [branch_a, branch_b1, branch_b2, branch_b3, branch_c1, branch_c2, branch_c3, branch_d]

    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
