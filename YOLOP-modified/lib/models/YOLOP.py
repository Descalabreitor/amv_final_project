import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized


# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[25],   #Da_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder

[ 16, Conv, [256, 128, 3, 1]],   #17
[ -1, Upsample, [None, 2, 'nearest']],  #18
[ -1, BottleneckCSP, [128, 64, 1, False]],  #19
[ -1, Conv, [64, 32, 3, 1]],    #20
[ -1, Upsample, [None, 2, 'nearest']],  #21
[ -1, Conv, [32, 16, 3, 1]],    #22
[ -1, BottleneckCSP, [16, 8, 1, False]],    #23
[ -1, Upsample, [None, 2, 'nearest']],  #24
[ -1, Conv, [8, 2, 3, 1]], #25 Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],   #26
[ -1, Upsample, [None, 2, 'nearest']],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 2, 3, 1]] #34 Lane line segmentation head

]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.seg_out_idx = block_cfg[0][0:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
        return out
            

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    dring_area_seg, lane_line_seg  = model_out
    Da_fmap, LL_fmap = SAD_out
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
 
