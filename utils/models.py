import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels = 9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        
        self.mlp = nn.ModuleList()
        # 激活函数 activation=nn.LeakyReLU(negative_slope=0.1)
        self.activation = activation

        # create mlp
        # mlp_layers 包含所有的每层通道 mlp_layers=[1, 30, 30, 1]
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        # __file__ = uitils/
        # os.path.join
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path) # 加载训练好的模型
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        # 创建批大小为 1 输入通道为 1
        # 
        # t 1 * N
        # x 1 * N * 1 
        x = x[None,...,None]
        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))
        # 最后一层在外面输入，因为没有激活函数
        x = self.mlp[-1](x)
        # 去掉维度为1的维度
        x = x.squeeze()
        return x
    # 初始化的时候就学习好了
    def init_kernel(self, num_channels):
        
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()
            ts.uniform_(-1, 1)
            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0] # 后面是索引 
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]
        gt_values[ts < -1.0 / ( num_channels - 1 ) ] = 0
        gt_values[ts > 1.0 / ( num_channels - 1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        
        nn.Module.__init__(self)
        
        # C
        self.value_layer = ValueLayer(mlp_layers,
                                      activation = activation,
                                      num_channels = dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        
        # B代表的是batch个数，一个索引代表一个文件
        B = int( ( 1 + events[-1,-1] ).item() )
        
        # 连乘 C H W
        # voxel_dimension = (9,180,240)
        num_voxels = int(2 * np.prod(self.dim) * B )
        # 为什么用events[0]
        # vox num_voxels 个 0
        vox = events[0].new_full([num_voxels  ,], fill_value=0)
        #
        C, H, W = self.dim
        
         
        # get values for each channel
        # 转置 方便的取出每一行 b 代表 的 第 几个 索引
        x, y, t, p, b = events.t()
        
        # 每一个索引内的所有t
        # normalizing timestamps
        # 注意切片判断时变成了行向量索引，很方便的对events修改
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()
            
        p = ( p + 1 ) / 2  # maps polarity to 0, 1
        # 2 * B * C * W * H
        
        # W * H * C * 2 * b 选定 b 层 
        # 变为还有 2 * C * W * H
        # W * H * C * p     选定 0 或 1 层
        # 变为还有 C * W * H
        #  当前0通道  x + W * y +  W * H * i_bin
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            # 得到全部的插值
            values = t * self.value_layer.forward( t - i_bin / ( C - 1 ) )
            # draw in voxel grid
            # 计算出了每个的对应位置
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)
        
        # B * 2 * C * H * W
        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes= 101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        
        # 预训练模型
        self.classifier = resnet34(pretrained=pretrained)
        # 
        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer
        input_channels =  2 * voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    # 更改体素小
    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]
        x = F.interpolate(x, size=output_resolution)
        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox


