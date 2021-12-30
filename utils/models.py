import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation = nn.ReLU(), num_channels = 9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        
        self.mlp = nn.ModuleList()
        # 激活函数 activation=nn.LeakyReLU(negative_slope=0.1)
        self.activation = activation
        # create mlp
        # mlp_layers 包含所有的每层通道 mlp_layers = [1, 30, 30, 1]
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels
        # init with trilinear kernel
        # os.path.join
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path) # 加载训练好的模型
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # x 为跟当前通道的差值 把它当作 |t|  delta t 为 通道间的距离
        # 这样对于任何通道，就变成了 |t| 有多大的区别
        
        # create sample of batchsize 1 and input channels 1
        # 每一行叫做样本，每一列叫做特征
        # t 1 * N
        # x 1 * N * 1 
        # 1个批次，n个样本，通道为1
        # 多层感知机输入数为通道
        x = x[None,...,None]
        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))
        x = self.mlp[-1](x) # 最后一层在外面输入，因为没有激活函数
        # 去掉维度为 1 的维度
        x = x.squeeze()
        return x
    
    def init_kernel(self, num_channels):    
        ts = torch.zeros((1, 2000))
        # self.parameters() ModuleList对应的参数
        optim = torch.optim.Adam(self.parameters() , lr = 1e-2)
        torch.manual_seed(1) # 保证随机数相同
        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()
            ts.uniform_(-1, 1)
            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)
            # pred
            # ts 同样为 跟 当前通道的插值
            values = self.forward(ts)
            # optimize
            loss = (values - gt_values).pow(2).sum()
            loss.backward()
            optim.step()
    
    def trilinear_kernel(self, ts, num_channels):
        # max(0,1 - |t| / delta t)  
        gt_values = torch.zeros_like(ts)
        gt_values[ts > 0] = ( 1 - (num_channels - 1) * ts)[ts > 0] # 后面是索引 
        gt_values[ts < 0] = ( (num_channels - 1) * ts + 1)[ts < 0]
        # 
        gt_values[ts < -1.0 / ( num_channels - 1 ) ] = 0
        gt_values[ts > 1.0 / ( num_channels - 1)] = 0
        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers = [1, 100, 100, 1],
                 activation = nn.LeakyReLU(negative_slope = 0.1)):
        nn.Module.__init__(self)
        # C H W
        self.value_layer = ValueLayer(mlp_layers,activation = activation,num_channels = dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        
        # B 代表 event bins
        # batch里有多个
        B = int( ( 1 + events[-1,-1] ).item() )
        # 连乘 C H W
        # voxel_dimension = ( 9 , 180 , 240 )
        num_voxels = int(2 * np.prod(self.dim) * B )
        # 为什么用events[0] ???
        vox = events[0].new_full([num_voxels  ,], fill_value = 0)# vox num_voxels 个 0
        C, H, W = self.dim
        
        # 转置 方便的取出每一行
        x, y, t, p, b = events.t()
        
        # normalizing timestamps 0 ~ 1
        # 注意切片判断时变成行向量索引，方便对events修改
        # 每一个event bins内的所有t
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()
        p = ( p + 1 ) / 2  # maps polarity to 0, 1
        
        # 实际上先按行存，再按列存，然后按通道数（时间）存
        # 组成 C * W * H
        # 然后分两个极性存  2 * C * W * H
        # 此时是一个event bin 对应的
        # 通过 x y p b 确定每个 event bin 的 x y p 所对应的 0 通道
        # 这样就能从 0 通道到当前 event bin 的 其他通道
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b # 注意这里计算了所有的
        # 一个 t 在每个通道上都有影响
        # get values for each channel
        for i_bin in range(C):
            # t * 插值结果
            values = t * self.value_layer.forward( t - i_bin / ( C - 1 ) )
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate = True)
        # 根据存的顺序反推
        # 先分 B ，再分极性，再分通道数，再分 H 和 W
        # B * 2 * C * H * W
        vox = vox.view(-1, 2, C, H, W)
        # 拼接少一维度 B * (2*C) * H * W
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension = (9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension = (224, 224),  # dimension of crop before it goes into classifier
                 num_classes = 101,
                 mlp_layers = [1, 30, 30, 1],
                 activation = nn.LeakyReLU(negative_slope = 0.1),
                 pretrained = True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        # 预训练模型
        self.classifier = resnet34(pretrained=pretrained)
        self.crop_dimension = crop_dimension
        # replace fc layer and first convolutional layer
        input_channels =  2 * voxel_dimension[0]
        # 输入通道数为 2 * C
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    # 更改体素小
    def crop_and_resize_to_resolution(self, x, output_resolution = (224, 224)):
        # B * (2*C) * H * W
        # 此时tensor变成多维
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            # // 整数除法
            h = W // 2 # 中心值 
            x = x[:, :, :, h - H // 2:h + H // 2]
        # 将 H W 较小者 裁减
        # 使得 H = W = R     
        # B 2*C R R
        x = F.interpolate(x, size = output_resolution)
        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        # 2 * C 通道
        pred = self.classifier.forward(vox_cropped)
        return pred, vox


