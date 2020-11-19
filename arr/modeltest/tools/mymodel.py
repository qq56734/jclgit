import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import models

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
    
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

#被替换的3*3卷积
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num = 1, nums_block_list = [3, 4, 6, 3], strides_list = [1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 128, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(128, 256, 256, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(256, 256, 512, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[3], stride=strides_list[3])
     
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(2048, class_num)
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up4 = double_conv(1024 + 512 , 512)
        self.dconv_up3 = double_conv(512 + 256 , 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1)
        
 
        
        
        
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)
      
 
    def forward(self, x):
        conv1 = self.basic_conv(x)
        conv2 = self.stage_1(conv1)
        conv3 = self.stage_2(conv2)
        conv4 = self.stage_3(conv3)
        fea = self.stage_4(conv4)
        
        fea = self.upsample(fea)
        fea = torch.cat([fea, conv4], dim=1)
        fea = self.dconv_up4(fea)
        
        
        fea = self.upsample(fea)
        fea = torch.cat([fea, conv3], dim=1)
        fea = self.dconv_up3(fea)
     
        fea = self.upsample(fea)  
        fea = torch.cat([fea, conv2], dim=1)       
        fea = self.dconv_up2(fea)
        
        fea = self.upsample(fea)        
        fea = self.dconv_up1(fea)
        
        
        
       
        out = self.conv_last(fea)
        

        
        return out
        
class SKv2(nn.Module):
    def __init__(self, class_num = 1, nums_block_list = [3, 4, 6, 3], strides_list = [1, 2, 2, 2]):
        super(SKv2, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 128, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(128, 256, 256, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(256, 256, 512, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[3], stride=strides_list[3])
     
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(2048, class_num)
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up4 = double_conv(1024 + 512 , 512)
        self.dconv_up3 = double_conv(512 + 256 , 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1)
        
 
        
        
        
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)
      
 
    def forward(self, x):
        conv1 = self.basic_conv(x)
        conv2 = self.stage_1(conv1)
        conv3 = self.stage_2(conv2)
        conv4 = self.stage_3(conv3)
        fea = self.stage_4(conv4)
        
        fea = self.upsample(fea)
        fea = torch.cat([fea, conv4], dim=1)
        fea = self.dconv_up4(fea)
        
        
        fea = self.upsample(fea)
        fea = torch.cat([fea, conv3], dim=1)
        fea = self.dconv_up3(fea)
     
        fea = self.upsample(fea)  
        fea = torch.cat([fea, conv2], dim=1)       
        fea = self.dconv_up2(fea)
        
        fea = self.upsample(fea)        
        fea = self.dconv_up1(fea)
        
        
        
       
        out = self.conv_last(fea)
        

        
        return out
    

    
    
    
class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale, mode='deconv'):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = None
        if mode == 'deconv':
            layers = self.make_deconv_layers(in_features, up_scale)
        elif mode == 'pixel_shuffle':
            layers = self.make_pixel_shuffle_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2))
            in_features = out_features
        return layers

    def make_pixel_shuffle_layers(self, in_features, up_scale):
        layers = []
        for i in range(up_scale):
            kernel_size = 2 ** (i + 1)
            out_features = self.compute_out_features(i, up_scale)
            in_features = int(in_features / (self.up_factor ** 2))
            layers.append(nn.PixelShuffle(self.up_factor))
            layers.append(nn.Conv2d(in_features, out_features, 1))
            if i < up_scale:
                layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)    
    

    
class SKv3(nn.Module):
    def __init__(self, class_num = 1, nums_block_list = [3, 6, 12, 3], strides_list = [1, 2, 2, 2]):
        super(SKv3, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 128, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(128, 256, 256, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(256, 256, 512, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[3], stride=strides_list[3])
     
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(2048, class_num)
        
        
    
        
        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(1024, 4)
        
        self.block_cat = nn.Conv2d(5, 1, kernel_size=1)
        
 
    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]
    
    


        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)
      
 
    def forward(self, x):
        conv1 = self.basic_conv(x)
        conv2 = self.stage_1(conv1)
        conv3 = self.stage_2(conv2)
        conv4 = self.stage_3(conv3)
        conv5 = self.stage_4(conv4)


        
        # upsampling blocks
        height, width = x.shape[-2:]
        slice_shape = (height, width)
        out_1 = self.slice(self.up_block_1(conv1), slice_shape)
        out_2 = self.slice(self.up_block_2(conv2), slice_shape)
        out_3 = self.slice(self.up_block_3(conv3), slice_shape)
        out_4 = self.slice(self.up_block_4(conv4), slice_shape)
        out_5 = self.slice(self.up_block_5(conv5), slice_shape)
        
        
        results = [out_1, out_2, out_3, out_4, out_5]
        # print(out_1.shape)
        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        
        
        return results
        
        

        



        
     
       
        
def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        # print("Applying custom weight initialization for nn.Conv2d layer...")
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)
        if m.weight.data.shape == torch.Size([1, 6, 1, 1]):
            torch.nn.init.constant_(m.weight, 0.2) # for fuse conv
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=1, stride=1, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, padding=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))
        # double check the norm1 comment if necessary and put norm after conv2

    def forward(self, x):
        x1, x2 = x
        # maybe I should put here a RELU
        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features



class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True  # XXX Unused
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = True
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class DexiNet(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(DexiNet, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        self.pre_dense_2 = SingleConvBlock(128, 256, 2,
                                           use_bs=False)  # by me, for left skip block4
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5_0 = SingleConvBlock(256, 512, 2,
                                             use_bs=False)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = nn.Conv2d(6, 1, kernel_size=1)

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]

    def forward(self, x):
        assert x.ndim == 4, x.shape

        # Block 1
        # print(f"x shape           : {x.shape}")
        block_1 = self.block_1(x)
        # print(f"block_1 shape     : {block_1.shape}")
        block_1_side = self.side_1(block_1)
        # print(f"block_1_side shape: {block_1_side.shape}")

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_4_pre_dense_256 = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(
            block_4_pre_dense_256 + block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(
            block_5_pre_dense_512 + block_4_down)
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side
#        block_5_side = self.side_5(block_5_add)

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
#        block_5_pre_dense_256 = self.pre_dense_6(block_5_add) # if error uncomment
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        height, width = x.shape[-2:]
        slice_shape = (height, width)
        out_1 = self.slice(self.up_block_1(block_1), slice_shape)
        out_2 = self.slice(self.up_block_2(block_2), slice_shape)
        out_3 = self.slice(self.up_block_3(block_3), slice_shape)
        out_4 = self.slice(self.up_block_4(block_4), slice_shape)
        out_5 = self.slice(self.up_block_5(block_5), slice_shape)
        out_6 = self.slice(self.up_block_6(block_6), slice_shape)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]
        # print(out_1.shape)

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results
        
    