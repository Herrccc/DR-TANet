import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
from attention import Temporal_Attention
from util import *

__all__ = ['get_encoder', 'get_attentionmodule', 'get_decoder']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

def block_function_factory(conv,norm,relu=None):
    def block_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return block_function

def do_efficient_fwd(block_f,x,efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f,x)
    else:
        return block_f(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_c,out_c,stride=1,downsample = None,efficient=True,use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(in_c,out_c,stride)
        self.bn1 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c,out_c)
        self.bn2 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        block_f1 = block_function_factory(self.conv1,self.bn1,self.relu)
        block_f2 = block_function_factory(self.conv2,self.bn2)

        out = do_efficient_fwd(block_f1,x,self.efficient)
        out = do_efficient_fwd(block_f2,out,self.efficient)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out,out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = block_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = block_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = block_function_factory(self.conv3, self.bn3)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out, out

class ResNet(nn.Module):

    def __init__(self, block, layers, efficient=False, use_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_bn = use_bn
        self.efficient = efficient

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x:x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [skip]
        return features

class AttentionModule(nn.Module):

    def __init__(self, local_kernel_size = 1, stride = 1, padding = 0, groups = 1,
                 drtam = False, refinement = False, channels = [64,128,256,512]):
        super(AttentionModule, self).__init__()

        if not drtam:
            self.attention_layer1 = Temporal_Attention(channels[0], channels[0], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer2 = Temporal_Attention(channels[1], channels[1], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer3 = Temporal_Attention(channels[2], channels[2], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer4 = Temporal_Attention(channels[3], channels[3], local_kernel_size, stride, padding, groups, refinement=refinement)
        else:
            self.attention_layer1 = Temporal_Attention(channels[0], channels[0], 7, 1, 3, groups, refinement=refinement)
            self.attention_layer2 = Temporal_Attention(channels[1], channels[1], 5, 1, 2, groups, refinement=refinement)
            self.attention_layer3 = Temporal_Attention(channels[2], channels[2], 3, 1, 1, groups, refinement=refinement)
            self.attention_layer4 = Temporal_Attention(channels[3], channels[3], 1, 1, 0, groups, refinement=refinement)


        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1]*2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2]*2, channels[3], stride=2)

    def forward(self, features):

        features_t0, features_t1 = features[:4], features[4:]

        fm1 = torch.cat([features_t0[0],features_t1[0]], 1)
        attention1 = self.attention_layer1(fm1)
        fm2 = torch.cat([features_t0[1], features_t1[1]], 1)
        attention2 = self.attention_layer2(fm2)
        fm3 = torch.cat([features_t0[2], features_t1[2]], 1)
        attention3 = self.attention_layer3(fm3)
        fm4 = torch.cat([features_t0[3], features_t1[3]], 1)
        attention4 = self.attention_layer4(fm4)

        downsampled_attention1 = self.downsample1(attention1)
        cat_attention2 = torch.cat([downsampled_attention1,attention2], 1)
        downsampled_attention2 = self.downsample2(cat_attention2)
        cat_attention3 = torch.cat([downsampled_attention2,attention3], 1)
        downsampled_attention3 = self.downsample3(cat_attention3)
        final_attention_map = torch.cat([downsampled_attention3,attention4], 1)
        
        features_map = [final_attention_map,attention4,attention3,attention2,attention1]
        return features_map


class Decoder(nn.Module):
    
    def __init__(self,channels=[64,128,256,512]):
        super(Decoder, self).__init__()
        self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3], num_maps_out=channels[3])
        self.upsample2 = Upsample(num_maps_in=channels[2]*2, skip_maps_in=channels[2], num_maps_out=channels[2])
        self.upsample3 = Upsample(num_maps_in=channels[1]*2, skip_maps_in=channels[1], num_maps_out=channels[1])
        self.upsample4 = Upsample(num_maps_in=channels[0]*2, skip_maps_in=channels[0], num_maps_out=channels[0])

    def forward(self, feutures_map):
        
        x = feutures_map[0]
        x = self.upsample1(x, feutures_map[1])
        x = self.upsample2(x, feutures_map[2])
        x = self.upsample3(x, feutures_map[3])
        x = self.upsample4(x, feutures_map[4])
        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    channels = [64,128,256,512]
    return model,channels

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    channels = [64, 128, 256, 512]
    return model,channels


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    channels = [256, 512, 1024, 2048]
    return model,channels

def get_encoder(arch,pretrained=True):
    if arch == 'resnet18':
        return resnet18(pretrained)
    elif arch == 'resnet34':
        return resnet34(pretrained)
    elif arch == 'resnet50':
        return resnet50(pretrained)
    else:
        print('Given the invalid architecture for ResNet...')
        exit(-1)

def get_attentionmodule(local_kernel_size = 1, stride = 1, padding = 0, groups = 1, drtam = False, refinement = False, channels=[64,128,256,512]):
    return AttentionModule(local_kernel_size=local_kernel_size,stride=stride, padding=padding, groups=groups,
                           drtam=drtam, refinement=refinement, channels=channels)
def get_decoder(channels=[64,128,256,512]):
    return Decoder(channels=channels)

