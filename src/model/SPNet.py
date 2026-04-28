from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.utils.initial import kaiming_init, constant_init
from src.loss.loss import JointsMSELoss


BN_MOMENTUM = 0.1

def load_config(config_file='config.yaml'):
    with open(config_file, 'r', encoding="UTF-8") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            # config = Dict2Obj(config)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config

class ConvModule(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            if padding_mode=='zero':
                self.padding_layer=nn.ZeroPad2d(padding)
            elif padding_mode=='reflect':
                self.padding_layer=nn.ReflectionPad2d(padding)
            elif padding_mode=='replicate':
                self.padding_layer==nn.ReplicationPad2d(padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding

        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=conv_padding, dilation=dilation,groups=groups, bias=bias)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if (norm_cfg['type']=='BN'):
                self.norm=nn.BatchNorm2d(norm_channels, momentum=BN_MOMENTUM)
            elif(norm_cfg['type']=='IBN'):
                self.norm=nn.InstanceNorm2d(norm_channels, momentum=BN_MOMENTUM)
        if self.with_activation:
            if act_cfg['type']=='ReLU':
                self.activate=nn.ReLU()
            elif act_cfg['type']=='Sigmoid':
                self.activate=nn.Sigmoid()

        self.init_weights()


    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,x) :
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x

####################################################################################
###############################_____depthwiseconv_____##############################

class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 dw_norm_cfg: Union[Dict, str] = 'default',
                 dw_act_cfg: Union[Dict, str] = 'default',
                 pw_norm_cfg: Union[Dict, str] = 'default',
                 pw_act_cfg: Union[Dict, str] = 'default',
                 **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.global_avgpool = GlobalAvgPool2d()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=-1, keepdim=True)
        return x.mean(dim=-2, keepdim=True)


class CrossResolutionWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = []

        for s in x[:-1]:
            ssize = s.size()[-2:]
            kernel = (int(ssize[0] / mini_size[0]), int(ssize[1] / mini_size[1]))
            assert kernel[0] == kernel[1]
            stride = (
                int((ssize[0] - kernel[0]) / (mini_size[0] - 1)), int((ssize[1] - kernel[1]) / (mini_size[1] - 1)))
            out.append(F.avg_pool2d(s, kernel, stride))
        out = out + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):
    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]

        return out


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        x = self.conv1(x)  # n, 32, 112, 112
        x1, x2 = x.chunk(2, dim=1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)  # n, 32, 56, 56
        x2 = self.linear_conv(x2)  # n, 16, 56, 56

        out = torch.cat((self.branch1(x1), x2), dim=1)

        out = channel_shuffle(out, 2)  # n, 32, 56, 56

        return out, x


class IterativeHead(nn.Module):

    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class ShuffleUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class LiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'),
                with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type='ReLU'),
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels[j],
                                out_channels=in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.Upsample(
                                scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[i])))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        return self.conv(self.up(x))


class SegmentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.up1 = Up(33, 32)
        self.up2 = Up(64, 16)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.up1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.up2(x)
        out = self.conv(x)

        return out


class Model_SPNet(nn.Module):
    def __init__(self,
                 conf,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super().__init__()
        cfg = load_config(r'/home/zyf/code/dhg/conf/Video-Skeleton/SPNet/litehrnet18_hand_192_192_DHGA_all.yaml')
        self.conf = conf
        self.frame_size = int(conf["input_frame_size"])
        self.extra = cfg['MODEL']['EXTRA']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.with_head = True

        self.stem = Stem(
            in_channels,
            stem_channels=32,
            out_channels=32,
            expand_ratio=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)


        self.stage0_cfg = self.extra['STAGE0']
        num_channels = self.stage0_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition0 = self._make_transition_layer([self.stem.out_channels], num_channels)
        self.stage0, num_pre_chanels = self._make_stage(self.stage0_cfg, num_channels)

        self.stage1_cfg = self.extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(num_pre_chanels, num_channels)
        self.stage1, num_pre_chanels = self._make_stage(self.stage1_cfg, num_channels)

        self.stage2_cfg = self.extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(num_pre_chanels, num_channels)
        self.stage2, num_pre_chanels = self._make_stage(self.stage2_cfg, num_channels)

        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_pre_chanels,
                norm_cfg=self.norm_cfg,
            )

        self.ske_head = nn.Conv2d(
            in_channels=num_pre_chanels[0],
            out_channels=21,
            kernel_size=1,
            stride=1,
            padding=0 # 1 if self.extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.seg_head = SegmentHead()

        if self.conf["mode"] in ["train", "train_multidataset"]:
            self.seg_criterian = nn.MSELoss(reduction='none')
            self.ske_criterian = JointsMSELoss(use_target_weight=False)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['NUM_MODULES']
        num_branches = stages_spec['NUM_BRANCHES']
        num_blocks = stages_spec['NUM_BLOCK']
        reduce_ratio = stages_spec['REDUCE_RATIO']
        with_fuse = stages_spec['WITH_FUSE']
        module_type = stages_spec['MODULE_TYPE']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
        in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def forward(self, data, is_train=False):
        fis = {}
        if is_train:
            data_rgb = data['RGB']  # Gray or RGB
            data_mask = data['mask']
            data_kpt = data['kpt_heatmap']
        else:
            data_rgb = data['RGB']

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data_rgb = data_rgb.cuda()
            data_rgb = data_rgb.view(-1, 3, self.frame_size, self.frame_size).contiguous()
            if is_train:
                data_mask = data_mask.cuda()
                data_kpt = data_kpt.cuda()
                data_mask = data_mask.view(-1, 3, self.frame_size, self.frame_size).contiguous()
                data_mask = data_mask[:, 0, :, :]
                data_kpt = data_kpt.view(-1, 21, 56, 56)

        """Forward function."""
        x_stem, x_high_res = self.stem(data_rgb)

        y_list = [x_stem]
        x_list = []
        for j in range(self.stage0_cfg['NUM_BRANCHES']):
            if self.transition0[j]:
                if j < len(y_list):
                    x_list.append(self.transition0[j](y_list[j]))
                else:
                    x_list.append(self.transition0[j](y_list[-1]))

        y_list = self.stage0(x_list)

        x_list = []
        for j in range(self.stage1_cfg['NUM_BRANCHES']):
            if self.transition1[j]:
                if j < len(y_list):
                    x_list.append(self.transition1[j](y_list[j]))
                else:
                    x_list.append(self.transition1[j](y_list[-1]))
            else:
                x_list.append(y_list[j])

        y_list = self.stage1(x_list)

        x_list = []
        for j in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition2[j]:
                if j < len(y_list):
                    x_list.append(self.transition2[j](y_list[j]))
                else:
                    x_list.append(self.transition2[j](y_list[-1]))
            else:
                x_list.append(y_list[j])

        y_list = self.stage2(x_list)

        if self.with_head:
            x = self.head_layer(y_list)

        skeleton = self.ske_head(x[0])  # b, 21, 56, 56
        segment = self.seg_head(skeleton, x_stem, x_high_res)  # b, c, 56, 56

        if is_train:
            seg_output = torch.flatten(segment.view(-1, *segment.shape[-2:]).contiguous(), 1)
            seg_target = torch.flatten(data_mask.view(-1, *data_mask.shape[-2:]).contiguous(), 1)
            seg_loss = self.seg_criterian(seg_output, seg_target)
            seg_loss = torch.mean(torch.mean(seg_loss, dim=1))

            ske_loss = self.ske_criterian(skeleton, data_kpt)

            fis['loss'] = seg_loss + ske_loss
            fis['skeleton'] = skeleton
            fis['segment'] = segment

        else:
            fis['skeleton'] = skeleton
            fis['segment'] = segment

        return fis
