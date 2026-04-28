# Demo Code for Paper:
# [Title]  - "Enhancing Perceptron Constancy for Real world Dynamic Hand Gesture Authentication"
# [Author] -Yufeng Zhang, Xilai Wang, Wenxiong Kang, Wenwei Song
# [Github] - https://github.com/SCUT-BIP-Lab/SSAF.git

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import ceil
from src.utils.graph_construction import Graph
from src.model.AGCN import TCN_GCN_unit
from src.loss.loss import AMSoftmax


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=49):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Recoupling_module(nn.Module):
    def __init__(self,  in_channel_m: int, in_channel_a: int, n_pos: int, reconstruct=True):
        """
        Parameters
        ----------
        in_channel_m: the input channels of motion feature
        in_channel_a: the input channels of appearance feature
        n_pos: the position of the embedding feature
        reconstruct: whether to reconstruct output to input shape (b, c, t, v)
        """
        super(Recoupling_module, self).__init__()
        # self.reduction = reduction
        self.in_channels = in_channel_m
        self.reconstruct = reconstruct

        self.pos = PositionalEncoding(in_channel_a, n_pos)

        self.convQ = nn.Conv2d(in_channel_m, in_channel_m, kernel_size=1)
        self.convK = nn.Conv2d(in_channel_a, in_channel_m, kernel_size=1)
        self.convV = nn.Conv2d(in_channel_a, in_channel_m, kernel_size=1)

        if self.reconstruct:
            self.conv_reconstruct = nn.Sequential(
                nn.Conv2d(in_channel_m, in_channel_m, kernel_size=1),
                nn.BatchNorm2d(in_channel_m)
            )

            nn.init.constant_(self.conv_reconstruct[1].weight, 0)
            nn.init.constant_(self.conv_reconstruct[1].bias, 0)

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor):
        """
        Parameters
        ----------
        x_m: feature from motion stream of shape (B, C, T, V)
            where C is the number of channels in the motion feature, T is the number of frames in the motion feature,
            and V is the number of joints in the motion feature.
        x_a: feature from appearance stream of shape (B, C_A, T_A, H, W)
            where T_A is the number of frames in the appearance feature, and H and W are the height and width of the feature map.
            T_A can be greater than or equal to T, the number of frames in the motion feature.
            If T_A is greater than T, the appearance feature will be split into multiple segments, each segment averaging over T frames.

        Returns
        -------
        Z: the enhanced motion feature of shape (B, C, T, V) after recoupling
        """
        b, c, t, v = x_m.size()  # t会随着层数的加深而降低，因此生理特征也需要时间降维
        assert c == self.in_channels, 'input channel not equal!'
        bt_a, c_a, h, w = x_a.size()
        t_a = bt_a // b
        if t_a >= t:
            len, _ = divmod(t_a, t)
            if not _ == 0:
                len = len + 1
            x_a = x_a.reshape(b, t_a, c_a, h, w)
            x2_split = x_a.split(len, dim=1)
            x2_mean = []
            for i, x in enumerate(x2_split):
                x = x.mean(1, keepdim=True)
                x2_mean.append(x)
            x_a = torch.cat(x2_mean, dim=1)
        else:
            x_a = x_a.reshape(b, t_a, c_a, h, w)
            x_a = x_a.repeat((1, (t // t_a), 1, 1, 1))
        x_a = x_a.permute(0, 1, 3, 4, 2).reshape(b * t, h * w, c_a)
        x_a = self.pos(x_a)
        x_a = x_a.reshape(b, t, h * w, c_a).permute(0, 3, 1, 2).contiguous()  # b, c, t, h*w

        cr = c

        Q = self.convQ(x_m)  # (b, cr, t, v)
        K = self.convK(x_a)  # (b, cr, t, hw)
        V = self.convV(x_a)  # (b, cr, t, hw)
        Q = Q.permute(0, 2, 3, 1).reshape(b*t, v, cr).contiguous()  # bt, v, cr
        K = K.permute(0, 2, 3, 1).reshape(b*t, h*w, cr).contiguous()  # bt, hw, cr
        V = V.permute(0, 2, 3, 1).reshape(b*t, h*w, cr).contiguous()  # bt, hw, cr

        correlation = torch.bmm(Q, K.permute(0, 2, 1))  # (b*t, v, h*w)
        correlation_attention = F.softmax(correlation, dim=-1)
        Z = torch.matmul(correlation_attention, V)  # bt, v, cr
        Z = Z.view(b, t, v, cr).permute(0, 3, 1, 2).contiguous()  # b, cr, t, v
        if self.reconstruct: Z = self.conv_reconstruct(Z)  # b, c, t, v
        Z = Z + x_m
        return Z


class Model_AMNet(torch.nn.Module):
    def __init__(self, conf):
        super(Model_AMNet, self).__init__()
        self.conf = conf
        self.frame_length = int(conf['frames_per_video'])  # the number of frames in each input video
        self.frame_size = 224  # the width (height) of each frame
        self.out_dim = 512  # the dimension of the output features for the two streams
        self.sample_rate = int(conf['sample_rate'])  # the down sampling rate for the Appearance stream, {1, 2, 4, 8} can be chosen
        self.frame_ds = ceil(self.frame_length / self.sample_rate)

        self.cv_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.cv_model.fc = nn.Linear(in_features=512 , out_features=self.out_dim)

        # importance weight generator
        self.weight_conv = nn.Linear(in_features=1024, out_features=2)
        # initialize the weights and biases of the generator to Zero
        nn.init.constant_(self.weight_conv.weight, 0)
        nn.init.constant_(self.weight_conv.bias, 0)

        # Graph
        self.graph = Graph(layout='skeleton', strategy='spatial')
        A = self.graph.A

        self.l1 = TCN_GCN_unit(2, 16, A, residual=False)
        self.l2 = TCN_GCN_unit(16, 32, A, stride=2)
        self.l3 = TCN_GCN_unit(32, 64, A, stride=2)
        self.l4 = TCN_GCN_unit(64, 128, A, stride=2)
        self.fc = nn.Linear(128, self.out_dim)

        self.recocpling = Recoupling_module(128, 512, 7*7)

        # loss
        if self.conf["mode"] in ["train", "train_multidataset"]:
            self.criterian_fusion = AMSoftmax(in_feats=1024, n_classes=int(self.conf["classes_num"]))
            self.criterian_appearance = AMSoftmax(in_feats=512, n_classes=int(self.conf["classes_num"]))
            self.criterian_motion = AMSoftmax(in_feats=512, n_classes=int(self.conf["classes_num"]))


    def dataTailoring(self, x):
        x = x.view((-1, self.frame_length) + x.shape[-3:])
        x = x.permute(0, 2, 1, 3, 4)
        # frame number to be padded
        pad_p = self.frame_ds * self.sample_rate - self.frame_length
        x_ds = F.pad(x, pad=(0, 0, 0, 0, 0, pad_p), mode='replicate').permute(0, 2, 1, 3, 4).contiguous()
        x_ds = x_ds.view((-1, self.sample_rate) + x_ds.shape[-3:])
        x_ds = x_ds[:, 0]  # take the middle frame from every 3 frames
        return x_ds


    def forward(self, data, label=None):
        """
        Parameters
        ----------
        data: data dict containing RGB video and Keypoint data of shape (B, T, C, H, W) and (B, T, V, 2), respectively. where B is the batch size, T is the number of frames,
                  C is the number of channels (3 for RGB), H and W are the height and width of each frame, 2 represents the x and y coordinates of each keypoint,
                  T is the number of frames, and V is the number of keypoints.
        lable: label data of shape (B,), which is None for testing
        Returns
        -------
        fis: A feature dict containing the identity feature for testing and loss for training.
        """

        fis = {}
        data_vid = data['RGB']  # B,T,3,H,W
        data_kpt = data['kpt']  # B,T,V,2
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            data_vid = data_vid.cuda()
            data_kpt = data_kpt.cuda()
            if label is not None:
                label = label.cuda()

        data_vid = data_vid.view(-1, 3, self.frame_size, self.frame_size).contiguous()  # B*T,3,H,W
        data_kpt = data_kpt.permute(0, 3, 1, 2).contiguous()  # B,2,T,V

        # get appearance feature
        data_vid = self.dataTailoring(data_vid)
        x_a = self.cv_model.conv1(data_vid)
        x_a = self.cv_model.bn1(x_a)
        x_a = self.cv_model.relu(x_a)
        x_a = self.cv_model.maxpool(x_a)

        x_a = self.cv_model.layer1(x_a)
        x_a = self.cv_model.layer2(x_a)
        x_a = self.cv_model.layer3(x_a)
        x_a = self.cv_model.layer4(x_a)
        x_a_d = x_a.detach()

        x_a = self.cv_model.avgpool(x_a)
        x_a = torch.flatten(x_a, 1)
        x_a = self.cv_model.fc(x_a)
        x_a = x_a.view(-1, self.frame_ds, self.out_dim)
        x_a = torch.mean(x_a, dim=1, keepdim=False)
        x_a_norm = torch.div(x_a, torch.norm(x_a, p=2, dim=1, keepdim=True).clamp(min=1e-12))

        # get motion feature
        x_k = self.l1(data_kpt)
        x_k = self.l2(x_k)
        x_k = self.l3(x_k)
        x_k = self.l4(x_k)

        # feature recoupling
        x_k_r = self.recocpling(x_k, x_a_d)
        b, c_new, t, v = x_k_r.size()
        x_m = x_k_r.view(b, c_new, t * v)
        x_m = x_m.mean(-1)
        x_m = self.fc(x_m)
        x_m_norm = torch.div(x_m, torch.norm(x_m, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization

        # generate importance weights
        x_weight = torch.cat((x_a, x_m), dim=-1).detach()  # block the gradients
        weight = self.weight_conv(x_weight)
        weight_soft = F.softmax(weight, dim=-1)
        weight_sqrt = weight_soft.sqrt()

        x_a_norm_d, x_m_norm_d = x_a_norm.detach(), x_m_norm.detach()
        x_a_norm_cat = x_a_norm_d * weight_sqrt[:, :1]
        x_m_norm_cat = x_m_norm_d * weight_sqrt[:, 1:]
        id_feature = torch.cat((x_a_norm_cat, x_m_norm_cat), dim=1)

        if label is None:
            fis["id_feature"] = id_feature
            return fis
        else:
            loss_f, _ = self.criterian_fusion(id_feature, label)
            loss_a, _ = self.criterian_appearance(x_a_norm, label)
            loss_m, _ = self.criterian_motion(x_m_norm, label)
            fis["loss"] = loss_f + loss_a + loss_m

            return fis
