# Demo Code for Paper:
# [Title]  - "Enhancing Perceptron Constancy for Real world Dynamic Hand Gesture Authentication"
# [Author] -Yufeng Zhang, Xilai Wang, Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/SSAF.git

import torch
from model.AMNet import Model_AMNet
from utils.loss import AMSoftmax

def feedforward_demo(model, out_dim, is_train=False):

    if is_train:
        # Using AMSoftmax loss function
        # There are 143 identities in the training set of the SCUT-DHGA dataset
        criterian_a = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_m = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_f = AMSoftmax(in_feats=out_dim * 2, n_classes=143)

    data_vid = torch.randn(2, 64, 3, 224, 224)  #batch, frame, channel, h, w
    data_kpt = torch.randn(2, 2, 64, 21)  # batch, channel, frame, joints_num
    data_vid = data_vid.view(-1, 3, 224, 224)  #regard the frame as batch (TSN paradigm)
    feature_fusion, feature_appearance, feature_motion = model(data_vid, data_kpt) # feedforward

    if is_train is False:
        # Use the id_feature to calculate the EER when testing
        return feature_fusion
    else:
        # Use the id_feature, x_r_norm, and x_d_norm to calculate loss when training
        label = torch.randint(0, 143, size=(2,))
        loss_a, _ = criterian_a(feature_appearance, label)
        loss_m, _ = criterian_m(feature_motion, label)
        loss_f, _ = criterian_f(feature_fusion, label)
        return loss_f + 0.5 * loss_a + 0.5 * loss_m

if __name__ == '__main__':
    # there are 64 frames in each dynamic hand gesture video
    frame_length = 64
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512
    # the spatial size of each frame
    frame_size = 224
    # the down sampling rate for the Appearance stream
    sample_rate = 1  # 1/2/4/8, 1 means no down sampling

    model = Model_AMNet(frame_length, frame_size, feature_dim, out_dim, sample_rate)
    # feedforward_test
    id_feature = feedforward_demo(model, out_dim, is_train=False)
    # feedforward_train
    loss = feedforward_demo(model, out_dim, is_train=True)
    print("Demo is finished!")
