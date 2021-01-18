import torch.nn as nn
import time
import torch
from torchvision import models
from collections import namedtuple
import warnings
from torch import Tensor
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torch.jit.annotations import Optional, Tuple
from torch.nn import init
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.nn.functional as F


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class VggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, maxpool):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if maxpool:
            self.conv.add_module("max_pool", nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        out = self.conv(x)
        return out


class DeepPCO(nn.Module):
    """
    input : [B,2,72,294], 1016 / 4 + 20 + 20 = 294
    """

    def __init__(self, input_channel, dropout_rate=0.1):  # [B,2,72,294]
        super(DeepPCO, self).__init__()

        block1 = [nn.Conv2d(in_channels=input_channel * 2, out_channels=64, kernel_size=3, padding=1, stride=2),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True),

                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True),

                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True),

                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True),

                  nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                  ]
        self.block1 = nn.Sequential(*block1)  # [B, 128, 5, 19]

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        fc1 = [nn.Linear(512, 512), nn.LeakyReLU(negative_slope=0.2, inplace=True),
               nn.Linear(512, 128), nn.LeakyReLU(negative_slope=0.2, inplace=True),
               nn.Linear(128, 64), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(dropout_rate),
               nn.Linear(64, 16), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout(dropout_rate),
               nn.Linear(16, 3), nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.fc1 = nn.Sequential(*fc1)

    def forward(self, x):
        x = self.block1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


class unDeepVO(nn.Module):
    """
    VGG-based
    """

    def __init__(self, in_channel, rot_channel, n_base_channels=16):
        super(unDeepVO, self).__init__()

        self.vgg_part = nn.ModuleList([  # (10, 5, 128, 416)
            VggBlock(in_channels=in_channel * 2, out_channels=n_base_channels, kernel_size=7, padding=3,
                     maxpool=False),  # 16 torch.Size([10, 16, 128, 416])
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels, kernel_size=7, padding=3,
                     maxpool=True),  # 16 torch.Size([10, 16, 64, 208])
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=False),  # 32 torch.Size([10, 32, 64, 208])
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=True),  # 32 torch.Size([10, 32, 32, 104])
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=False),  # 64 torch.Size([10, 64, 32, 104])
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=True),  # 64 torch.Size([10, 64, 16, 52])
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=False),  # 128 torch.Size([10, 128, 16, 52])
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=True),  # 128 torch.Size([10, 128, 8, 26])
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 8, 26])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=True),  # 256 torch.Size([10, 256, 4, 13])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 4, 13])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=True),  # 256 torch.Size([10, 256, 2, 6])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=False),  # 512 torch.Size([10, 512, 2, 6])
            VggBlock(in_channels=n_base_channels * 32, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=True),  # 512 torch.Size([10, 512, 1, 3])
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 14))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(n_base_channels * 32 * 7 * 14, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, rot_channel)

        self.transl1 = nn.Linear(n_base_channels * 32 * 7 * 14, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, target_frame, reference_frame):
        x = torch.cat([target_frame, reference_frame], dim=1)
        i = 0
        for block in self.vgg_part:
            x = block(x)
            print(i, ": ", x.shape)
            i += 1

        x = self.avgpool(x)
        # print('a ',x.shape)

        out = self.flatten(x)
        # print('b ',out.shape)

        out_rot = 0.01 * self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        out_transl = 0.01 * self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))
        # print('rot: ',out_rot.shape)
        # print('transl', out_transl.shape)
        return out_rot, out_transl


class unDeepVO_ResNet(nn.Module):
    """
    resNet-based
    """

    def __init__(self, n_base_channels=16, pretrained=True, input_images=2):
        super(unDeepVO_ResNet, self).__init__()
        self._first_layer = nn.Conv2d(3 * input_images, 64, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        resnet = models.resnet18(pretrained=pretrained)
        self.resnet_part = nn.Sequential(*list(resnet.children())[1:-2])
        if pretrained:
            loaded_weights = resnet.state_dict()["conv1.weight"]
            loaded_weights = torch.cat([loaded_weights] * input_images, 1) / input_images
            self._first_layer.load_state_dict({"weight": loaded_weights})

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()

        self.rot1 = nn.Linear(512 * 6 * 6, 512)
        self.rot2 = nn.Linear(512, 512)
        self.rot3 = nn.Linear(512, 3)

        self.transl1 = nn.Linear(512 * 6 * 6, 512)
        self.transl2 = nn.Linear(512, 512)
        self.transl3 = nn.Linear(512, 3)

    def forward(self, target_frame, reference_frame):
        x = torch.cat([target_frame, reference_frame], dim=1)
        x = self._first_layer(x)
        x = self.resnet_part(x)
        x = self.avgpool(x)
        out = self.flatten(x)

        out_rot = 0.01 * self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        out_transl = 0.01 * self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))

        return (out_rot, out_transl)


class FlowNetS(nn.Module):
    def __init__(self, input_channels=12, size=(64,256), isDebug=False, batchNorm=True):
        super(FlowNetS, self).__init__()
        self.isDebug = isDebug
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        # self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        # self.conv4_1 = conv(self.batchNorm, 512, 512)
        # self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        # self.conv5_1 = conv(self.batchNorm, 512, 512)
        # self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        # self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        #
        # self.deconv5 = deconv(1024, 512)
        # self.deconv4 = deconv(1026, 256)
        # self.deconv3 = deconv(770, 128)
        # self.deconv2 = deconv(386, 64)
        self.deconv2 = deconv(256, 64)

        # self.predict_flow6 = predict_flow(1024)
        # self.predict_flow5 = predict_flow(1026)
        # self.predict_flow4 = predict_flow(770)
        # self.predict_flow3 = predict_flow(386)
        self.predict_flow3 = predict_flow(256)
        # self.predict_flow2 = predict_flow(194)

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.resize = F.interpolate
        self.size = size
    def forward(self, x):

        x = self.resize(x, size=self.size)
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        # out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow6 = self.predict_flow6(out_conv6)
        # flow6_up = self.upsampled_flow6_to_5(flow6)
        # out_deconv5 = self.deconv5(out_conv6)
        # concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)  # [B, 1026, 2, 8]   16416
        # flow5 = self.predict_flow5(concat5)
        # flow5_up = self.upsampled_flow5_to_4(flow5)
        # out_deconv4 = self.deconv4(concat5)
        #
        # concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        # flow4 = self.predict_flow4(concat4)
        # flow4_up = self.upsampled_flow4_to_3(flow4)
        # out_deconv3 = self.deconv3(concat4)
        # concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)  #  [B, 386, 8, 32] , 98816
        flow3 = self.predict_flow3(out_conv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(out_conv3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        # flow2 = self.predict_flow2(concat2)

        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        #     return flow2
        if self.isDebug:
            return concat2, out_conv1, out_conv2, out_conv3, out_deconv2
        else:
            return concat2

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}
_GoogLeNetOutputs = GoogLeNetOutputs


class PoseNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, in_channel, out_channel=7, aux_logits=True, transform_input=False, init_weights=True,
                 blocks=None):
        super(PoseNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, out_channel)
            self.aux2 = inception_aux_block(528, out_channel)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, out_channel)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout1(x))
        x = self.fc2(self.dropout2(x))
        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)




def test():
    net = FlowNetS(input_channels=4)
    x1 = torch.randn(10, 4, 72, 294)
    y = net(x1)
    print(y[0].shape)
    print(y[1].shape)


if __name__ == "__main__":
    test()
