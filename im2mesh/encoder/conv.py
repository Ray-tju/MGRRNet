import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from im2mesh.common import normalize_imagenet
from im2mesh.encoder import batchnet as bnet
import torch
import math
from im2mesh.encoder import attention_1d

import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from PIL import Image
import numpy as np
import torch

import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float().cuda()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.7
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
    # Save colored heatmap
    path_to_file = os.path.join('results', file_name + '_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('results', file_name + '_Cam_On_Image_hot.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('results', file_name + '_Cam_Grayscale.png')
    print(path_to_file)
    save_image(activation_map, path_to_file)


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.extracted_layers = None
        self.features = models.resnet18(pretrained=True)
        self.features_head1 = models.resnet18(pretrained=True)
        self.features_head2 = models.resnet18(pretrained=True)
        self.vis_conv_feature = True
        self.one_hot_vis_feature = True
        self.target_layer = 1
        # self.features = bnet.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.features.avgpool = nn.Sequential()

        self.features_head1.fc = nn.Sequential()
        self.features_head1.layer4 = nn.Sequential()
        self.features_head1.avgpool = nn.Sequential()

        self.features_head2.fc = nn.Sequential()
        self.features_head2.avgpool = nn.Sequential()

        self.max1 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.max2 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.max3 = nn.MaxPool2d(kernel_size=7, stride=7)

        self.attention = attention_1d.SELayer(768, 8)
        self.attention1 = attention_1d.SELayer(256, 8)

        self.conv_block1 = nn.Sequential(
            BasicConv(256, 1024, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(1024, 512, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(512, 1024, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(1024, 512, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(512, 1024, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(1024, 512, kernel_size=3, stride=1, padding=1, relu=True)
        )
        # self.gru = nn.GRU(256, 256, 3, batch_first=True)
        # self.gru = nn.GRU(256, 512, 3, batch_first=True)
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
            self.fc1 = nn.Linear(512, c_dim)
            self.fc2 = nn.Linear(512, c_dim)
            # self.fc_concat = nn.Linear(256, c_dim)
            self.fc_concat1 = nn.Linear(768, 512)
            self.fc_concat2 = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.features._modules.items():
            # print(module_pos)
            x = module(x)  # Forward
            # if int(module_pos) == self.target_layer:
            #     conv_output = x  # Save the convolution output on that layer

        for module_pos, module in self.conv_block3._modules.items():
            x = module(x.view(x.size(0), -1, 7, 7))  # Forward
            # print('conv_block3', module_pos)
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        x = self.max3(x)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # print(x.size())
        # Forward pass on the classifier
        x = self.fc(x)
        return conv_output, x

    def forward(self, x, x1, x2):
        if self.normalize:
            x0 = normalize_imagenet(x)
            x1 = normalize_imagenet(x1)
            x2 = normalize_imagenet(x2)
        x = self.features(x0)
        x = self.conv_block3(x.view(x.size(0), -1, 7, 7))
        x = self.max3(x)
        x = x.view(x.size(0), -1)
        x_ori = self.fc(x)

        xfg1 = self.features_head1(x1)
        xfg1 = self.conv_block1(xfg1.view(xfg1.size(0), -1, 14, 14))
        xfg1 = self.max1(xfg1)
        xfg1 = xfg1.view(xfg1.size(0), -1)
        xfg1 = self.fc1(xfg1)

        xfg2 = self.features_head2(x2)
        xfg2 = self.conv_block2(xfg2.view(xfg2.size(0), -1, 7, 7))
        xfg2 = self.max2(xfg2)
        xfg2 = xfg2.view(xfg2.size(0), -1)
        xfg2 = self.fc2(xfg2)

        # xfg3 = self.conv_block3(xfg3)
        # xfg3 = self.max3(xfg3)
        # xfg3 = xfg3.view(xfg3.size(0), -1)
        # xfg3 = self.fc3(xfg3)

        # concat_input = torch.cat(
        #     (xfg1.view(x0.size(0), -1, 1), xfg2.view(x0.size(0), -1, 1), (x0.view(x0.size(0), -1, 1))), -1)
        # gru_input = concat_input.transpose(-2, -1)

        # h_0 = torch.randn(6, x.size(0), 256).cuda()
        # h_0 = torch.randn(3, x.size(0), 512).cuda()
        # output, h1 = self.gru(gru_input, h_0)
        # concat_out = self.fc_concat(self.attention(output[:, -1, :].unsqueeze(-1)).squeeze())
        concat_input = torch.cat((xfg1, xfg2, x_ori), dim=-1)
        # print(concat_input.size())
        concat_out = self.attention(concat_input.unsqueeze(-1))
        # print(concat_out.size())
        concat_out = self.fc_concat1(concat_out.squeeze(-1))
        concat_out = self.fc_concat2(concat_out)
        concat_out = self.attention1(concat_out.unsqueeze(-1))

        # if self.vis_conv_feature == True:
        #     outputs = {}
        #     dst = './feautures'
        #     therd_size = 224
        #     for name, module in self.features._modules.items():
        #         if "fc" in name:
        #             x0 = x0.view(x0.size(0), -1)
        #
        #         x0 = module(x0)
        #         # print(name)
        #         if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
        #             x0 = x0.view(x0.size(0), -1)
        #             x0=self.fc(x)
        #             outputs[name] = x0
        #
        #     for k, v in outputs.items():
        #         # print(k)
        #         features = v[0]
        #         iter_range = features.shape[0]
        #         # print(iter_range)
        #         for i in range(iter_range):
        #             # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
        #             if 'fc' in k:
        #                 continue
        #
        #             feature = features.data.cpu().numpy()
        #             feature_img = feature[i, :, :]
        #             feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        #
        #             dst_path = os.path.join(dst, k)
        #
        #             make_dirs(dst_path)
        #             feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        #             if feature_img.shape[0] < therd_size:
        #                 tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
        #                 tmp_img = feature_img.copy()
        #                 tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
        #                 cv2.imwrite(tmp_file, tmp_img)
        #
        #             dst_file = os.path.join(dst_path, str(i) + '.png')
        #             cv2.imwrite(dst_file, feature_img)
        #
        # if self.one_hot_vis_feature == True:
        #     img_path = '/data1/lilei/occupancy_networks-master_finegrained_mutilbrach_LossCon_True/chair_1.jpg'
        #     target_class = None
        #     # Read image
        #     original_image = Image.open(img_path).convert('RGB')
        #     original_image = original_image.resize((224, 224), Image.ANTIALIAS)
        #     # Process image
        #     prep_img = preprocess_image(original_image)
        #     conv_output, model_output = self.forward_pass(prep_img)
        #     print(' conv_output', conv_output.size())
        #     if target_class is None:
        #         target_class = np.argmax(model_output.data.cpu().numpy())
        #         print(target_class)
        #     # Get convolution outputs
        #     target = conv_output[0]
        #     # Create empty numpy array for cam
        #     cam = np.ones(target.shape[1:], dtype=np.float32)
        #     # Multiply each weight with its conv output and then, sum
        #     print(len(target))
        #     for i in range(len(target)):
        #         # Unsqueeze to 4D
        #         saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)
        #         # Upsampling to input size
        #         saliency_map = F.interpolate(saliency_map, size=(224, 224))
        #         if saliency_map.max() == saliency_map.min():
        #             continue
        #         # Scale between 0-1
        #         norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        #         # Get the target score
        #         w = F.softmax(self.forward_pass(prep_img * norm_saliency_map)[1], dim=1)[0][target_class]
        #         cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        #     cam = np.maximum(cam, 0)
        #     cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        #     cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        #     cam = np.uint8(Image.fromarray(cam).resize((prep_img.shape[2],
        #                                                 prep_img.shape[3]), Image.ANTIALIAS)) / 255
        #     # Save mask
        #     save_class_activation_images(original_image, cam, 'sssss')
        #     print('Score cam completed')

        return xfg1, xfg2, x_ori, concat_out.squeeze(-1)


class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
