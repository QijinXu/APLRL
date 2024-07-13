import torch
import torch.nn as nn
import torch.nn.functional as F
import models.generator.basicblock as B
import math
from utils.misc import extract_patches


class RAL(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10.):
        super(RAL, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale

    def forward(self, background, foreground):
        
        # accelerated calculation 
        foreground = F.interpolate(foreground, scale_factor=1. / self.rate, mode='bilinear', align_corners=True)

        foreground_size, background_size = list(foreground.size()), list(background.size())

        background_kernel_size = 2 * self.rate
        background_patches = extract_patches(background, kernel_size=background_kernel_size, stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1, 
            background_size[1], background_kernel_size, background_kernel_size)
        background_patches_list = torch.split(background_patches, 1, dim=0)

        foreground_list = torch.split(foreground, 1, dim=0)
        foreground_patches = extract_patches(foreground, kernel_size=self.kernel_size, stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1,
            foreground_size[1], self.kernel_size, self.kernel_size)
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)

        output_list = []
        padding = 0 if self.kernel_size == 1 else 1
        escape_NaN = torch.FloatTensor([1e-4])
        if torch.cuda.is_available():
            escape_NaN = escape_NaN.cuda()

        for foreground_item, foreground_patches_item, background_patches_item in zip(
            foreground_list, foreground_patches_list, background_patches_list
        ):

            foreground_patches_item = foreground_patches_item[0]
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt((foreground_patches_item * foreground_patches_item).sum([1, 2, 3], keepdim=True)), escape_NaN)

            score_map = F.conv2d(foreground_item, foreground_patches_item_normed, stride=1, padding=padding)
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                foreground_size[2], foreground_size[3])
            attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]
            output_item = F.conv_transpose2d(attention_map, background_patches_item, stride=self.rate, padding=1) / 4.
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)
        return output


class MSFA(nn.Module):
    '''Multi-scale feature aggregation.'''

    def __init__(self, in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(MSFA, self).__init__()

        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):

            self.__setattr__('dilated_conv_{:d}'.format(_), nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate),
                nn.ReLU(inplace=True))
            )

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        weight_map = self.weight_calc(x)

        x_feature_list =[]
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(
                self.__getattr__('dilated_conv_{:d}'.format(_))(x)
            )
        
        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2] + \
                 weight_map[:, 3:4, :, :] * x_feature_list[3]

        return output


class CFA(nn.Module):
    '''Contextual Feature Aggregation.'''

    def __init__(self, 
        kernel_size=3, stride=1, rate=2, softmax_scale=10.,
        in_channels=64, out_channels=64, dilation_rate_list=[1, 2, 4, 8]):
        super(CFA, self).__init__()

        self.ral = RAL(kernel_size=kernel_size, stride=stride, rate=rate, softmax_scale=softmax_scale)
        self.msfa = MSFA(in_channels=in_channels, out_channels=out_channels, dilation_rate_list=dilation_rate_list)
        
    def forward(self, background, foreground):

        output = self.ral(background, foreground)
        output = self.msfa(output)
        
        return output
    
class FEM_and_GFAM(nn.Module):
    def __init__(self, nc=64, n_fem_res=2, n_heads=4, ksize=8, stride_1=8, stride_2=8, bias=True):
        super(FEM_and_GFAM, self).__init__()
        FEM = [
            B.ResBlock(
                nc, nc
            ) for _ in range(n_fem_res)
        ]
        self.FEM = nn.Sequential(
            *FEM
        )
        # stage 1 (4 head)
        self.GFAM = nn.ModuleList([GFAM(ksize=ksize, stride_1=stride_1, stride_2=stride_2, in_channels=nc, inter_channels=nc) for _ in range(n_heads)])
        self.merge = nn.Conv2d(nc*n_heads, nc, 1, 1, 0)
    def forward(self, x):
        # 4head-3stages
        out = self.FEM(x)
        out = torch.cat([att(x) for att in self.GFAM], dim=1)
        out = self.merge(out)+x
        return out
# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings
class GFAM(nn.Module):
    def __init__(self, ksize=4, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=32,use_multiple_size=False,use_topk=False,add_SE=False,num_edge = 50):
        super(GFAM, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE
        self.num_edge = num_edge
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=1)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=inter_channels),#将N个向量投影成原来的1/28后再相乘计算分数
            nn.ELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=ksize**2*inter_channels,out_features=inter_channels),
            nn.ELU()
        )
        
        self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
        self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
        # self.conv_sim = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=self.ksize, stride=self.stride_1,
        #                    padding=1)#卷积相似度
        # self.pos_embedding = nn.Parameter(torch.randn(1, in_channels, 32, 32))#可学习的位置编码\
        # self.fc3 = nn.Linear(inter_channels*2,1)
        self.Graph_W = nn.Parameter(torch.zeros(self.in_channels*ksize**2, self.in_channels*ksize**2))
        # self.Graph_W = nn.Parameter(torch.eye(self.in_channels*ksize**2))
    def forward(self, b):
        # b=b+self.pos_embedding
        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = b1
        # b1_group = torch.split(b1,1,dim=0)
        # b3_group = torch.split(b3,1,dim=0)
        raw_int_bs = list(b1.size())  # b*c*h*w
        b4, _ = same_padding(b,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
        soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
        soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)
        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')#原本三个都是same
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)#B N C H W
        patch_28_group = torch.split(patch_28, 1, dim=0)#按照batchsize维度分

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        y = []
        w, h = raw_int_bs[2], raw_int_bs[3]
        _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])
        
        for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
            #b1i,b3i;,b1_group,b3_group
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            #点积相似度
            wi = self.fc1(wi.view(wi.shape[1],-1))#N*vector of patches
            wi = F.normalize(wi, p=2, dim=1)# nomalized
            xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
            xi = F.normalize(xi, p=2, dim=0)
            score_map = torch.matmul(wi,xi)#N*N，计算注意力分数
            # for i in range(wi.shape[0]):
            #     for j in range(xi.shape[1]):
            #         score_map[i,j]=self.fc3(torch.cat([wi[i,:].unsqueeze(0),xi[:,j].unsqueeze(0)],dim=1))
            #         # print(score_map[i,j])
            # print(score_map[1,1])
            #卷积相似度
            # b1i = self.conv_sim(b1i).view(-1,int(w*h/4)).permute(1,0)
            # b3i = self.conv_sim(b3i).view(-1,int(w*h/4))
            # score_map = torch.matmul(b1i,b3i)#N*N，计算注意力分数
            
            score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
                                       math.ceil(h / self.stride_2))#原本两个都是math.ceil
            b_s, l_s, h_s, w_s = score_map.shape
            yi = score_map.view(l_s, -1)
            
            mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
            mask_b = (mask!=0.).float()

            yi = yi * mask
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mask_b
            
            pi = pi.view(h_s * w_s, -1)#(N,C*H*W)
            yi = torch.mm(yi, pi)#使用注意力系数的新的特征图
            yi = torch.mm(yi, self.Graph_W)# 引入图权重W，x = A*x*W
            
            yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]#N,C,H,W
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)#B,C*H*W,N
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
            out_mask += (out_mask==0.).float()
            zi = zi / out_mask#由于重叠部分在fold过程中会相加，因此将重叠部分去除。
            y.append(zi)
        y = torch.cat(y, dim=0)
        return y
