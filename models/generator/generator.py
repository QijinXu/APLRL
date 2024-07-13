import math


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import weights_init
from models.generator.cfa import FEM_and_GFAM, CFA
from models.generator.bigff import BiGFF
from models.generator.pconv import PConvBNActiv
from models.generator.projection import Feature2Structure, Feature2Texture

class AAI_Extractor(nn.Module):

    def __init__(self, in_channels=3, mid_channels=64, out_channels=64):
        super(AAI_Extractor, self).__init__()
        self.projection = PConvBNActiv(in_channels, mid_channels, activ='leaky')
        # self.projection = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.res_layer1 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer2 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer3 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        self.res_layer4 = PConvBNActiv(mid_channels, mid_channels, activ='leaky')
        # self.res_layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.res_layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.res_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.res_layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.res_layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.out_layer = PConvBNActiv(mid_channels, out_channels, activ='leaky')

    def forward(self, image, mask):
        
        image, mask = self.projection(image, mask)
        aai_1, mask1 = self.res_layer1(image, mask)
        aai_1 += image 
        aai_2, mask2 = self.res_layer2(aai_1, mask1)
        aai_2 += aai_1
        aai_3, mask3 = self.res_layer3(aai_2, mask2)
        aai_3 += aai_2
        aai_4, mask4 = self.res_layer4(aai_3, mask3)
        aai_4 += aai_3
        # aai_5 = self.res_layer5(aai_4)+aai_4
        aai, mask = self.out_layer(aai_4, mask4)

        return aai
class Extractor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=64):
        super(Extractor, self).__init__()
        self.aai_extractor_1 = AAI_Extractor()
    def forward(self, image, mask):
        image = self.aai_extractor_1(image, mask)
        return image.detach()
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 全连接层
        return x * y.expand_as(x)  # 乘法操作，将通道注意力应用到输入特征图上



class Generator(nn.Module):

    def __init__(self, image_in_channels=3, edge_in_channels=2, out_channels=3, init_weights=True, aai_channels=64):
        super(Generator, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # texture encoder-decoder
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')

        # -------------------------
        # structure encoder-decoder
        # # -------------------------
        # self.ec_structure_1 = PConvBNActiv(edge_in_channels, 64, bn=False, sample='down-7')
        # self.ec_structure_2 = PConvBNActiv(64, 128, sample='down-5')
        # self.ec_structure_3 = PConvBNActiv(128, 256, sample='down-5')
        # self.ec_structure_4 = PConvBNActiv(256, 512, sample='down-3')
        # self.ec_structure_5 = PConvBNActiv(512, 512, sample='down-3')
        # self.ec_structure_6 = PConvBNActiv(512, 512, sample='down-3')
        # self.ec_structure_7 = PConvBNActiv(512, 512, sample='down-3')

        # self.dc_structure_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        # self.dc_structure_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        # self.dc_structure_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        # self.dc_structure_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        # self.dc_structure_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        # self.dc_structure_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        # self.dc_structure_1 = PConvBNActiv(64 + 2, 64, activ='leaky')

        # -------------------------
        # adaptive auxiliary information encoder-decoder
        # -------------------------
        self.aai_extractor_1 = AAI_Extractor()
        self.ec_aai_1 = PConvBNActiv(aai_channels, 64, bn=False, sample='down-7')
        self.ec_aai_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_aai_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_aai_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_aai_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_aai_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_aai_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_aai_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_aai_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_aai_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_aai_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_aai_1 = PConvBNActiv(64 + aai_channels, 64, activ='leaky')
        # -------------------
        # Projection Function
        # -------------------
        # self.structure_feature_projection = Feature2Structure()
        # self.texture_feature_projection = Feature2Texture()

        # -----------------------------------
        # Bi-directional Gated Feature Fusion
        # -----------------------------------
        # self.cfa_stru = FEM_and_GFAM(nc=64, n_fem_res=1, n_heads=1, ksize=16, stride_1=16, stride_2=16)
        # self.cfa_text = FEM_and_GFAM(nc=64, n_fem_res=1, n_heads=1, ksize=16, stride_1=16, stride_2=16)
        # self.cfa_stru = CFA(in_channels=64, out_channels=64)
        # self.cfa_text = CFA(in_channels=64, out_channels=64)
        # self.cfa2 = FEM_and_GFAM(nc=64, n_fem_res=1, n_heads=4)
        self.bigff = BiGFF(in_channels=64, out_channels=64)
        # self.bigff_2 = BiGFF(in_channels=64, out_channels=64)
        # self.bigff_3 = BiGFF(in_channels=64, out_channels=64)
        # self.se = SEBlock(in_channels=128)
        # ------------------------------
        # Contextual Feature Aggregation
        # ------------------------------
        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # self.fusion_layer2 = nn.Sequential(
        #     nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        # self.fusion_layer3 = nn.Sequential(
        #     nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        self.cfa_texture = FEM_and_GFAM(nc=3, n_fem_res=1, n_heads=4, ksize=8, stride_1=8, stride_2=8)
        self.cfa_aai = FEM_and_GFAM(nc=64, n_fem_res=1, n_heads=4, ksize=8, stride_1=8, stride_2=8)    
        # self.cfa = FEM_and_GFAM(nc=128, n_fem_res=1, n_heads=4, ksize=16, stride_1=8, stride_2=8) 
        self.fusion_layer4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 3, kernel_size=1),
            nn.Tanh()
        )

        if init_weights:
           self.apply(weights_init())


    def forward(self, input_image, input_edge, mask, input_aai, is_train):
        

        if is_train == True:    
            # aai_feature = self.aai_extractor_1(input_aai, torch.ones_like(input_aai))
            aai_feature = self.aai_extractor_1(input_aai*mask, mask.expand_as(input_aai))
            aai_input = aai_feature
            aai_mask = mask.expand_as(aai_feature)
            # aai_input = aai_feature*aai_mask 
        else:
            #输入经过提取和掩码的图片
            aai_feature = input_aai
            aai_mask = mask.expand_as(aai_feature)
            aai_input = aai_feature

        
        ec_textures = {}
        # ec_structures = {}
        ec_aai = {}
        
        input_texture_mask = torch.cat((mask, mask, mask), dim=1)
        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
        ec_textures['ec_t_0'] = self.cfa_texture(ec_textures['ec_t_0'])
        ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'])
        ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'])
        ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'])
        ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'])
        ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'])
        ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'])
        ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'])

        # input_structure_mask = torch.cat((mask, mask), dim=1)
        # ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'] = input_edge, input_structure_mask
        # ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'] = self.ec_structure_1(ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'])
        # ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'] = self.ec_structure_2(ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'])
        # ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'] = self.ec_structure_3(ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'])
        # ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'] = self.ec_structure_4(ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'])
        # ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'] = self.ec_structure_5(ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'])
        # ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'] = self.ec_structure_6(ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'])
        # ec_structures['ec_s_7'], ec_structures['ec_s_masks_7'] = self.ec_structure_7(ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'])


        ec_aai['ec_a_0'], ec_aai['ec_a_masks_0'] = aai_input, aai_mask
        ec_aai['ec_a_0'] = self.cfa_aai(ec_aai['ec_a_0'])
        ec_aai['ec_a_1'], ec_aai['ec_a_masks_1'] = self.ec_aai_1(ec_aai['ec_a_0'], ec_aai['ec_a_masks_0'])
        ec_aai['ec_a_2'], ec_aai['ec_a_masks_2'] = self.ec_aai_2(ec_aai['ec_a_1'], ec_aai['ec_a_masks_1'])
        ec_aai['ec_a_3'], ec_aai['ec_a_masks_3'] = self.ec_aai_3(ec_aai['ec_a_2'], ec_aai['ec_a_masks_2'])
        ec_aai['ec_a_4'], ec_aai['ec_a_masks_4'] = self.ec_aai_4(ec_aai['ec_a_3'], ec_aai['ec_a_masks_3'])
        ec_aai['ec_a_5'], ec_aai['ec_a_masks_5'] = self.ec_aai_5(ec_aai['ec_a_4'], ec_aai['ec_a_masks_4'])
        ec_aai['ec_a_6'], ec_aai['ec_a_masks_6'] = self.ec_aai_6(ec_aai['ec_a_5'], ec_aai['ec_a_masks_5'])
        ec_aai['ec_a_7'], ec_aai['ec_a_masks_7'] = self.ec_aai_7(ec_aai['ec_a_6'], ec_aai['ec_a_masks_6'])


        dc_texture_out, dc_tecture_mask = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
        dc_texture = F.interpolate(dc_texture_out, scale_factor=2, mode='bilinear')
        dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')
        for _ in range(6, 0, -1):
            ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)
            ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_texture_{:d}'.format(_)

            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
            dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')

            dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)
            dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]), dim=1)

            dc_texture, dc_tecture_mask = getattr(self, dc_conv)(dc_texture, dc_tecture_mask)#get the attribution named 'dc_conv of' self and then use it; 

        # dc_structure, dc_structure_masks =  ec_structures['ec_s_7'], ec_structures['ec_s_masks_7']
        
        # for _ in range(7, 0, -1):

        #     ec_structure_skip = 'ec_s_{:d}'.format(_ - 1)
        #     ec_structure_masks_skip = 'ec_s_masks_{:d}'.format(_ - 1)
        #     dc_conv = 'dc_structure_{:d}'.format(_)

        #     dc_structure = F.interpolate(dc_structure, scale_factor=2, mode='bilinear')
        #     dc_structure_masks = F.interpolate(dc_structure_masks, scale_factor=2, mode='nearest')

        #     dc_structure = torch.cat((dc_structure, ec_structures[ec_structure_skip]), dim=1)
        #     dc_structure_masks = torch.cat((dc_structure_masks, ec_structures[ec_structure_masks_skip]), dim=1)

        #     dc_structure, dc_structure_masks = getattr(self, dc_conv)(dc_structure, dc_structure_masks)

################## adaptive auxiliary information encoder-decoder#########################
        dc_aai, dc_aai_masks =  ec_aai['ec_a_7'], ec_aai['ec_a_masks_7']
        
        for _ in range(7, 0, -1):

            ec_aai_skip = 'ec_a_{:d}'.format(_ - 1)
            ec_aai_masks_skip = 'ec_a_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_aai_{:d}'.format(_)

            dc_aai = F.interpolate(dc_aai, scale_factor=2, mode='bilinear')
            dc_aai_masks = F.interpolate(dc_aai_masks, scale_factor=2, mode='nearest')

            dc_aai = torch.cat((dc_aai, ec_aai[ec_aai_skip]), dim=1)
            dc_aai_masks = torch.cat((dc_aai_masks, ec_aai[ec_aai_masks_skip]), dim=1)

            dc_aai, dc_aai_masks = getattr(self, dc_conv)(dc_aai, dc_aai_masks)
        # -------------------
        # Projection Function
        # -------------------
        projected_image = 0
        # projected_edge = self.structure_feature_projection(dc_structure)

        
        # add graph model before bigff
        # dc_texture = dc_texture+self.cfa_texture(dc_texture)*(1-mask)*0.1
        # dc_aai = dc_aai+self.cfa_aai(dc_aai)*(1-mask)*0.1
        # output_bigff = self.bigff(dc_texture, dc_structure)
        # output_bigff = self.se(output_bigff)
        
        # output = self.fusion_layer1(output_bigff)
        # dc_aai=dc_aai+
        output_bigff_2 = self.bigff(dc_texture, dc_aai)
        # output_bigff_2 = output_bigff_2 + self.cfa(output_bigff_2)
        output_2 = self.fusion_layer1(output_bigff_2)
        
        # output_bigff_3 = self.bigff_3(output, output_2)
        # output_3 = self.fusion_layer3(output_bigff_3)
        
        # output_atten = self.cfa(output_2)
        # output_3 = self.fusion_layer4(torch.cat((output_2, output_atten), dim=1))#消融实验：在这里cat output 和output_atten
        # output = F.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.out_layer(torch.cat((output_2, output_bigff_2), dim=1))

        # dc_texture is the latent vector
        return output, projected_image, aai_feature.detach(), dc_aai

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
