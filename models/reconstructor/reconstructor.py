import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import weights_init
from models.generator.cfa import CFA
from models.generator.bigff import BiGFF
from models.generator.pconv import PConvBNActiv
from models.generator.projection import Feature2Structure, Feature2Texture

class Reconstructor(nn.Module):

    def __init__(self, image_in_channels=3, out_channels=3, init_weights=True):
        super(Reconstructor, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # encoder-decoder reconstruct
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_texture_7 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 , 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 , 256, activ='leaky')
        self.dc_texture_3 = PConvBNActiv(256 , 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 , 64, activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64 , 64, activ='leaky')

        # -------------------------
        # structure encoder-decoder
        # -------------------------

        # -------------------
        # Projection Function
        # -------------------
        self.texture_feature_projection = Feature2Texture()

        # -----------------------------------
        # Bi-directional Gated Feature Fusion
        # -----------------------------------
        # self.bigff = BiGFF(in_channels=64, out_channels=64)

        # ------------------------------
        # Contextual Feature Aggregation
        # ------------------------------
        # self.fusion_layer1 = nn.Sequential(
        #     nn.Conv2d(64 + 64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        # self.cfa = CFA(in_channels=64, out_channels=64)
        # self.fusion_layer2 = nn.Sequential(
        #     nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        self.out_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

        if init_weights:
           self.apply(weights_init())


    def forward(self, input_image, mask):

        ec_textures = {}
        input_texture_mask = mask

        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
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

        dc_texture_out, dc_tecture_mask = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
        dc_texture = F.interpolate(dc_texture_out, scale_factor=2, mode='bilinear')
        dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')
        
        dc_texture, dc_tecture_mask = self.dc_texture_7(dc_texture, dc_tecture_mask)#get the attribution named 'dc_conv of' self and then use it; 

        for _ in range(6, 0, -1):
            # ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)
            # ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_texture_{:d}'.format(_)

            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
            dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')

            # dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)
            # dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]), dim=1)

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

        # -------------------
        # Projection Function
        # -------------------
        projected_image = self.texture_feature_projection(dc_texture)
        # projected_edge = self.structure_feature_projection(dc_structure)

        # output_bigff = self.bigff(dc_texture, dc_structure)

        # output = self.fusion_layer1(output_bigff)
        # output_atten = self.cfa(output, output)
        # output = self.fusion_layer2(torch.cat((output, output_atten), dim=1))
        # output = F.interpolate(output, scale_factor=2, mode='bilinear')
        output = self.out_layer(dc_texture)

        return output, projected_image, dc_texture_out

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
