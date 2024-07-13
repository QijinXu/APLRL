import torch
import torch.nn as nn

from models.rec_discriminator.structure_branch import EdgeDetector, StructureBranch
from models.rec_discriminator.texture_branch import TextureBranch


class Discriminator(nn.Module):

    def __init__(self, image_in_channels):
        super(Discriminator, self).__init__()

        self.texture_branch = TextureBranch(in_channels=image_in_channels)
        # self.structure_branch = StructureBranch(in_channels=edge_in_channels)
        # self.edge_detector = EdgeDetector()

    def forward(self, output):


        texture_pred = self.texture_branch(output)


        return texture_pred
        

