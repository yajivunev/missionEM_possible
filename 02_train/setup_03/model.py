from funlib.learn.torch.models import UNet, ConvPass
import torch


num_fmaps = 10
fmap_inc_factor = 5

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.unet = UNet(
                in_channels=1,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=[
                    (1,2,2),
                    (1,2,2),
                    (1,2,2)],
                kernel_size_down=[
                    [[3,3,3],[3,3,3]],
                    [[1,3,3],[1,3,3]],
                    [[1,3,3],[1,3,3]],
                    [[1,3,3],[1,3,3]]],
                kernel_size_up=[
                    [[1,3,3],[1,3,3]],
                    [[1,3,3],[1,3,3]],
                    [[3,3,3],[3,3,3]]],
                constant_upsample=True,
                num_heads=4)

        self.endo_affs_head = ConvPass(num_fmaps, 3, [[1,1,1]], activation='Sigmoid')
        self.lyso_affs_head = ConvPass(num_fmaps, 3, [[1,1,1]], activation='Sigmoid')
        self.endo_lsds_head = ConvPass(num_fmaps, 10, [[1,1,1]], activation='Sigmoid')
        self.lyso_lsds_head = ConvPass(num_fmaps, 10, [[1,1,1]], activation='Sigmoid')

    def forward(self, input):
        
        z = self.unet(input)

        endo_affs = self.endo_affs_head(z[0])
        lyso_affs = self.lyso_affs_head(z[1])
        endo_lsds = self.endo_lsds_head(z[2])
        lyso_lsds = self.lyso_lsds_head(z[3])

        return endo_affs, lyso_affs, endo_lsds, lyso_lsds 
