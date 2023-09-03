from funlib.learn.torch.models import UNet, ConvPass
import torch


num_fmaps = 6
fmap_inc_factor = 3

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.unet = UNet(
                in_channels=1,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=[
                    (1,2,2),
                    (1,2,2)],
                kernel_size_down=[
                    [[3,3,3],[3,3,3]],
                    [[1,3,3],[1,3,3]],
                    [[1,3,3],[1,3,3]]],
                kernel_size_up=[
                    [[1,3,3],[1,3,3]],
                    [[3,3,3],[3,3,3]]],
                constant_upsample=True,
                num_heads=2)

        self.endo_head = ConvPass(num_fmaps, 1, [[1,1,1]], activation='Sigmoid')
        self.lyso_head = ConvPass(num_fmaps, 1, [[1,1,1]], activation='Sigmoid')

    def forward(self, input):
        
        z = self.unet(input)

        endo = self.endo_head(z[0])
        lyso = self.lyso_head(z[1])

        return endo, lyso        
