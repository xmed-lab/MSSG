import torch
from torch import nn


class SpatialLoss(nn.Module):
    def __init__(self, h, w, channels):
        super(SpatialLoss, self).__init__()
        self.mpy_target = torch.zeros(h - 1, w, channels).cuda()
        self.mpz_target = torch.zeros(h, w - 1, channels).cuda()
        self.loss_mpy = torch.nn.L1Loss(reduction='mean')
        self.loss_mpz = torch.nn.L1Loss(reduction='mean')

    def forward(self, output_map):
        mpy = (output_map[1:, :, :] - output_map[0:-1, :, :]).cuda()
        mpz = (output_map[:, 1:, :] - output_map[:, 0:-1, :]).cuda()
        lmpy = self.loss_mpy(mpy, self.mpy_target)
        lmpz = self.loss_mpz(mpz, self.mpz_target)

        return lmpy + lmpz
