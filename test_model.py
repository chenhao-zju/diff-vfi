import torch

import option
from model import diffmae, diffusion

if __name__ == '__main__':
    args = option.Options().gather_options()

    diff = diffusion.Diffusion(schedule='cosine')
    model = diffmae.DiffMAE(args, diff).cuda()

    inputs = torch.randn(4, 40, 203).cuda()

    pred, loss, ids_restore, mask, ids_masked, _ = model(inputs)

    print(pred.shape)
    print(loss)

