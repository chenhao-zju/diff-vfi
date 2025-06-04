from functools import partial
import einops
import copy
import torch
import torch.nn as nn

from model.modules import PatchEmbed, EncoderBlock, DecoderBlock, get_2d_sincos_pos_embed, get_sincos_positional_encoding
from model.l2_loss import L2_LOSS

class DiffMAE(nn.Module):
    def __init__(self, args, diffusion, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.fixed_number = args.fixed_number
        self.start_index = args.start_index

        self.embed_dim = args.emb_dim
        self.dec_emb_dim = args.dec_emb_dim
        self.num_heads = args.num_heads
        self.decoder_num_heads = args.decoder_num_heads
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth

        self.mask_ratio = args.mask_ratio
        self.grad_weight = args.grad_weight
        self.transl_weight = args.transl_weight
        self.level = args.level
        self.extra_edge = args.extra_edge
        self.norm_pix_loss = args.norm_pix_loss

        self.in_chans = args.n_channels

        # self.patch_embed = PatchEmbed(args.img_size, args.patch_size,
        #                               args.n_channels, args.emb_dim)
        
        # self.num_patches = int(args.img_size // args.patch_size) ** 2
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, args.emb_dim))
        self.in_layer = nn.Linear(self.in_chans, self.embed_dim, bias=True)

        self.pos_embed = nn.Parameter(torch.randn(1, self.fixed_number, self.embed_dim) * .02)
        self.norm = norm_layer(self.embed_dim)

        self.blocks = nn.ModuleList([
            EncoderBlock(self.embed_dim, self.num_heads) for i in range(self.depth)])

        self.decoder_embed = nn.Linear(self.embed_dim, self.dec_emb_dim, bias=True)
        # num_masked_patches = self.num_patches - int(self.num_patches * (1 - args.mask_ratio))
        # self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_masked_patches + 1, args.dec_emb_dim) * .02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.fixed_number, self.dec_emb_dim) * .02)
        self.decoder_norm = norm_layer(self.dec_emb_dim)
        self.decoder_pred = nn.Linear(self.dec_emb_dim, self.in_chans, bias=True)

        self.dec_blocks = nn.ModuleList([
            DecoderBlock(self.embed_dim, self.dec_emb_dim, self.decoder_num_heads) for i in range(self.decoder_depth)])

        self.diffusion = diffusion

        learn_log_variance=dict(flag=True, channels=self.in_chans, logvar_init=0., requires_grad=True)
        self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        pos_embed = get_sincos_positional_encoding(self.fixed_number, self.embed_dim )
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.data.copy_( pos_embed.float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(int(self.patch_embed.num_patches*self.mask_ratio)**.5), cls_token=False)
        decoder_pos_embed = get_sincos_positional_encoding(self.fixed_number, self.dec_emb_dim )       
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_( decoder_pos_embed.float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.in_layer.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x, masked_length=20, start_index=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L - masked_length)
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # ids_restore = torch.argsort(ids_shuffle, dim=1)

        if start_index != None:
            samples = [start_index] * N
        else:
            samples = torch.randint(10, len_keep-10, (N,))
            
        ids_shuffle = torch.zeros([N, L], dtype=torch.int64).to(x.device)
        for i, j in enumerate(samples):
            ids_shuffle[i, 0:j] = torch.arange(0, j).to(x.device)
            ids_shuffle[i, j:L-masked_length] = torch.arange(j+masked_length, L).to(x.device)
            ids_shuffle[i, L-masked_length:L] = torch.arange(j, j+masked_length).to(x.device)
        # ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        self.samples = samples
        self.masked_length = masked_length
        self.len_keep = len_keep

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]

        visible_tokens = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        masked_tokens = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_tokens, masked_tokens, mask, ids_restore, ids_masked, ids_keep

    def forward_edge_loss(self, imgs, pred, level, extra_edge=True):
        recons_imgs = imgs
        losses = 0
        for i, index in enumerate(self.samples):
            end_index = index + self.masked_length
            recons_imgs[i, :index] = pred[i, :index]
            recons_imgs[i, end_index:] = pred[i, end_index:]

            if extra_edge:
                for j in range(1, level+1):
                    losses += ( ( recons_imgs[i, index-j:index+j].diff(axis=0, n=j) - pred[i, index-j:index+j].diff(axis=0, n=j) ).pow(2) ).mean() / (level * len(self.samples))

                    losses += ( ( recons_imgs[i, end_index-j:end_index+j].diff(axis=0, n=j) - pred[i, end_index-j:end_index+j].diff(axis=0, n=j) ).pow(2) ).mean() / (level * len(self.samples))
            else:
                for j in range(1, level+1):
                    losses += ( ( recons_imgs[i, index-j:end_index+j].diff(axis=0, n=j) - pred[i, index-j:end_index+j].diff(axis=0, n=j) ).pow(2) ).mean() / (level * len(self.samples))

        # losses = losses.mean() / (level * len(self.samples))
        return losses

    def forward_loss(self, imgs, pred, mask, weight=None, grad_weight=None, level=1, extra_edge=True):
        """
        imgs: [B, N, C]
        pred: [B, N, C]
        mask: [N, C], 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:
            mean = imgs.mean(dim=-1, keepdim=True)
            var = imgs.var(dim=-1, keepdim=True)
            imgs = (imgs - mean) / (var + 1.e-6)**.5

        # loss = (pred - imgs) ** 2
        loss = self.loss_gen(pred, imgs, weight=weight)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # print('loss: ', loss.shape)

        if grad_weight != None:
            if extra_edge:
                gradient_diff_loss = (imgs.diff(axis=1)-pred.diff(axis=1)).pow(2)
                # print(imgs.shape, pred.shape, mask.shape, gradient_diff_loss.shape)
                extra_edge_loss = self.forward_edge_loss(imgs, pred, level=level, extra_edge=extra_edge)
                gradient_diff_loss = gradient_diff_loss.mean() + extra_edge_loss
                
            else:
                gradient_diff_loss = self.forward_edge_loss(imgs, pred, level=level, extra_edge=extra_edge)

            total_loss = (1-grad_weight) * loss + grad_weight * gradient_diff_loss
        else:
            total_loss = loss

        # print('gradient_diff_loss: ', gradient_diff_loss.shape)

        return total_loss, loss, gradient_diff_loss
    
    def forward(self, inputs, weight=10):
        t = self.diffusion.sample_timesteps(inputs.shape[0])

        x = self.in_layer(inputs)
        x += self.pos_embed

        x, mask_token, mask, ids_restore, ids_masked, ids_keep = self.random_masking(x, self.mask_ratio, start_index=self.start_index)

        mask_token, noise = self.diffusion.noise_samples(mask_token, t)

        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        outputs[-1] = self.norm(outputs[-1])
        
        mask_token = self.decoder_embed(mask_token)
        # mask_token += self.decoder_pos_embed[:, 1:, :]
        decoder_pos_embed = nn.Parameter(
            torch.gather(self.decoder_pos_embed.repeat(mask_token.shape[0], 1, 1), dim=1,
                         index=ids_masked.unsqueeze(-1).repeat(1, 1, mask_token.shape[-1])))
        mask_token += decoder_pos_embed
        for dec_block, enc_output in zip(self.dec_blocks, reversed(outputs)):
            mask_token = dec_block(mask_token, enc_output)
        x8 = self.decoder_norm(mask_token)
        preds = self.decoder_pred(x8)

        # print('ids_keep: ', ids_keep)
        # print('ids_masked: ', ids_masked)
        # print('ids_restore: ', ids_restore)
        # print('mask: ', mask)

        visual_token = torch.gather(inputs, dim=1, index=ids_keep[:, :, None].expand(-1, -1, self.in_chans))
        new_preds = torch.cat([visual_token, preds], dim=1)
        new_preds = torch.gather(new_preds, dim=1, index=ids_restore[:, :, None].expand(-1, -1, self.in_chans)) # to unshuffle

        # print('inputs: ', inputs[0, :, 10])
        # print('visible_tokens: ', visual_token[0, :, 10])
        # print('new_preds: ', new_preds[0, :, 10])

        loss = self.forward_loss(inputs, new_preds, mask, weight=weight, grad_weight=self.grad_weight, level=self.level, extra_edge=self.extra_edge)

        return preds, loss, ids_restore, mask, ids_masked, ids_keep
