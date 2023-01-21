import math
import time
import torch
import torch.nn as nn
from functools import partial
from furnace.engine_for_pretraining_cs import get_combination_of

from models.modeling_finetune import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models.modeling_cae_helper import *

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()

        self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        # alignment constraint
        self.teacher = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.patch_embed.num_patches

        self.pretext_neck = VisionTransformerNeck(patch_size=patch_size, num_classes=args.decoder_num_classes, embed_dim=args.decoder_embed_dim, depth=args.regressor_depth,
            num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, num_patches=self.num_patches, init_std=init_std, args=args)

        # encoder to decoder projection, borrowed from mae.
        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not args.fix_init_weight:
            self.apply(self._init_weights)
        self._init_teacher()
        
        
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.encoder.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    '''
    Input shape:
        x: [bs, 3, 224, 224]
        bool_masked_pos: [bs, num_patch * num_patch]
    '''
    def forward(self, x, bool_masked_pos, return_all_tokens=None):
        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        x_unmasked = self.encoder(x, bool_masked_pos=~bool_masked_pos) # x_unmasked = self.encoder(x, bool_masked_pos=~bool_masked_pos) 
        # print()
        # encoder to decoder projection
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)

        '''
        Alignment constraint
        '''
        with torch.no_grad():
            latent_target = self.teacher(x, bool_masked_pos=(~bool_masked_pos)) # latent_target = self.teacher(x, bool_masked_pos=(~bool_masked_pos))
            
            latent_target = latent_target[:, 1:, :] # remove class token
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

            self.momentum_update(self.args.base_momentum) # self.args.base_momentum=0
        # print(f'x_unmasked:{x_unmasked.shape},latent_target:{latent_target.shape} ') 
        '''
        Latent contextual regressor and decoder
        '''
        b, num_visible_plus1, dim = x_unmasked.shape
        # remove class token
        x_unmasked = x_unmasked[:, 1:, :]

        # num_masked_patches = self.num_patches - (num_visible_plus1-1)
        num_masked_patches = (num_visible_plus1-1)



        # generate position embeddings.
        pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches+1, dim).cuda(x_unmasked.device)
        # print(pos_embed.shape) # 32 65 768 
        # print(bool_masked_pos.shape) # 32 64 
        # print(f'pos_embed[:,1:][bool_masked_pos_a].shape:{pos_embed[:,1:][bool_masked_pos].shape}') # ([512, 768])



        # print(f'pos_embed.shape:{pos_embed.shape}') # 32 65 768 
        # print(f'bool_masked_pos_a.shape:{bool_masked_pos.shape}') # 32 64  
        # print(f'pos_embed[:,1:][bool_masked_pos_a].shape:{pos_embed[:,1:][bool_masked_pos].shape}') # ([512, 768])
        # pos_embed_masked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, dim) 
        
        # print(f'pos_embed_masked.shape:{pos_embed_masked.shape}')
        # # pos embed for unmasked patches
        # pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, dim) 





        # pos embed for masked patches
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, dim) 

         # Original 
        # pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, dim) 

        # pos embed for unmasked patches
        pos_embed_unmasked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, dim)  
       


        # masked embedding '''
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)
        # print(f'pretext input shape:{x_masked.shape, x_unmasked.shape, pos_embed_masked.shape, pos_embed_unmasked.shape,bool_masked_pos.shape}')
        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos)
        logits = logits.view(-1, logits.shape[2])
        # print(f'pretext return shape:{logits.shape, latent_pred.shape, latent_target.shape}') 
        return logits, latent_pred, latent_target


class ComplementaryCAE_Naive(VisionTransformerForMaskedImageModeling):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, vocab_size=8192, 
    embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0,
     attn_drop_rate=0, drop_path_rate=0, norm_layer=None, init_values=None, attn_head_dim=None, use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__(img_size, patch_size, in_chans, vocab_size, embed_dim, depth, num_heads, 
        mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim, use_abs_pos_emb, init_std, args, **kwargs)
        self.encoder = ViTEncoder_CS(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
        self.teacher = ViTEncoder_CS(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
    def forward(self, x, bool_masked_pos_a,bool_masked_pos_b):

        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        # print(f'x.shape:{x.shape}') # ([32, 3, 32, 32]) 

        
        x_unmasked = self.encoder(x, bool_masked_pos_a,bool_masked_pos_b) # x_unmasked = x[bool_masked_pos_b]
        # print(f'x_unmasked.shape:{x_unmasked.shape}') # ([32, 49, 384]) 
        # encoder to decoder projection
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)

        '''
        Alignment constraint
        '''
        with torch.no_grad():
            latent_target = self.teacher(x, bool_masked_pos_b,bool_masked_pos_a)
            latent_target = latent_target[:, 1:, :] # remove class token
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

            self.momentum_update(self.args.base_momentum)

        '''
        Latent contextual regressor and decoder
        '''
        b, num_visible_plus1, dim = x_unmasked.shape
        # remove class token
        x_unmasked = x_unmasked[:, 1:, :]

        # num_masked_patches = self.num_patches - (num_visible_plus1-1) # 64 - 
        num_masked_patches = num_visible_plus1-1
        # generate position embeddings.
        pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches+1, dim).cuda(x_unmasked.device)

        # pos embed for masked patches
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos_a].reshape(batch_size, -1, dim) 

        # pos embed for unmasked patches
        pos_embed_unmasked = pos_embed[:,1:][bool_masked_pos_b].reshape(batch_size, -1, dim) 

        # masked embedding '''
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)
        # print(f'pretext input shape:{x_masked.shape, x_unmasked.shape, pos_embed_masked.shape, pos_embed_unmasked.shape, bool_masked_pos_a.shape}')
        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos_a)
        logits = logits.view(-1, logits.shape[2])

        return logits, latent_pred, latent_target

class CSCAE_Encoder_Helper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()
        
        self.encoder = ViTEncoder_CSv2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
        self.teacher = ViTEncoder_CSv2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
                # encoder to decoder projection, borrowed from mae.
        self.num_patches = 64#self.encoder.patch_embed.num_patches
        self.init_std = init_std
        self.args = args
        
        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None
        self._init_teacher()
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.encoder.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)
    def forward(self, x, bool_visable_pos):
        # teacher encoder 用一个bool pos
        x_unmasked = self.encoder(x, bool_visable_pos) # x_unmasked = x[bool_masked_pos_b]
        # print(f'x_unmasked.shape:{x_unmasked.shape}') # ([32, 49, 384]) 
        # encoder to decoder projection
        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)

        '''
        Alignment constraint
        '''
        with torch.no_grad():
            if False:
                latent_target = self.teacher(x, bool_visable_pos)
                latent_target = latent_target[:, 1:, :] # remove class token
                if self.encoder_to_decoder is not None:
                    latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

                self.momentum_update(self.args.base_momentum)
            else:
                latent_target = self.encoder(x, bool_visable_pos)
                latent_target = latent_target[:, 1:, :] # remove class token
                if self.encoder_to_decoder is not None:
                    latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

                self.momentum_update(self.args.base_momentum)
                # generate position embeddings.
        batch_size, num_visible_plus1, dim = x_unmasked.shape
        # num_masked_patches = self.num_patches - (num_visible_plus1-1)
        pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches+1,dim).cuda(x_unmasked.device)
        
        return x_unmasked,latent_target,pos_embed


class CSCAE_Decoder_Helper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()


        self.init_std = init_std
        self.args = args
        self.num_patches = 64

        self.pretext_neck = VisionTransformerNeck(patch_size=patch_size, num_classes=args.decoder_num_classes, embed_dim=args.decoder_embed_dim, depth=args.regressor_depth,
            num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, num_patches=self.num_patches, init_std=init_std, args=args)

        # encoder to decoder projection, borrowed from mae.
        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not args.fix_init_weight:
            self.apply(self._init_weights)
        # self._init_teacher()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000., use_cls_token=False):
        h, w = (8,8)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        if not use_cls_token:
            pos_embed = nn.Parameter(pos_emb)
        else:
            pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
            pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def forward(self, x_unmasked, pos_embed,bool_masked_pos_a,bool_masked_pos_b):
        '''
        Latent contextual regressor and decoder
        '''
        # print('------------------------forward 了 1 次-----------------------------')
        batch_size, num_visible_plus1, dim = x_unmasked.shape
        # remove class token
        x_unmasked = x_unmasked[:, 1:, :]

        num_masked_patches = (num_visible_plus1-1)
        
        # generate position embeddings.
        pos_embed = self.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(int(batch_size), int(self.num_patches+1), int(dim)).cuda(x_unmasked.device)

        # pos embed for masked patches
        
        # print(f'pos_embed.shape:{pos_embed[:,1:].shape}') # 32 65 768 
        # print(f'bool_masked_pos_a.shape:{bool_masked_pos_a.shape}') # 32 64  
        # print(bool_masked_pos_a.dtype)
        # print(f'pos_embed[:,1:][bool_masked_pos_a].shape:{pos_embed[:,1:][bool_masked_pos_a].shape}') # ([512, 768])
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos_a].reshape(batch_size, -1, dim) 
        
        # print(f'pos_embed_masked.shape:{pos_embed_masked.shape}')
        # pos embed for unmasked patches
        pos_embed_unmasked = pos_embed[:,1:][bool_masked_pos_b].reshape(batch_size, -1, dim) 

        # masked embedding '''
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)
        # print(f'pretext input shape:{x_masked.shape, x_unmasked.shape, pos_embed_masked.shape, pos_embed_unmasked.shape,bool_masked_pos_a.shape,bool_masked_pos_b.shape}')
        logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos_a)
        logits = logits.view(-1, logits.shape[2])

        return logits, latent_pred

class ComplementaryCAE(VisionTransformerForMaskedImageModeling):
    # 耦合
    def __init__(self, img_size=32, patch_size=4, in_chans=3, vocab_size=8192, 
        embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0,
        attn_drop_rate=0, drop_path_rate=0, norm_layer=None, init_values=None, attn_head_dim=None, use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):

        super().__init__(img_size, patch_size, in_chans, vocab_size, embed_dim, depth, num_heads, 
        mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim, use_abs_pos_emb, init_std, args, **kwargs)

        self.encoder = ViTEncoder_CSv2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)
        self.teacher = ViTEncoder_CSv2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)


    def forward(self, x, bool_masked_pos_lst):

        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        visable_fearures = []
        lentent_targets = []
        # batch_size number len
        for vis_id in range(4):
            visable_pos = bool_masked_pos_lst[:,vis_id,:]
            # print(combination_set)
            # pivot_pos = combination_set[0]
            # free_pos = combination_set[1]
            # pos_a = bool_masked_pos_lst[:,pivot_pos,:]
            # pos_b = bool_masked_pos_lst[:,free_pos,:]
            # print(f'bool_masked_pos_lst[:,0,:].shape:{bool_masked_pos_lst[:,0,:].shape}')

            # print(f'x.shape:{x.shape}') # ([32, 3, 32, 32]) 
            visable_fearure = self.encoder(x, visable_pos)

            # x1 x2 x3 x4 = self.encoder(*, bool_masked_pos_*,bool_masked_pos_*)
            # print(f'x_unmasked.shape:{x_unmasked.shape}') # ([32, 49, 384]) 
            # encoder to decoder projection
            if self.encoder_to_decoder is not None:
                visable_fearure = self.encoder_to_decoder(visable_fearure)
                visable_fearure = self.encoder_to_decoder_norm(visable_fearure)
            visable_fearures.append(visable_fearure)
            '''
            Alignment constraint
            '''
            with torch.no_grad():
                latent_target = self.teacher(x, visable_pos)
                latent_target = latent_target[:, 1:, :] # remove class token
                if self.encoder_to_decoder is not None:
                    latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

                self.momentum_update(self.args.base_momentum)
            lentent_targets.append(latent_target)
        '''
        Latent contextual regressor and decoder
        '''
        logitss = []
        latent_preds = []
        pos_masks = []
        for combination_set in get_combination_of(4):
            # (1,2) (1,3) (2,3)
            vis_id = combination_set[0]
            masked_id = combination_set[1]
            pos_vis = bool_masked_pos_lst[:,vis_id,:]
            pos_masked = bool_masked_pos_lst[:,masked_id,:]
            x_unmasked = visable_fearures[vis_id]

            b_s, num_visible_plus1, dim = x_unmasked.shape
            # remove class token
            x_unmasked = x_unmasked[:, 1:, :]

            # num_masked_patches = self.num_patches - (num_visible_plus1-1) # 64 - 
            num_masked_patches = num_visible_plus1-1
            # generate position embeddings.
            pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand(batch_size, self.num_patches+1, dim).cuda(x_unmasked.device)

            # pos embed for masked patches
            pos_embed_masked = pos_embed[:,1:][pos_masked].reshape(batch_size, -1, dim) 

            # pos embed for unmasked patches
            pos_embed_unmasked = pos_embed[:,1:][pos_vis].reshape(batch_size, -1, dim) 

            # masked embedding '''
            x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1)
            # print(f'pretext input shape:{x_masked.shape, x_unmasked.shape, pos_embed_masked.shape, pos_embed_unmasked.shape, bool_masked_pos_a.shape}')
            logits, latent_pred = self.pretext_neck(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, pos_masked)
            logits = logits.view(-1, logits.shape[2])
            logitss.append(logits)
            latent_preds.append(latent_pred)
            pos_masks.append(pos_masked)
            
        return logits, latent_pred, lentent_targets,pos_masks

@register_model
def cae_small_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cae_base_cifar(pretrained=False, **kwargs):
    model = VisionTransformerForMaskedImageModeling(img_size=32,
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def com_cae_cifar(pretrained=False, **kwargs):
    model = ComplementaryCAE(img_size=32,
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

    #ComplementaryCAE_Naive
@register_model
def com_cae_cifar_naive(pretrained=False, **kwargs):
    model = ComplementaryCAE_Naive(img_size=32,
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def com_encoder(**kwargs):
    model = CSCAE_Encoder_Helper(img_size=32,
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def com_decoder(**kwargs):
    model = CSCAE_Decoder_Helper(img_size=32,
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    return model