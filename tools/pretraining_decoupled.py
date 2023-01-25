import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import shutil
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from pathlib import Path
from tqdm import tqdm
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from furnace.optim_factory import create_optimizer

from furnace.datasets import build_cae_pretraining_dataset
from furnace.engine_for_pretraining import train_one_epoch
from furnace.utils import NativeScalerForDecoupled as NativeScaler
from furnace.engine_for_pretraining_cs import train_one_epoch_cs, train_one_epoch_decoupled
import furnace.utils as utils
from models import modeling_cae
import torch.distributed as dist
import logging
def init_logger(out_pth: str = 'logs'):
    '''
    初始化日志类
    :param out_pth: 输出路径，默认为调用文件的同级目录logs
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 输出到日志
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger = init_logger(r'')
    logger.info("Start print log") #一般信息
    logger.debug("Do something") #调试显示
    logger.warning("Something maybe fail.")#警告
    logger.info("Finish")
    '''
    return logger

def get_args():
    parser = argparse.ArgumentParser('pre-training script for Complementary sampling', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e", help='[dall-e, vqgan_gumbel_f8_8192, customized]')
    parser.add_argument('--dvae_num_layers', default=3, type=int)

    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--sincos_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=64, type=int, # 8 * patch num should equal to 64
                        help='images input size for discrete vae') 
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.98, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/opt/data/private/data', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--exp_name', default='', type=str, help='it is used when save the checkpoint')
    parser.add_argument('--enable_multi_print', action='store_true',default=False, help='allow each gpu to print something')

    '''
    Data augmentation
    '''
    # crop size
    parser.add_argument('--crop_min_size', type=float, default=0.08, help='min size of crop')
    parser.add_argument('--crop_max_size', type=float, default=1.0, help='max size of crop')
    # color jitter
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT', help='Color jitter factor (default: 0)')
    
    '''
    Mask strategy
    '''
    parser.add_argument('--mask_generator', default='complementary', type=str,
                        help='block or random')
    # 1. if use block mask, set the num_mask_patches
    parser.add_argument('--num_mask_patches', default=32, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=32)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=2)
    # 2. if use random mask, set the mask ratio
    parser.add_argument('--ratio_mask_patches', default=None, type=float, help="mask ratio")

    '''
    CAE hyper-parameters
    '''
    parser.add_argument('--regressor_depth', default=4, type=int, help='depth of the regressor')
    parser.add_argument('--decoder_depth', default=4, type=int, help='depth of the decoder')
    parser.add_argument('--decoder_embed_dim', default=768, type=int,
                        help='dimensionaltiy of embeddings for decoder')
    parser.add_argument('--decoder_num_heads', default=12, type=int,
                        help='Number of heads for decoder')
    parser.add_argument('--decoder_num_classes', default=8192, type=int,
                        help='Number of classes for decoder')
    parser.add_argument('--decoder_layer_scale_init_value', default=0.1, type=float,
                        help='decoder layer scale init value')

    # alignment constraint
    parser.add_argument('--align_loss_weight', type=float, default=2, help='loss weight for the alignment constraint')
    parser.add_argument('--base_momentum', type=float, default=0, help='ema weight for the dual path network')

    # init func, borrowed from BEiT
    parser.add_argument('--fix_init_weight', action='store_true', default=False, help='if true, the fix_init_weight() func will be activated')


    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        args=args,
    )

    return model

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2471, 0.2435, 0.2616]
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 100, 'input_size': (3, 32, 32), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': cifar10_mean, 'std': cifar10_std,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
default_cfgs = {
    'vit_small_cifar': _cfg(
        url=None, input_size=(3, 32, 32)
    ),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}
def vit_small_cifar(**kwargs):
    # if pretrained:
    #     # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
    #     kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(
        img_size=32,patch_size=4, num_classes=100,embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    # model.default_cfg = default_cfgs['vit_small_cifar']
    # if pretrained:
    #     load_pretrained(
    #         model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
def vit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        img_size=32,patch_size=4, num_classes=100,embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_small_cifar']

    return model

def vit_base_patch4(**kwargs):

    model = VisionTransformer(
        img_size=32,patch_size=4, num_classes=100,embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,hybrid_backbone=None, **kwargs)
    return model
# vit_base_for_cifar = modeling_cae.cae_base_cifar()
def main(args):
    
    utils.init_distributed_mode(args)

    print(args)
    # args.distributed = False
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    args.model = 'com_encoder'
    encoder = get_model(args)
    args.model = 'com_decoder'
    decoder = get_model(args)
    '''
            args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        args=args,
    '''

    # model = vit_base_for_cifar
    patch_size = encoder.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_cae_pretraining_dataset(args)

    # prepare discrete vae
    d_vae = utils.create_d_vae(
        weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
        device=device, image_size=args.second_input_size, args=args)

    if  args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        
    else:
        log_writer = None

    if global_rank == 0:
        logger = init_logger(opts.output_dir)
#sampler=sampler_train,
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        
    )

    encoder.to(device)
    encoder_without_ddp = encoder

    decoder.to(device)
    decoder_without_ddp = decoder
    encoder_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print("Model = %s" % str(encoder_without_ddp))
    print('number of params:', encoder_parameters)

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], find_unused_parameters=True)
        encoder_without_ddp = encoder.module
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.gpu], find_unused_parameters=True)
        decoder_without_ddp = decoder.module

    encoder_optimizer = create_optimizer(
        args, encoder_without_ddp)
    decoder_optimizer = create_optimizer(
        args, decoder_without_ddp)

    loss_scaler = NativeScaler() # for decoupled

    # loss_scaler_decoder = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=encoder, model_without_ddp=encoder_without_ddp, optimizer=encoder_optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        one_epoch_start_time =   time.time()      
        train_stats = train_one_epoch_decoupled(
            encoder,decoder, d_vae, data_loader_train,
            encoder_optimizer,decoder_optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
        )
        one_epoch_end_time =   time.time()   
        logger.info(f"One Eopch Time:{one_epoch_end_time-one_epoch_start_time}") 
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=encoder, model_without_ddp=encoder_without_ddp, optimizer=encoder_optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, exp_name=args.exp_name)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'encoder_parameters': encoder_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
