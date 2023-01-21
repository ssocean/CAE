import math
import sys
import time
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm
import itertools
import furnace.utils as utils
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from math import factorial
def loss_selector(loss_type, pred, target):
    if loss_type == 'mse':
        return F.mse_loss(pred, target, reduction="mean")
    elif loss_type == 'kld':
        return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='mean')

def get_combination_of(n:int=4,m=2)->list:
    '''
    This func returns the combination of the designated integer n.

    '''
    iter_lst = [i for i in range(n)]
    return list(itertools.combinations(iter_lst, m))


 
def get_comb(n=4,m=2):
    '''
    works for Python < 3.8.
    If your installed version of Python > 3.8, use math.comb(n,2) instead.
    '''
    if m <= n:
        rst=factorial(n)/(factorial(n-m) * factorial(m))
    return int(rst)

# res = get_comb(4, 2)

    
def train_one_epoch_cs_naive(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print(data_loader)
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):#
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in tqdm(enumerate(optimizer.param_groups)):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        # print('DONE')
        samples, images, bool_masked_pos_lst = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        # print(f'bool_masked_pos_lst.shape:{bool_masked_pos_lst.shape}')
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        # print('DONE')
        
        # forward 一次
        # 记录时间(epoch batch)


        for combination_set in get_combination_of(int(1/0.25)):
            # print(combination_set)
            pivot_pos = combination_set[0]
            free_pos = combination_set[1]
            pos_a = bool_masked_pos_lst[:,pivot_pos,:]
            pos_b = bool_masked_pos_lst[:,free_pos,:]
            # print(f'bool_masked_pos_lst[:,0,:].shape:{bool_masked_pos_lst[:,0,:].shape}')
            pos_a = pos_a.to(device, non_blocking=True)
            pos_b = pos_b.to(device, non_blocking=True)
            with torch.no_grad():
                input_ids = d_vae.get_codebook_indices(images).flatten(1)
                bool_masked_pos_a = pos_a.flatten(1).to(torch.bool)
                bool_masked_pos_b = pos_b.flatten(1).to(torch.bool)
                # print(bool_masked_pos_a.shape) # 8 * 8
                # print(input_ids.shape) # 32 * 64
                labels = input_ids[bool_masked_pos_a]
        # print('DONE')
            # print('-'*50)
            # print(bool_masked_pos_a*1)
            # print(bool_masked_pos_b*1)
            # print('-'*50)
            with torch.cuda.amp.autocast():
                outputs, latent, latent_target = model(samples, bool_masked_pos_a,bool_masked_pos_b)
                # print(f'latent.shape,latent_target.shape:{latent.shape,latent_target.shape}')
                loss_main = nn.CrossEntropyLoss()(input=outputs.float(), target=labels)
                loss_align = args.align_loss_weight * loss_selector('mse', latent.float(), latent_target.detach().float())
                loss = loss_main + loss_align
            # print('DONE')
            loss_value = loss.item()
            loss_main_value = loss_main.item()
            loss_align_value = loss_align.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        # print('DONE')
        mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
        metric_logger.update(mlm_acc=mlm_acc)
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")


        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_align=loss_align_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss=loss_main_value, head="loss_main")
            log_writer.update(loss=loss_align_value, head="loss_align")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}










def train_one_epoch_cs(model: torch.nn.Module, d_vae: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                log_writer=None, lr_scheduler=None, start_steps=None,
                lr_schedule_values=None, wd_schedule_values=None, args=None):
    '''
    Coupled way
    '''
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print(data_loader)
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):#
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in tqdm(enumerate(optimizer.param_groups)):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        # print('DONE')
        samples, images, bool_masked_pos_lst = batch
        bool_masked_pos_lst = bool_masked_pos_lst.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        # print(f'bool_masked_pos_lst.shape:{bool_masked_pos_lst.shape}')
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        # print('DONE')
        
        # forward 一次
        # 记录时间(epoch batch)

        with torch.cuda.amp.autocast():
            outputs, latents, latent_targets,pos_masks = model(samples, bool_masked_pos_lst)
        # for pos_mask in bool_masked_pos_lst:
        #     # print(combination_set)


        #     # print(f'bool_masked_pos_lst[:,0,:].shape:{bool_masked_pos_lst[:,0,:].shape}')
        #     pos_mask = pos_mask.to(device, non_blocking=True)
        for item in zip(outputs, latents, latent_targets,pos_masks):
            output = item[0]
            latent = item[1]
            latent_target = item[2]
            pos_mask = item[3]
            with torch.no_grad():
                input_ids = d_vae.get_codebook_indices(images).flatten(1)
                pos_mask = pos_mask.flatten(1).to(torch.bool)
                # print(bool_masked_pos_a.shape) # 8 * 8
                # print(input_ids.shape) # 32 * 64
                label = input_ids[pos_mask]
            # print('DONE')
                # print('-'*50)
                # print(bool_masked_pos_a*1)
                # print(bool_masked_pos_b*1)
                # print('-'*50)
            with torch.cuda.amp.autocast():
                # outputs, latent, latent_target = model(samples, bool_masked_pos_lst)
                # print(f'latent.shape,latent_target.shape:{latent.shape,latent_target.shape}')
                loss_main = nn.CrossEntropyLoss()(input=output.float(), target=label)
                loss_align = args.align_loss_weight * loss_selector('mse', latent.float(), latent_target.detach().float())
                loss = loss_main + loss_align
            # print('DONE')
            loss_value = loss.item()
            loss_main_value = loss_main.item()
            loss_align_value = loss_align.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        # print('DONE')
        #  这里的log有点bug output不是6个的平均
        mlm_acc = (output.max(-1)[1] == label).float().mean().item()
        metric_logger.update(mlm_acc=mlm_acc)
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")


        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_align=loss_align_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss=loss_main_value, head="loss_main")
            log_writer.update(loss=loss_align_value, head="loss_align")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_decoupled(encoder: torch.nn.Module, decoder: torch.nn.Module, d_vae: torch.nn.Module,
                data_loader: Iterable, encoder_optimizer: torch.optim.Optimizer,decoder_optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                log_writer=None, lr_scheduler=None, start_steps=None,
                lr_schedule_values=None, wd_schedule_values=None, args=None):
    encoder.train()
    decoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print(data_loader)
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):#
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(encoder_optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
            for i, param_group in enumerate(decoder_optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        # print('DONE')
        samples, images, bool_masked_pos_lst = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        # print(f'bool_masked_pos_lst.shape:{bool_masked_pos_lst.shape}')
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        # print('DONE')
        
        # forward 一次
        # 记录时间(epoch batch)

        with torch.no_grad():
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
        # for combination_set in get_combination_of(int(1/0.25)):
        #     # print(combination_set) [(0,1),(0,2)]
        #     pivot_pos = combination_set[0]
        #     free_pos = combination_set[1]
        #     pos_a = bool_masked_pos_lst[:,pivot_pos,:]
        #     pos_b = bool_masked_pos_lst[:,free_pos,:]
        #     # print(f'bool_masked_pos_lst[:,0,:].shape:{bool_masked_pos_lst[:,0,:].shape}')
        #     pos_a = pos_a.to(device, non_blocking=True)
        #     pos_b = pos_b.to(device, non_blocking=True)
        #     with torch.no_grad():
                
        #         bool_masked_pos_a = pos_a.flatten(1).to(torch.bool)
        #         bool_masked_pos_b = pos_b.flatten(1).to(torch.bool)
        #         # print(bool_masked_pos_a.shape) # 8 * 8
        #         # print(input_ids.shape) # 32 * 64
        #         labels = input_ids[bool_masked_pos_a]
        # print('DONE')
            # print('-'*50)
            # print(bool_masked_pos_a*1)
            # print(bool_masked_pos_b*1)
            # print('-'*50)
            pivot_bool_pos_lst = []
            labels = []
            x_unmasked_lst = []
            latent_target_lst = []
            pos_embed_lst = []
        for pivot_pos in range(4):
            pivot_bool_pos = bool_masked_pos_lst[:,pivot_pos,:]
            pivot_bool_pos = pivot_bool_pos.to(device, non_blocking=True)
            
            pivot_bool_pos = pivot_bool_pos.flatten(1).to(torch.bool)
            label = input_ids[pivot_bool_pos]

            with torch.cuda.amp.autocast():
                x_unmasked, latent_target,  pos_embed = encoder(samples, pivot_bool_pos)

            pivot_bool_pos_lst.append(pivot_bool_pos)
            labels.append(label)
            
            x_unmasked_lst.append(x_unmasked)
            latent_target_lst.append(latent_target)
            pos_embed_lst.append(pos_embed)
        for combination_set in get_combination_of(4):
            # [(0,1),(0,2),...,(3,4)]
            pivot_pos = combination_set[0]
            free_pos = combination_set[1]
            # pos_a = bool_masked_pos_lst[:,pivot_pos,:]
            # pos_b = bool_masked_pos_lst[:,free_pos,:]
            pos_a = pivot_bool_pos_lst[0]
            pos_b = pivot_bool_pos_lst[1]
            with torch.cuda.amp.autocast():

                logits, latent = decoder(x_unmasked_lst[pivot_pos],pos_embed_lst[pivot_pos], pos_a,pos_b)
                # print(f'latent.shape,latent_target.shape:{latent.shape,latent_target.shape}')
                loss_main = nn.CrossEntropyLoss()(input=logits.float(), target=labels[free_pos])
                loss_align = args.align_loss_weight * loss_selector('mse', latent.float(), latent_target_lst[free_pos].detach().float())
                loss = loss_main + loss_align
            # print('DONE')
        
            loss_value = loss.item()
            loss_main_value = loss_main.item()
            loss_align_value = loss_align.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(encoder_optimizer, 'is_second_order') and encoder_optimizer.is_second_order

            _ = loss_scaler(loss_main,loss_align, encoder_optimizer,decoder_optimizer, clip_grad=max_norm,
                                    parameters_encoder=encoder.parameters(), parameters_decoder=decoder.parameters(), 
                                    create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]
        grad_norm = loss_scaler(loss_main,loss_align, encoder_optimizer,decoder_optimizer, clip_grad=max_norm,
                                    parameters_encoder=encoder.parameters(), parameters_decoder=decoder.parameters(), 
                                    create_graph=is_second_order,need_step=True)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        torch.cuda.synchronize()
        # print('DONE')
        # mlm_acc = (logits.max(-1)[1] == labels).float().mean().item()
        # metric_logger.update(mlm_acc=mlm_acc)
        # if log_writer is not None:
        #     log_writer.update(mlm_acc=mlm_acc, head="loss")


        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_main=loss_main_value)
        metric_logger.update(loss_align=loss_align_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in encoder_optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        for group in decoder_optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in encoder_optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        for group in decoder_optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss=loss_main_value, head="loss_main")
            log_writer.update(loss=loss_align_value, head="loss_align")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}