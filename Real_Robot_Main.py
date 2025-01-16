import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import json
from time import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import clip
from Real_Robot_LMDBDataset_jpeg import LMDBDataset as LMDBdst_jpeg,get_norm_stats
from Real_Robot_LMDBDataset_jpeg import DataPrefetcher as DataPrefetcher_jpeg
from PreProcess import Real_Robot_PreProcess,Real_Robot_2D_PreProcess
import models.vision_transformer as vits
from models.real_robot_gr1 import GR1 
from AccelerateFix import AsyncStep
from tqdm import tqdm
from pickle import loads
import lmdb
import argparse
# fsc
def masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):
    if skip_frame == 0:
        new_pred = pred
    else:
        new_pred = pred[:, :-skip_frame]
    new_target = target[:, skip_frame:]
    new_mask = mask[:, skip_frame:]
    data_shape, mask_shape = new_pred.shape, new_mask.shape
    loss = loss_func(new_pred, new_target, reduction='none')
    for _ in range(len(data_shape) - len(mask_shape)):
        new_mask = new_mask.unsqueeze(-1)
    loss = (loss*new_mask).sum() / new_mask.sum() / math.prod(data_shape[len(mask_shape):])
    return loss

def train(acc, train_prefetcher, test_prefetcher, preprocessor, model, env, eva, eval_dir, optimizer, scheduler, device, cfg, step, writer,stats):
    '''
    prof = profile(
        schedule = torch.profiler.schedule(
            wait=20,
            warmup=3,
            active=4,
            repeat=1,
        ),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(cfg['save_path']+'prof'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    prof.start()
    '''

    train_dataset_len = len(train_prefetcher.loader.dataset)
    test_dataset_len = len(test_prefetcher.loader.dataset)
    eval_steps = train_dataset_len // test_dataset_len
    avg_reward = 0.0
    for epoch in range(cfg['num_epochs']):
        log_loss = {
            'rgb_camera_left': 0,
            'rgb_camera_right': 0,
            'rgb_camera_top': 0,
            'action_arm': 0,
            'action_gripper': 0,
            **({'traj_2d_preds': 0} if cfg['use_2d_traj'] else {}),
            'total_loss': 0
        }
        eval_log_loss = {
            'rgb_camera_left': 0,
            'rgb_camera_right': 0,
            'rgb_camera_top': 0,
            'action_arm': 0,
            'action_gripper': 0,
            **({'traj_2d_preds': 0} if cfg['use_2d_traj'] else {}),
            'total_loss':0
        }
        for key in log_loss:
            log_loss[key] = torch.tensor(0).float().to(device)
        for key in eval_log_loss:
            eval_log_loss[key] = torch.tensor(0).float().to(device)
        cum_load_time = 0 
        clock = time()
        batch_idx = 0
        with tqdm(total=train_dataset_len, desc=f"Train Epoch {epoch+1}", leave=False,disable= not acc.is_main_process) as pbar:
            batch, load_time = train_prefetcher.next()
            while batch is not None:
                # training
                with acc.accumulate(model):
                    model.train()
                    optimizer.zero_grad()
                    if cfg['use_2d_traj']:
                        rgb_camera_left, rgb_camera_right, rgb_camera_top, new_action_2d_top, new_action_2d_top_rest = preprocessor.rgb_process(batch['rgb_camera_left'], batch['rgb_camera_right'], batch['rgb_camera_top'], batch['actions_2d'],batch['traj_2d_preds'],train=True)
                    else:
                        rgb_camera_left, rgb_camera_right, rgb_camera_top = preprocessor.rgb_process(batch['rgb_camera_left'], batch['rgb_camera_right'], batch['rgb_camera_top'],train=True)
                        new_action_2d_top = None
                    # import cv2
                    # rgb_vis = batch['rgb_camera_top'][0][0].permute(1, 2, 0).cpu().numpy().copy()
                    # for index, point_2d in enumerate(batch['actions_2d'][0,0,:,:].to(device)):
                    #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                    #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=(225,206,135), thickness=-1)
                    # cv2.imwrite("traj_predict/script/visualization/tmp.png", rgb_vis)  
                    obs_mask = batch['mask'][..., 0]
                    pred = model(
                        rgb_left=rgb_camera_left,
                        rgb_right=rgb_camera_right,
                        rgb_top=rgb_camera_top,
                        state={'arm': batch['arm_state'], 'gripper': batch['gripper_state']},
                        language=batch['inst_token'],
                        attention_mask=obs_mask,
                        action_2d = new_action_2d_top, # 224*224的坐标系
                    )
                    loss = {}
                    if cfg['use_2d_traj']:
                        loss['traj_2d_preds'] = masked_loss(pred['traj_2d_preds'], new_action_2d_top_rest[:,:,:pred['traj_2d_preds'].shape[2]]/224, obs_mask, 0, F.smooth_l1_loss)
                    loss['rgb_camera_left'] = masked_loss(pred['obs_preds'], pred['obs_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)
                    loss['rgb_camera_right'] = masked_loss(pred['obs_right_preds'], pred['obs_right_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)
                    loss['rgb_camera_top'] = masked_loss(pred['obs_top_preds'], pred['obs_top_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)

                    loss['action_arm'] = masked_loss(pred['arm_action_preds'], batch['actions'][..., :7], batch['mask'], 0, F.smooth_l1_loss)
                    loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][..., -1:], batch['mask'], 0, F.binary_cross_entropy_with_logits)

                    loss['total_loss'] = (loss['traj_2d_preds'] if cfg['use_2d_traj'] else 0) \
                                   + loss['rgb_camera_left'] + loss['rgb_camera_right'] + loss['rgb_camera_top'] \
                                   + cfg['arm_loss_ratio']*loss['action_arm'] + loss['action_gripper'] 
                    acc.backward(loss['total_loss'])
                    optimizer.step(optimizer)
                    for key in log_loss:
                        log_loss[key] += loss[key].detach() / cfg['print_steps']
                    cum_load_time += load_time / cfg['print_steps']
            	# evaluation test dataset
                if batch_idx % eval_steps == 0:
                    with torch.no_grad():
                        model.eval()
                        batch, _ = test_prefetcher.next_without_none()
                        if cfg['use_2d_traj']:
                            rgb_camera_left, rgb_camera_right, rgb_camera_top, new_action_2d_top, new_action_2d_top_rest = preprocessor.rgb_process(batch['rgb_camera_left'], batch['rgb_camera_right'], batch['rgb_camera_top'], batch['actions_2d'],batch['traj_2d_preds'],train=False)
                        else:
                            rgb_camera_left, rgb_camera_right, rgb_camera_top = preprocessor.rgb_process(batch['rgb_camera_left'], batch['rgb_camera_right'], batch['rgb_camera_top'],train=False)
                            new_action_2d_top = None

                        obs_mask = batch['mask'][..., 0]
                        pred = model(
                            rgb_left=rgb_camera_left,
                            rgb_right=rgb_camera_right,
                            rgb_top=rgb_camera_top,
                            state={'arm': batch['arm_state'], 'gripper': batch['gripper_state']},
                            language=batch['inst_token'],
                            attention_mask=obs_mask,
                            action_2d = new_action_2d_top, # 224*224的坐标系
                        )
                        loss = {}
                        if cfg['use_2d_traj']:
                            loss['traj_2d_preds'] = masked_loss(pred['traj_2d_preds'], new_action_2d_top_rest[:,:,:pred['traj_2d_preds'].shape[2]]/224, obs_mask, 0, F.smooth_l1_loss)
                        loss['rgb_camera_left'] = masked_loss(pred['obs_preds'], pred['obs_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)
                        loss['rgb_camera_right'] = masked_loss(pred['obs_right_preds'], pred['obs_right_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)
                        loss['rgb_camera_top'] = masked_loss(pred['obs_top_targets'], pred['obs_top_targets'], obs_mask, cfg['skip_frame'], F.mse_loss)

                        loss['action_arm'] = masked_loss(pred['arm_action_preds'], batch['actions'][..., :7], batch['mask'], 0, F.smooth_l1_loss)
                        loss['action_gripper'] = masked_loss(pred['gripper_action_preds'], batch['actions'][..., -1:], batch['mask'], 0, F.binary_cross_entropy_with_logits)
                        loss['total_loss'] = (loss['traj_2d_preds'] if cfg['use_2d_traj'] else 0) \
                                    + loss['rgb_camera_left'] + loss['rgb_camera_right'] + loss['rgb_camera_top'] \
                                    + cfg['arm_loss_ratio']*loss['action_arm'] + loss['action_gripper'] 
                        for key in eval_log_loss:
                            eval_log_loss[key] += loss[key].detach() / cfg['print_steps'] * eval_steps
            	# print steps log
                if batch_idx % cfg['print_steps'] == 0 and batch_idx != 0:
                    for key in log_loss:
                        log_loss[key] = acc.gather_for_metrics(log_loss[key]).mean()
                    for key in eval_log_loss:
                        eval_log_loss[key] = acc.gather_for_metrics(eval_log_loss[key]).mean()
                    load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(device)
                    load_pecnt = acc.gather_for_metrics(load_pecnt).mean()
                    fps = (cfg['bs_per_gpu']*cfg['print_steps']*cfg['seq_len']) / (time()-clock)
                    fps = acc.gather_for_metrics(torch.tensor(fps).to(device)).sum()

                    text = 'Train Epoch: {} [{}/{} ({:.0f}%)] Reward: {:.5f} FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
                        epoch+1, 
                        batch_idx * cfg['bs_per_gpu'] * acc.num_processes, 
                        train_dataset_len, 
                        100. * batch_idx * cfg['bs_per_gpu'] * acc.num_processes / train_dataset_len, 
                        avg_reward,
                        fps,
                        load_pecnt,
                        scheduler.get_last_lr()[0],
                    )
                    for key in log_loss:
                        text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
                    for key in eval_log_loss:
                        text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
                    acc.print(text)
                    # write in tensorboard
                    if acc.is_main_process:
                        for key in log_loss:
                            writer.add_scalar(key+'_loss', log_loss[key], step)
                        for key in eval_log_loss:
                            writer.add_scalar('eval_'+key+'_loss', eval_log_loss[key], step)
                        writer.add_scalar("reward", avg_reward, step)
                        writer.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
                        writer.add_scalar("FPS", fps, step)
                        writer.add_scalar("loading time in total time", load_pecnt, step)
                        with open(cfg['save_path']+'step.json', 'w') as json_file:
                            json.dump(step, json_file)

                    for key in log_loss:
                        log_loss[key] = torch.tensor(0).float().to(device)
                    for key in eval_log_loss:
                        eval_log_loss[key] = torch.tensor(0).float().to(device)
                    cum_load_time = 0
                    clock = time()
                    scheduler.step()
                
                pbar.set_postfix(
                    ordered_dict={
                        "epoch": epoch+1,
                        "total loss": total_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                pbar.update(cfg['bs_per_gpu']*acc.num_processes)
                batch_idx += 1
                step += 1
                batch, load_time = train_prefetcher.next()
                '''
                prof.step()
                if batch_idx == 28:
                    prof.stop()
                '''
        # 每cfg['save_epochs']个epoch都保存，除了初始第一个epoch
        if (epoch+1) % cfg['save_epochs'] == 0:
            acc.wait_for_everyone()
            unwrapped_model = acc.unwrap_model(model)
            modules_to_exclude = ['model_mae', 'model_clip']
            if hasattr(unwrapped_model, '_orig_mod'):
                state_dict = {k: v for k, v in unwrapped_model._orig_mod.state_dict().items() if not any(module_name in k for module_name in modules_to_exclude)}
            else:
                state_dict = {k: v for k, v in unwrapped_model.state_dict().items() if not any(module_name in k for module_name in modules_to_exclude)}
            acc.save({'state_dict': state_dict,
                      'stats':stats}, cfg['save_path']+'GR1_{}.pth'.format(epoch+1+cfg['load_epoch']))
        else:
            acc.wait_for_everyone()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load config JSON file")
    parser.add_argument('--config', type=str, default='Real_Robot_configs_debug.json', help='Path to the config JSON file')
    parser.add_argument('--train_ratio', type=float, default=0.98, help='Training ratio')
    parser.add_argument('--val_ratio', type=float, default=1, help='Training ratio')
    args = parser.parse_args()
    # 确保文件路径存在
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The config file at {config_path} was not found.")
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Preparation
    # cfg = json.load(open('Real_Robot_configs_scratch.json'))

    # The timeout here is 3600s to wait for other processes to finish the simulation
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        mixed_precision="bf16",
        kwargs_handlers=[init_pg_kwargs, ddp_kwargs]
    )
    device = acc.device
    if cfg['use_2d_traj']:
        preprocessor = Real_Robot_2D_PreProcess(
            cfg['rgb_static_pad'],
            cfg['rgb_shape'],
            cfg['rgb_mean'],
            cfg['rgb_std'],
            device,
        )
    else:
        preprocessor = Real_Robot_PreProcess(
            cfg['rgb_static_pad'],
            cfg['rgb_shape'],
            cfg['rgb_mean'],
            cfg['rgb_std'],
            device,
        )
    # 获得stats
    env = lmdb.open(cfg['LMDB_path'], readonly=True, create=False, lock=False)
    with env.begin() as txn:
        stats = loads(txn.get(b'stats'))
    env.close()

    train_dataset = LMDBdst_jpeg(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0,
        end_ratio = args.train_ratio, 
    )
    test_dataset = LMDBdst_jpeg(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = args.train_ratio,
        end_ratio = args.val_ratio, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    model_clip, _ = clip.load(cfg['clip_ckpt']) 
    model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
    checkpoint = torch.load(cfg['mae_ckpt'],weights_only=False)
    model_mae.load_state_dict(checkpoint['model'], strict=False)
    if cfg['fwd_pred'] and cfg['fwd_pred_multiview']:
        training_target = ['act_pred', 'fwd_pred', 'fwd_pred_multiview']
    else:
        training_target = ['act_pred']
    model = GR1(
        model_clip,
        model_mae,
        state_dim=cfg['state_dim'],
        act_dim=cfg['act_dim'],
        hidden_size=cfg['embed_dim'],
        sequence_length=cfg['seq_len'],
        chunk_size=cfg['chunk_size'],
        training_target=training_target,
        img_feat_dim=cfg['img_feat_dim'],
        patch_feat_dim=cfg['patch_feat_dim'],
        lang_feat_dim=cfg['lang_feat_dim'],
        resampler_params={
            'depth': cfg['resampler_depth'],
            'dim_head': cfg['resampler_dim_head'],
            'heads': cfg['resampler_heads'],
            'num_latents': cfg['resampler_num_latents'],
            'num_media_embeds': cfg['resampler_num_media_embeds'],
        },
        without_norm_pixel_loss=False,
        use_multi_rgb=cfg['use_multi_rgb'],
        use_2d_traj = cfg['use_2d_traj'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_inner=4*cfg['embed_dim'],
        activation_function=cfg['activation_function'],
        n_positions=cfg['n_positions'],
        resid_pdrop=cfg['dropout'],
        attn_pdrop=cfg['dropout'],
    ).to(device)
    if cfg['load_bytedance_ckpt']:
        pretrained_dict = torch.load(cfg['bytedance_ckpt_path'],map_location=device)['state_dict']
       # 过滤掉与 model_mae 和 text_encoder 相关的层
        filtered_pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'model_mae' not in k and 'text_encoder' not in k
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_pretrained_dict, strict=False)

        acc.print('load ', cfg['bytedance_ckpt_path'], '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
        # 删除不再需要的变量以释放内存
        del pretrained_dict,filtered_pretrained_dict
        torch.cuda.empty_cache()
    elif os.path.isfile(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch'])):
        state_dict = torch.load(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']))['state_dict'] 
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        acc.print('load ', cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']),  '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
    if cfg['compile_model']:
        model = torch.compile(model)
    if os.path.isfile(cfg['save_path']+'step.json'):
        with open(cfg['save_path']+'step.json', 'r') as json_file:
            step = json.load(open(cfg['save_path']+'step.json'))
    else:
        step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_max'], weight_decay=cfg['weight_decay'], fused=True)
    total_prints_per_epoch = len(train_dataset) // (cfg['print_steps'] * cfg['bs_per_gpu'] * acc.num_processes)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=cfg['num_warmup_epochs']*total_prints_per_epoch,
        num_training_steps=cfg['num_epochs']*total_prints_per_epoch,
    )
    model, optimizer, train_loader, test_loader = acc.prepare(
        model, 
        optimizer, 
        train_loader, 
        test_loader, 
        device_placement=[True, True, False, False],
    )
    optimizer.step = AsyncStep
    train_prefetcher = DataPrefetcher_jpeg(train_loader, device)
    test_prefetcher = DataPrefetcher_jpeg(test_loader, device)
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    eval_dir = cfg['save_path']+f'eval{torch.cuda.current_device()}/'
    os.makedirs(eval_dir, exist_ok=True)

    env = None
    eva = None
    writer = SummaryWriter(cfg['save_path'] + 'logs')

    # 把设置好的json文件保存在save路径下
    os.makedirs(cfg['save_path'], exist_ok=True)
    save_file_path = os.path.join(cfg['save_path'], "config_saved.json")
    with open(save_file_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Configuration saved to {save_file_path}")

    # Train
    train(acc, train_prefetcher, test_prefetcher, preprocessor, model, env, eva, eval_dir, optimizer, scheduler, device, cfg, step, writer,stats)
