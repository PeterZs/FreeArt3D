import os
import sys
sys.path.append('TRELLIS')

import numpy as np
import torch
import trellis.models as models 
import json
import trimesh

from scipy import ndimage
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
from omegaconf import OmegaConf
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from network.hash_grid import HashGridVoxel, generate_image_grid_3d, generate_surround_offset
 
from pipelines.utils.seeding import seed_everything
from pipelines.utils.transforms import GridTransformer, calc_grid_weight 
from pipelines.utils.visualization import visualize_voxels_two_parts, visualize_voxels
from pipelines.utils.postprocessing import flood_fill_exterior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize(base_dir, cfg):
 
    seed_everything()
    torch.use_deterministic_algorithms(True)
    
    print(f"[SDS] Initializing HashGrids...")
    if os.path.exists(f'{base_dir}/initialization/model.pth'):
        return  
 
    voxels = np.load(f'{base_dir}/recon/recon_voxel_joint_00_state_00.npy', allow_pickle=True)
    voxels = torch.tensor(voxels, device=device).float()[None, None]
    
    model = HashGridVoxel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0.0001)

    with torch.enable_grad():
        pbar = tqdm(range(1, 10000))    
        for i in pbar:
            
            grid = generate_image_grid_3d(64).to(device)
            grid += (torch.rand_like(grid) - 0.5) / 63.   
            grid = grid.view(-1, 3)
            
            voxels_out = model(grid).unsqueeze(0)
            loss = 0.5 * F.mse_loss(voxels_out, voxels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % cfg.print_interval == 0:
                pbar.set_postfix(loss=loss.item())

    voxels_out = torch.where(voxels_out > 0.5, 1, 0)
    visualize_voxels(voxels_out[0, 0].detach().cpu().numpy(), f'{base_dir}/initialization/voxel.png')
    torch.save(voxels, f'{base_dir}/initialization/voxel.pth')
    torch.save(model.state_dict(), f'{base_dir}/initialization/model.pth')

def train(base_dir, output_dir, joint_num, cfg):
     
    seed_everything()
    torch.use_deterministic_algorithms(True)

    if os.path.exists(f"{output_dir}/states/qpos_05.glb"):
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[SDS] Optimize HashGrids & Joints...") 

    init_ests = []
    joint_types = []
    transform_scales = []
    joint_positions = []
    joint_axes = []
    qpos_lists = []

    for joint_id in range(joint_num):

        init_ests.append(np.load(f'{base_dir}/initialization/joint_{joint_id:02d}_est.npy', allow_pickle=True).item())
        joint_types.append(init_ests[-1]['joint_type'])
    
        ## Initialize joint axis & transform scale
        # It's easy to optimize the transform scale (take absolute value) from a large initialization,
        # but difficult vice versa. So we set a minimum value for the init transform scale.
        if joint_types[-1] == 0: # Translate    
            transform_init = min(init_ests[-1]['transform_scale'], -0.3) 
            transform_scales.append(transform_init) 
            joint_positions.append(np.zeros(3))
            joint_axes.append(init_ests[-1]['joint_axis'][::-1].astype(np.float32).copy()) 
        elif joint_types[-1] == 1: # Rotate
            if init_ests[-1]['transform_scale'] < 0:
                transform_init = min(init_ests[-1]['transform_scale'], -1.0)
            else:
                transform_init = max(init_ests[-1]['transform_scale'], 1.0)
            transform_scales.append(transform_init)  
            joint_positions.append((init_ests[-1]['joint_position'][::-1] + 0.5) * 63.)
            joint_axes.append(-init_ests[-1]['joint_axis'][::-1].astype(np.float32).copy())
            axis_idx = np.argmax(np.abs(joint_axes[-1])) 
            joint_positions[-1][axis_idx] = 32.0
        qpos_lists.append(torch.tensor(init_ests[-1]['qpos'], device=device).float())
         
    joint_positions = torch.tensor(joint_positions, device=device).float()
    joint_positions = [nn.Parameter(joint_positions[i], requires_grad=True) for i in range(len(joint_positions))]
    transform_scales = [nn.Parameter(torch.tensor(transform_scales[i], device=device).float(),
        requires_grad=True) for i in range(len(transform_scales))]
    qpos_lists_train = [nn.Parameter(qpos_lists[i][1:-1], requires_grad=True) for i in range(len(qpos_lists))] 

    coeff = torch.tensor(cfg.sample_coeff, device=device)
    sample_interval = torch.tensor(cfg.sample_interval, device=device) 
    
    for joint_id in range(joint_num):
        print(f'[Joint {joint_id:02d} Info] Max Scale: {transform_scales[joint_id].data.cpu().numpy()} ' \
              f'Axis: {joint_axes[joint_id]} Position: {joint_positions[joint_id].data.cpu().numpy()} ' \
              f'Type: {joint_types[joint_id]}')
        print(f'[Joint {joint_id:02d} Info] qpos: {qpos_lists[joint_id].cpu().numpy()}')
    OmegaConf.save(cfg, f'{output_dir}/config.yaml')

    encoder = models.from_pretrained("JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").to(device)
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    decoder = pipe.models['sparse_structure_decoder']
    diffusion = pipe.models['sparse_structure_flow_model']
    
    conds = [] 
    for joint_id in range(joint_num):
        target_image = []
        for i in range(cfg.train_num_state):
            target_image.append(Image.open(f'{base_dir}/renderings/rendering_joint_{joint_id:02d}_state_{i:02d}.png'))
        target_image = [pipe.preprocess_image(img) for img in target_image]
        conds.append(pipe.get_cond(target_image))
    
    model_base = HashGridVoxel().to(device) 
    model_base.load_state_dict(torch.load(f'{base_dir}/initialization/model.pth', weights_only=True))
    model_articulateds = []
    for joint_id in range(joint_num):
        model_articulateds.append(HashGridVoxel().to(device))
        model_articulateds[-1].load_state_dict(torch.load(f'{base_dir}/initialization/model.pth', weights_only=True))
     
    optimized_paras = [{'params': model_base.parameters(), 'lr': cfg.base_lr}]
    for joint_id in range(joint_num):
        if joint_types[joint_id] == 0:
            optimized_paras.append({'params': transform_scales[joint_id], 'lr': cfg.scale_lr_prismatic})
        else:
            optimized_paras.append({'params': joint_positions[joint_id], 'lr': cfg.pivot_lr})  
            optimized_paras.append({'params': transform_scales[joint_id], 'lr': cfg.scale_lr_revolute})
        optimized_paras.append({'params': qpos_lists_train[joint_id] , 'lr': cfg.qpos_lr})
        optimized_paras.append({'params': model_articulateds[joint_id].parameters(), 'lr': cfg.art_lr})
    optimizer = torch.optim.Adam(optimized_paras)

    output_check_list = [{'transform_scale': [], 'joint_position': [], 'qpos': []} for _ in range(joint_num)]
    zero_offset = torch.zeros((1, 1, 3), device=device)

    # Init Grid Transformers
    grid_transformers = []
    for joint_id in range(joint_num):
        grid_transformers.append(GridTransformer(joint_types[joint_id], joint_axes[joint_id], opt_dir=False))

    pbar = tqdm(range(1, cfg.total_iters + 1))    
    for i in pbar:
    
        sample_offset = generate_surround_offset(sample_interval).to(device)
        if sample_offset.shape[0] % 2 == 0:
            sample_offset = torch.cat([zero_offset, sample_offset], dim=0)  

        joint_idx = torch.randint(0, joint_num, (cfg.batch_size,), device=device) 
        joint_idx_item = joint_idx.item()
 
        idx = torch.randint(0, cfg.train_num_state, (cfg.batch_size,), device=device)
        idx_item = idx.item()
        
        grid = generate_image_grid_3d(64).to(device) # [64, 64, 64, 3] 
        grid = grid.view(1, -1, 3).repeat(cfg.batch_size, 1, 1) # [bs, 262144, 3]
        
        grid_base = grid.clone()
        grid_articulateds = []
        for joint_id in range(joint_num):
            grid_articulateds.append(grid.clone())
        
        if idx == 0:
            qpos = torch.zeros_like(transform_scales[joint_idx] * qpos_lists_train[joint_idx][idx])
        elif idx == cfg.train_num_state - 1:
            qpos = transform_scales[joint_idx] * torch.ones_like(transform_scales[joint_idx] * qpos_lists_train[joint_idx][idx-2])  
        else:
            qpos = transform_scales[joint_idx] * qpos_lists_train[joint_idx][idx-1]   

        voxels_base = model_base(grid_base.view(-1, 3)).unsqueeze(1)
        voxels_articulateds = []

        for joint_id in range(joint_num):
            current_qpos = qpos if joint_id == joint_idx_item else torch.zeros_like(qpos)
            grid_articulated = grid_transformers[joint_id].transform(grid_articulateds[joint_id],
                current_qpos, joint_positions[joint_id])
            grid_articulated_sampled = grid_articulated.clone().unsqueeze(0).repeat(27, 1, 1, 1)
            grid_articulated_sampled = (grid_articulated_sampled + sample_offset).detach() 
            sample_weight_articulated = calc_grid_weight(grid_articulated_sampled, grid_articulated.unsqueeze(0), coeff)
            voxels_articulateds.append((sample_weight_articulated * \
                model_articulateds[joint_id](grid_articulated_sampled.view(-1, 3)).view( 
                27, cfg.batch_size, 64, 64, 64)).sum(dim=0).unsqueeze(1))

        loss_intersect = 0
        voxels_in = voxels_base.clone()
        for joint_id in range(joint_num):
            voxels_in = torch.max(voxels_in, voxels_articulateds[joint_id]) 
            mask_intersect = torch.logical_and(voxels_base > 0.5, voxels_articulateds[joint_id] > 0.5)
            loss_intersect += torch.sum(voxels_base[mask_intersect] * voxels_articulateds[joint_id][mask_intersect]) 

        t = torch.rand(1, device=device) * (cfg.noise_end - cfg.noise_start) + cfg.noise_start
        t = 3 * t / (1 + 2 * t) # Rescale
        w = cfg.sds_weight
        t = t[:, None, None, None, None]
        
        latents = encoder(voxels_in, sample_posterior=False)
        noise = torch.randn((1, 8, 16, 16, 16), device=device)
        x_t = (1 - t) * latents + t * noise
        
        batch_cond = {'cond': conds[joint_idx_item]['cond'][idx], 'neg_cond': conds[joint_idx_item]['neg_cond'][idx]}
        x_0_pred, noise_pred = pipe.sparse_structure_sampler.sample_once_eps(diffusion, x_t, t,
            **batch_cond, cfg_strength=cfg.cfg_strength, cfg_interval=[0.5, 1.0])
        
        loss = 0
        loss_latent = w * F.mse_loss(latents, x_0_pred.detach())
        loss_latent = torch.nan_to_num(loss_latent) 
        
        voxels_pred = decoder(x_0_pred)
        voxels_activated = torch.sigmoid(voxels_pred)
        
        loss_voxel = 0
        masks = []
        masks.append(voxels_base == voxels_in)
        
        for joint_id in range(joint_num):
            masks.append(voxels_base == voxels_articulateds[joint_id])
        
        for j, mask in enumerate(masks):
            if torch.sum(mask) == 0:
                continue
            mask_coeff = 1 / len(masks)
            loss_voxel += mask_coeff * w * F.mse_loss(voxels_in[mask], voxels_activated[mask].detach())

        loss_voxel = torch.nan_to_num(loss_voxel * cfg.voxel_weight)
        loss_intersect = loss_intersect * cfg.intersect_weight
 
        # Total loss
        loss = loss_latent + loss_voxel + loss_intersect 
 
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        with torch.no_grad():
 
            output_check_list[joint_idx_item]['transform_scale'].append( \
                transform_scales[joint_idx_item].data.item())
            output_check_list[joint_idx_item]['qpos'].append(qpos_lists_train[joint_idx_item].data.cpu().numpy().tolist())
            if joint_types[joint_idx_item] == 1:
                output_check_list[joint_idx_item]['joint_position'].append( \
                joint_positions[joint_idx_item].data.cpu().numpy().tolist())
           
            if i % cfg.print_interval == 0:

                if joint_types[0] == 0:
                    pbar.set_postfix(scale=transform_scales[0].data)
                else:
                    pbar.set_postfix(scale=transform_scales[0].data, joint_position=joint_positions[0].data.cpu().numpy().tolist())

            if i % cfg.vis_interval == 0:
                vis_dir = f'{output_dir}/vis'
                os.makedirs(vis_dir, exist_ok=True)

                voxels_base_ori = model_base(grid_base[:1].view(-1, 3)).unsqueeze(1)
                voxels_articulated_oris = []
    
                for joint_id in range(joint_num):
                    if joint_id == joint_idx_item:
                        grid_articulated_sampled = grid_articulated.clone().unsqueeze(0).repeat(27, 1, 1, 1) # [27, bs, 262144, 3]
                        grid_articulated_sampled = (grid_articulated_sampled + sample_offset).detach()
                        sample_weight_articulated = calc_grid_weight(grid_articulated_sampled, grid_articulated.unsqueeze(0), coeff) # [27, bs, 64, 64, 64]
                        voxels_articulated_oris.append((sample_weight_articulated[:, :1] * \
                            model_articulateds[joint_id](grid_articulated_sampled[:, :1].reshape(-1, 3)).view(
                            27, 1, 64, 64, 64)).sum(dim=0).unsqueeze(1))
                    else:
                        voxels_articulated_oris.append(model_articulateds[joint_id](grid_articulateds[joint_id][:1].view(-1,
                            3)).unsqueeze(1))

                voxels_base_cond = voxels_base_ori > 0.5
                voxels_articulated_conds = []
                for joint_id in range(joint_num):
                    voxels_base_cond = torch.logical_and(voxels_base_cond, voxels_base_ori > voxels_articulated_oris[joint_id])
                    voxels_articulated_conds.append(torch.logical_and(voxels_articulated_oris[joint_id] > 0.5, 
                        voxels_articulated_oris[joint_id] > voxels_base_ori))
                    
                voxels_base_ori = torch.where(voxels_base_cond, 1, 0)
                voxels_articulateds_ori = []
                for joint_id in range(joint_num):
                    voxels_articulateds_ori.append(torch.where(voxels_articulated_conds[joint_id], 1, 0))

                for joint_id in range(joint_num):
                    visualize_voxels_two_parts(voxels_base_ori[0, 0].detach().cpu().numpy(),
                        voxels_articulateds_ori[joint_id][0, 0].detach().cpu().numpy(),
                        f'{vis_dir}/voxels_{i}_idx={idx_item:02d}_joint={joint_id:02d}.png', grid_normalizer=None)
                    
            if i % cfg.save_interval == 0:
                with open(f'{output_dir}/output_check_list.json', 'w') as f:
                    json.dump(output_check_list, f)
                torch.save(model_base.state_dict(), f'{output_dir}/model_base.pth')
                for joint_id in range(joint_num):
                    torch.save(model_articulateds[joint_id].state_dict(), f'{output_dir}/model_articulated_{joint_id:02d}.pth')
                with open(f'{output_dir}/output_check_list.json', 'w') as f:
                    json.dump(output_check_list, f)

    torch.save(model_base.state_dict(), f'{output_dir}/model_base.pth')
    for joint_id in range(joint_num):
        torch.save(model_articulateds[joint_id].state_dict(), f'{output_dir}/model_articulated_{joint_id:02d}.pth')
    with open(f'{output_dir}/output_check_list.json', 'w') as f:
        json.dump(output_check_list, f)

def recon(base_dir, output_dir, joint_num, cfg):
    
    if os.path.exists(f"{output_dir}/states/qpos_05.glb"):
        return

    seed_everything()
    torch.use_deterministic_algorithms(True)
    
    print(f"[SDS] Final Reconstructing...") 
 
    with open(f'{output_dir}/output_check_list.json', 'r') as f:
        output_check_list = json.load(f)
    
    init_ests = []
    joint_types = []
    transform_scales = []
    joint_positions = []
    joint_axes = [] 

    num_avg = 100
    joint_info = []

    for joint_id in range(joint_num):

        init_ests.append(np.load(f'{base_dir}/initialization/joint_{joint_id:02d}_est.npy', allow_pickle=True).item())
        joint_types.append(init_ests[-1]['joint_type']) 

        if joint_types[-1] == 0: # Prismatic
            joint_positions.append(np.zeros(3))
            joint_axes.append(init_ests[-1]['joint_axis'][::-1].astype(np.float32).copy())
        elif joint_types[-1] == 1: # Revolute
            joint_positions.append(output_check_list[joint_id]['joint_position'][-num_avg:])
            joint_positions[-1] = np.mean(joint_positions[-1], axis=0)
            joint_positions[-1] = torch.tensor(joint_positions[-1], device=device).float()
            joint_axes.append(-init_ests[-1]['joint_axis'][::-1].astype(np.float32).copy())    
        transform_scales.append(output_check_list[joint_id]['transform_scale'][-num_avg:])
        transform_scales[-1] = np.mean(transform_scales[-1]) 

        # Joint info
        current_joint_info = {}
        current_joint_info['type'] = 'prismatic' if joint_types[-1] == 0 else 'revolute'
        current_joint_info['range'] = [0, transform_scales[-1]]
        current_joint_info['axis'] = {}
        current_joint_info['axis']['origin'] = [0.0, 0.0, 0.0] if joint_types[-1] == 0 else (joint_positions[-1] / 63. -0.5).tolist()[::-1]
        current_joint_info['axis']['direction'] = (-joint_axes[-1][::-1]).tolist()
        joint_info.append(current_joint_info)
        
    with open(f'{output_dir}/joint_info.json', 'w') as f:
        json.dump(joint_info, f, indent=4)

    model_base = HashGridVoxel().to(device) 
    model_base.load_state_dict(torch.load(f'{output_dir}/model_base.pth', weights_only=True))
    model_articulateds = []
    for joint_id in range(joint_num):
        model_articulateds.append(HashGridVoxel().to(device))
        model_articulateds[-1].load_state_dict(torch.load(f'{output_dir}/model_articulated_{joint_id:02d}.pth', weights_only=True))
    model_base.eval()
    for joint_id in range(joint_num):
        model_articulateds[joint_id].eval()
     
    encoder = models.from_pretrained("JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").to(device)
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
    decoder = pipe.models['sparse_structure_decoder']
    diffusion = pipe.models['sparse_structure_flow_model']

    if os.path.exists(f'{base_dir}/renderings/rendering_joint_-1_state_00.png'):
        target_image = [Image.open(f'{base_dir}/renderings/rendering_joint_-1_state_00.png')]
        target_image_pure = [Image.open(f'{base_dir}/renderings/rendering_pure_joint_-1_state_00.png')]
    else:
        target_image = [Image.open(f'{base_dir}/renderings/rendering_joint_00_state_{cfg.train_num_state-1:02d}.png')]
        target_image_pure = [Image.open(f'{base_dir}/renderings/rendering_pure_joint_00_state_{cfg.train_num_state-1:02d}.png')]
    target_image = [pipe.preprocess_image(img) for img in target_image]
    target_image_pure = [pipe.preprocess_image(img) for img in target_image_pure]
    cond = [pipe.get_cond([img]) for img in target_image]
    cond_pure = [pipe.get_cond([img]) for img in target_image_pure] 

    model_base = HashGridVoxel().to(device) 
    model_base.load_state_dict(torch.load(f'{output_dir}/model_base.pth', weights_only=True))
    model_articulateds = []
    for joint_id in range(joint_num):
        model_articulateds.append(HashGridVoxel().to(device))
        model_articulateds[-1].load_state_dict(torch.load(f'{output_dir}/model_articulated_{joint_id:02d}.pth', weights_only=True))
    model_base.eval()
    for joint_id in range(joint_num):
        model_articulateds[joint_id].eval()
    
    coeff = torch.tensor(cfg.sample_coeff, device=device)
    sample_interval = torch.tensor(cfg.sample_interval, device=device)
    zero_offset = torch.zeros((1, 1, 3), device=device)

    sample_offset = generate_surround_offset(sample_interval).to(device)
    if sample_offset.shape[0] % 2 == 0:
        sample_offset = torch.cat([zero_offset, sample_offset], dim=0)  

    # Init Grid Transformer  
    grid_transformers = []
    for joint_id in range(joint_num):
        grid_transformers.append(GridTransformer(joint_types[joint_id], joint_axes[joint_id], opt_dir=False))
    idx = torch.tensor([cfg.train_num_state-1], device=device)

    grid = generate_image_grid_3d(64).to(device)
    grid = grid.view(-1, 3)
    
    grid_base = grid.clone()
    grid_articulateds = []

    qpos_list = []
    for joint_id in range(joint_num):
        qpos = transform_scales[joint_id] * idx / (cfg.inference_num_state - 1)
        qpos_list.append(qpos)
        grid_articulateds.append(grid.clone())

    voxels_base = model_base(grid_base.view(-1, 3)).unsqueeze(1)
    voxels_articulateds = []

    for joint_id in range(joint_num):
    
        current_qpos = qpos_list[joint_id] 
        grid_articulated = grid_transformers[joint_id].transform(grid_articulateds[joint_id],
            current_qpos, joint_positions[joint_id])
        grid_articulated_sampled = grid_articulated.clone().unsqueeze(0).repeat(27, 1, 1, 1)
        grid_articulated_sampled = (grid_articulated_sampled + sample_offset).detach() 
        sample_weight_articulated = calc_grid_weight(grid_articulated_sampled, grid_articulated.unsqueeze(0), coeff)
        voxels_articulateds.append((sample_weight_articulated * \
            model_articulateds[joint_id](grid_articulated_sampled.view(-1, 3)).view( 
            27, cfg.batch_size, 64, 64, 64)).sum(dim=0).unsqueeze(1)) 
    
    voxels_max = voxels_base.clone()
    for joint_id in range(joint_num):
        voxels_max = torch.max(voxels_max, voxels_articulateds[joint_id])

    part_mask = torch.zeros_like(voxels_base) 
    base_mask = voxels_base == voxels_max
    part_mask[base_mask] = 1
    art_masks = []
    for joint_id in range(joint_num):
        art_masks.append(voxels_articulateds[joint_id] == voxels_max) 
        part_mask[art_masks[-1]] = 2 + joint_id

    voxels_in = voxels_max  
    refine_noise = cfg.refine_noise  

    t = torch.ones(1, device=device) * refine_noise
    t = 3 * t / (1 + 2 * t) # Rescale
    t = t[:, None, None, None, None]
    latents = encoder(voxels_in, sample_posterior=False)
    noise = torch.randn((1, 8, 16, 16, 16), device=device)
    x_t = (1 - t) * latents + t * noise 
     
    latents = pipe.sparse_structure_sampler.sample_partial(diffusion, x_t, refine_noise, 25, 3.0,
        **cond[0], cfg_strength=cfg.cfg_strength, cfg_interval=[0.5, 1.0])
    voxels_refined = decoder(latents)
    voxels_refined = torch.where(voxels_refined > 0.5, 1, 0)
    visualize_voxels(voxels_refined[0, 0].detach().cpu().numpy(), f'{output_dir}/voxels_refined.png')
    
    # Post-processing
    print(f"[SDS] Post-processing...")

    # Iteratively delete the small disks
    first_del = True
    coords = torch.argwhere(voxels_refined > 0)[:, [0, 2, 3, 4]].int()
    disk_cond = torch.logical_or(torch.logical_or(coords[:, 1] <= 2, coords[:, 2] <= 2),
        torch.logical_or(coords[:, 1] >= 61, coords[:, 2] >= 61))
    
    while disk_cond.sum() > 0:

        coords_disk = coords[disk_cond]
        if not first_del and coords_disk[:, 3].min().int() - disk_height > 5:
            break
        disk_height = coords_disk[:, 3].min().int()
         
        if voxels_refined[:, :, :, :, disk_height].sum() < 100:
            voxels_refined[:, :, :, :, disk_height] = 0
            coords = torch.argwhere(voxels_refined > 0)[:, [0, 2, 3, 4]].int()
            disk_cond = torch.logical_or(torch.logical_or(coords[:, 1] <= 2, coords[:, 2] <= 2),
                torch.logical_or(coords[:, 1] >= 61, coords[:, 2] >= 61))
            continue
        
        coords_bottom = coords[coords[:, 3] == disk_height+1]
        if coords_bottom.shape[0] == 0:
            min_x, max_x = 32, 32
            min_y, max_y = 32, 32
        else:
            min_x, max_x = coords_bottom[:, 1].min().int(), coords_bottom[:, 1].max().int()
            min_y, max_y = coords_bottom[:, 2].min().int(), coords_bottom[:, 2].max().int()
        min_x, min_y = max(min_x, 2), max(min_y, 2)
        max_x, max_y = min(max_x, 61), min(max_y, 61)
        print(f'[Remove Disk] Disk height: {disk_height}, Disk size: {coords_disk.shape[0]}')
        print(f'[Recover Voxel] Min x: {min_x}, Max x: {max_x}, Min y: {min_y}, Max y: {max_y}') 

        if first_del:
            voxels_refined[:, :, :, :, :disk_height+1] = 0 
        else:
            voxels_refined[:, :, :, :, disk_height] = 0  
        
        coords = torch.argwhere(voxels_refined > 0)[:, [0, 2, 3, 4]].int()
        disk_cond = torch.logical_or(torch.logical_or(coords[:, 1] <= 2, coords[:, 2] <= 2),
            torch.logical_or(coords[:, 1] >= 61, coords[:, 2] >= 61))
        
        # Create a mask for the current disk height level
        xy_mask = torch.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=torch.bool, device=coords.device)
        
        # Mark the occupied edges on the mask
        for coord in coords_bottom:
            x, y = coord[1].int() - min_x, coord[2].int() - min_y
            if 0 <= x < xy_mask.shape[0] and 0 <= y < xy_mask.shape[1]:
                xy_mask[x, y] = True
        
        # Fill in the edges regardless of first_del
        for i in range(xy_mask.shape[0]):
            for j in range(xy_mask.shape[1]):
                if xy_mask[i, j]:
                    if min_x + i != 0 and min_y + j != 0:
                        voxels_refined[:, :, min_x + i, min_y + j, disk_height] = 1 
                        part_mask[:, :, min_x + i, min_y + j, disk_height] = \
                            torch.where(part_mask[:, :, min_x + i, min_y + j, disk_height+1] == 0,
                            1, part_mask[:, :, min_x + i, min_y + j, disk_height+1])

        # If first_del is True, also fill in the closures
        if first_del:
            # Flood fill from the boundaries to identify exterior regions
            filled_mask = flood_fill_exterior(xy_mask)
            
            # Invert the filled mask to get the closures (interior regions)
            closures = ~filled_mask & ~xy_mask
            
            # Apply the closures to voxels_refined
            for i in range(closures.shape[0]):
                for j in range(closures.shape[1]):
                    if closures[i, j]:
                        if min_x + i != 0 and min_y + j != 0:
                            voxels_refined[:, :, min_x + i, min_y + j, disk_height] = 1 
                            part_mask[:, :, min_x + i, min_y + j, disk_height] = \
                                torch.where(part_mask[:, :, min_x + i, min_y + j, disk_height+1] == 0,
                                1, part_mask[:, :, min_x + i, min_y + j, disk_height+1]) 
        
        first_del = False
    
    # Expand the articulated parts & fixed part
    def update_voxels(voxels_refined, part_mask):
        voxels_refined_arts = []
        voxel_refined_base_cond = torch.logical_and(voxels_refined > 0, part_mask == 1)
        voxels_refined_base = torch.where(voxel_refined_base_cond, voxels_refined, 0)
        voxels_refined_old = voxels_refined.clone()
        voxels_refined = voxels_refined_base.clone()
        for joint_id in range(joint_num): 
            voxel_refined_art_cond = torch.logical_and(voxels_refined_old > 0, part_mask == 2 + joint_id)
            voxels_refined_arts.append(torch.where(voxel_refined_art_cond, voxels_refined_old, 0))
            voxels_refined = torch.max(voxels_refined, voxels_refined_arts[joint_id])    
        return voxels_refined, voxels_refined_base, voxels_refined_arts

    def expand_parts(joint_id, voxels_refined_base, voxels_refined_arts, part_mask):
        
        expand_iteration = 100
        
        if joint_id == -1: # Base
            coords = torch.argwhere(voxels_refined_base > 0) 
            threshold = cfg.expand_threshold_base
            large_iteration = cfg.large_iteration_base
        else:
            coords = torch.argwhere(voxels_refined_arts[joint_id] > 0)
            threshold = cfg.expand_threshold_prismatic if joint_types[joint_id] == 0 else cfg.expand_threshold_revolute
            large_iteration = cfg.large_iteration_prismatic if joint_types[joint_id] == 0 else cfg.large_iteration_revolute
        
        print(f'[Expand Parts] Joint {joint_id}, Threshold: {threshold}')
        
        if coords.shape[0] == 0:
            return part_mask

        for iter in range(expand_iteration): 

            # Create a kernel for 26-neighborhood check
            kernel = torch.ones((3, 3, 3), device=voxels_refined.device)
            kernel[1, 1, 1] = 0  # Don't count the center voxel
            
            # Get the current parts and other parts
            current_mask = torch.logical_and(voxels_refined > 0, part_mask == 2 + joint_id)
            other_mask = torch.logical_and(voxels_refined > 0, torch.logical_and(part_mask >= 1,
                part_mask != 2 + joint_id))
            total_mask = torch.logical_and(voxels_refined > 0, part_mask >= 1)
            
            # Count different types of neighbors for each voxel
            current_neighbors = torch.nn.functional.conv3d(
                current_mask[:, :, :, :, :].float(), 
                kernel.view(1, 1, 3, 3, 3),
                padding=1
            )
            total_neighbors = torch.nn.functional.conv3d(
                total_mask[:, :, :, :, :].float(), 
                kernel.view(1, 1, 3, 3, 3),
                padding=1
            )   

            if iter < large_iteration:
                change_to_current = torch.logical_and(other_mask, current_neighbors / \
                    (total_neighbors + 1e-5) >= threshold - 0.05)
            else:
                change_to_current = torch.logical_and(other_mask, current_neighbors / \
                    (total_neighbors + 1e-5) >= threshold)
            part_mask[change_to_current] = 2 + joint_id
        
        return part_mask

    # Iteratively expand the parts with best order (base -> joints)
    voxels_refined, voxels_refined_base, voxels_refined_arts = update_voxels(voxels_refined, part_mask) 
    part_mask = expand_parts(-1, voxels_refined_base, voxels_refined_arts, part_mask)
    voxels_refined, voxels_refined_base, voxels_refined_arts = update_voxels(voxels_refined, part_mask)
    for joint_id in range(joint_num):
        part_mask = expand_parts(joint_id, voxels_refined_base, voxels_refined_arts, part_mask)
        voxels_refined, voxels_refined_base, voxels_refined_arts = update_voxels(voxels_refined, part_mask)  

    # Purge isolate parts according to connectivity
    purge_threshold = 80
    voxels_refined_base_np = voxels_refined_base[0, 0].detach().cpu().numpy()
    base_mask_np = (voxels_refined_base_np > 0).astype(np.int32)
    base_labeled, base_num_features = ndimage.label(base_mask_np)  

    base_sizes = {}
    for i in range(1, base_num_features + 1):
        base_sizes[i] = (base_labeled == i).sum()
        if base_sizes[i] < purge_threshold: 
            voxels_refined_base_np[base_labeled == i] = 0
            print(f"Purge base domain with size {base_sizes[i]}")
    voxels_refined_base[0, 0] = torch.from_numpy(voxels_refined_base_np).to(voxels_refined_base.device)

    for joint_id in range(joint_num):
        voxels_refined_art_np = voxels_refined_arts[joint_id][0, 0].detach().cpu().numpy()
        art_mask_np = (voxels_refined_art_np > 0).astype(np.int32)
        art_labeled, art_num_features = ndimage.label(art_mask_np)  
        art_sizes = {}
        for i in range(1, art_num_features + 1):
            art_sizes[i] = (art_labeled == i).sum()
            if art_sizes[i] < purge_threshold: 
                print(f"Purge articulated domain {joint_id:02d} with size {art_sizes[i]}")
                voxels_refined_art_np[art_labeled == i] = 0 
        voxels_refined_arts[joint_id][0, 0] = torch.from_numpy(voxels_refined_art_np).to(voxels_refined_arts[joint_id].device)
    
    voxels_refined = voxels_refined_base.clone()
    for joint_id in range(joint_num):
        voxels_refined = torch.max(voxels_refined, voxels_refined_arts[joint_id])
    
    # Update part_mask based on the refined voxels
    part_mask = torch.zeros_like(voxels_refined)
    part_mask[voxels_refined_base == voxels_refined] = 1
    for joint_id in range(joint_num): 
        part_mask[voxels_refined_arts[joint_id] == voxels_refined] = 2 + joint_id

    for joint_id in range(joint_num):
        visualize_voxels_two_parts(voxels_refined_base[0, 0].detach().cpu().numpy(),
            voxels_refined_arts[joint_id][0, 0].detach().cpu().numpy(),
            f'{output_dir}/voxels_refined_postprocessed_{joint_id:02d}.png')
 
    pipe.models['slat_decoder_mesh'].part_mask = part_mask
    pipe.models['slat_decoder_mesh'].joint_num = joint_num
    coords_refined = torch.argwhere(voxels_refined == 1)[:, [0, 2, 3, 4]].int() 
    slat = pipe.sample_slat(cond_pure[0], coords_refined, {})
    outputs = pipe.decode_slat(slat) 

    meshes = [] 
    for i in range(len(outputs['mesh'])):
        if outputs['mesh'][i] is not None:
            mesh = postprocessing_utils.to_glb(
                outputs['gaussian'][0], 
                outputs['mesh'][i],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            ) 
            meshes.append(mesh) 
        else:
            meshes.append(None)

    os.makedirs(f"{output_dir}/part_meshes", exist_ok=True)
    meshes[0].export(f"{output_dir}/part_meshes/fixed.glb")
    for joint_id in range(joint_num):
        if meshes[joint_id + 1] is not None:
            meshes[joint_id + 1].export(f"{output_dir}/part_meshes/articulated_{joint_id:02d}.glb")  

    os.makedirs(f"{output_dir}/states", exist_ok=True)
    pbar = tqdm(range(cfg.inference_num_state)) 
    for j in pbar:
        scene = trimesh.Scene()
        scene.add_geometry(meshes[0], geom_name='fixed_part')
        for joint_id in range(joint_num):
            if meshes[joint_id+1] is None:
                continue
            if joint_types[joint_id] == 1:
                axis_dir = -joint_axes[joint_id][::-1]
            else:
                axis_dir = joint_axes[joint_id]
            qpos = transform_scales[joint_id] * j / (cfg.inference_num_state - 1)
            transformed_mesh = meshes[joint_id+1].copy()
            if joint_types[joint_id] == 0:  # Prismatic 
                vertices = transformed_mesh.vertices.copy()
                vertices[:, [1, 2]] = vertices[:, [2, 1]]
                vertices[:, 1] = -vertices[:, 1]
                translation_vector = axis_dir * qpos 
                vertices += translation_vector
                vertices[:, 1] = -vertices[:, 1]
                vertices[:, [1, 2]] = vertices[:, [2, 1]] 
                transformed_mesh.vertices = vertices
            elif joint_types[joint_id] == 1:  # Revolute
                vertices = transformed_mesh.vertices.copy()
                vertices[:, [1, 2]] = vertices[:, [2, 1]]
                vertices[:, 1] = -vertices[:, 1]
                vertices -= (joint_positions[joint_id].cpu().numpy()[::-1] / 63. - 0.5)  
                cos_theta = np.cos(qpos)
                sin_theta = np.sin(qpos)
                one_minus_cos = 1 - cos_theta
                ux, uy, uz = axis_dir 
                rotation_matrix = np.array([
                    [cos_theta + ux**2 * one_minus_cos, ux * uy * one_minus_cos - uz * sin_theta, ux * uz * one_minus_cos + uy * sin_theta],
                    [uy * ux * one_minus_cos + uz * sin_theta, cos_theta + uy**2 * one_minus_cos, uy * uz * one_minus_cos - ux * sin_theta],
                    [uz * ux * one_minus_cos - uy * sin_theta, uz * uy * one_minus_cos + ux * sin_theta, cos_theta + uz**2 * one_minus_cos]
                ])
                rotated_vertices = np.dot(vertices, rotation_matrix.T)
                rotated_vertices += joint_positions[joint_id].cpu().numpy()[::-1] / 63. - 0.5
                rotated_vertices[:, 1] = -rotated_vertices[:, 1]
                rotated_vertices[:, [1, 2]] = rotated_vertices[:, [2, 1]] 
                transformed_mesh.vertices = rotated_vertices
            scene.add_geometry(transformed_mesh, geom_name=f'articulated_part_{joint_id:02d}')
        output_idx = j if idx == 0 else cfg.inference_num_state - j - 1
        scene.export(f"{output_dir}/states/qpos_{output_idx:02d}.glb")

def run_sds(base_dir, joint_num, cfg, stage='all'):
   
    output_dir = f'{base_dir}/sds_output'  

    if stage == 'all' or stage == 'train':
        initialize(base_dir, cfg) 
        train(base_dir, output_dir, joint_num, cfg)  
    if stage == 'all' or stage == 'recon':
        recon(base_dir, output_dir, joint_num, cfg) 

    full_mesh = f'{output_dir}/states/qpos_05.glb'
    fixed_part = f'{output_dir}/part_meshes/fixed.glb'
    articulated_part = f'{output_dir}/part_meshes/articulated_00.glb'

    return full_mesh, fixed_part, articulated_part
