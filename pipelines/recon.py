import os 
import sys
sys.path.append('TRELLIS')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
 
from PIL import Image
from trellis.modules import sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from pipelines.utils.seeding import seed_everything
from pipelines.utils.visualization import visualize_voxels
from pipelines.render import run_rendering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def run_recon(image_paths, output_dir, joint_idx=0, multi_joints=False, min_azi=np.pi / 8, max_azi=3 * np.pi / 8, app=False):
    
    os.makedirs(output_dir, exist_ok=True)
    total_states = len(image_paths)
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()
      
    target_image = []
    for i in range(total_states):
        target_image.append(Image.open(image_paths[i]))
    target_image = [pipe.preprocess_image(img) for img in target_image]
    cond = pipe.get_cond(target_image)
    
    sampler_params = {}
    sampler_params = {**pipe.sparse_structure_sampler_params, **sampler_params}
    sampler_params['cfg_interval'] = [0.5, 1.0]
    sampler_params['cfg_strength'] = 10

    # Voxel Reconstruction
    if not os.path.exists(f'{output_dir}/recon_voxel_joint_{joint_idx:02d}_state_{total_states-1:02d}.npy'): 
        
        seed_everything()
        noise = torch.randn((1, 8, 16, 16, 16), device=device).repeat(total_states, 1, 1, 1, 1) 
        sample = pipe.sparse_structure_sampler.sample(pipe.models['sparse_structure_flow_model'], noise,
            **cond, **sampler_params, verbose=True).samples
        voxels = pipe.models['sparse_structure_decoder'](sample)
        voxels = torch.where(voxels > 0.5, 1.0, 0.0)
    
        for i in range(total_states): 
            np.save(f'{output_dir}/recon_voxel_joint_{joint_idx:02d}_state_{i:02d}.npy', voxels[i, 0].detach().cpu().numpy())
            visualize_voxels(voxels[i, 0].detach().cpu().numpy(), f'{output_dir}/recon_voxel_joint_{joint_idx:02d}_state_{i:02d}.png')
            
    # Mesh Reconstruction
    for i in range(total_states):

        if os.path.exists(f'{output_dir}/recon_mesh_joint_{joint_idx:02d}_state_{i:02d}.glb'):
            continue
        
        seed_everything()
        torch.use_deterministic_algorithms(True)
        voxel = np.load(f'{output_dir}/recon_voxel_joint_{joint_idx:02d}_state_{i:02d}.npy')
        voxel = torch.from_numpy(voxel)[None, None].to(device)
        current_cond = {'cond': cond['cond'][i:i+1], 'neg_cond': cond['neg_cond'][i:i+1]}
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        slat = pipe.sample_slat(current_cond, coords, pipe.slat_sampler_params, noise=None)
        outputs = pipe.decode_slat(slat)
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
            baking_mode='fast',
        )
        glb.export(f'{output_dir}/recon_mesh_joint_{joint_idx:02d}_state_{i:02d}.glb') 

    # Render meshes
    glb_paths = [f'{output_dir}/recon_mesh_joint_{joint_idx:02d}_state_{i:02d}.glb' for i in range(total_states)]
    recon_voxel_paths = [f'{output_dir}/recon_voxel_joint_{joint_idx:02d}_state_{i:02d}.png' for i in range(total_states)]

    rendering_dir = output_dir.replace('/recon', '/renderings_recon')
    if multi_joints:
        glb_paths.append(f'{output_dir}/recon_mesh_joint_-1_state_00.glb')  
    
    if app:
        return glb_paths, rendering_dir, recon_voxel_paths
    else:
        recon_mesh_paths = run_rendering(glb_paths, rendering_dir, min_azi=min_azi, max_azi=max_azi, recon=True, joint_idx=joint_idx)
        return recon_voxel_paths, recon_mesh_paths