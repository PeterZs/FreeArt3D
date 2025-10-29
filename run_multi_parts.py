import os 
import argparse
import shutil
import json
import numpy as np

from omegaconf import OmegaConf
from pipelines.recon import run_recon
from pipelines.estimate import run_estimate
from pipelines.sds import run_sds
from pipelines.urdf import write_urdf
from pipelines.utils.seeding import seed_everything
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml') 
    parser.add_argument('--meta_data', type=str, default='datasets/multi_joint/meta.json')
    parser.add_argument('--input_dir', type=str, default='datasets/multi_joint/46180') 
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--stage', type=str, default='all') 
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    seed_everything(args.seed) 

    data_name = args.input_dir.split('/')[-1]
    output_dir = f'{args.output_dir}/{data_name}'
    recon_dir = f'{args.output_dir}/{data_name}/recon'

    with open(args.meta_data, 'r') as f:
        meta_data = json.load(f)

    joint_types = meta_data['joint_types'][data_name]
    num_joints = len(joint_types)  
    cfg.ransac_threshold = 0.03
    cfg.min_azi = -np.pi / 4
    cfg.max_azi = np.pi / 4  

    for joint_id in range(num_joints):

        image_paths = [f'{args.input_dir}/joint_{joint_id:02d}_state_{i:02d}.png' for i in range(cfg.train_num_state)]
        image_paths_pure = [f'{args.input_dir}/joint_{joint_id:02d}_state_{i:02d}_pure.png' for i in range(cfg.train_num_state)]
        os.makedirs(f'{output_dir}/renderings', exist_ok=True)
        for i in range(len(image_paths)):
            shutil.copy(image_paths[i], f'{output_dir}/renderings/rendering_joint_{joint_id:02d}_state_{i:02d}.png') 
        shutil.copy(f'{args.input_dir}/joint_all_state_05.png', f'{output_dir}/renderings/rendering_joint_-1_state_00.png')
        shutil.copy(f'{args.input_dir}/joint_all_state_05_pure.png', f'{output_dir}/renderings/rendering_pure_joint_-1_state_00.png')
    
    if args.stage == 'all' or args.stage == 'recon':
        image_paths = [f'{output_dir}/renderings/rendering_joint_-1_state_00.png']
        run_recon(image_paths, recon_dir, joint_idx=-1, min_azi=cfg.min_azi, max_azi=cfg.max_azi)
        for joint_idx in range(num_joints):
            image_paths = [f'{output_dir}/renderings/rendering_joint_{joint_idx:02d}_state_{i:02d}.png' for i in range(cfg.train_num_state)]
            run_recon(image_paths, recon_dir, joint_idx=joint_idx, multi_joints=True, min_azi=cfg.min_azi, max_azi=cfg.max_azi)

    if args.stage == 'all' or args.stage == 'estimate':
        for joint_idx in range(num_joints):
            image_paths = [f'{output_dir}/renderings/rendering_joint_{joint_idx:02d}_state_{i:02d}.png' for i in range(cfg.train_num_state)]
            run_estimate(image_paths, output_dir, joint_types[joint_idx], cfg=cfg, joint_idx=joint_idx) 

    if args.stage == 'all' or args.stage == 'sds':
        run_sds(output_dir, num_joints, cfg, stage='all')
    
    write_urdf(
        base_mesh=f"{output_dir}/sds_output/part_meshes/fixed.glb",
        part_meshes=[f"{output_dir}/sds_output/part_meshes/articulated_{i:02d}.glb" for i in range(num_joints)],
        joint_config=f'{output_dir}/sds_output/joint_info.json',
        urdf_path=f"{output_dir}/output.urdf",
        robot_name=data_name,
    )