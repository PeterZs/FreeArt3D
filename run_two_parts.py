import os 
import argparse
import shutil
import numpy as np
import json

from omegaconf import OmegaConf
from pipelines.recon import run_recon
from pipelines.estimate import run_estimate
from pipelines.sds import run_sds
from pipelines.urdf import write_urdf 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml') 
    parser.add_argument('--partnet_config', type=str, default='configs/partnet.json') 
    parser.add_argument('--joint_type', type=str, default='prismatic')
    parser.add_argument('--input_dir', type=str, default='examples/cabinet') 
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--stage', type=str, default='all') 
    parser.add_argument('--partnet', action='store_true')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)  

    data_name = args.input_dir.split('/')[-1]
    output_dir = f'{args.output_dir}/{data_name}'
    recon_dir = f'{args.output_dir}/{data_name}/recon' 
 
    if os.path.exists(f'{args.input_dir}/05_seg.png'): # Example  
        image_paths = [f'{args.input_dir}/{i:02d}_seg.png' for i in range(cfg.train_num_state)]
        image_paths.append(f'{args.input_dir}/05_pure.png')
    else: # PartNet default rendering name
        image_paths = [f'{args.input_dir}/rendering_joint_00_state_{i:02d}.png' for i in range(cfg.train_num_state)]
        image_paths.append(f'{args.input_dir}/rendering_pure_joint_00_state_05.png')

    os.makedirs(f'{output_dir}/renderings', exist_ok=True)
    for i in range(len(image_paths) - 1):
        shutil.copy(image_paths[i], f'{output_dir}/renderings/rendering_joint_00_state_{i:02d}.png')
    shutil.copy(image_paths[-1], f'{output_dir}/renderings/rendering_pure_joint_00_state_05.png')
    image_paths = image_paths[:-1]
    
    if args.partnet: # Use config for PartNet-Mobility Articulation test set
        with open(args.partnet_config, "r") as f:
            partnet_meta = json.load(f)
        if data_name not in partnet_meta['category_ids']:
            raise ValueError(f"Not data from PartNet test set")
        joint_type = 'prismatic' if data_name in partnet_meta['prismatic']['obj_ids'] else 'revolute'
        category = partnet_meta['category_ids'][data_name]
        args.joint_type = joint_type  

        # We found the joint initialization is sensitive to the RANSAC threshold, the default value works for most cases,
        # but you may tune it if the optimization failed (we tried 0.03 & 0.1 and it works better for some cases)
        # cfg.ransac_threshold = 0.1

        # It's also important to render the coarse reconstruction at a good view to avoid blind spots.
        # Here we only found the WashingMachine need to be rendered at default views
        if category in ['WashingMachine']: 
            cfg.min_azi = -np.pi / 3
            cfg.max_azi = -np.pi / 4

    if args.stage == 'all' or args.stage == 'recon':
        run_recon(image_paths, recon_dir, min_azi=cfg.min_azi, max_azi=cfg.max_azi)
    if args.stage == 'all' or args.stage == 'estimate':
        run_estimate(image_paths, output_dir, args.joint_type, cfg) 
    if args.stage == 'all' or args.stage == 'sds':
        run_sds(output_dir, 1, cfg, stage='all')
    
    write_urdf(
        base_mesh=f"{output_dir}/sds_output/part_meshes/fixed.glb",
        part_meshes=[f"{output_dir}/sds_output/part_meshes/articulated_00.glb"],
        joint_config=f"{output_dir}/sds_output/joint_info.json",
        urdf_path=f"{output_dir}/output.urdf",
        robot_name=data_name,
    )
