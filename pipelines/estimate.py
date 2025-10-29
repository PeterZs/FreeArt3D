import sys
sys.path.append('gim')
from networks.dkm.models.model_zoo.DKMv3 import DKMv3
from tools import get_padding_size

import numpy as np
import torch 
import os 
import pyexr 

from PIL import Image
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

from pipelines.utils.seeding import seed_everything

### Feature Matching
class FeatureMatchingConfig:

    def __init__(self, num_states, joint_type): 

        self.base_min_diff = 0.05  
        self.base_max_diff = 0.2 
        self.step_min_increment = 0.1 / (num_states - 1)
        self.step_max_increment = 0.5 / (num_states - 1) 
        self.floor_threshhold = 0.03 
        self.depth_threshhold = 0.02
        self.min_corr = 500 if joint_type == 'prismatic' else 200

def update_points(keypoints_0, keypoints_1, pos_0, pos_1, valid):

    keypoints_0, keypoints_1 = keypoints_0[valid], keypoints_1[valid]
    pos_0, pos_1 = pos_0[valid], pos_1[valid]
    return keypoints_0, keypoints_1, pos_0, pos_1

def match_key_points(image_0, image_1, matcher):
    
    width, height = 672, 896
    orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image_0, width, height)
    orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image_1, width, height)
    
    image_0_ = torch.nn.functional.pad(image_0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
    image_1_ = torch.nn.functional.pad(image_1, (pad_left1, pad_right1, pad_top1, pad_bottom1))
    dense_matches, dense_certainty = matcher.match(image_0_, image_1_)  
    sparse_matches, mconf = matcher.sample(dense_matches, dense_certainty, 5000)
    
    height0, width0 = image_0_.shape[-2:]
    height1, width1 = image_1_.shape[-2:]

    kpts0 = sparse_matches[:, :2]
    kpts0 = torch.stack((width0 * (kpts0[:, 0] + 1) / 2,
        height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
    kpts1 = sparse_matches[:, 2:]
    kpts1 = torch.stack((width1 * (kpts1[:, 0] + 1) / 2,
        height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)
    b_ids = torch.where(mconf[None])[0]

    # before padding
    kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
    kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
    mask_ = (kpts0[:, 0] > 0) & \
            (kpts0[:, 1] > 0) & \
            (kpts1[:, 0] > 0) & \
            (kpts1[:, 1] > 0)
    mask_ = mask_ & \
            (kpts0[:, 0] <= (orig_width0 - 1)) & \
            (kpts1[:, 0] <= (orig_width1 - 1)) & \
            (kpts0[:, 1] <= (orig_height0 - 1)) & \
            (kpts1[:, 1] <= (orig_height1 - 1))

    mconf = mconf[mask_]
    b_ids = b_ids[mask_]
    kpts0 = kpts0[mask_]
    kpts1 = kpts1[mask_]
    return kpts0, kpts1, mconf

def get_position(keypoints, pos_map, depth_map, floor_height, cfg):
    
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    offsets = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1],[1, -1], [1, 0], [1, 1]])
    neighbor_coords = np.stack([ys, xs], axis=1)[:, None, :] + offsets[None, :, :]
    H, W = pos_map.shape[:2]
    neighbor_coords[..., 0] = np.clip(neighbor_coords[..., 0], 0, H - 1)
    neighbor_coords[..., 1] = np.clip(neighbor_coords[..., 1], 0, W - 1)
    neighbor_ys = neighbor_coords[..., 0]
    neighbor_xs = neighbor_coords[..., 1]

    pos = pos_map[neighbor_ys, neighbor_xs] # (N, 9, 3)
    depth = depth_map[neighbor_ys, neighbor_xs] # (N, 9)
    depth_min = depth.min(axis=1) # (N,)
    norms = np.linalg.norm(pos, axis=-1)
    diff_floor = np.abs(pos[..., 2] - floor_height)
    diff_depth = np.abs(depth - depth_min[..., None])
    mask = (norms > 0) & (diff_floor > cfg.floor_threshhold) & (diff_depth < cfg.depth_threshhold)

    valid_pos = pos * mask[..., None]
    valid_counts = mask.sum(axis=1)  # (N,)
    sum_pos = valid_pos.sum(axis=1)  # (N, 3)
    mean_pos = np.zeros_like(sum_pos)
    valid_mask = valid_counts > 0
    mean_pos[valid_mask] = sum_pos[valid_mask] / valid_counts[valid_mask, None]
    
    return mean_pos

def calculate_shift(output_dir, num_states, view_idx, joint_idx, matcher, cfg):

    shift = [np.zeros(3)]
 
    for i in tqdm(range(1, num_states)):

        image_0_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_00_view_{view_idx:02d}.png'
        image_1_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}.png'
        image_0_pos_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_00_view_{view_idx:02d}_position0000.exr'
        image_1_pos_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}_position{i:04d}.exr'
        image_0_depth_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_00_view_{view_idx:02d}_depth0000.exr'
        image_1_depth_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}_depth{i:04d}.exr'

        image_0_rgba = Image.open(image_0_path).convert('RGBA')
        image_1_rgba = Image.open(image_1_path).convert('RGBA') 
        image_0_rgba = np.array(image_0_rgba)
        image_1_rgba = np.array(image_1_rgba)
        image_0 = torch.from_numpy(image_0_rgba)[..., :3].unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        image_1 = torch.from_numpy(image_1_rgba)[..., :3].unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        image_0 = image_0.cuda()
        image_1 = image_1.cuda()
        image_0_pos = pyexr.open(image_0_pos_path).get()
        image_1_pos = pyexr.open(image_1_pos_path).get()
        image_0_depth = pyexr.open(image_0_depth_path).get()[:, :, 0]
        image_1_depth = pyexr.open(image_1_depth_path).get()[:, :, 0]

        # Get floor height
        pos_length = np.linalg.norm(image_0_pos, axis=2)
        image_0_pos_valid = image_0_pos[pos_length > 1e-5]
        floor_height = image_0_pos_valid[:, 2].min() 

        keypoints_0, keypoints_1, conf = match_key_points(image_0, image_1, matcher)
        keypoints_0 = keypoints_0.cpu().numpy().astype(np.int32)
        keypoints_1 = keypoints_1.cpu().numpy().astype(np.int32)
        conf = conf.cpu().numpy()

        pos_0 = get_position(keypoints_0, image_0_pos, image_0_depth, floor_height, cfg)
        pos_1 = get_position(keypoints_1, image_1_pos, image_1_depth, floor_height, cfg)
        pos_0_length = np.linalg.norm(pos_0, axis=1)    
        pos_1_length = np.linalg.norm(pos_1, axis=1)

        diff = np.linalg.norm(pos_0 - pos_1, axis=1)
        valid_pos = np.logical_and(pos_0_length > 1e-5, pos_1_length > 1e-5)
        match_mask = np.logical_and(valid_pos, conf > 0.0)

        if len(pos_0[match_mask]) == 0:
            shift.append(np.zeros(3))
            continue

        least_match_diff = sorted(diff[match_mask])[int(len(diff[match_mask]) * 0.1)]
        static_mask = np.logical_and(valid_pos, np.logical_and(conf > 0.0, diff < least_match_diff)) 

        if len(pos_0[static_mask]) == 0:
            shift.append(np.zeros(3))
            continue

        pos_0_static = pos_0[static_mask]
        pos_1_static = pos_1[static_mask]
        shift.append((pos_0_static - pos_1_static).mean(axis=0))

    return shift

def find_correspondence(output_dir, num_states, view_idx, joint_idx, matcher, cfg, shift):

    images_rgb, images_grey, images_pos, images_depth = [], [], [], []
    for i in range(num_states):
        image_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}.png'
        image_pos_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}_position{i:04d}.exr'
        image_depth_path = f'{output_dir}/renderings_recon/rendering_recon_joint_{joint_idx:02d}_state_{i:02d}_view_{view_idx:02d}_depth{i:04d}.exr'
        images_rgb.append(np.array(Image.open(image_path).convert('RGB')))
        images_grey.append(np.array(Image.open(image_path).convert('L')))
        images_pos.append(pyexr.open(image_pos_path).get())
        images_depth.append(pyexr.open(image_depth_path).get()[:, :, 0])
    images_rgb = np.stack(images_rgb, axis=0)
    images_grey = np.stack(images_grey, axis=0)
    images_pos = np.stack(images_pos, axis=0)
    
    max_interval = 2
    intervals = range(1, max_interval+1)    
    total_correspondence = 0 

    for interval in intervals:
        for i in tqdm(range(num_states - interval)):

            image_0_rgb = images_rgb[i]
            image_1_rgb = images_rgb[i + interval]
            image_0 = torch.from_numpy(image_0_rgb).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
            image_1 = torch.from_numpy(image_1_rgb).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
            image_0 = image_0.cuda()
            image_1 = image_1.cuda()
            image_0_pos = images_pos[i]
            image_1_pos = images_pos[i + interval]
            image_0_depth = images_depth[i]
            image_1_depth = images_depth[i + interval]  

            # Get floor height
            pos_length = np.linalg.norm(image_0_pos, axis=2)
            image_0_pos_valid = image_0_pos[pos_length > 1e-5]
            floor_height = image_0_pos_valid[:, 2].min() 

            keypoints_0, keypoints_1, conf = match_key_points(image_0, image_1, matcher)
            keypoints_0 = keypoints_0.cpu().numpy().astype(np.int32)
            keypoints_1 = keypoints_1.cpu().numpy().astype(np.int32)
            conf = conf.cpu().numpy()

            pos_0 = get_position(keypoints_0, image_0_pos, image_0_depth, floor_height, cfg)
            pos_1 = get_position(keypoints_1, image_1_pos, image_1_depth, floor_height, cfg)
            pos_0_length = np.linalg.norm(pos_0, axis=1)
            pos_1_length = np.linalg.norm(pos_1, axis=1) 
            pos_0_shift = pos_0 + shift[i][None]
            pos_1_shift = pos_1 + shift[i+interval][None]

            diff = np.linalg.norm(pos_0_shift - pos_1_shift, axis=1)
            valid_pos = np.logical_and(pos_0_length > 1e-5, pos_1_length > 1e-5)
            
            base_min_diff = cfg.base_min_diff
            base_max_diff = cfg.base_max_diff
            min_increment = cfg.step_min_increment * interval
            max_increment = cfg.step_max_increment * interval 
            dynamic_mask = np.logical_and(valid_pos, np.logical_and(conf > 0.0,
                np.logical_and(diff > base_min_diff + min_increment, diff < base_max_diff + max_increment)))

            keypoints_0, keypoints_1, pos_0, pos_1 = update_points(keypoints_0, keypoints_1, pos_0, pos_1, dynamic_mask)
            if len(pos_0) == 0:
                diff = None
            else:
                diff = np.linalg.norm((pos_0 + shift[i][None]) - (pos_1 + shift[i+interval][None]), axis=1)
                diff_max = sorted(diff)[-1]  

            # Visualization
            H0, W0, H1, W1 = *image_0_rgb.shape[:2], *image_1_rgb.shape[:2]
            img = np.concatenate((np.array(image_0_rgb), np.array(image_1_rgb)), axis=1)
            fig, ax = plt.subplots(dpi=300)
            ax.imshow(img)
            cmap = plt.get_cmap('jet')
        
            N = keypoints_0.shape[0]
            for p in range(N):
                (x0, y0), (x1, y1) = keypoints_0[p].T, keypoints_1[p].T
                ax.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(diff[p] / diff_max), scalex=False, 
                    scaley=False, linewidth=0.5, markersize=0.1)  
            fig.savefig(f'{output_dir}/correspondences/correspondence_joint_{joint_idx:02d}_state_{i:02d}-{i+interval:02d}_view_{view_idx:02d}.png', bbox_inches='tight')
            correspondence = np.concatenate((pos_0 + shift[i][None], pos_1 + shift[i+interval][None]), axis=1) 
            np.save(f'{output_dir}/correspondences/correspondence_joint_{joint_idx:02d}_state_{i:02d}-{i+interval:02d}_view_{view_idx:02d}.npy', correspondence)
            total_correspondence += correspondence.shape[0] 

    print(f"[Joint {joint_idx:02d} View {view_idx:02d}] Find Total correspondence: {total_correspondence}") 
    if total_correspondence < cfg.min_corr or cfg.base_min_diff <= 0.01:
        print(f"[Joint {joint_idx:02d} View {view_idx:02d}] Not enough correspondence, relax the constraint")
        cfg.base_min_diff -= 0.01
        cfg.step_min_increment -= 0.01 
        return False
    return True

### Estimate scale

def rodrigues_rotation_matrix(v, theta):

    v = v / np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    I = np.eye(3).reshape(1, 3, 3)
    K = K.reshape(1, 3, 3)
    theta = theta.reshape(-1, 1, 1)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R

def rotation_residual(params, X1, X2, axis, qpos_X1, qpos_X2, num_states):

    theta = params[0]
    p = params[1:4].reshape(1, 3, 1)
    qpos = np.concatenate([np.zeros(1), params[4:], np.ones(1)], axis=0) 
    theta_X1 = theta * qpos[qpos_X1]
    theta_X2 = theta * qpos[qpos_X2]
    R_X1 = rodrigues_rotation_matrix(axis, theta_X1)
    R_X2 = rodrigues_rotation_matrix(axis, theta_X2)
    X1 = X1.reshape(-1, 3, 1)
    X2 = X2.reshape(-1, 3, 1)
    X1_rot = R_X1 @ (X1 - p) + p
    X2_rot = R_X2 @ (X2 - p) + p
    return (X1_rot - X2_rot).ravel()

def solve_rotation_angle_and_axis_position(X1, X2, axis, qpos_X1, qpos_X2, num_states):
    
    X1 = np.asarray(X1)  # shape [N, 3]
    X2 = np.asarray(X2)
    axis = axis / np.linalg.norm(axis)

    init_theta = np.zeros(1)
    init_qpos = np.arange(1, num_states - 1) / (num_states - 1)
    init_p = np.zeros(3)
    x0 = np.concatenate([init_theta, init_p, init_qpos], axis=0)  # shape 1 + 3 + 4
    min_bounds = np.concatenate([np.ones(1) * -3.14, np.ones(3) * -0.2, np.zeros(num_states-2)], axis=0)
    max_bounds = np.concatenate([np.ones(1) * 3.14, np.ones(3) * 0.2, np.ones(num_states-2)], axis=0)
    result = least_squares(rotation_residual, x0, args=(X1, X2, axis, qpos_X1, qpos_X2, num_states),
        bounds=(min_bounds, max_bounds), method='trf') 
    theta_opt = result.x[0]
    p_opt = result.x[1:4] 
    qpos_opt = np.concatenate([np.zeros(1), result.x[4:], np.ones(1)], axis=0)

    return theta_opt, p_opt, qpos_opt

def compute_rotation_reprojection_error(X1, X2, theta, p, qpos, axis, qpos_X1, qpos_X2): 

    p = p.reshape(1, 3, 1)
    theta_X1 = theta * qpos[qpos_X1]
    theta_X2 = theta * qpos[qpos_X2]
    R_X1 = rodrigues_rotation_matrix(axis, theta_X1)
    R_X2 = rodrigues_rotation_matrix(axis, theta_X2)
    X1 = X1.reshape(-1, 3, 1)
    X2 = X2.reshape(-1, 3, 1)
    X1_rot = R_X1 @ (X1 - p) + p
    X2_rot = R_X2 @ (X2 - p) + p
    errors = np.linalg.norm(X1_rot - X2_rot, axis=1)
    return errors

def translation_residual(params, X1, X2, axis, qpos_X1, qpos_X2):

    d = params[0]
    qpos = np.concatenate([np.zeros(1), params[1:], np.ones(1)], axis=0) 
    d_X1 = d * qpos[qpos_X1]
    d_X2 = d * qpos[qpos_X2] 
    d_X1 = d_X1.reshape(-1, 1)
    d_X2 = d_X2.reshape(-1, 1)
    axis = axis.reshape(1, 3)
    X1_translate = X1 + d_X1 * axis
    X2_translate = X2 + d_X2 * axis
    return (X1_translate - X2_translate).ravel()

def solve_translation_scale(X1, X2, axis, qpos_X1, qpos_X2, num_states):
    
    X1 = np.asarray(X1)  # shape [N, 3]
    X2 = np.asarray(X2)
    axis = axis / np.linalg.norm(axis)

    init_d = np.zeros(1)
    init_qpos = np.arange(1, num_states - 1) / (num_states - 1)
    x0 = np.concatenate([init_d, init_qpos],axis=0)  # shape 1 + 18
    min_bounds = np.concatenate([np.ones(1) * -1, np.zeros(num_states-2)], axis=0)
    max_bounds = np.concatenate([np.ones(1) * 1, np.ones(num_states-2)], axis=0)
    result = least_squares(translation_residual, x0, args=(X1, X2, axis, qpos_X1, qpos_X2),
        bounds=(min_bounds, max_bounds), method='trf')
    d_opt = result.x[0]
    qpos_opt = np.concatenate([np.zeros(1), result.x[1:], np.ones(1)], axis=0)
    return d_opt, qpos_opt

def compute_translation_reprojection_error(X1, X2, d, qpos, axis, qpos_X1, qpos_X2): 

    d_X1 = d * qpos[qpos_X1]
    d_X2 = d * qpos[qpos_X2]
    d_X1 = d_X1.reshape(-1, 1)
    d_X2 = d_X2.reshape(-1, 1)
    axis = axis.reshape(1, 3)
    X1_translate = X1 + d_X1 * axis
    X2_translate = X2 + d_X2 * axis
    errors = np.linalg.norm(X1_translate - X2_translate, axis=1)
    return errors

def estimate_scale(output_dir, num_states, axis_type, joint_idx, cfg):
 
    max_interval = 2
    total_views = 3
    intervals = range(1, max_interval+1)
    
    # Load correspondences
    corr = []
    qpos_table_X1 = []
    qpos_table_X2 = []
    for view_idx in range(total_views):
        for interval in intervals:
            for i in range(num_states - interval): 
                current_corr = np.load(f"{output_dir}/correspondences/correspondence_joint_{joint_idx:02d}_state_{i:02d}-{i+interval:02d}_view_{view_idx:02d}.npy") 
                corr.append(current_corr)
                qpos_table_X1.append(np.ones(current_corr.shape[0]) * i)
                qpos_table_X2.append(np.ones(current_corr.shape[0]) * (i + interval))
    corr = np.concatenate(corr, axis=0)
    qpos_table_X1 = np.concatenate(qpos_table_X1, axis=0).astype(np.int32)
    qpos_table_X2 = np.concatenate(qpos_table_X2, axis=0).astype(np.int32)
    X1, X2 = corr[:, :3], corr[:, 3:6]
    
    # Predict the joint axis (it should be axis-aligned)
    diff_vectors = X1 - X2  
    if axis_type == 'revolute': 
        axis_candidates = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dot_products = []
        for candidate_axis in axis_candidates: 
            dots = np.abs(np.dot(diff_vectors, candidate_axis)) 
            dot_sum = np.sum(dots)
            dot_products.append(dot_sum) 
        target_axis = np.argmin(dot_products) 
    elif axis_type == 'prismatic':
        diff_axis = np.mean(np.abs(diff_vectors), axis=0) 
        target_axis = np.argmax(diff_axis)  
    axis = np.array([0, 0, 0])
    axis[target_axis] = -1 
    print(f"[Joint {joint_idx:02d}] Estimation] Predicted joint axis: {axis}")

    # Apply RANSAC to find the best model 
    max_iterations = cfg.ransac_iteration
    threshold = cfg.ransac_threshold 
    min_samples = cfg.ransac_sample 

    best_inliers = [] 
    N = X1.shape[0]
    print(f"Total Number of matches: {N}")

    for i in tqdm(range(max_iterations)):

        # Random sample
        idx = np.random.choice(N, min_samples, replace=False)
        X1_sample, X2_sample = X1[idx], X2[idx]
        qpos_table_X1_sample, qpos_table_X2_sample = qpos_table_X1[idx], qpos_table_X2[idx]

        # Solve minimal model
        if axis_type == 'revolute':
            theta, p, qpos = solve_rotation_angle_and_axis_position(X1_sample, X2_sample, axis,
                qpos_table_X1_sample, qpos_table_X2_sample, num_states)
            errors = compute_rotation_reprojection_error(X1, X2, theta, p, qpos, axis, qpos_table_X1, qpos_table_X2)
        else:
            d, qpos = solve_translation_scale(X1_sample, X2_sample, axis, qpos_table_X1_sample, qpos_table_X2_sample, num_states)
            errors = compute_translation_reprojection_error(X1, X2, d, qpos, axis, qpos_table_X1, qpos_table_X2)
        inliers = np.where(errors < threshold)[0]
        
        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers 
            print(f"[Joint {joint_idx:02d}] Estimation] Best inliers: {len(best_inliers)}") 

    # Refit using all inliers
    X1_inliers = X1[best_inliers]
    X2_inliers = X2[best_inliers]
    qpos_X1_inliers = qpos_table_X1[best_inliers]
    qpos_X2_inliers = qpos_table_X2[best_inliers]
    
    if axis_type == 'revolute':
        d_opt, p_opt, qpos_opt = solve_rotation_angle_and_axis_position(X1_inliers, X2_inliers, axis, 
            qpos_X1_inliers, qpos_X2_inliers, num_states)
        p_opt[target_axis] = 0
        qpos_opt = np.clip(qpos_opt, 0, 1)
        error = compute_rotation_reprojection_error(X1_inliers, X2_inliers, d_opt, p_opt, qpos_opt, 
            axis, qpos_X1_inliers, qpos_X2_inliers).mean()
        print(f"[Joint {joint_idx:02d}] Estimation] Max rotation: {d_opt} Joint Position: {p_opt} RANSAC Cost: {error}") 
        print(f"[Joint {joint_idx:02d}] Estimation] qpos: {qpos_opt}")
        joint_type = np.ones(1)
    else:
        d_opt, qpos_opt = solve_translation_scale(X1_inliers, X2_inliers, axis, 
            qpos_X1_inliers, qpos_X2_inliers, num_states)
        p_opt = np.zeros(3)
        qpos_opt = np.clip(qpos_opt, 0, 1)
        error = compute_translation_reprojection_error(X1_inliers, X2_inliers, d_opt, qpos_opt, 
            axis, qpos_X1_inliers, qpos_X2_inliers).mean()
        print(f"[Joint {joint_idx:02d}] Estimation] Max translation: {d_opt} RANSAC Cost: {error}") 
        print(f"[Joint {joint_idx:02d}] Estimation] qpos: {qpos_opt}")
        joint_type = np.zeros(1)
        
    np.save(f"{output_dir}/initialization/joint_{joint_idx:02d}_est.npy", 
        {"joint_axis": axis, "joint_position": p_opt, "transform_scale": d_opt, "qpos": qpos_opt,
         "joint_type": joint_type}) 

def run_estimate(image_paths, output_dir, joint_type, cfg, joint_idx=0):
 
    seed_everything()
    os.makedirs(f"{output_dir}/correspondences", exist_ok=True)
    os.makedirs(f"{output_dir}/initialization", exist_ok=True)
 
    total_views = 3
    num_states = len(image_paths)
    feature_matching_cfg = FeatureMatchingConfig(num_states, joint_type)  

    # Initialize DKMv3
    matcher = DKMv3(weights=None, h=672, w=896)
    state_dict = torch.load('gim/weights/gim_dkm_100h.ckpt', map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if 'encoder.net.fc' in k:
            state_dict.pop(k)
    matcher.load_state_dict(state_dict)
    matcher = matcher.eval().cuda() 

    if not os.path.exists(f"{output_dir}/initialization/joint_{joint_idx:02d}_est.npy"):
        
        shifts = []
        print(f"Calculate shifts for {total_views} recon views...")
        for view_idx in range(total_views):
            shifts.append(calculate_shift(output_dir, num_states, view_idx, joint_idx, matcher, feature_matching_cfg))
        shift = np.array(shifts)
        shift = np.mean(shift, axis=0)

        print(f"Find correspondence for {total_views} recon views...")
        while True:
            for view_idx in range(total_views):
                flag = find_correspondence(output_dir, num_states, view_idx, joint_idx, matcher, feature_matching_cfg, shift)
                if not flag:
                    break
            if flag:
                break
            
        print(f"Estimate initial joint parameters...")
        estimate_scale(output_dir, num_states, joint_type, joint_idx, cfg)
    
    est_info = np.load(f"{output_dir}/initialization/joint_{joint_idx:02d}_est.npy", allow_pickle=True).item()
    info_dict = {}
    info_dict['joint_type'] = 'Revolute' if est_info['joint_type'] == 1 else 'Prismatic'
    info_dict['joint_axis'] = est_info['joint_axis']
    info_dict['joint_position'] = est_info['joint_position']
    info_dict['joint_scale'] = est_info['transform_scale']
    info_dict['joint_qpos'] = est_info['qpos']

    matching_examples = [f'{output_dir}/correspondences/correspondence_joint_{joint_idx:02d}_state_{i:02d}-{i+1:02d}_view_{view_idx:02d}.png' \
        for view_idx in range(total_views) for i in range(num_states-1)]
    return matching_examples, info_dict