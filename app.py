import os 
import shutil
import json 
import numpy as np  
import gradio as gr
import spaces
import time
  
from gradio_litmodel3d import LitModel3D
from pipelines.render import run_rendering
from pipelines.recon import run_recon 
from pipelines.estimate import run_estimate
from pipelines.sds import run_sds
from typing import * 
from omegaconf import OmegaConf

partnet_dir = f'datasets/PartNet'
real_world_dir = f'examples'
multi_joint_dir = f'datasets/multi_joint'
 
with open(f"configs/partnet.json") as f:
    data_info = json.load(f) 
cfg = OmegaConf.load('configs/default.yaml') 

labels = {
    'Cabinet (real-world)': 'cabinet',
    'Cabinet2 (real-world)': 'cabinet2',
    'Box (100247)': '100247',
    'Dishwasher (12614)': '12614',
    'Laptop (10270)': '10270',
    'Lighter (100309)': '100309',
    'Microwave (7320)': '7320',
    'Oven (102001)': '102001',
    'Refrigerator (11231)': '11231',
    'Safe (102301)': '102301',
    'Stapler (103111)': '103111',
    'StorageFurniture (47183)': '47183',
    'Table (20411)': '20411',
    'WashingMachine (100283)': '100283',
}
 
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
 
def start_session(req: gr.Request): 
    return req.session_hash
    
def end_session(req: gr.Request):
    pass

def get_available_ids():
    """Get list of available object IDs for the dropdown"""
    # TODO: Support multi-joints
    total_list = list(labels.keys())
    return total_list

def load_renderings_with_id(id_input: str, session_hash: str) -> Tuple[List[str], str]:
    
    selected_id = labels[id_input]
    print(f'[Selected Object] {selected_id}')
    cfg.train_num_state = 6

    rendering_paths = []
    if 'cabinet' in selected_id:
        base_dir = real_world_dir
    elif selected_id in os.listdir(partnet_dir):
        base_dir = partnet_dir
    else:
        base_dir = multi_joint_dir

    for i in range(cfg.train_num_state):
        rendering_paths.append(f'{base_dir}/{selected_id}/{i:02d}_seg.png') 
    return rendering_paths, selected_id 

def handle_mesh_upload(files):
    """Handle uploaded mesh files and create dropdown choices"""
    if not files or len(files) == 0:
        return gr.Dropdown(choices=[], value=None), None
    choices = [f"Mesh State {i:02d} - {os.path.basename(file.name)}" for i, file in enumerate(files)]
    # yield gr.update(), gr.update(None)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), files[0].name

def switch_mesh_view(selected_file, files):
    """Switch the 3D viewer to show the selected mesh file"""
    if not files or not selected_file:
        return None
    
    # Extract the file index from the selection
    try:
        file_index = int(selected_file.split(" ")[2])
        if 0 <= file_index < len(files):
            return files[file_index].name
    except:
        # Fallback: try to match by basename
        try:
            selected_base = os.path.basename(selected_file)
            for idx, f in enumerate(files):
                if os.path.basename(getattr(f, 'name', '')) == selected_base:
                    return f.name
        except Exception:
            pass
    
    return None
 
def process_and_update_gallery(mesh_files, session_hash):
    mesh_paths = [f.name for f in mesh_files] if mesh_files else []
    cfg.train_num_state = len(mesh_paths)
    rendering_paths = run_rendering(mesh_paths, output_dir=f'{TMP_DIR}/{session_hash}') 
    return rendering_paths 

def format_meta_info_text(info_dict):
    """Format joint meta info into 4 rich-text lines (larger font; bold title for Joint Axis)."""
    def to_str(v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, dict, tuple)):
            try:
                return json.dumps(v)
            except Exception:
                return str(v)
        return "" if v is None else str(v)

    axis_val = info_dict.get("joint_axis", "")
    pos_val = info_dict.get("joint_position", "")
    scale_val = info_dict.get("joint_scale", "")
    qpos_val = info_dict.get("joint_qpos", "")

    axis_str = to_str(axis_val)
    pos_str = to_str(pos_val)
    scale_str = to_str(-scale_val)
    qpos_str = to_str(qpos_val)

    # Use inline HTML for size and emphasis; rendered by gr.Markdown
    return (
        "<div style=\"font-size: 18px; line-height: 1.5;\">"
        f"<div><strong>Joint Axis:</strong> {axis_str}</div>"
        f"<div><strong>Joint Pivot:</strong> {pos_str}</div>"
        f"<div><strong>Max Scale:</strong> {scale_str}</div>"
        f"<div><strong>qpos:</strong> {qpos_str}</div>"
        "</div>"
    )

def image_gallery_state_change(p_state): 
    base_dir = f'outputs/{p_state.get("sel_id")}_{p_state.get("session_hash")}' 
    if p_state.get("step") == "running": 
        while True:
            with open(f'{base_dir}/state_list.json', 'r') as f:
                state_list = json.load(f)
            if state_list[0] == "done":
                break
            time.sleep(0.5)
        return gr.update(), gr.update(), \
             gr.update(), gr.update(), \
             gr.update(), gr.update(), \
             gr.update(), gr.update(), gr.update()
    if p_state.get("step") == "done": 
        return p_state.get("gallery", gr.update()), p_state.get("sel_id", gr.update()), \
            gr.update(value=None), gr.update(value=None), \
            gr.update(value=None), gr.update(value=None), \
            gr.update(value=None), gr.update(value=None), gr.update(value=None)
    return gr.update(), gr.update(), \
        gr.update(), gr.update(), \
        gr.update(), gr.update(), \
        gr.update(), gr.update(), gr.update()

def recon_state_change(p_state): 
    base_dir = f'outputs/{p_state.get("sel_id")}_{p_state.get("session_hash")}' 
    if p_state.get("step") == "running": 
        while True:
            with open(f'{base_dir}/state_list.json', 'r') as f:
                state_list = json.load(f) 
            if state_list[1] == "done":
                break
            time.sleep(0.5)
        return gr.update(), gr.update()
    if p_state.get("step") == "done":
        return p_state.get("vox", gr.update()), p_state.get("recon", gr.update())   
    return gr.update(), gr.update()
 
def estimate_state_change(p_state): 
    base_dir = f'outputs/{p_state.get("sel_id")}_{p_state.get("session_hash")}' 
    if p_state.get("step") == "running": 
        while True:
            with open(f'{base_dir}/state_list.json', 'r') as f:
                state_list = json.load(f)
            if state_list[2] == "done":
                break
            time.sleep(0.5)
        return gr.update(), gr.update()
    if p_state.get("step") == "done":
        return p_state.get("match", gr.update()), p_state.get("html", gr.update())
    return gr.update(), gr.update()

def sds_state_change(p_state): 
    base_dir = f'outputs/{p_state.get("sel_id")}_{p_state.get("session_hash")}' 
    if p_state.get("step") == "running": 
        while True:
            with open(f'{base_dir}/state_list.json', 'r') as f:
                state_list = json.load(f)
            if state_list[3] == "done":
                break
            time.sleep(0.5)
        return gr.update(), gr.update(), gr.update()
    if p_state.get("step") == "done":
        return p_state.get("full", gr.update()), p_state.get("fixed", gr.update()), p_state.get("art", gr.update())
    return gr.update(), gr.update(), gr.update()

def save_gallery_to_renderings(gallery_value, selected_id, session_hash):
    """Persist gallery images into outputs/<selected_id>/renderings with SDS-expected names.
    Ensures there are cfg.train_num_state frames by padding with the last image if needed.
    Also writes the required 'rendering_pure_joint_00_state_{T-1}.png' using the last frame.
    """
    try:
        image_paths = []
        if gallery_value:
            for item in gallery_value:
                path = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
                if isinstance(path, str) and os.path.exists(path):
                    image_paths.append(path)
        if len(image_paths) == 0:
            return

        cfg.train_num_state = len(image_paths)
        base_dir = f'outputs/{selected_id}_{session_hash}' 
        if "uploaded" in base_dir:
            os.system(f"rm -rf {base_dir}/*")
            
        rendering_dir = f'{base_dir}/renderings'
        os.makedirs(rendering_dir, exist_ok=True)

        # Normalize count to T frames
        for i in range(cfg.train_num_state):
            src = image_paths[i] if i < len(image_paths) else image_paths[-1]
            dst = f'{rendering_dir}/rendering_joint_00_state_{i:02d}.png'
            shutil.copy(src, dst)

        # Write the 'pure' image as the last uploaded (or last padded) frame
        if os.path.exists(image_paths[-1].replace('_seg.png', '_pure.png')):
            pure_src = image_paths[-1].replace('_seg.png', '_pure.png')
        else:
            pure_src = image_paths[-1]
        pure_dst = f'{rendering_dir}/rendering_pure_joint_00_state_{(cfg.train_num_state - 1):02d}.png'
        shutil.copy(pure_src, pure_dst)
    except Exception as e:
        print(f"[Gallery->Renderings] Failed to persist images: {e}")
        return

@spaces.GPU(duration=240)
def run_recon_trellis(image_paths, output_dir): 
    return run_recon(image_paths, output_dir=output_dir, app=True)

def recon_trellis_meshes(input_image_gallery, output_dir): 
    # Carefully extract file paths from Gradio Gallery elements, which may be dicts, file objects, or strings.
    image_paths = [img[0] for img in input_image_gallery]  
    glb_paths, rendering_dir, recon_voxel_paths = run_recon_trellis(image_paths, output_dir=output_dir)
    recon_mesh_paths = run_rendering(glb_paths, rendering_dir, recon=True)
    return recon_voxel_paths, recon_mesh_paths 

@spaces.GPU(duration=120)
def estimate_initial_joints(input_image_gallery, joint_type, output_dir):
    image_paths = [img[0] for img in input_image_gallery]
    matching_examples, info_dict = run_estimate(image_paths, output_dir=output_dir, cfg=cfg, joint_type=joint_type) 
    return matching_examples, info_dict

@spaces.GPU(duration=800)
def articulated_generation_sds(selected_id, session_hash):

    if 'cabinet' in selected_id:
        input_dir = real_world_dir
    elif selected_id in os.listdir(partnet_dir):
        input_dir = partnet_dir
    else:
        input_dir = multi_joint_dir

    base_dir = f'outputs/{selected_id}_{session_hash}' 
    rendering_dir = f'{base_dir}/renderings'

    os.makedirs(rendering_dir, exist_ok=True)
    if os.path.exists(f'{input_dir}/{selected_id}/05_seg.png'):
        for i in range(6):
            shutil.copy(f'{input_dir}/{selected_id}/{i:02d}_seg.png', f'{rendering_dir}/rendering_joint_00_state_{i:02d}.png')
        shutil.copy(f'{input_dir}/{selected_id}/05_pure.png', f'{rendering_dir}/rendering_pure_joint_00_state_05.png')
  
    full_mesh, fixed_part, articulated_part = run_sds(base_dir, 1, cfg)
    return full_mesh, fixed_part, articulated_part

def pipeline(id_value, mesh_files, uploaded_image_gallery, joint_type_in, session_hash, run_idx):
    """Server-side pipeline that yields state only; UI updates are applied via a mapper."""
 
    # 1) Prepare inputs  
    if id_value is not None:
        sel_id = f"{labels[id_value]}_{run_idx:02d}"
    elif mesh_files is not None:
        sel_id = f"mesh_uploading_{run_idx:02d}"
    elif uploaded_image_gallery is not None:
        sel_id = f"image_uploading_{run_idx:02d}" 

    base_dir = f'outputs/{sel_id}_{session_hash}' 
    os.makedirs(base_dir, exist_ok=True)
    state_list_output_path = f'{base_dir}/state_list.json'

    state_list = ["running", "prepare", "prepare", "prepare"]
    with open(state_list_output_path, 'w') as f:
        json.dump(state_list, f)
    yield (
        {"step": state_list[0], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[1], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[2], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[3], "sel_id": sel_id, "session_hash": session_hash}, run_idx, \
        f"<h3>Preparing input images...<p style=\"color:red\">(<u>~3 min</u>)</p></h3>"
    )

    if id_value is not None:
        rendering_paths, _ = load_renderings_with_id(id_value, session_hash)
    elif mesh_files is not None:
        rendering_paths = process_and_update_gallery(mesh_files, session_hash) 
    elif uploaded_image_gallery is not None:
        rendering_paths = [img[0] for img in uploaded_image_gallery]

    gallery = [[p, None] for p in rendering_paths] 
    save_gallery_to_renderings(gallery, sel_id, session_hash)

    state_list = ["done", "running", "prepare", "prepare"]
    with open(state_list_output_path, 'w') as f:
        json.dump(state_list, f)
    yield (
        {"step": state_list[0], "gallery": gallery, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[1], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[2], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[3], "sel_id": sel_id, "session_hash": session_hash}, run_idx, \
        f"<h3>Stage 1: Initial reconstruction... <span style=\"color:red\">(<u style=\"color:red\">~3 min</u>)</span></h3>"
    )
 
    # 3) Reconstruction
    vox, recon = recon_trellis_meshes(gallery, f"{base_dir}/recon")

    state_list = ["done", "done", "running", "running"]
    with open(state_list_output_path, 'w') as f:
        json.dump(state_list, f)
    yield (
        {"step": state_list[0], "gallery": gallery, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[1], "vox": vox, "recon": recon, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[2], "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[3], "sel_id": sel_id, "session_hash": session_hash}, run_idx, \
        f"<h3>Stage 2: Joint initialization... <span style=\"color:red\">(<u style=\"color:red\">~1 min</u>)</span></h3>"
    )

    # 4) Joint estimation
    if joint_type_in is not None:
        joint_type = joint_type_in
    else:
        if 'cabinet' in sel_id:
            joint_type = 'revolute' if 'cabinet2' in sel_id else 'prismatic'
        else:
            joint_type = 'prismatic' if (sel_id in data_info['prismatic']['obj_ids']) else 'revolute'
    print(f"[Joint Type] {joint_type}")
    match, meta = estimate_initial_joints(gallery, joint_type, base_dir)
    
    # 5) Format meta info
    html = format_meta_info_text(meta)

    state_list = ["done", "done", "done", "running"]
    with open(state_list_output_path, 'w') as f:
        json.dump(state_list, f)
    yield (
        {"step": state_list[0], "gallery": gallery, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[1], "vox": vox, "recon": recon, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[2], "match": match, "html": html, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[3], "sel_id": sel_id, "session_hash": session_hash}, run_idx, \
        f"<h3>Stage 3: SDS optimization... <span style=\"color:red\">(<u style=\"color:red\">~10 min</u>)</span></h3>"
    )

    # 6) SDS refinement 
    full, fixed, art = articulated_generation_sds(sel_id, session_hash)
    
    state_list = ["done", "done", "done", "done"]
    with open(state_list_output_path, 'w') as f:
        json.dump(state_list, f)
    yield (
        {"step": state_list[0], "gallery": gallery, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[1], "vox": vox, "recon": recon, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[2], "match": match, "html": html, "sel_id": sel_id, "session_hash": session_hash},
        {"step": state_list[3], "full": full, "fixed": fixed, "art": art, "sel_id": sel_id, "session_hash": session_hash},
        run_idx+1, \
        f"<h3>Done!</h3>"
    )

with gr.Blocks(delete_cache=(600, 600), css="""
    .gallery-container {
        overflow-x: auto !important;
        overflow-y: auto !important;
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }
    /* Ensure inner wrappers can scroll vertically */
    .gallery-container > div {
        max-height: 100% !important;
        overflow-y: auto !important;
    }
    .gallery-container .grid, 
    .gallery-container .grid-wrap, 
    .gallery-container .thumbnail-grid, 
    .gallery-container .gallery, 
    .gallery-container .container {
        max-height: 100% !important;
        overflow-y: auto !important;
    }
    .gallery-container::-webkit-scrollbar {
        height: 8px;
        width: 8px;
    }
    .gallery-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .gallery-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .gallery-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    /* Ensure images are fully visible in galleries (no cropping) */
    .gallery-container img {
        object-fit: contain !important;
        object-position: center center;
    }
""") as demo:

    gr.Markdown("""
    ## FreeArt3D Demo
    ### You can use the following three methods to generate articulated 3D models.
    1. Choose an example object (from PartNet-Mobility or real-world).
    2. Upload multiple mesh files (e.g., .obj, .ply, .stl, .glb, .gltf) at different articulation states (images will be rendered automatically).
    3. Upload multiple input images with **a segmented object on the disk** at different articulation states.
    
    ### The total generation time is *~10 min*. Be patient :)
    """)

    with gr.Row(equal_height=True): 
        with gr.Tab("Load by Example Object"):
            with gr.Row():
                id_input = gr.Dropdown(
                    choices=get_available_ids(),
                    value='Cabinet (real-world)',
                    label="Example Object",
                    info='Select a real-world object or an object from the PartNet-Mobility dataset. (e.g., Box (100214))',
                )
            with gr.Row():
                id_input_btn = gr.Button("Generate Articulated Object", variant="primary") 
        with gr.Tab("Upload Mesh Files"):
            with gr.Row(equal_height=True): 
                with gr.Column():
                    mesh_files = gr.File(label="Upload Mesh Files", file_count="multiple", file_types=[".obj", ".ply", ".stl", ".glb", ".gltf"], height=300)
                    mesh_selector = gr.Dropdown(label="Select Mesh to View", choices=[], value=None, interactive=True)
                with gr.Column():
                    mesh_viewer = LitModel3D(label="Selected Mesh", exposure=10.0, height=300)
            with gr.Row():
                joint_type_in_mesh = gr.Dropdown(label="joint_type", choices=["revolute", "prismatic"], value=None)
            with gr.Row():
                mesh_input_btn = gr.Button("Generate Articulated Object", variant="primary") 
        with gr.Tab("Upload Image Files"):
            with gr.Row(equal_height=True):
                uploaded_image_gallery = gr.Gallery(label="Input Images", show_label=True, elem_id="gallery", height=300, columns=10, allow_preview=True, elem_classes=["gallery-container"])
            with gr.Row():
                joint_type_in_image = gr.Dropdown(label="joint_type", choices=["revolute", "prismatic"], value=None)
            with gr.Row():
                image_input_btn = gr.Button("Generate Articulated Object", variant="primary") 

    with gr.Row():  
        with gr.Accordion("Current Stage", open=True):
            current_stage = gr.HTML(value="<h3>Waiting for input...</h3>")


    with gr.Row(equal_height=True):
        with gr.Column(): 
            input_image_gallery = gr.Gallery(label="Input Images", show_label=True, elem_id="gallery", height=300, columns=3, allow_preview=True, interactive=False, elem_classes=["gallery-container"])
        with gr.Column():
            voxel_gallery = gr.Gallery(label="Reconstructed Voxels", show_label=True, elem_id="gallery", height=360, columns=3, allow_preview=True, interactive=False, elem_classes=["gallery-container"])
        with gr.Column():
            recon_mesh_gallery = gr.Gallery(label="Reconstructed Meshes", show_label=True, elem_id="gallery", height=360, columns=3, allow_preview=True, interactive=False, elem_classes=["gallery-container"])
 
    with gr.Row():
        matching_examples = gr.Gallery(label="Correspondences", show_label=True, elem_id="gallery", height=400, interactive=False, columns=5, allow_preview=True, elem_classes=["gallery-container"])
    
    with gr.Row(): 
        meta_info = gr.State(
            value={
                "joint_axis": "",
                "joint_position": "",
                "joint_scale": "",
                "joint_qpos": ""
            }
        )
        with gr.Accordion("Joint Information", open=True):
            joint_info_html = gr.HTML(value="")
     
    with gr.Row():
        with gr.Column():
            full_mesh = LitModel3D(label="Full Mesh", exposure=10.0, height=300)
        with gr.Column():
            fixed_part = LitModel3D(label="Fixed Part", exposure=10.0, height=300)
        with gr.Column():
            articulated_part = LitModel3D(label="Articulated Part", exposure=10.0, height=300)

    selected_id_state = gr.State(value="")
    session_hash_state = gr.State(value="")
    run_idx_state = gr.State(value=0)
    
    image_gallery_state = gr.State(value="prepare")
    recon_state = gr.State(value="prepare")
    estimate_state = gr.State(value="prepare")
    sds_state = gr.State(value="prepare")
    none_val = gr.State(value=None)

    # Handlers
    demo.load(start_session, outputs=session_hash_state)
    demo.unload(end_session)

    id_input_btn.click(
        start_session,
        outputs=session_hash_state,
    ).then(
        pipeline,
        inputs=[id_input, none_val, none_val, none_val, session_hash_state, run_idx_state],
        outputs=[image_gallery_state, recon_state, estimate_state, sds_state, run_idx_state, current_stage],
    )

    mesh_input_btn.click(
        start_session,
        outputs=session_hash_state,
    ).then(
        pipeline,
        inputs=[none_val, mesh_files, none_val, joint_type_in_mesh, session_hash_state, run_idx_state],
        outputs=[image_gallery_state, recon_state, estimate_state, sds_state, run_idx_state, current_stage],
    )

    image_input_btn.click(
        start_session,
        outputs=session_hash_state,
    ).then(
        pipeline,
        inputs=[none_val, none_val, uploaded_image_gallery, joint_type_in_image, session_hash_state, run_idx_state],
        outputs=[image_gallery_state, recon_state, estimate_state, sds_state, run_idx_state, current_stage],
    )

    # Watch state and update only relevant components per step
    image_gallery_state.change(
        image_gallery_state_change,
        inputs=image_gallery_state,
        outputs=[input_image_gallery, full_mesh, 
            voxel_gallery, recon_mesh_gallery,
            matching_examples, joint_info_html,
            fixed_part, articulated_part, selected_id_state],
    )

    recon_state.change(
        recon_state_change,
        inputs=recon_state,
        outputs=[voxel_gallery, recon_mesh_gallery],
    ) 

    estimate_state.change(
        estimate_state_change,
        inputs=estimate_state,
        outputs=[matching_examples, joint_info_html],
    )

    sds_state.change(
        sds_state_change,
        inputs=sds_state,
        outputs=[full_mesh, fixed_part, articulated_part],
    )
    
    # Handle mesh file uploads
    mesh_files.change(
        handle_mesh_upload,
        inputs=mesh_files,
        outputs=[mesh_selector, mesh_viewer]
    )

    # Handle dropdown selection for mesh view
    mesh_selector.change(
        switch_mesh_view,
        inputs=[mesh_selector, mesh_files],
        outputs=mesh_viewer
    )

    # Ensure downstream uses the generic 'uploaded' namespace for output paths
    uploaded_image_gallery.change(
        lambda *_: "uploaded",
        inputs=None,
        outputs=selected_id_state
    )
 
# Launch the Gradio app
if __name__ == "__main__":   
    demo.launch()