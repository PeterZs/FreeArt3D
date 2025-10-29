pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install plotly==6.0.1 blenderproc kornia pyexr psutil
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/

cd TRELLIS
bash ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install git+https://github.com/jukgei/diff-gaussian-rasterization.git@b1e1cb83e27923579983a9ed19640c6031112b94
pip install git+https://github.com/openai/CLIP.git
pip install lpips open3d OmegaConf spaces pyexr gradio_litmodel3d multipledispatch loguru mathutils open_clip_torch
pip install gradio==5.34.2
pip install diffusers
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install flash_attn==2.7.4.post1
pip install kaleido==0.2.1