# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

ENV PYTHON_VERSION=3.12

ENV INDEX_URL=https://download.pytorch.org/whl/cu124
ENV TORCH_VERSION=2.6.0+cu124
ENV XFORMERS_VERSION=0.0.29.post3

# Install Python from deadsnakes PPA
add-apt-repository ppa:deadsnakes/ppa
apt install -y --no-install-recommends \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv" \
    "python3-tk"

# Link Python
rm /usr/bin/python
ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
rm /usr/bin/python3
ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

# Upgrade pip
python3 -m pip install --upgrade --no-cache-dir pip

# Create symlink for pip3
rm -f /usr/bin/pip3
ln -s /usr/local/bin/pip3 /usr/bin/pip3

# Install git and other necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    ffmpeg

# Install Torch and xformers
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} --index-url ${INDEX_URL}

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli using Python 3.10
RUN python -m pip install --upgrade pip && python -m pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.4 --manager-url https://github.com/ltdrdata/ComfyUI-Manager@3.33.8 --nvidia --version 0.3.57

# Optout analytics tracking
RUN comfy tracking disable

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod and necessary libraries
RUN python -m pip install runpod requests scikit-image opencv-python matplotlib imageio_ffmpeg

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Install custom nodes manually
RUN comfy --workspace /comfyui node install comfyui-art-venture comfyui_ipadapter_plus ComfyUI-IPAdapter-Flux comfyui_controlnet_aux comfyui-videohelpersuite ComfyUI-GGUF

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

# Install git and other necessary tools
RUN apt-get update && apt-get install -y git git-lfs wget 

ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui
# Create necessary directories
RUN mkdir -p models/{checkpoints,controlnet,vae,loras,clip,clip_vision,unet,diffusion_models,ipadapter,text_encoders,upscale_models}
RUN mkdir -p models/ipadapter

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      git lfs install && \
      git clone https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter models/ipadapter-flux && \
      wget -O models/unet/flux1-dev-Q8_0.gguf https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf && \
      wget -O models/clip/t5-v1_1-xxl-encoder-Q8_0.gguf https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/flux_ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors && \
      wget -O models/loras/Ars_MidJourney_Watercolor.safetensors "https://civitai.com/api/download/models/742802?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/PixelArtStylesFlux.safetensors "https://civitai.com/api/download/models/779124?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Cartoonillustration_flux_lora_v1.safetensors "https://civitai.com/api/download/models/734299?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/RealAnime.safetensors "https://civitai.com/api/download/models/1549230?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/New_Fantasy_CoreV4_FLUX.safetensors "https://civitai.com/api/download/models/1264088?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Digital_Impressionist_Flux.safetensors "https://civitai.com/api/download/models/1466567?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Comic_book_opus_IV.safetensors "https://civitai.com/api/download/models/1277654?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Inkwash-Fusion_v30-000030.safetensors "https://civitai.com/api/download/models/1524366?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Studio_Ghibli_Flux.safetensors "https://civitai.com/api/download/models/755852?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}"; \
    fi
    
RUN if [ "$MODEL_TYPE" = "wan21" ]; then \
      wget -O models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
      wget -O models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors && \
      wget -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "refiner" ]; then \
      wget -O models/upscale_models/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
      wget -O models/upscale_models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth; \
    fi

RUN if [ "$MODEL_TYPE" = "sd" ]; then \
      wget -O models/checkpoints/dreamshaper_8.safetensors "https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16" && \
      wget -O models/controlnet/control_v11p_sd15_canny_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors && \
      wget -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors && \
      wget -O models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors && \
      wget -O models/clip_vision/clip-vit-large-patch14-336.bin https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/image_encoder/pytorch_model.bin && \
      wget -O models/ipadapter/ip-adapter_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors && \
      wget -O models/ipadapter/ip-adapter_sd15_light_v11.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin && \
      wget -O models/ipadapter/ip-adapter-plus_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors && \
      wget -O models/ipadapter/ip-adapter-plus-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors && \
      wget -O models/ipadapter/ip-adapter-full-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors && \
      wget -O models/ipadapter/ip-adapter_sd15_vit-G.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors && \
      wget -O models/ipadapter/ip-adapter-faceid_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin && \
      wget -O models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin && \
      wget -O models/ipadapter/ip-adapter-faceid-portrait-v11_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin && \
      wget -O models/loras/ip-adapter-faceid_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors && \
      wget -O models/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
