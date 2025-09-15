# syntax=docker/dockerfile:1.6

# Stage 1: Base image with common dependencies
# Allow overriding CUDA/Ubuntu variants via build args
ARG CUDA_VERSION=12.4.1
ARG CUDNN_FLAVOR=cudnn-runtime
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_FLAVOR}-ubuntu${UBUNTU_VERSION} AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Core version args (overridable via Bake or --build-arg)
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_SUFFIX=cu124
ARG TORCH_VERSION=2.6.0+cu124
ARG XFORMERS_VERSION=0.0.29.post3

# Expose as environment variables for convenience
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV INDEX_URL=https://download.pytorch.org/whl/${TORCH_CUDA_SUFFIX}
ENV TORCH_VERSION=${TORCH_VERSION}
ENV XFORMERS_VERSION=${XFORMERS_VERSION}

ARG AWS_REGION=ap-northeast-1
# Set up AWS region mirror to speed up the build
RUN sed -i "s|http://archive.ubuntu.com/ubuntu/|http://${AWS_REGION}.ec2.archive.ubuntu.com/ubuntu/|g" /etc/apt/sources.list \
    && sed -i "s|http://security.ubuntu.com/ubuntu/|http://${AWS_REGION}.ec2.archive.ubuntu.com/ubuntu/|g" /etc/apt/sources.list \
    && apt-get update

# Install git and other necessary tools
RUN apt-get update && apt-get install -y \
        software-properties-common \
        python-is-python3 \
        python3-pip \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y --no-install-recommends \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-venv" \
        "python3-tk" \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install pip && Upgrade pip && Create symlink for pip3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 -m pip install --upgrade --no-cache-dir pip \
    && rm -f /usr/bin/pip3 && ln -s /usr/local/bin/pip3 /usr/bin/pip3

# Install Torch and xformers
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} && \
    pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} --index-url ${INDEX_URL}

# Install comfy-cli using Python 3.10
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir comfy-cli runpod requests scikit-image opencv-python matplotlib imageio_ffmpeg

# Install ComfyUI
# Use CUDA_SHORT (e.g., 12.4) to match CUDA toolchain for prebuilt wheels
ARG CUDA_SHORT=12.4
RUN /usr/bin/yes | comfy --workspace /comfyui install --skip-torch-or-directml --manager-url https://github.com/ltdrdata/ComfyUI-Manager@3.33.8 --nvidia --version 0.3.57 && comfy tracking disable

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Install custom nodes manually
RUN comfy --workspace /comfyui node install comfyui-art-venture comfyui_ipadapter_plus comfyui_controlnet_aux comfyui-videohelpersuite

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
RUN mkdir -p models/{checkpoints,controlnet,vae,loras,clip,clip_vision,unet,diffusion_models,ipadapter,text_encoders,upscale_models} && mkdir -p models/ipadapter

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "flux" ]; then \
      python --version && \
      pip freeze | grep -E "torch|torchvision|torchaudio|xformers" > constraints.txt && \
      git clone https://github.com/mit-han-lab/ComfyUI-nunchaku custom_nodes/nunchaku_nodes && \
      pip3 install --no-cache-dir -r custom_nodes/nunchaku_nodes/requirements.txt -c constraints.txt && \
      wget https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.6-cp312-cp312-linux_x86_64.whl && \
      pip3 install --no-cache-dir nunchaku-1.0.0+torch2.6-cp312-cp312-linux_x86_64.whl && \
      rm nunchaku-1.0.0+torch2.6-cp312-cp312-linux_x86_64.whl && \
      wget -O models/diffusion_models/svdq-int4_r32-flux.1-dev.safetensors https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev/resolve/main/svdq-int4_r32-flux.1-dev.safetensors && \
      wget -O models/text_encoders/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors && \
      wget -O models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors && \
      wget -O models/loras/mjV6.safetensors https://huggingface.co/strangerzonehf/Flux-Midjourney-Mix2-LoRA/resolve/main/mjV6.safetensors && \
      wget -O models/loras/Ars_MidJourney_Watercolor.safetensors "https://civitai.com/api/download/models/742802?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/PixelArtStylesFlux.safetensors "https://civitai.com/api/download/models/779124?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Cartoonillustration_flux_lora_v1.safetensors "https://civitai.com/api/download/models/734299?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/RealAnime.safetensors "https://civitai.com/api/download/models/1549230?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/New_Fantasy_CoreV4_FLUX.safetensors "https://civitai.com/api/download/models/1264088?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Digital_Impressionist_Flux.safetensors "https://civitai.com/api/download/models/1466567?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Comic_book_opus_IV.safetensors "https://civitai.com/api/download/models/1277654?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Inkwash-Fusion_v30-000030.safetensors "https://civitai.com/api/download/models/1524366?type=Model&format=SafeTensor&token=${CIVITAI_ACCESS_TOKEN}" && \
      wget -O models/loras/Ghibili-Cartoon-Art.safetensors https://huggingface.co/strangerzonehf/Ghibli-Flux-Cartoon-LoRA/resolve/main/Ghibili-Cartoon-Art.safetensors; \
    fi
    
RUN if [ "$MODEL_TYPE" = "wan" ]; then \
      wget -O models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
      wget -O models/clip_vision/clip_vision_h.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors && \
      wget -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors; \
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
      wget -O models/loras/ip-adapter-faceid-plusv2_sd15_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors && \
      wget -O models/upscale_models/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
      wget -O models/upscale_models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
