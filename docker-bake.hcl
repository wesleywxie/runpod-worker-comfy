variable "DOCKERHUB_REPO" {
  default = "yelsewx"
}

variable "DOCKERHUB_IMG" {
  default = "runpod-worker-comfy"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

variable "CIVITAI_ACCESS_TOKEN" {
  default = ""
}

# Build variants (can be overridden via CLI --set)
variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "CUDA_VERSION" {
  default = "12.4.1"
}

variable "TORCH_CUDA_SUFFIX" {
  default = "cu124"
}

variable "TORCH_VERSION" {
  default = "2.6.0+cu124"
}

variable "XFORMERS_VERSION" {
  default = "0.0.29.post3"
}

group "default" {
  targets = ["sd"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  target = "base"
  platforms = ["linux/amd64"]
  args = {
    PYTHON_VERSION     = "${PYTHON_VERSION}"
    CUDA_VERSION       = "${CUDA_VERSION}"
    TORCH_CUDA_SUFFIX  = "${TORCH_CUDA_SUFFIX}"
    TORCH_VERSION      = "${TORCH_VERSION}"
    XFORMERS_VERSION   = "${XFORMERS_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-base"]
}

target "flux" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    MODEL_TYPE = "flux"
    HUGGINGFACE_ACCESS_TOKEN = "${HUGGINGFACE_ACCESS_TOKEN}"
    CIVITAI_ACCESS_TOKEN = "${CIVITAI_ACCESS_TOKEN}"
    PYTHON_VERSION     = "${PYTHON_VERSION}"
    CUDA_VERSION       = "${CUDA_VERSION}"
    TORCH_CUDA_SUFFIX  = "${TORCH_CUDA_SUFFIX}"
    TORCH_VERSION      = "${TORCH_VERSION}"
    XFORMERS_VERSION   = "${XFORMERS_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-flux"]
  inherits = ["base"]
}

target "qwen" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    MODEL_TYPE = "qwen"
    HUGGINGFACE_ACCESS_TOKEN = "${HUGGINGFACE_ACCESS_TOKEN}"
    CIVITAI_ACCESS_TOKEN = "${CIVITAI_ACCESS_TOKEN}"
    PYTHON_VERSION     = "${PYTHON_VERSION}"
    CUDA_VERSION       = "${CUDA_VERSION}"
    TORCH_CUDA_SUFFIX  = "${TORCH_CUDA_SUFFIX}"
    TORCH_VERSION      = "${TORCH_VERSION}"
    XFORMERS_VERSION   = "${XFORMERS_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-qwen"]
  inherits = ["base"]
}

target "sd" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    MODEL_TYPE = "sd"
    HUGGINGFACE_ACCESS_TOKEN = "${HUGGINGFACE_ACCESS_TOKEN}"
    CIVITAI_ACCESS_TOKEN = "${CIVITAI_ACCESS_TOKEN}"
    PYTHON_VERSION     = "${PYTHON_VERSION}"
    CUDA_VERSION       = "${CUDA_VERSION}"
    TORCH_CUDA_SUFFIX  = "${TORCH_CUDA_SUFFIX}"
    TORCH_VERSION      = "${TORCH_VERSION}"
    XFORMERS_VERSION   = "${XFORMERS_VERSION}"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-sd"]
  inherits = ["base"]
}

# WAN 2.1 needs CUDA 12.8; override CUDA/Torch for this target only
target "wan" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    MODEL_TYPE = "wan"
    HUGGINGFACE_ACCESS_TOKEN = "${HUGGINGFACE_ACCESS_TOKEN}"
    CIVITAI_ACCESS_TOKEN = "${CIVITAI_ACCESS_TOKEN}"
    PYTHON_VERSION     = "3.12"
    # Override CUDA to 12.8 for WAN 2.1 only
    CUDA_VERSION       = "12.8.1"
    TORCH_CUDA_SUFFIX  = "cu128"
    # Choose a Torch version that matches cu128 wheels
    TORCH_VERSION      = "2.7.1+cu128"
    XFORMERS_VERSION   = "0.0.31.post1"
  }
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-wan"]
  inherits = ["base"]
}
