# cerberus-neuro — first-time setup

One-time setup for the four resources this project uses: GitHub (code), Hugging Face Hub (artifacts), Google Colab (interactive compute), Docker (portable training).

If you set up cellduet first, you've already done the GitHub + HF + Colab steps. The new piece here is **Docker** for full-scale training and platform portability.

## Execution model

```
Edit locally (VS Code)
        │
        │  git push
        ▼
GitHub  ─────────────┬──────► Colab (T4 Free / Pro)   ──┐
                     │                                  │
                     └──────► Docker (local /           │
                              Lambda / RunPod /         │
                              Paperspace)               │
                                                        ▼
                                            Hugging Face Hub
                                            (datasets, models)
```

Day-to-day development on Colab. Full-scale training (when v0 demands it) on Docker container running on rented A100. Code stays portable across both.

## 1. GitHub, Hugging Face, Colab

If you went through `~/Sandbox/cellduet/docs/SETUP.md` already, these are done. Quick verification:

```bash
gh auth status          # GitHub CLI auth
hf auth whoami          # HF CLI auth (should show: patrickjreed)
```

For Colab, verify by opening any notebook on colab.research.google.com and confirming the `HF_TOKEN` secret is set.

If any of these are not done, start with `~/Sandbox/cellduet/docs/SETUP.md` Sections 1–3.

## 2. Push cerberus-neuro to GitHub

```bash
cd ~/Sandbox/cerberus-neuro
gh repo create PatrickJReed/cerberus-neuro --public --source=. --remote=origin --push
```

The repo URL becomes `https://github.com/PatrickJReed/cerberus-neuro`.

## 3. Docker setup

### Why Docker for this project

Two reasons:

1. **Portability across cheap GPU rentals.** The same container runs on Lambda Cloud, RunPod, Paperspace, or any Linux box with NVIDIA Container Toolkit. When a full-scale training run needs an A100 hour or two ($1–3), the container makes deployment a one-command thing.
2. **Reproducibility.** Anyone forking the repo can train the same model with the same env. Important for a citable artifact.

Docker is **not** used inside Google Colab — Colab provides its own runtime. The Dockerfile is for local-or-cloud execution outside Colab.

### Local Docker setup (one time)

Install Docker Desktop for Mac:

```bash
brew install --cask docker
open -a Docker      # start the daemon
```

After Docker is running, build the image:

```bash
cd ~/Sandbox/cerberus-neuro
docker build -t cerberus-neuro:latest .
```

First build pulls the PyTorch CUDA base image (~5-8 GB) and takes 10–20 min. Subsequent builds are fast.

### Verify the local image

```bash
docker run --rm cerberus-neuro:latest python -c "import cerberus_neuro; print(cerberus_neuro.__version__)"
```

Should print the version. (CPU-only on a Mac; the image is ready for GPU when run on a GPU host.)

### NVIDIA Container Toolkit (Linux GPU hosts only)

When deploying to a Linux box with an NVIDIA GPU (Lambda, RunPod, Paperspace, your own workstation):

```bash
# install on the host (NOT in the container)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Most GPU rental services have this preinstalled. Lambda Cloud images come with it.

### Run with GPU

```bash
docker run --gpus all --rm -it \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -e HF_TOKEN=$HF_TOKEN \
    cerberus-neuro:latest \
    python -m cerberus_neuro.train --config configs/default.yaml
```

(Training config doesn't exist yet; this is the pattern.)

### Cheap-A100 cloud workflow

When v0 ships and you need a full-scale training run:

1. Push code to GitHub.
2. Spin up a Lambda Cloud A100 instance ([cloud.lambda.ai](https://cloud.lambda.ai)) — ~$1.10/hr.
3. SSH in: `ssh ubuntu@<lambda-ip>`
4. `git clone https://github.com/PatrickJReed/cerberus-neuro.git && cd cerberus-neuro`
5. `docker build -t cerberus-neuro:latest .` (or pull if you've pushed to a registry)
6. `docker run --gpus all -e HF_TOKEN=$HF_TOKEN cerberus-neuro:latest python -m cerberus_neuro.train ...`
7. Checkpoint pushed to HF mid-run; can stop the instance once the run completes.

Total cost for a converged run: usually under $15.

Alternative providers with similar workflows: RunPod, Paperspace, vast.ai. RunPod and vast.ai often cheaper for older GPUs (A6000, V100) which are sufficient for ResNet34.

## 4. VS Code workflow

Same pattern as cellduet:

1. Edit `src/cerberus_neuro/*.py` and notebooks in VS Code locally.
2. Commit and push to GitHub.
3. Choose the runtime:
   - **Colab** (small / interactive work): `https://colab.research.google.com/github/PatrickJReed/cerberus-neuro/blob/main/notebooks/<notebook>.ipynb`
   - **Local Docker** (testing the container): `docker run --rm -it -p 8888:8888 cerberus-neuro:latest jupyter lab --ip=0.0.0.0 --allow-root --no-browser`
   - **Cloud Docker** (full training): SSH in, `docker run --gpus all ...`

## Per-session checklist

```
Local (VS Code):
  [ ] git pull
  [ ] edit code/notebooks
  [ ] git add + commit + push
  [ ] (if container changed) docker build -t cerberus-neuro:latest .

Colab (interactive work):
  [ ] open notebook from github.com/PatrickJReed/cerberus-neuro
  [ ] Runtime → T4 GPU
  [ ] run install + HF login + Drive mount cells
  [ ] do work; checkpoint to Drive AND/OR HF before disconnect

Cloud GPU (full training, when needed):
  [ ] spin up A100 instance on Lambda / RunPod / Paperspace
  [ ] git clone, docker build (or pull image)
  [ ] docker run --gpus all -e HF_TOKEN=... cerberus-neuro:latest python -m cerberus_neuro.train ...
  [ ] verify HF checkpoints landing
  [ ] tear down instance when run completes
```

## Known sharp edges

- **Colab does not run the Dockerfile.** Colab gives you its own runtime with pre-installed CUDA and PyTorch. Notebooks installing `cerberus_neuro` from GitHub via `pip install -q git+...` work directly without the container. The Docker setup is for local + cloud GPU rentals only.
- **HF write tokens stored in Lambda/RunPod containers** should be passed via `-e HF_TOKEN=$HF_TOKEN` at runtime, not baked into the image. The Dockerfile in this repo never embeds the token.
- **Lambda Cloud / RunPod instances are ephemeral.** Save artifacts to HF (or to attached persistent disk) before tearing down. The container itself is recreated each run.
- **First Docker build is slow** (~10-20 min on Mac due to base-image pull). Subsequent builds with cached layers are seconds.
- **Image size**: PyTorch CUDA images are 5–8 GB. If you push to Docker Hub or GitHub Container Registry, this matters. For Lambda / RunPod, building on the host is usually fine and free.
