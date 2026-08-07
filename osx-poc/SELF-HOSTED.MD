# Self-hosted GPU runner — setup

This document describes how to configure the machine that will run the
**`full-gpu-tests`** job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
(`runs-on: [self-hosted, gpu]`, triggered only manually via `workflow_dispatch`).

Not needed for day-to-day development — `cpu-tests` runs on GitHub-hosted
`ubuntu-latest` runners and requires no setup. This is only for whoever wants
to actually exercise the GPU-marked tests (`@pytest.mark.gpu`) from CI instead
of running `make test` locally.

---

## Hardware requirements

Matches the project's dev target (see `README.md` → *Dev environment
constraints*):

- NVIDIA GPU with ≥ 22 GB VRAM (reference: **RTX 3090, 24 GB**)
- Enough disk for the `osx-poc:dev` image (~20 GB on disk, ~7 GB content) plus
  Docker build cache

## Software prerequisites

The runner itself doesn't need Python, torch, or any project dependency
directly — the job only runs `docker compose build` and
`docker compose run ...`, everything else happens inside the container. The
host needs:

1. **Docker Desktop** (Windows) or **Docker Engine** (Linux), with the
   WSL2 backend if on Windows
2. **NVIDIA driver** supporting CUDA 12.1.1 (matches `Dockerfile`'s
   `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image)
3. **NVIDIA Container Toolkit**, so `docker compose`'s `runtime: nvidia` can
   actually see the GPU
4. **Git**

### Verify before registering the runner

```bash
# 1. Driver + GPU visible to the host
nvidia-smi

# 2. GPU visible *inside* a container (this is the part that actually matters)
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Both must succeed and show the GPU before continuing. If step 2 fails while
step 1 works, the NVIDIA Container Toolkit isn't installed/configured — the
runner will build fine but every `docker compose run` will fail or silently
run CPU-only.

---

## Register the runner

1. On GitHub: repo → **Settings → Actions → Runners → New self-hosted runner**
2. Pick the OS matching the machine (Windows/Linux) and follow the
   download/configure commands GitHub generates — they include a
   short-lived registration token, so copy them fresh from that page rather
   than reusing old ones.
3. When prompted for labels during `config.cmd`/`config.sh`, add `gpu` in
   addition to the default `self-hosted` label. `ci.yml` targets
   `runs-on: [self-hosted, gpu]` — without the `gpu` label the job will queue
   forever with no matching runner.
4. Install it as a service (`./svc.sh install && ./svc.sh start` on Linux, or
   the equivalent via `config.cmd` on Windows) so it survives reboots and
   keeps listening without a logged-in session.

## Verify the runner picked up the job

```bash
gh api repos/danielesalpietro/vMemoryFabric/actions/runners
```

Should list the new runner with `"status": "online"`. Then trigger the job
manually:

```bash
gh workflow run ci.yml --ref develop
```

`full-gpu-tests` only runs on `workflow_dispatch` (see `if:` condition in
`ci.yml`) — a normal push will never dispatch work to this runner.

---

## Security note

Self-hosted runners execute arbitrary workflow code on this machine with
whatever permissions the runner process has — this is why `full-gpu-tests`
is gated to manual `workflow_dispatch` instead of running on every push/PR
(a PR from an untrusted fork could otherwise run code on your GPU box). Keep
it that way: don't add `pull_request` to this job's triggers without also
restricting who can open PRs against this repo.
