# Binder Configuration

This directory contains configuration files for [mybinder.org](https://mybinder.org).

## Files

- **`Dockerfile`**: Points to the pre-built Docker image from GitHub Container Registry
  - This is automatically generated and should not be edited manually
  - It tells Binder to use `ghcr.io/lorenzsp/emri-fom-binder:latest` instead of building from scratch

## How It Works

1. **GitHub Actions** (`.github/workflows/binder-build.yml`) builds a Docker image using `repo2docker`
2. The image is pushed to GitHub Container Registry (GHCR) at `ghcr.io/lorenzsp/emri-fom-binder`
3. When a user clicks the Binder badge, Binder reads this `Dockerfile` and pulls the pre-built image
4. **Result**: Launch time reduced from 5-10 minutes to ~1 minute

## Why Pre-build?

Without pre-building, mybinder.org has to:
- Install all Python packages from `requirements.txt`
- Configure the Jupyter environment
- This takes 5-10 minutes every time the repo changes

With pre-building:
- The heavy installation happens once in GitHub Actions
- Binder just pulls a ready-to-use image
- Launch time is typically under 1 minute

## Updating the Username

If you fork this repository, update the Docker image name in:
1. `.binder/Dockerfile` - Change `lorenzsp` to your GitHub username
2. `.github/workflows/binder-build.yml` - The workflow will automatically use your username

## More Information

- [Binder Documentation](https://mybinder.readthedocs.io/)
- [repo2docker Documentation](https://repo2docker.readthedocs.io/)
- [Reducing Binder startup time](https://discourse.jupyter.org/t/how-to-reduce-mybinder-org-repository-startup-time/4956)
