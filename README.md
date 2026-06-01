# Volumetric Gaussian Splatting: Direct Optimization in Voxel Space for Efficient Medical Data Visualization

Avis Gaussian Splatting is a research fork of 3D Gaussian Splatting adapted for
volume-supervised training on 3D medical-style data such as CT or MR volumes.
Instead of supervising Gaussians with multi-view RGB images, this fork
optimizes a 3D Gaussian representation against a target volume inside a mask
region of interest.

This README is written around the workflow implemented in this repository today:
environment setup, volume and mask expectations, reference training commands,
evaluation, export, and viewing. The legacy image-based 3DGS code path still
exists in the tree, but the reproduction steps below target the volume pipeline.

## Relation To The Original Paper

This repository inherits its core representation from Kerbl et al.,
"3D Gaussian Splatting for Real-Time Radiance Field Rendering". In the original paper,
the task is novel-view synthesis from calibrated multi-view images.

For readers coming from the paper, the main differences in this fork are:

- Input data is a 3D volume plus aligned ROI mask, not calibrated multi-view
  RGB images.
- Gaussian initialization samples points from voxels inside the mask, rather
  than from sparse SfM points produced during camera calibration.
- Supervision is volumetric and mask-aware, using losses on rasterized voxel
  grids rather than photometric losses on rendered camera views.
- Evaluation is centered on masked volume metrics and exported PLY inspection,
  rather than novel-view synthesis benchmarks.

In other words, this codebase keeps the Gaussian parameterization and much of
the optimization/export machinery familiar from 3DGS, but retargets training to
masked 3D reconstruction from medical-style volumes rather than real-time
free-viewpoint image rendering.

## What This Repository Does

- Loads a target volume from `--volume_path` and an aligned ROI mask from
  `--mask_path`.
- Samples initial Gaussian centers from voxels inside the mask.
- Rasterizes Gaussians back into a voxel grid for supervision.
- Optimizes toward mask density, CT intensity, or both via
  `--supervision_target {mask,ct,joint}`.
- Supports learned or sampled intensity and opacity values.
- Can periodically export Gaussian PLY snapshots for inspection or downstream
  evaluation.
- Includes utilities for masked MSE evaluation and a standalone PLY viewer.

## Reproduction Checklist

To reproduce a run from scratch:

1. Clone the repository with submodules.
2. Create the Conda environment from `environment.yml`.
3. Prepare a volume and a mask that are spatially aligned.
4. Run `train.py` with an explicit `--model_path`, `--volume_path`, and
   `--mask_path`.
5. Use the generated `cfg_args` file under the model directory as the source of
   truth for later evaluation or reruns.
6. Evaluate the result with `evaluate_ply_masked_mse.py` or compare produced
   volumes with `compare_volumes.py`.

## Requirements

- Windows commands are shown below because that is the main workflow in this
  repository. The same scripts also work on Linux with the usual shell syntax
  adjustments.
- NVIDIA GPU recommended for training.
- CUDA-compatible PyTorch runtime.
- Conda or Miniconda.
- Visual Studio Build Tools on Windows for compiling CUDA extensions.
- A CUDA toolkit compatible with the PyTorch CUDA runtime in `environment.yml`.

The checked-in environment targets:

- Python 3.12
- PyTorch 2.7.0
- CUDA 12.6 via `pytorch-cuda=12.6`

## Setup

Clone the repository with submodules:

```shell
git clone --recursive <repo-url>
cd <repo-dir>
```

If the repository is already cloned without submodules:

```shell
git submodule update --init --recursive
```

Create and activate the environment:

```shell
SET DISTUTILS_USE_SDK=1
conda env create --file environment.yml
conda activate avis_gaussian_splatting
```

`environment.yml` installs the CUDA extensions from `submodules/` in editable
mode, so the build step depends on a working compiler toolchain.

Quick sanity checks:

```shell
python train.py --help
python evaluate_ply_masked_mse.py --help
python compare_volumes.py --help
```

## Repository Entry Points

The main scripts used for reproduction are:

| Path | Purpose |
| --- | --- |
| `train.py` | Volume-supervised training entrypoint |
| `arguments.py` | Unified CLI surface for training and export options |
| `evaluate_ply_masked_mse.py` | Evaluate an exported or external PLY with the same masked full-ROI path used during training |
| `compare_volumes.py` | Compare two volumes inside a mask with masked MSE and PSNR |
| `gs_viewer/` | Standalone viewer for Gaussian PLY exports |
| `gaussian_splatting/data/volume_loader.py` | Input loading, normalization, and volume resampling logic |
| `gaussian_splatting/utils/volume_supervisor.py` | Volume supervision, ROI handling, and masked evaluation logic |

## Data Requirements And Conventions

Supported input formats include `.nii`, `.nii.gz`, `.npy`, and `.mhd`.

Important conventions:

- `--volume_path` is the target volume used for supervision.
- `--mask_path` is the ROI mask used for initialization and masked losses.
- Volume and mask should be spatially aligned and describe the same subject.
- Volumes are normalized to `[0, 1]` by the loader. Do not assume Hounsfield
  Units are preserved unless you change the loading path.
- Volume tensor order is `[D, H, W] = [Z, Y, X]`.
- Gaussian coordinates are normalized `[x, y, z]` positions in `[0, 1]^3`.
- `_input_/` and `_output_/` are conventions used in this repository, not hard
  requirements.

Reproducibility notes:

- `train.py` seeds Python, NumPy, and PyTorch through `safe_state(...)`.
- Each training run writes the resolved CLI namespace to
  `<model_path>/cfg_args`.
- If you need to reproduce a previous run exactly, keep both the command line
  and the emitted `cfg_args` file.
- GPU kernels can still introduce small nondeterministic differences depending
  on hardware and library versions.

## Training

At minimum, provide:

- `--model_path`
- `--volume_path`
- `--mask_path`

### Reference Reproduction Command

The following PowerShell command is a good baseline for joint supervision at
full volume resolution without densification. Replace the input paths with your
own data.

```shell
python train.py `
  --model_path _output_/repro-joint-250k `
  --mask_path _input_/your_mask.nii.gz `
  --volume_path _input_/your_volume.nii.gz `
  --iterations 1000 `
  --init_n_points 250000 `
  --medical_mode none `
  --supervision_target joint `
  --mask_loss_weight 0.2 `
  --ct_loss_weight 1.0 `
  --volume_downscale_factor 1 `
  --volume_render_downscale_factor 1 `
  --disable_volume_overflow_guard `
  --save_ply_every 100 `
  --enable_diagnostics `
  --intensity_mode sampled `
  --opacity_mode sampled `
  --eval_masked_mse_full_roi_interval 100 `
  --eval_masked_mse_full_roi_target ct `
  --eval_masked_mse_full_roi_downscale_factor 1 `
  --max_points_per_iter 4000 `
  --disable_densification
```

What this baseline does:

- Trains 250k initial Gaussians sampled from the mask.
- Uses joint mask and CT supervision.
- Renders on the full working grid by setting
  `--volume_render_downscale_factor 1`.
- Periodically exports PLYs and runs masked full-ROI evaluation.
- Uses sampled intensity and opacity values from the input data.
- Disables densification to keep the optimization trace easier to reproduce.

### Higher-Detail Densification Run

If you want topology changes during training, use a configuration like this:

```shell
python train.py `
  --model_path _output_/repro-joint-densify `
  --mask_path _input_/your_mask.nii.gz `
  --volume_path _input_/your_volume.nii.gz `
  --iterations 1000 `
  --checkpoint_iterations 800 1200 1600 2000 `
  --init_n_points 200000 `
  --medical_mode none `
  --supervision_target joint `
  --mask_loss_weight 0.2 `
  --ct_loss_weight 1.0 `
  --enable_densification `
  --densify_from_iter 100 `
  --densification_interval 50 `
  --densify_max_new_points 10000 `
  --prune_min_opacity 0.001 `
  --max_points_per_iter 8000 `
  --volume_downscale_factor 1 `
  --volume_render_downscale_factor 1 `
  --disable_volume_overflow_guard `
  --save_ply_every 50 `
  --enable_diagnostics `
  --intensity_mode sampled `
  --opacity_mode sampled
```

### Common Training Knobs

The most important flags for reproducing behavior are:

- `--supervision_target {mask,ct,joint}`
  Selects whether optimization matches the mask field, CT intensity, or both.
- `--volume_loss_type {mse,dice,tversky,kl}`
  Loss for the mask branch.
- `--ct_loss_type {mse,dice,tversky,kl}`
  Loss for the CT branch when using `ct` or `joint` supervision.
- `--mask_loss_weight` and `--ct_loss_weight`
  Balance the two losses in joint mode.
- `--init_n_points`
  Initial Gaussian count sampled from the mask.
- `--max_points_per_iter`
  Caps the number of Gaussians updated per iteration to control VRAM usage.
- `--volume_downscale_factor`
  Downscales the loaded volume and mask.
- `--volume_render_downscale_factor`
  Downscales the internal working grid used for rasterization.
- `--medical_mode {none,organ,vessel,vessel_anisotropy}`
  Applies preset behavior on top of the general CLI.
- `--enable_densification` and `--disable_densification`
  Control topology updates.
- `--densify_from_iter`, `--densification_interval`,
  `--densify_max_new_points`, and `--prune_min_opacity`
  Tune densification and pruning.
- `--intensity_mode {learned,sampled,sampled_mean_covered}`
  Controls how per-Gaussian intensity is produced.
- `--opacity_mode {learned,sampled,sampled_mean_covered}`
  Controls how per-Gaussian opacity is produced.
- `--enable_diagnostics`
  Enables extra logging, parameter monitoring, and richer TensorBoard output.

### What A Successful Run Produces

Training writes artifacts under `--model_path`:

- `cfg_args`
  Resolved arguments for the run. Keep this file for reproduction.
- TensorBoard event files
  Written when TensorBoard is available in the environment.
- `ply_sequence/`
  PLY snapshots saved at iteration 1, every `--save_ply_every`, and the final
  iteration.
- `chkpnt*.pth`
  Checkpoints written for `--checkpoint_iterations`.

To inspect TensorBoard logs:

```shell
tensorboard --logdir _output_
```

## Evaluation

### Periodic In-Training Evaluation

`train.py` can run masked full-ROI MSE during training:

- `--eval_masked_mse_full_roi_interval`
- `--eval_masked_mse_full_roi_target {auto,mask,ct}`
- `--eval_masked_mse_full_roi_downscale_factor`

Use `ct` or `auto` for joint CT runs, and `mask` or `auto` for mask-only runs.

### Evaluate An Exported Or External PLY

Use `evaluate_ply_masked_mse.py` to score a PLY with the same volume rasterizer
used during training.

If you already have a training output directory with `cfg_args`:

```shell
python evaluate_ply_masked_mse.py `
  --ply_path path\to\model.ply `
  --training_model_path _output_/repro-joint-250k
```

If you want to evaluate directly from explicit inputs:

```shell
python evaluate_ply_masked_mse.py `
  --ply_path path\to\model.ply `
  --volume_path _input_/your_volume.nii.gz `
  --mask_path _input_/your_mask.nii.gz `
  --target ct
```

`--intensity_source auto` prefers the exported `intensity_01` attribute when it
exists and falls back to SH/DC features for standard 3DGS PLY files.

### Compare Two Volumes Inside A Mask

Use `compare_volumes.py` when you already have two voxel grids and want masked
MSE plus PSNR:

```shell
python compare_volumes.py `
  --volume_path _input_/reference.nii.gz `
  --mask_path _input_/roi_mask.nii.gz `
  --comparative_volume_path path\to\candidate.nii.gz
```

## Export And Viewing

PLY export is controlled primarily by:

- `--save_ply_every`
- `--ply_output_prefix`
- `--export_ao`
- `--export_ao_method {isotropic,normal}`
- `--export_ao_radius_vox`
- `--export_ao_n_samples`
- `--export_ao_strength`

Ambient occlusion is export-only. When enabled, AO is sampled from the mask and
baked into the exported appearance.

### Standalone PLY Viewer

Install the viewer dependencies:

```shell
uv pip install -r gs_viewer/requirements.txt
uv pip install -e gs_viewer
```

Run the viewer from the repository root:

```shell
gs-viewer --ply path\to\model.ply
```

Or run the script directly:

```shell
python gs_viewer\run_viewer.py --ply path\to\model.ply
```

The viewer expects the PLY schema exported by `scene/gaussian_model.py`,
including fields such as position, SH/DC appearance, scale, rotation, opacity,
and optionally `intensity_01` and `ao`.

See `gs_viewer/README.md` for controls and transfer-function editing.

## Optional: Accelerated Rasterizer For `sparse_adam`

This repository supports `--optimizer_type sparse_adam`, but it requires the
accelerated rasterizer branch.

```shell
uv pip uninstall diff-gaussian-rasterization -y
cd submodules\diff-gaussian-rasterization
git checkout 3dgs_accel
uv pip install -e .
```

Then run training with:

```shell
python train.py --model_path _output_/sparse-adam-run --volume_path ... --mask_path ... --optimizer_type sparse_adam
```

## Validation Notes

The fastest repository-level validation commands for this workflow are:

```shell
python train.py --help
python evaluate_ply_masked_mse.py --help
python compare_volumes.py --help
```

The current workspace does not include checked-in Python test files under
`tests/`, so CLI validation and smoke training runs are the practical checks
available here.

## Acknowledgements

This project builds on the original 3D Gaussian Splatting work by Kerbl et al.

- Paper and project page:
  <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- Full citation:
  Kerbl, Bernhard, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis.
  "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM
  Transactions on Graphics 42, no. 4 (2023).

## License

Based on original 3DGS repository, see `LICENSE.md`.
