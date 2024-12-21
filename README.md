# Bridge Matching for Paired Tasks
In this project we implement Bridge Matching for one-to-many generation for solving paired
tasks like inpainting, super-resolution, etc. We support three setups: 1) a generalized Brownian Bridge with a triangular schedule for a diffusion coefficient $g(t)$
2) a Brownian Bridge with constant $g(t) = \sqrt{\gamma}$ 3) Flow Matching with $g(t) = 0$.


## Installation
1. Install required libraries to your environment
```
pip install -r requirements.txt
```
2. Clone EDM repository to the root of the project
```
git clone https://github.com/NVlabs/edm.git
```
3. Prepare FFHQ dataset by running, which saves it to `PROJECT_DIR/datasets/ffhq`
```
python bridge_matching/dataset/ffhq.py
```
4. Calculate statistics of a dataset for FID calculation. By default it saves statistics to `stats/ffhq`.
```
PYTHONPATH=".:edm" python calculate_dataset_stats.py --config-name stats_ffhq.yaml
```

## Training
To train the model for the inpainting task on FFHQ simply run. Other configs can be found at `bridge_matching/config`.
```
WANDB_API_KEY=YOUR_KEY PYTHONPATH=".:edm" accelerate launch --config_file accelerate_config.yaml --num_processes 1 train.py --config-name ffhq_inpainting.yaml
```

## Evaluation
FID metric can be calculated by running `eval_fid.py` script. One can set model checkpoint and a dataset in `bridge_matching/config/eval_fid.yaml`.
```
PYTHONPATH=".:edm" python eval_fid.py --config-name eval_fid.yaml
```
