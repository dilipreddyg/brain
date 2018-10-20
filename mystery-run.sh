#!/usr/bin/env bash

# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.01
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.05
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.08
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.1
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.5
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.8
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 1.0
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 1.5
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 5.0
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.001
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.005
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.008
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-2 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.1
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 5e-2 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.1
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 5e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.1
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 5e-4 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.1
# sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 1e-3 --lr_h1h2 1e-3 --lr_h2h3 1e-3 --lr_h3dout 1e-3 --lr_douth3 1e-3 --lr_h3h2 1e-3 --lr_h2h1 1e-3 --sigma 0.1
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.002
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.0001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.0002
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.0005
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.0008
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.00001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.00002
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.00005
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 1e-3 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.00008
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 3e-4 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 1e-4 --lr_h1h2 1e-4 --lr_h2h3 1e-4 --lr_h3dout 1e-4 --lr_douth3 1e-3 --lr_h3h2 1e-4 --lr_h2h1 1e-4 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-4 --lr_h2h3 3e-4 --lr_h3dout 3e-4 --lr_douth3 3e-4 --lr_h3h2 3e-4 --lr_h2h1 3e-4 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 1e-4 --lr_h1h2 1e-4 --lr_h2h3 1e-4 --lr_h3dout 1e-4 --lr_douth3 1e-4 --lr_h3h2 1e-4 --lr_h2h1 1e-4 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 1e-3 --lr_h1h2 1e-3 --lr_h2h3 1e-3 --lr_h3dout 1e-3 --lr_douth3 1e-3 --lr_h3h2 1e-3 --lr_h2h1 1e-3 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-5 --lr_h1h2 3e-5 --lr_h2h3 3e-5 --lr_h3dout 3e-5 --lr_douth3 1e-3 --lr_h3h2 3e-5 --lr_h2h1 3e-5 --sigma 0.001
sbatch --gres=gpu --qos=high --mem=6000 --time=0:45:00 run.sh --epochs 100 --lr_inh1 3e-4 --lr_h1h2 3e-5 --lr_h2h3 3e-5 --lr_h3dout 3e-5 --lr_douth3 3e-5 --lr_h3h2 3e-5 --lr_h2h1 3e-5 --sigma 0.001

$@