#!/bin/bash
#SBATCH --job-name=remote-access
#SBATCH --out slurm_out/jupyter.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --exclude tir-0-[32,36],tir-1-[32,36]

source activate device_benchmarking;

PORT=$(shuf -i8000-9999 -n1)
HOST=$(hostname -i)

echo -e "
Copy/Paste this in your local terminal to ssh tunnel with remote
------------------------------------------------------------------
ssh -N -L $PORT:$HOST:$PORT jaredfer@tir.lti.cs.cmu.edu
------------------------------------------------------------------
Then open a browser on your local machine to the following address
(prefix w/ https:// if using password)
------------------------------------------------------------------
------------------------------------------------------------------
localhost:$PORT
------------------------------------------------------------------
"

/usr/bin/ssh -N -f -R $PORT:$HOST:$PORT jaredfer@tir.lti.cs.cmu.edu;

jupyter-lab --no-browser --port $PORT --ip=$HOST;
