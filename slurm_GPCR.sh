#!/bin/bash
#SBATCH --job-name=GPCR_ACseqGAN  
#SBATCH --partition=smp             
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32        
#SBATCH --time=48:00:00              
#SBATCH --mem=16G                  
#SBATCH --output=GPCR_penalty.log       
#SBATCH --error=GPCR_penalty_ERROR.log  

module load gcc/8.2.0

source /ihome/jwang/hat170/miniconda3/etc/profile.d/conda.sh
conda activate tf1.4

python /ihome/jwang/hat170/ACORGAN/condi_GPCR.py