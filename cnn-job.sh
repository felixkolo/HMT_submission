#!/bin/bash -login
#PBS -N create_training_data_python
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=4gb
#PBS -j oe

# change to the directory where the job was submitted
cd $PBS_O_WORKDIR

cd $BIGWORK/felix/Python/

## work
module load GCC/7.3.0-2.30
module load CUDA/9.2.88
module load OpenMPI/3.1.1
module load PyTorch/1.2.0-Python-3.6.6

## don't work
module load matplotlib/3.0.0-Python-3.6.6
module load Pillow/5.3.0-Python-3.6.6
# module load scikit-image/0.14.1-Python-3.6.6

# run program
echo "I am running on $HOSTNAME"
python cnnDO.py
