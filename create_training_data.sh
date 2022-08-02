#!/bin/bash -login
#PBS -N create_training_data_python
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=4
#PBS -l mem=32gb
#PBS -m bae
#PBS -j oe

# change to the directory where the job was submitted
cd $PBS_O_WORKDIR

cd $BIGWORK/felix/Sim/DataGen/4/Python

module load GCC/7.3.0-2.30  OpenMPI/3.1.1
# module load matplotlib/3.0.0-Python-3.6.6
module load scikit-image/0.14.1-Python-3.6.6
# module load CUDA/9.2.88
# module load PyTorch/1.2.0-Python-3.6.6
module load ABAQUS/2018

# run program
echo "I am running on $HOSTNAME"
chmod +x voronoi_loop.sh
dos2unix voronoi_loop.sh
./voronoi_loop.sh
