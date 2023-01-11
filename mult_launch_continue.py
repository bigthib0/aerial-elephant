import subprocess
import os

def write(filename, text):
    f = open(filename, "w+")
    f.write(text)
    f.close()


# gpu_type = ['ampere', 'titan', 'turing', 'pascal']
# gpu_nb = [1, 2, 4]
# for i, t in enumerate(gpu_type):
#     for nb in gpu_nb:

# for i in range(0, 2):
for i, name in enumerate(os.listdir('checkpoint')):
    print(name)
    text = f'#!/bin/sh\n#SBATCH --job-name train\n#SBATCH --output tt-continue{i}.o%j\n#SBATCH --ntasks 4\n#SBATCH --partition shared-gpu\n#SBATCH --time 12:00:00\n#SBATCH --gpus 1\n#SBATCH --mem-per-gpu 40000\n'
    #text += '#SBATCH --constraint="COMPUTE_TYPE_AMPERE|COMPUTE_TYPE_TITAN"\n'
    text += 'module load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0\nmodule load SciPy-bundle/2021.05\nmodule import matplotlib/3.4.2\n'
    text += f'NAME="f{i}"\n'
    text += 'echo "Running on $SLURM_NODELIST"\necho NAME = $NAME\n'
    text += f'python ./train_elephant.py --epochs 50 --complex {"False" if i==0 else "True"} --name_continue_training {name}'
    write(f'launch_{i}.sh', text)
    subprocess.call(['sbatch', f'launch_{i}.sh'])