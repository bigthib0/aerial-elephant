import subprocess

def write(filename, text):
    f = open(filename, "w+")
    f.write(text)
    f.close()

def complexity_launch():
    for i in [1, 2, 4]:
        text = f'#!/bin/sh\n#SBATCH --job-name train\n#SBATCH --output xtrain{i}.o%j\n#SBATCH --ntasks 8\n#SBATCH --partition shared-gpu\n#SBATCH --time 12:00:00\n#SBATCH --gpus 1\n#SBATCH --mem-per-gpu 40000\n'
        # text += '#SBATCH --constraint="COMPUTE_TYPE_AMPERE|COMPUTE_TYPE_TITAN"\n'
        text += '#SBATCH --exclusive\n'
        text += 'module load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0\nmodule load SciPy-bundle/2021.05\nmodule load matplotlib/3.4.2\n'
        text += f'NAME="f{i}"\n'
        text += 'echo "Running on $SLURM_NODELIST"\necho NAME = $NAME\n'
        text += f'python3 ./train_elephant.py --epochs 15 --model_complexity {i} --mini_dataset_testing True'
        write(f'new_launch_{i}.sh', text)
        subprocess.call(['sbatch', f'new_launch_{i}.sh'])

def fine_tuning_launch(arr, param):
    for a in arr:
        text = f'#!/bin/sh\n#SBATCH --job-name finetune\n#SBATCH --output tuning/finetune_{param[2]}{a}.o%j\n#SBATCH --ntasks 8\n#SBATCH --partition shared-gpu\n#SBATCH --time 12:00:00\n#SBATCH --gpus 1\n#SBATCH --mem-per-gpu 40000\n'
        text += '#SBATCH --exclusive\n'
        text += 'module load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0\nmodule load SciPy-bundle/2021.05\nmodule load matplotlib/3.4.2\n'
        text += f'NAME="f{a}"\n'
        text += 'echo "Running on $SLURM_NODELIST"\necho NAME = $NAME\n'
        text += f'python3 ~/train_elephant.py --epochs 15 {param} {a}'
        write(f'tuning/launch_{a}.sh', text)
        subprocess.call(['sbatch', f'tuning/launch_{a}.sh'])


def classic():
    for a in [1,2]:
        text = f'#!/bin/sh\n#SBATCH --job-name {a}\n#SBATCH --output classic{a}.o%j\n#SBATCH --ntasks 8\n#SBATCH --partition shared-gpu\n#SBATCH --time 06:00:00\n#SBATCH --gpus 1\n#SBATCH --mem-per-gpu 40000\n'
        text += 'module load GCC/10.3.0\nmodule load OpenMPI/4.1.1\nmodule load TensorFlow/2.6.0\nmodule load SciPy-bundle/2021.05\nmodule load matplotlib/3.4.2\n'
        text += f'NAME="f{a}"\n'
        text += 'echo "Running on $SLURM_NODELIST"\necho NAME = $NAME\n'
        text += f'python3 ~/train_elephant.py --epochs 25 --elephant_only {False if a == 0 else True} --mini_dataset_testing 0.2 '
        write(f'launch_{a}.sh', text)
        subprocess.call(['sbatch', f'launch_{a}.sh'])


# compl = [1, 2, 4]
# activation = [1, 2, 3]
# learn = [0.001, 0.01, 0.1]
# data = [0.2, 0.7]

# fine_tuning_launch(compl, "--model_complexity")
# fine_tuning_launch(activation, "--activation_id")
# fine_tuning_launch(learn, "--lr")
# fine_tuning_launch(data, "--mini_dataset_testing")

classic()