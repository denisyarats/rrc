# RRC

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```
source activate rrc
``` 


## Instructions

To train
```
python train.py 
```

This will produce the `exp_local` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```
tensorboard --logdir runs
```

The console output is also available in a form:
```
| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132
```
