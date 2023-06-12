```
conda create -n recbole python=3.7
conda activate recbole
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c aibox recbole
pip install ray
python data_creator.py
python train.py --model=RecVAE --config_files 'RecVAE.yaml' 'env.yaml' 'dataset.yaml' 'wandb.yaml'
python inference.py --type=G --config_files 'RecVAE.yaml' 'env.yaml' 'dataset.yaml' 'wandb.yaml'
```