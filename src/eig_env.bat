conda create -n eig-new-py python=3.9
conda activate eig-new-py
conda install matplotlib
conda install scipy
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install configparser
conda install h5py        
conda install pandas
conda install scikit-learn
set PYTHONPATH=%PYTHONPATH%;C:\Users\nick\source\changlab\ilker_collab\EIG-faces