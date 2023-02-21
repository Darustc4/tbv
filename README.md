# :star: Total Brain Volume Predictor (TBV)
**Total Brain Volume (TBV)** estimation from 3D ultrasound imaging for neborn babies using safe deep learning techniques.

Features:
- :tv: Modern and easy to use GUI powered by Custom TKinter. Visualize 3D rasters and predict brain volume with 1 click.
- :brain: 30 layer 3D convolution ResNet trained on 294 brain volumes and validaded on 76 more using **PyTorch**.
- :arrow_up: 3D Data augmentation on the go using TorchIO.
- :x: Noisy or tricky 3D volumes are automatically rejected using a bayesian network approximation.
- :watch: Fast inference on CPU, no GPU needed. High security inference might take a few minutes, though. 

Structure:
- :file_folder: Models: All the models and architectures tested tested in the making of this project, including graphs and a tester script.
- :file_folder: Visualizer: The final distributable program + network weights.


## Instructions for WSL

```console
cd ~
sudo apt update
sudo apt upgrade
ssh-keygen -t ed25519 -C "email@domain.com"
cat .ssh/id_ed25519.pub
```

Add this ssh key to GitHub's allowed SSH keys before continuing.

```console
git clone git@github.com:Darustc4/tbv.git
sudo apt install python3-venv gnupg2 wget
sudo usermod -a -G video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/21.40.2/ubuntu/focal/amdgpu-install_21.40.2.40502-1_all.deb
sudo apt-get install ./amdgpu-install_21.40.2.40502-1_all.deb
sudo amdgpu-install --usecase=rocm,hip

cd tbv
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scikit-learn opencv-python wget torchvision
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install torch-directml pynrrd tqdm
```

## Instructions for Ubuntu 22

```console
cd ~
sudo apt update
sudo apt upgrade
ssh-keygen -t ed25519 -C "email@domain.com"
cat .ssh/id_ed25519.pub
```

Add this ssh key to GitHub's allowed SSH keys before continuing.

```console
git clone git@github.com:Darustc4/tbv.git
sudo apt install python3-venv gnupg2 wget
sudo usermod -a -G video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb
sudo apt-get install ./amdgpu-install_5.4.50400-1_all.deb
sudo amdgpu-install --usecase=rocm,hip
sudo apt-get install miopen-hip

cd tbv
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scikit-learn opencv-python wget torchvision
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install torch-directml pynrrd tqdm

cd models
sudo HSA_OVERRIDE_GFX_VERSION=10.3.0 ../venv/bin/python3 conv3d_mono_no_age.py


```

Useful command:
sudo sh -c "sleep 150m; HSA_OVERRIDE_GFX_VERSION=10.3.0 ../venv/bin/python3 conv3d_no_age_simple.py"
