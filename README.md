# :star: Total Brain Volume Predictor (TBV)
**Total Brain Volume (TBV)** estimation from 3D ultrasound imaging for neborn babies using safe deep learning techniques.

Features:
- :tv: Modern and easy to use GUI powered by Custom TKinter. Visualize 3D rasters and predict brain volume with 1 click.
- :brain: 30 layer 3D convolution ResNet trained on 294 brain volumes and validaded on 76 more using **PyTorch**.
- :arrow_up: 3D Data augmentation on the go using TorchIO.
- :x: Noisy or tricky 3D volumes are automatically rejected using a bayesian network approximation.
- :watch: Fast inference on CPU, no GPU needed. High security inference might take a few minutes, though. 

Structure:
- :file_folder: Models: All the models and architectures tested tested in the making of this project, including a tester script that benchmarks the models (measure errors).
- :file_folder: Visualizer: The final distributable program + network weights.