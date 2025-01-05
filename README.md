# ECG Classification with Grad-CAM

This repository implements a comprehensive pipeline for classifying ECG signals and visualizing feature importance using Grad-CAM. It is based on the implementation of the paper **"ECG Heartbeat Classification: A Deep Transferable Representation"** ([arXiv:1805.00794](https://arxiv.org/abs/1805.00794)), with the added feature of 1D Grad-CAM for model interpretability. The dataset used in this project is available [here](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).


## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Grad-CAM Visualization](#grad-cam-visualization)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)


## Project Overview

This project aims to classify ECG signals into different heart conditions and explain the model’s predictions using Grad-CAM. The workflow includes:

- Implementing the deep learning model from the referenced paper.
- Training the model on the MIT-BIH dataset.
- Fine-tuning the model on the PTB dataset using transfer learning.
- Adding 1D Grad-CAM visualization for interpretability.


## Model Architecture

The implemented 1D convolutional neural network (`ConvNet`) includes:

- Convolutional layers for feature extraction.
- Fully connected layers for classification.
- A modifiable classifier for transfer learning on PTB data.

### Grad-CAM Integration

- Implements Grad-CAM for 1D CNNs, enabling visualization of the most influential regions in the ECG signal for predictions.



## Grad-CAM Visualization
Grad-CAM effectively highlights regions in the ECG signals that influence the model's decisions, providing better interpretability for predictions.

1. Generate Grad-CAM for an ECG signal:

   ```python
   from gradcam import GradCAM1D

   gradcam = GradCAM1D(model, target_layer=model.convnet[4].conv2)
   cam = gradcam.generate_cam(input_tensor, target_class=0)
   ```

2. Visualize the results:

   ```python
   from utils import plot_ecg_with_background_cam

   plot_ecg_with_background_cam(cam, ecg_signal)
   ```


## Results

### MIT-BIH Dataset

- **Accuracy**: Achieved **95.96%** on the test set.

### PTB Dataset (Transfer Learning)

- **Accuracy**: Achieved **91.28%** on the test set.


## Acknowledgements

This project uses data from:

- [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/)
- [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/)

This repository implements the work described in the paper:

**"ECG Heartbeat Classification: A Deep Transferable Representation"** ([arXiv:1805.00794](https://arxiv.org/abs/1805.00794))

Additionally, it includes a Grad-CAM implementation to enhance model explainability. If you find this project useful, feel free to cite or acknowledge the original authors and contribute to the repository via issues or pull requests.

