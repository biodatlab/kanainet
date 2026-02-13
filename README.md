## KAN-AINet: Kolmogorov-Arnold Network with Adaptive Illumination Modulation for Generalizable Polyp Segmentation


KAN-AINet is a novel polyp segmentation architecture that leverages Kolmogorov-Arnold Networks (KAN) for adaptive illumination modulation and boundary-aware attention. Unlike standard neural networks that use fixed activation functions, KAN learns optimal per-task activation functions, enabling more expressive feature transformations for challenging colonoscopy images.

## 🔥 Highlights

- **State-of-the-Art Performance**  
  Achieves **XX% improvement in mDice** and **XX% improvement in mIoU** over prior SOTA models on external validation benchmarks:  
  `Kvasir-Sessile`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, and `PolypGen-C6`.

- **KAN-IMM (Illumination Modulation Module)**  
  Learns adaptive per-channel scaling to effectively handle specular reflections, shadows, and varying illumination conditions in colonoscopy images.

- **KAN-BAM (Boundary Attention Module)**  
  Utilizes multi-scale edge-aware attention (3×3, 5×5, 7×7 receptive fields) to accurately differentiate true polyp boundaries from illumination artifacts.

- **Interpretable Learned Functions**  
  KAN-based activation functions are directly visualizable, providing model interpretability.  
  Notably, **98% of the learned functions are novel and task-specific**, demonstrating strong adaptation to polyp segmentation

## Installation

```bash
git clone https://github.com//KAN-ACNet.git
cd KAN-ACNet
pip install -r requirements.txt
```
## Training

We used the same training dataset as ESPNet. The dataset can be accessed from the official ESPNet GitHub repository:  
[ESPNet Polyp Segmentation Repository](https://github.com/Raneem-MT/ESPNet_Polyp_Segmentation)

You can use the default training configuration or modify the hyperparameters in `config.py`.

### Train KAN-IANet

To train KAN-IANet with the default configuration:

```bash
python train_threshold.py --configs config.py

