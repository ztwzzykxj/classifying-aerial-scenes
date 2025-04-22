# Aerial Scene Classification with Traditional and Deep Learning Methods

This repository presents an evaluation of multiple approaches for classifying aerial scene images. We explore both classical computer vision methods (LBP, SIFT, color histograms) and state-of-the-art deep learning models (ResNet, EfficientNet, SENet). We further tackle class imbalance and assess robustness via perturbation-based testing.

## 📁 Dataset

We used a custom-organized version of the [Aerial Landscape Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/aerial-cvpr2021).  
The dataset was manually split into `/train` and `/test` folders under:

```
D:\archive\Aerial_Landscapes_Split\
```

Each subfolder corresponds to a different scene category (e.g., forest, beach, farmland, city, etc.).

## ⚙️ Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- scikit-learn
- OpenCV
- scikit-image
- Pillow (PIL)
- NumPy

To install required packages:

```bash
pip install -r requirements.txt
```

## 🧠 Models and Methods

All methods were implemented and executed in the Jupyter Notebook:  
📓 **`train and test.ipynb`**

### 🔹 Traditional Machine Learning Approaches

- **LBP + SVM**: Uses Local Binary Patterns as texture descriptors, followed by a linear Support Vector Machine classifier.
- **SIFT + KNN**: Uses Scale-Invariant Feature Transform descriptors pooled by mean, classified using k-Nearest Neighbors.
- **LBP + Color Histogram + SVM**: Concatenates LBP and 3D color histogram features for a hybrid descriptor and uses SVM.

These methods are easy to interpret and serve as solid baselines.

### 🔸 Deep Learning Architectures

Implemented using pretrained models from `torchvision.models`:

- **ResNet-18 / ResNet-50**
- **EfficientNet-B0**
- **SENet (SE-ResNet)**

Each model is fine-tuned using the aerial dataset within the notebook.

## ⚖️ Class Imbalance Handling

In `train and test.ipynb`, class imbalance is addressed by:

- Computing per-class weights and applying them to the loss function
- Using a weighted sampler in the DataLoader to balance training batches

These methods increase the influence of underrepresented classes during training.

## 🧪 Robustness Testing

The notebook also includes a robustness evaluation section that tests model predictions under the following perturbations:

- **Blur**: Gaussian blur
- **Occlude**: Black box occlusion
- **Noise**: Random Gaussian noise
- **Darken**: Brightness reduction
- **Contrast**: Contrast enhancement

Results are printed in the notebook showing perturbation-specific accuracies.

## 📈 Evaluation

We report:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Optional: Confusion Matrix
- Robustness under perturbation

Evaluation is performed using `sklearn.metrics.classification_report`.

## ▶️ How to Run

Open the notebook:

```bash
jupyter notebook "train and test.ipynb"
```

Inside the notebook, you can:
- Execute classical models (LBP, SIFT)
- Train and test deep CNNs (ResNet, EfficientNet, SENet)
- Run robustness testing at the end

## 🧾 References

- [1] Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.
- [2] Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints*. IJCV.
- [3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- [4] Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- [5] Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR.
- [7] Buda, M., Maki, A., & Mazurowski, M. A. (2018). *A systematic study of the class imbalance problem in CNNs*. Neural Networks.
- [8] Dodge, S., & Karam, L. (2017). *A study and comparison of human and deep learning recognition performance under visual distortions*. ICCCN.
