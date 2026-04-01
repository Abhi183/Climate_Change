# Automated Detection of Atmospheric Circulation Patterns Using Machine Learning for Climate Change Analysis

**Machine Learning Capstone — Final Project**

---

## Abstract

Climate change is altering global atmospheric circulation patterns in uncertain yet consequential ways. Among these patterns, cut-off lows (COLs), cut-off highs (COHs), and closed lows (CLs) are particularly significant due to their association with extreme precipitation events. This study presents a comparative evaluation of supervised and unsupervised machine learning approaches for the automated detection and classification of these circulation patterns from gridded climate model output. Using a dataset of 737 labeled spatial pressure fields — and in extended experiments, combined pressure and upper-level wind fields — derived from simulated 19th-century (cold) and late 21st-century (warm) climate conditions, we train and evaluate Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), K-Means clustering, and Self-Organizing Maps (SOMs). Results show that supervised deep learning models significantly outperform unsupervised methods, with the Simple RNN trained on combined pressure and wind features achieving the highest classification accuracy of **88.74%**. Our findings demonstrate that machine learning offers a scalable and effective framework for automating climate pattern detection and for quantifying shifts in circulation regime frequency under climate change.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
   - [4.1 Data Preprocessing](#41-data-preprocessing)
   - [4.2 Unsupervised Methods](#42-unsupervised-methods)
   - [4.3 Supervised Deep Learning Methods](#43-supervised-deep-learning-methods)
5. [Experimental Setup](#5-experimental-setup)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

The intensification of the global hydrological cycle under anthropogenic warming is expected to alter the frequency, duration, and spatial distribution of extreme weather events. Cut-off lows (COLs) — closed cyclonic vortices detached from the polar jet stream — are a particularly important class of circulation pattern, as they are responsible for intense and prolonged precipitation in mid-latitude regions. Similarly, cut-off highs (COHs) and generic closed lows (CLs) modulate regional weather variability in ways that remain poorly constrained under future climate scenarios.

Traditional detection of these features relies on subjective expert identification or rule-based algorithms applied to reanalysis products. Such methods are labor-intensive, inconsistent across studies, and difficult to apply at scale to the large ensembles of climate model output needed to characterize future changes in circulation. Machine learning offers an attractive alternative: once trained on a labeled dataset, a model can classify thousands of circulation snapshots in seconds, with reproducible, quantitative outputs.

This project addresses two objectives:

1. **Automated detection**: Develop and compare machine learning classifiers — including CNNs, RNNs, LSTMs, K-Means, and SOMs — for identifying COLs, COHs, CLs, and non-regions of interest (NROIs) from gridded geopotential/pressure fields.
2. **Climate change analysis**: Use the trained classifiers to analyze whether and how the frequency of these circulation patterns shifts between simulated past (19th-century, cold) and projected future (late 21st-century, warm) climate conditions.

---

## 2. Background and Related Work

**Atmospheric circulation patterns and climate change.** Cut-off lows form when an upper-level trough breaks off from the main westerly flow, creating an isolated cyclone. Their association with extreme rainfall (e.g., Viale & Nuñez, 2011; Reboita et al., 2010) makes them a critical target for climate impact studies. However, their response to global warming is uncertain: while some studies project an equatorward shift and frequency decrease, others suggest regional increases in intensity (Nieto et al., 2005).

**Machine learning for weather pattern classification.** Deep learning has been applied to atmospheric science with increasing success. CNNs have been used to detect tropical cyclones and atmospheric rivers in climate model output (Liu et al., 2016; Racah et al., 2017). SOMs have a longer history in synoptic climatology for unsupervised weather typing (Hewitson & Crane, 2002). This work extends these approaches to the specific task of COL/COH/CL detection using both supervised and unsupervised paradigms, and evaluates the added value of incorporating wind data alongside pressure fields.

---

## 3. Dataset

The dataset consists of gridded climate model output representing atmospheric circulation patterns for two climate periods:

- **Historical period**: Multiple decades of simulated 19th-century conditions (cold, pre-industrial climate).
- **Future period**: Multiple decades of simulated late 21st-century conditions (warm, high-emission scenario).

Each sample corresponds to a **15 × 15 spatial grid** centered on a detected feature location, extracted from daily or 6-hourly model output. Two atmospheric variables are available:

| Variable | Description |
|----------|-------------|
| Surface/500 hPa Pressure | Geopotential or pressure field capturing the circulation structure |
| CESMU200 (Upper Wind) | 200 hPa zonal wind field capturing jet stream interaction |

**Class distribution** (737 total samples):

| Class | Label | Count | Proportion |
|-------|-------|-------|------------|
| Cut-Off Low (COL) | 2 | 327 | 44.4% |
| Closed Low (CL) | 0 | 178 | 24.2% |
| Cut-Off High (COH) | 3 | 161 | 21.8% |
| No Region of Interest (NROI) | 1 | 71 | 9.6% |

The dataset exhibits moderate class imbalance, with COLs comprising nearly half of all samples and NROIs being the rarest class. This imbalance is physically meaningful — COLs occur most frequently in the regions sampled — but it does present a challenge for classifiers.

The full dataset is provided as `First2YearRawAndNormData.zip`, containing raw and min-max normalized versions of both pressure and wind fields in CSV format organized by class.

---

## 4. Methodology

### 4.1 Data Preprocessing

All experiments apply a consistent preprocessing pipeline:

1. **Loading**: CSV files are read from class-labeled subdirectories (CL, COL, COH, NROI).
2. **Normalization**: Each feature is scaled to [0, 1] using min-max normalization:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

3. **Reshaping**: The 225 flattened pressure features (or 450 for pressure + wind) are reshaped into spatial tensors:
   - Single-feature: `(N, 15, 15, 1)` — treating the 15×15 grid as a single-channel image.
   - Dual-feature: `(N, 15, 15, 2)` — stacking pressure and wind as two channels.
4. **Train/test split**: For supervised models, 70% of samples are used for training (515 samples) and 30% for testing (222 samples), stratified to preserve class proportions.
5. **Label encoding**: Integer labels — CL=0, NROI=1, COL=2, COH=3 — are used with sparse categorical cross-entropy loss.

### 4.2 Unsupervised Methods

#### K-Means Clustering

K-Means partitions the dataset into *k* = 4 clusters (matching the number of known classes) by minimizing within-cluster sum-of-squared distances. The algorithm is initialized with random state 42 and run on the full normalized, flattened feature matrix. Because K-Means produces unlabeled clusters, each cluster is assigned the plurality class label of its members for evaluation purposes. This approach does not require a train/test split, as the algorithm is unsupervised.

#### Self-Organizing Map (SOM)

A SOM is an unsupervised neural network that projects high-dimensional input onto a low-dimensional (here, 15×15) topological grid, preserving input-space distances. Our SOM implementation follows the standard algorithm:

1. **Initialization**: Weights $W \in \mathbb{R}^{15 \times 15 \times D}$ (where *D* = 225 or 450) are randomly initialized.
2. **Competition**: For each input vector $\mathbf{x}$, the best matching unit (BMU) is the neuron with minimum Euclidean distance to $\mathbf{x}$.
3. **Adaptation**: Weights of the BMU and its Manhattan-distance neighborhood are updated:
$$W_i(t+1) = W_i(t) + \eta(t) \cdot h(i, \text{BMU}, t) \cdot (\mathbf{x} - W_i(t))$$
   where $\eta(t)$ decays from 0.5 and the neighborhood radius decays from 4 over 150,000 steps.
4. **Labeling**: Each neuron is labeled by the most frequent class among its mapped training samples.

Accuracy is measured at checkpoints: {500, 1,000, 5,000, 10,000, 20,000, 50,000, 75,000, 100,000, 125,000, 150,000} steps.

### 4.3 Supervised Deep Learning Methods

All supervised models are trained with:
- **Optimizer**: Adam (default learning rate)
- **Loss function**: Sparse categorical cross-entropy
- **Batch size**: 32
- **Early stopping**: Training is halted if validation loss does not improve for a patience window of 30–50 epochs, with a maximum of 1,000 epochs.

#### Convolutional Neural Network (CNN)

CNNs exploit local spatial structure in the 15×15 pressure grid through learned convolutional filters. The architecture for the pressure-only model:

| Layer | Configuration |
|-------|---------------|
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool |
| Conv2D | 128 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dense | 64 units, ReLU |
| Dense (output) | 4 units, Softmax |

For the pressure + wind model, the input shape is extended to `(15, 15, 2)` to accommodate the dual-channel tensor, and the final convolutional block is simplified (Dropout 0.25 added after pooling).

#### Simple Recurrent Neural Network (RNN)

A SimpleRNN treats each row of the 15×15 spatial grid as a timestep, processing the grid as a sequence of 15 vectors of length 15. Architecture:

| Layer | Configuration |
|-------|---------------|
| SimpleRNN | 64 units, input shape (15, 15) |
| Dense (output) | 4 units, Sigmoid/Softmax |

For the dual-feature model, a Lambda layer extracts the primary channel before the RNN.

#### Long Short-Term Memory (LSTM)

LSTMs extend the RNN with gating mechanisms that alleviate the vanishing gradient problem, enabling learning of longer-range dependencies across the spatial sequence. Architecture:

| Layer | Configuration |
|-------|---------------|
| Lambda | Channel extraction, input (15, 15, 2) |
| LSTM | 64 units, return sequences=True |
| Dropout | 0.1 |
| LSTM | 64 units |
| Dense (output) | 4 units, Softmax |

---

## 5. Experimental Setup

Eight experimental configurations are evaluated, varying the model type and input feature set:

| Experiment | Model | Input Features | Feature Dimension |
|------------|-------|----------------|-------------------|
| 1 | CNN | Pressure only | 15×15×1 |
| 2 | SimpleRNN | Pressure only | 15×15 |
| 3 | K-Means | Pressure only | 225 |
| 4 | SOM | Pressure only | 225 |
| 5 | SOM | Pressure + Wind | 450 |
| 6 | CNN | Pressure + Wind | 15×15×2 |
| 7 | SimpleRNN | Pressure + Wind | 15×15×2 |
| 8 | LSTM | Pressure + Wind | 15×15×2 |

All experiments use the same 737-sample dataset. Supervised models use a fixed 70/30 train/test split (515 training, 222 testing). Unsupervised models use all 737 samples for fitting and evaluation. Model performance is primarily assessed via **test accuracy** and the **confusion matrix**.

---

## 6. Results

### 6.1 Overall Performance Comparison

| Model | Input | Test Accuracy |
|-------|-------|:-------------:|
| K-Means | Pressure | ~62% |
| SOM | Pressure | ~76% |
| SOM | Pressure + Wind | ~77% |
| CNN | Pressure | 86.94% |
| CNN | Pressure + Wind | 86.04% |
| SimpleRNN | Pressure | 87.84% |
| LSTM | Pressure + Wind | 88.29% |
| **SimpleRNN** | **Pressure + Wind** | **88.74%** |

### 6.2 Confusion Matrices — Supervised Models

All confusion matrices report counts in the order: **CL, NROI, COL, COH** (rows = true, columns = predicted).

**CNN — Pressure only (Accuracy: 86.94%)**

```
                Predicted
              CL  NROI  COL  COH
True   CL   [ 52    0    6    0 ]
       NROI [  0   12    7    2 ]
       COL  [  8    3   81    0 ]
       COH  [  0    2    1   48 ]
```

**SimpleRNN — Pressure only (Accuracy: 87.84%)**

```
                Predicted
              CL  NROI  COL  COH
True   CL   [ 52    0    6    0 ]
       NROI [  0   17    1    3 ]
       COL  [  8    7   77    0 ]
       COH  [  0    2    0   49 ]
```

**CNN — Pressure + Wind (Accuracy: 86.04%)**

```
                Predicted
              CL  NROI  COL  COH
True   CL   [ 47    0   11    0 ]
       NROI [  0   12    7    2 ]
       COL  [  6    1   84    1 ]
       COH  [  0    1    2   48 ]
```

**SimpleRNN — Pressure + Wind (Accuracy: 88.74%)**

```
                Predicted
              CL  NROI  COL  COH
True   CL   [ 53    0    5    0 ]
       NROI [  0   16    3    2 ]
       COL  [  7    5   80    0 ]
       COH  [  0    3    0   48 ]
```

**LSTM — Pressure + Wind (Accuracy: 88.29%)**

```
                Predicted
              CL  NROI  COL  COH
True   CL   [ 53    0    5    0 ]
       NROI [  0   14    5    2 ]
       COL  [  7    5   80    0 ]
       COH  [  0    1    1   49 ]
```

### 6.3 Per-Class Analysis

Examining the confusion matrices across models reveals consistent patterns:

- **COL (Cut-Off Low)**: The dominant and best-classified category. With 327 training samples, all supervised models achieve ~80–84 correct predictions out of ~92 test cases (precision ≈ 87–91%).
- **COH (Cut-Off High)**: Also well-classified, with 48–49 correct out of ~51 test cases (precision ≈ 94–96%). Occasional confusion with COL reflects the physical similarity between these two circulation types.
- **CL (Closed Low)**: Consistently confused with COL (5–11 misclassifications), suggesting that closed lows and cut-off lows share overlapping spatial signatures in pressure fields — a physically interpretable finding.
- **NROI (No Region of Interest)**: The most challenging class, with only 71 training examples. Models correctly identify 12–17 of ~21 test cases. The SimpleRNN achieves the best NROI recall (81% with pressure-only features), indicating stronger temporal feature extraction compared to CNN.

### 6.4 Impact of Adding Wind Features

| Model | Pressure Only | Pressure + Wind | Change |
|-------|:-------------:|:---------------:|:------:|
| CNN | 86.94% | 86.04% | −0.90% |
| SimpleRNN | 87.84% | 88.74% | +0.90% |

Adding upper-level wind data (CESMU200) yields a marginal improvement for the RNN (+0.9%) but a slight decrease for the CNN (−0.9%). This divergence suggests that the sequential feature extraction of RNNs is better suited to exploit the complementary information in wind fields, while the CNN architecture used here may not capture the cross-channel interactions effectively with only two channels and a simplified convolutional block.

---

## 7. Discussion

### 7.1 Supervised vs. Unsupervised Approaches

The supervised deep learning models substantially outperform unsupervised clustering methods across all configurations. K-Means, limited to linear decision boundaries in feature space, achieves approximately 62% accuracy — only slightly above a majority-class baseline of ~44%. The SOM improves on this by ~14 percentage points, leveraging its topological structure to create a more nuanced partition of the feature space. Nevertheless, both methods fall well short of the supervised models, which benefit from direct optimization against labeled data.

This result is expected: the four circulation classes (COL, COH, CL, NROI) are not linearly separable in the raw pressure field feature space, and the unsupervised methods lack the label information needed to orient their representations accordingly. Despite this, the SOM's label maps offer an interpretable visualization of the feature space structure and may be valuable for exploratory analysis of unlabeled future-climate data.

### 7.2 Spatial vs. Sequential Modeling

The near-parity between CNN and RNN/LSTM performance is notable. CNNs explicitly model 2D spatial structure through convolution, which is well-matched to the 15×15 pressure grid. RNNs, by contrast, treat each row of the grid as a timestep in a sequence — a less natural inductive bias for spatial data. Yet the RNNs match or slightly exceed CNN performance, suggesting that the row-wise sequential structure of pressure patterns (e.g., latitudinal gradients associated with COLs and COHs) carries sufficient discriminative information. LSTMs do not offer a significant advantage over SimpleRNNs at this sequence length (15 timesteps), as the vanishing gradient problem is not severe enough over such short sequences to warrant the additional complexity.

### 7.3 Class Imbalance and Minority Class Performance

The NROI class (9.6% of samples) is consistently the most difficult to classify. With only ~71 training samples, models are prone to misclassifying NROIs as COLs or COHs. Future work could address this through oversampling (e.g., SMOTE), class-weighted loss functions, or data augmentation (rotation, flipping of the spatial grids). Improving NROI recall is important for the downstream climate change analysis, as failing to correctly exclude non-events will inflate estimates of pattern frequency.

### 7.4 Physical Interpretation

The primary confusion axes — CL ↔ COL and NROI ↔ COL/COH — are physically meaningful. Closed lows and cut-off lows represent a continuum of cyclone detachment from the jet stream, and their pressure fields can appear structurally similar at the spatial scale of the 15×15 window. This suggests that additional features (e.g., the temporal evolution of the pattern, or larger spatial context) could improve discrimination. The inclusion of wind fields partially addresses this, as the 200 hPa wind pattern is more clearly differentiated between COLs (which show closed anticyclonic flow above) and CLs.

---

## 8. Conclusion

This study demonstrates the feasibility of automated machine learning-based detection of atmospheric circulation patterns relevant to climate change. Across eight experimental configurations, we find that:

1. **Supervised deep learning substantially outperforms unsupervised clustering**: CNN, RNN, and LSTM models achieve 86–89% accuracy, compared to ~62–77% for K-Means and SOM. Labeled training data is essential for effective classification at this accuracy level.

2. **The Simple RNN with pressure + wind features achieves the highest accuracy (88.74%)**, slightly outperforming the CNN and LSTM. The row-wise sequential processing of spatial grids is an effective, if non-obvious, inductive bias for this problem.

3. **Adding upper-level wind data provides a modest, model-dependent benefit**: RNNs benefit from dual-channel input (+0.9%), while CNNs show a marginal regression (−0.9%), indicating that model architecture mediates the utility of additional atmospheric variables.

4. **CL–COL confusion is the dominant error mode**, reflecting genuine physical ambiguity between these circulation types in pressure-only representations. Larger spatial contexts, temporal sequences, or additional atmospheric variables (e.g., vorticity, geopotential height) may further reduce this confusion.

5. **The NROI class remains challenging** due to limited training samples. Addressing class imbalance through augmentation or resampling is a priority for future work.

These results establish a strong baseline for automated circulation pattern detection and lay the groundwork for large-scale application to multi-ensemble climate model output. By applying trained classifiers to full historical and future climate simulations, it becomes feasible to quantify statistically robust changes in COL, COH, and CL frequency, seasonality, and regional distribution under anthropogenic warming — a key step toward understanding the consequences of climate change for extreme weather.

**Future directions** include: (i) expanding the dataset with additional climate model ensembles to improve minority-class coverage; (ii) incorporating temporal sequences of pressure fields for dynamic pattern tracking; (iii) applying the trained models to the full climate simulation archive to generate frequency climatologies; and (iv) exploring attention-based architectures (e.g., Vision Transformers) that may better capture global spatial relationships within the pressure grid.

---

## 9. References

Hewitson, B. C., & Crane, R. G. (2002). Self-organizing maps: Applications to synoptic climatology. *Climate Research*, 22(1), 13–26.

Liu, Y., Racah, E., Correa, J., Khosrowshahi, A., Lavers, D., Kunkel, K., ... & Collins, W. (2016). Application of deep convolutional neural networks for detecting extreme weather in climate datasets. *arXiv preprint arXiv:1605.01156*.

Nieto, R., Gimeno, L., de la Torre, L., Ribera, P., Gallego, D., García-Herrera, R., ... & Anel, J. A. (2005). Climatological features of cutoff low systems in the Northern Hemisphere. *Journal of Climate*, 18(16), 3085–3103.

Racah, E., Beckham, C., Maharaj, T., Kahou, S. E., Prabhat, M., & Pal, C. (2017). ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events. *Advances in Neural Information Processing Systems*, 30.

Reboita, M. S., Nieto, R., Gimeno, L., da Rocha, R. P., Ambrizzi, T., Garreaud, R., & Kruger, L. F. (2010). Climatological features of cutoff low systems in the Southern Hemisphere. *Journal of Geophysical Research: Atmospheres*, 115(D17).

Viale, M., & Nuñez, M. N. (2011). Climatology of winter orographic precipitation over the subtropical central Andes and associated synoptic and regional characteristics. *Journal of Hydrometeorology*, 12(4), 481–507.

---

## Repository Structure

```
Climate_Change/
├── README.md                          # This document
├── First2YearRawAndNormData.zip       # Dataset (raw + normalized pressure/wind CSVs)
├── Pressure/                          # Single-feature experiments (pressure only)
│   ├── CNN_pressure.ipynb             # CNN classifier — pressure features
│   ├── RNN_pressure.ipynb             # SimpleRNN classifier — pressure features
│   └── KMeans_SOM_pressure.ipynb      # K-Means and SOM — pressure features
└── Pressure + Wind/                   # Dual-feature experiments (pressure + wind)
    ├── CNN_Pressure_Wind.ipynb        # CNN classifier — pressure + wind
    ├── RNN_Pressure_Wind.ipynb        # SimpleRNN + LSTM — pressure + wind
    └── SOM_Pressure_Wind.ipynb        # SOM — pressure + wind
```

## Requirements

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install dependencies via:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```
