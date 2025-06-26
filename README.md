# Deep Adaptive Learning Predicts and Diagnoses CSVD-Related Cognitive Decline using Radiomics from T2-FLAIR: A Multi-Centre Study

## Overview
This repository provides the official inference code and example data for the study:

**Deep Adaptive Learning Predicts and Diagnoses CSVD-Related Cognitive Decline using Radiomics from T2-FLAIR: A Multi-Centre Study**

The project implements a transformer-based deep learning model for the detection and prediction of cognitive impairment related to cerebral small vessel disease (CSVD-CI) using radiomics features extracted from white matter hyperintensity (WMH) regions on T2-FLAIR MRI. The model leverages domain adaptation for robust generalization across multiple centers.

## Authors
Lili Huang#, Zhuoyuan Li#, Xiaolei Zhu#, Hui Zhao, Chenglu Mao, Zhihong Ke, Yuting Mo, Dan Yang, Yue Cheng, Ruomeng Qin, Zheqi Hu, Pengfei Shao, Ying Chen, Min Lou*, Kelei He*, Yun Xu*

# These authors contributed equally
* Corresponding authors: Yun Xu (xuyun20042001@aliyun.com), Kelei He (hkl@nju.edu.cn), Min Lou (lm99@zju.edu.cn)

## Reference
If you use this code or data, please cite our work:
> Deep Adaptive Learning Predicts and Diagnoses CSVD-Related Cognitive Decline using Radiomics from T2-FLAIR: A Multi-Centre Study

## Abstract
Early identification of cerebral small vessel disease related cognitive impairment (CSVD-CI) is crucial for timely clinical intervention. We developed a transformer-based deep learning model using WMH radiomics features from T2-FLAIR images to detect CSVD-CI. The model was trained and validated on 783 subjects from three centers, using domain adaptation for generalization. The model achieved high AUCs and outperformed conventional machine learning models. Key radiomics features, especially logarithm-transformed gray level size zone matrix features, were identified as important contributors and correlated with clinical and imaging markers. This approach provides an automated, interpretable, and noninvasive tool for CSVD-CI detection.

## Directory Structure
- `example_model/`: Pre-trained model weights (e.g., `bestv_ci.pth`).
- `example_data/`: Example de-identified data for testing the inference pipeline.
- `inference.py`: Inference script to load the model and data, and output prediction results.
- `requirements.txt`: List of required Python packages.
- `result.txt`: Output file for inference results.

## Installation
1. Clone this repository and enter the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the inference script with example data and model:
```bash
python inference.py --data_dir example_data/ --model_path example_model/bestv_ci.pth
```
- You can specify additional arguments such as `--batch_size` and `--out_txt` as needed.
- The script will output prediction results and accuracy statistics to `result.txt` by default.

## Data Format
- Example data is provided in `example_data/` for testing the inference workflow.
- To use your own data, follow the structure and format of the example data directory.

## Model Description
- The model is a hierarchical transformer architecture that takes 1,316 radiomics features per subject, grouped into 85 feature classes, as input.
- Domain adaptation is used to ensure robust performance across different centers.
- The code is for inference only. 

## Contact
For questions or requests (including code for training or feature extraction), please contact:
- Yun Xu: xuyun20042001@aliyun.com
- Kelei He: hkl@nju.edu.cn
- Min Lou: lm99@zju.edu.cn

---

**Acknowledgements:**
This work was supported by the National Natural Science Foundation of China and other funding bodies. See the full manuscript for details.

**Disclaimer:**
The datasets analyzed during the current study are available from the corresponding author on reasonable request. The code is provided for academic and non-commercial use only. 