## Introduction
The module has designed to recognize left champion in croped images like below:
![image1](/readme_images/Ahri_278220660753197_round6_Ahri_06-02-2021.mp4_10_2.jpg)

![image2](/readme_images/Ashe_231705051716794_round3_Ashe_06-07-2021.mp4_26_2.jpg)

Accurancy achieved 94% on the test set

## Pipeline
The pipline has three phase with ideal that using image processing function to detect champion face area, using L2 distance to compare and predict through features extracted by Resnet50 (weight Imagenet).
![pipeline](/readme_images/pipeline.png)
## Installation
You can install requirements through Virtual venv or Conda env.
```bash
pip install -r requirements.txt
```
## Usage
Using file predict_test.py for testing the test images fold
```bash
python predict_test.py --test_fold "path-to-test-folder" --batch_size n
```
