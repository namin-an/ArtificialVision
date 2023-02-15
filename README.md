# ArtificialVision


# Assistive Machine Learning Approaches for Simulating Human Psychophysical Test of Low-Resolution Artificial Vision
Na Min An, Hyeonhee Roh, Sein Kim, Jae Hun Kim, and Maesoon Im
<br />


## Main figure
<p align="center" width="100%"><img src="https://github.com/namin-an/ArtificialVision/blob/main/images/Fig1.png"></img></p>   
🌃: https://pixabay.com
🌁: https://www.flaticon.com
<br />


## Abstract video
[![IMAGE ALT TEXT](https://github.com/namin-an/ArtificialVision/blob/main/images/cover.png)](https://www.youtube.com/watch?v=kHdlyUNurds)
<br />


## Datasets
> Machine data: [K-face dataset](https://aihub.or.kr)   
> Human data: [ours](https://github.com/namin-an/ArtificialVision/tree/main/data/Human_Expert/211202)


## Usage
1. Clone this repository.
```
git clone https://github.com/namin-an/ArtificialVision.git   
cd ArtificialVision   
```


2. Setup the conda environment (python 3.9.13).
```
conda create -n artificialvision python=3.9   
conda activate artificialvision   
pip install -r requirements.txt   
```


3. Preprocess the K-face datasets by following the step-by-step process from the jupyter notebooks below:

    - Step 1. Unzip all the original files.
    ```
    unzipAIHubData.py
    ```

    - Step 2. SGBt 4,972 images that are recognizable and crop them into the dimension of 128 x 128.
    ```
    selCropPhotos.ipynb  
    ```

    - Step 3. Make customized masks for K-Face datasets using the U-Net pretrained on [CelebA-HQ dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and remove noisy backgrounds.
    ```
    genMaskremBack.ipynb
    ```

    - Step 4. Make low-resolution phosphene images (contrast enhancement + grayscaling + pixelation + phosphenizing).
    ```
    downsamp.ipynb
    ```
          

4. Build and test machine learning (ML) models.   

    - Step 1. Train ML models using high-resolution images.   
    - Step 2. Evaluate their performances on low-resolution images (DEFAULT: opt/CNN_SVC/16).   
    ```
    scripts/
    ```


5. (OPTIONAL) Reproduce several figures from the manuscript.  

  ```
  Visualization/
  ```
<br />


## Code structure
```
├── Preprocessing (usage #3)
│   ├── unzipAIHubData.py
│   ├── selCropPhotos.ipynb (Fig. S7b-c)
│   ├── genMaskremBack.ipynb (Fig. S7d and the first three steps in Fig. S7e)
│   ├── downsamp.ipynb (the last three steps in Fig. S7e, S7f, and S7g)
│
├── main.py (usage #4)
├── loadData.py  
├── expModels.py 
├── mypackages
│   ├── pytorchtools.py
├── scripts (training options)
│   ├── cnn_svc_4_GB.sh
│   ├── cnn_svc_4_NGB.sh
│   ├── cnn_svc_16_GB.sh
│   ├── cnn_svc_16_NGB.sh
│
├── Visualization (optional usage #5)
│   ├── ColormapsPIXGS.ipynb (Figs. 1a, S1a, 3a, S3a, Ext. Data Fig. 1a, and 1b)
│   ├── Parallel.ipynb (Figs. 1e and S1e) 
│   ├── Prediction.ipynb (Figs 1e, S1e, 3a-c, and S3a-c)  
│   ├── Wordclouds.ipynb (Figs 1c, S1c, 5d, and S5d)
│
├── checkpoints
│   ├── saved-unet_model-02-0.09.hdf5 (usage #3: U-Net - can be given if requested)
│   ├── Checkpoint_3.h5 (usage #5: Shallow ML model for non-Gaussian-blurred version)
│   ├── Checkpoint_4.h5 (usage #5: Shallow ML model for Gaussian-blurred version)
```
<br />


## Adaptations
> Grad-CAM [*/pytorch_grad_cam*](https://github.com/jacobgil/pytorch-grad-cam) (Figs. 5a and S5a)  
> Early-stopping [*/mypackages/pytorchtools.py*](https://github.com/Bjarten/early-stopping-pytorch) (for training)   
<br />


## Citation
If you want to report the results of our method or implement the framework, please cite as follows:   
```
@INPROCEEDINGS{?,
  author    = {An, Na Min and Roh, Hyeonhee and Kim, Sein and Kim, Jae Hun and Im, Maesoon},
  booktitle = {?}, 
  title     = {Assistive Machine Learning Approaches for Simulating Human Psychophysical Test of Low-Resolution Artificial Vision},
  year      = {2023},
  volume    = {?},
  pages     = {?},
  doi       = {?}
}
```
<br />


## Contact
If you have any questions regarding the code or paper, feel free to reach us at:
```
namin0202@gmail.com (Na Min An)
```