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
> Human data: [*/data/Human_Expert/211202/*]


## Code structure
```
├── main.py (Usage #3)
├── loadData.py  
├── trainANNs.py 
├── testPhosphenes.py  
├── mypackages
│   ├── pytorchtools.py
├── scripts (training options)
│   ├── cnn_svc_4_elec.sh
│   ├── cnn_svc_4_opt.sh
│   ├── cnn_svc_16_elec.sh
│   ├── cnn_svc_16_opt.sh
│
├── Visualization (Optional usage #4)
│   ├── ColormapsPIXGS.ipynb (Figs. 1a, S1a, 3a, S3a, Ext. Data Fig. 1a, and 1b)
│   ├── Parallel.ipynb (Figs. 1e and S1e) 
│   ├── Prediction.ipynb (Figs 1e, S1e, 3a-c, and S3a-c)  
│   ├── Wordclouds.ipynb (Figs 1c, S1c, 5d, and S5d)
├── checkpoints (freezed parameters for shallow ML model) 
```
<br />


## Adaptations
> Grad-CAM [*/pytorch_grad_cam*](https://github.com/jacobgil/pytorch-grad-cam) (Figs. 5a and S5a)  
> Early-stopping [*/mypackages/pytorchtools.py*](https://github.com/Bjarten/early-stopping-pytorch) (for training)   
<br />


## Usage
1. Clone this repository.
```
git clone https://github.com/namin-an/ArtificialVision.git   
cd ArtificialVision   
```

2. Setup the conda environment.
```
conda create -n artificialvision python=3.9   
conda activate artificialvision   
pip install -r requirements.txt   
```

3. Preprocess the downloaded K-face datasets by following the step-by-step process from the jupyter notebook below:
```
selectTrainableImgs.ipynb # Select 4,972 images that are trainable (Fig. S7b).   
processBeforeDownsamp.ipynb # Crop the images into squares, remove noisy backgrounds, and enhance the contrast of the images (Fig. S7c-d and the first three steps in Fig. S7e).   
downsampling.ipynb # Make low-resolution phosphene images (the last three steps in Fig. S7e, S7f, and S7g).
```

3. Build and evaluate various machine learning (ML) models.
```
python main.py -demo_type train # 1. To train various ML models using high-resolution images (DEFAULT: opt/CNN_SVC/16).   
python main.py -demo_type test # 2. To test the performances of ML models on low-resolution images (DEFAULT: opt/CNN_SVC/16).   
```

4. (OPTIONAL) Reproduce several figures from the manuscript.   
```
Visualization/
```
<br />


## Citation
If you want to report the results of our method or implement the framework, please cite as follows:   
```
@INPROCEEDINGS{?,
  author    = {An, Na Min and Roh, Hyeonhee and Kim, Sein and Kim, Jae Hun and Im, Maesoon},
  booktitle={?}, 
  title     = {Assistive Machine Learning Approaches for Simulating Human Psychophysical Test of Low-Resolution Artificial Vision},
  year      = {2023},
  volume    = {?},
  pages     = {?},
  doi       = {?}
}
```
<br />


## Contact
Feel free to ask questions or leave comments through this email:
```
namin0202@gmail.com (Na Min An)
```