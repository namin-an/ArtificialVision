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


## Code structure
```
├── scripts
│   ├── cnn_svc_4_elec.sh
│   ├── cnn_svc_4_opt.sh
│   ├── cnn_svc_16_elec.sh
│   ├── cnn_svc_16_opt.sh
├── **process.py** (Usage #2)
├── **train.py** (Usage #3) 
├── **test.py** (Usage #4) 
├── loadData.py  
├── mypackages
│   ├── pytorchtools.py
├── checkpoints (freezed parameters for shallow ML model) 
├── Visualization (Optional usage #5)
│   ├── ColormapsPIXGS.ipynb (Figs. 1a, S1a, 3a, and S3a)
│   ├── Parallel.ipynb (Figs. 1e and S1e) 
│   ├── Prediction.ipynb (Figs 1e, S1e, 3a-c, and S3a-c)  
│   ├── Wordclouds.ipynb (Figs 1c, S1c, 5d, and S5d)
```
<br />


## Dataset
We used K-face data, which can be downloaded from https://aihub.or.kr.
<br />


## Adaptations
> Grad-CAM [*/pytorch_grad_cam*](https://github.com/jacobgil/pytorch-grad-cam)   
> Early-stopping [*/mypackages/pytorchtools.py*](https://github.com/Bjarten/early-stopping-pytorch)
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

3. Process high-resolution original to low-resolution phosphene images.
```
python process.py
```

4. Train various machine learning (ML) models using high-resolution images.
```
python train.py
```

5. Test the performances of ML models on low-resolution images.
```
python test.py
```

6. (OPTIONAL) Reproduce several figures from the manuscript.
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