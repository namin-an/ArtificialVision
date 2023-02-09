# ArtificialVision

# Assistive Machine Learning Approaches for Simulating Human Psychophysical Test of Low-Resolution Artificial Vision
Na Min An, Hyeonhee Roh, Sein Kim, Jae Hun Kim, and Maesoon Im

## Main figure
<p align="center" width="100%"><img src="https://github.com/namin-an/ArtificialVision/blob/main/images/Fig1.png"></img></p>   
🌃: https://pixabay.com
🌁: https://www.flaticon.com
<br />

## Abstract video
[![IMAGE ALT TEXT](https://github.com/namin-an/ArtificialVision/blob/main/images/cover.png)](https://www.youtube.com/watch?v=kHdlyUNurds)
<br />

## Dataset
We used K-face dataset, which can be downloaded from https://aihub.or.kr.
<br />

## Code structure
```
├── loadData.py  
├── trainANNs.py  
├── fm.py  
├── test.py  
├── checkpoints (Freezed parameters for shallow ML model) 
├── Visualization
│   ├── ColormapsPIXGS.ipynb (Figs. 1a, S1a, 3a, and S3a)
│   ├── Parallel.ipynb (Figs. 1e and S1e) 
│   ├── Prediction.ipynb (Figs 1e, S1e, 3a-c, and S3a-c)  
│   ├── Wordclouds.ipynb (Figs 1c, S1c, 5d, and S5d)
```

### Adaptations
> Grad-CAM [*/pytorch_grad_cam*](https://github.com/jacobgil/pytorch-grad-cam)   
> Early-stopping [*/mypackages/pytorchtools.py*](https://github.com/Bjarten/early-stopping-pytorch)
<br />


### Citation
-----
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