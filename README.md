# Deep_CV_segmentation
Official code for "Unsupervised Deep Learning Meets Chan-Vese Model". https://arxiv.org/pdf/2204.06951.pdf

## Fg/Bg segmentation
Download the Weimann dataset from https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/dl.html.

Use saliency detection methods such as https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Saliency/Saliency.html and https://sites.google.com/site/salientobjectdetection/need-to-knows to initialize our model. 

## Dataset based segmentation
The dataset preparation is the same with the ReDO method. See https://github.com/mickaelChen/ReDO.

### Validate pretrained model
We provide the trained Deep_CV model in */dataset_segmentation/model_weights, run:
```
cd ./dataset_segmentation/
python eval.py
```

### Training model
Run:
```
cd ./dataset_segmentation/
python run.py
```
