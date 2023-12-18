Make sure all recent versions of dependencies are installed before continuing with yolov5.
ONLY train one of the models at a time.

It is RECOMMENDED that you train and use the custom model with yolov5s.
DO NOT run both training lines at the same time.

Dataset: ***https://www.kaggle.com/datasets/lahbibfedi/fashion-dataset-with-annotation***
- Includes 13 classes with 10k Training images/labels and 2k validation images/labels

******************************************************************************************************************************************
!python yolov5/train.py --img-size 640 --batch-size 16 --epochs 50 --data dataset/dataset.yaml --cfg yolov5/models/yolov5s.yaml --device 0
#Train 50 epochs with a batch size of 16 using the yolov5s model
******************************************************************************************************************************************
