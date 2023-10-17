# AA-trans: Core attention aggregating transformer with informationentropy selector for fine-grained visual classification
>The task of fine-grained visual classification (FGVC) is to distinguish targets from subordinate classifica-tions. Since fine-grained images have the inherent characteristic of large inter-class variances and >smallintra-class variances, it is considered an extremely difficult task. To resolve this problem, we redesigned an attention aggregating transformer (AA-Trans) to better capture minor differences among images by >improving the ViT structure.Extensive experiments showthat our proposed model structure can achieve a new state-of-the-art performance on several mainstreamdatasets.
## Environment Requirements
```
cxxfilt>=0.2.0
tqdm>=4.28.1
numpy>=1.15.3
PyYAML>=5.1
pytest>=3.5.1
packaging>=14.0
flake8>=3.7.9
Sphinx>=3.0.3
```
## DataSets
### Caltech-UCSD Birds-200-2011 (CUB-200-2011)
Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.<br>
DownLoad Link: https://www.vision.caltech.edu/datasets/cub_200_2011/
>1. Number of categories: 200
>2. Number of images: 11,788
>3. Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box
### Stanford Dogs Dataset
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. <br>
Download Link: http://vision.stanford.edu/aditya86/ImageNetDogs/
>1. Number of categories: 120
>2. Number of images: 20,580
>3. Annotations: Class labels, Bounding boxesx
