# U-Net Architecture 

The project is was inspired from the original U-Net Architecture paper "U-Net: Convolutional Networks for Biomedical 
Image Segmentation"[https://arxiv.org/abs/1505.04597]

- The Architecture of the model implemented can be found in the following Image


![Alt text](images/u-net-architecture.png?raw=true "Optional Title")

## Prepare the dataset

Downloading from Kaggle

```bash
kaggle competitions download -c carvana-image-masking-challenge
```
- Extract the dataset 
- create a New folder "valid"
- select copy and paste the last 48 Images in the train folder to "valid"

## Run Locally

Clone the project

```bash
  git clone darthvader2/U-Net-implementation-from-scratch
```

Go to the project directory

```bash
  cd U-Net-implementation-from-scratch
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Training the model

```bash
  python train.py
```

## Changing the image directory in train.py

![Alt Text]("images/carbon.png)

