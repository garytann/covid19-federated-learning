# COVID-19 Federated Learning Screening Application for LUS

A decentralised deep learning application to detect Covid-19, Pneumonia or Healthy lungs based on Lung Ultrasound Images. This repo contains the code to run the application. This is a research studies conducted during the final year of my bachelors.

## Motivation

The reason for this work is so that medical organisations can collaboratively share a deep learning model without sharing the datasets. This improves data privacy and also allows the model to be continuously trained to be more accurate with contributions of more datasets.

## Contribution

The federated learning application is realised through a client-server architecture built using REST (FastAPI), and a user-interface built using NextJS to allow clients to upload dataset, train, perform federated learning algorithms and lastly perform inferencing to achieve a score.


## Dataset

Lung Ultrasound images are acquired from publicly available resources. 

NOTE: Some of the dataset network might not be publicly available.

- https://github.com/jannisborn/covid19_ultrasound
- https://github.com/nrc-cnrc/COVID-US

## Installation

```
conda create -n covidlus python=3.8 -y
conda activate covidlus
pip install -r requirements.txt
```

## Application

### FastAPI
```
cd fl-module
python app.py
```
### Front End 
Uses material design from ant Design
```
cd front-end/fl-app
npm install antd --save
npm run build
npm start
```
## Tutorial

To understand the backbone architecture of the deep learning network. I first built a Convolutional Neural Network (CNN) that compares between pretrained models from **VGG16**, **ResNet50** and a **custom CNN** architecture. You can explore the algorithms in by only changing the `DATA_DIR` path in `ML_Approach(CNN).ipynb`. 

Alternatively, you can run run both the Jupyter notebook or script, `covidLUS.ipynb` or `covidlus.py`. The notebook and script was both built to run in Google Colab and Mac-M1. 

## Contact

Feel free to contact for collaboration or raise a issue, please reach out: `garytannjy@gmail.com` 
