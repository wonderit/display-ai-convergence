## 디스플레이 AI 융합 인재 양성 교육
이미지 분석 딥러닝 및 광학소자 분석 - 이미지기반 광학 소자 특성 에측 실습 <br>
생성모델 및 광학소자 역설계 - 광학소자 역설계 실습

## 📚 프로젝트 설명

1. 수치기반 광학소자 특성 예측 <br>
2. 이미지기반 광학소자 특성 예측 <br>
3. 광학소자 역설계 실습 <br>

## 📝 사용 언어 및 툴 

- Language : Python
- IDE Tool : Jupyter Notebook, Colab

## Colab

### 1. ML
- colab link : https://colab.research.google.com/github/wonderit/display-ai-convergence/blob/main/1_ml_maxwellfdfd_colab.ipynb


### 2. CNN
- colab link : https://colab.research.google.com/github/wonderit/display-ai-convergence/blob/main/2_cnn_maxwellfdfd_colab.ipynb

### 3. GAN
- colab link : https://colab.research.google.com/github/wonderit/display-ai-convergence/blob/main/3_wgan_maxwellfdfd_colab.ipynb



## Jupyter Notebook 

### 1. 소스 받기
```
git clone https://github.com/wonderit/display-ai-convergence
```

### 2. 데이터 다운로드 및 압축 해제
```
./download.sh
```

### 3. Conda 환경 설정
```
conda create -n DAIC python=3.8
conda activate DAIC
```

### 3. pip dependencies
```
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tqdm 
pip install imageio
# for cpu 
pip3 install torch torchvision torchaudio
# for gpu
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


#### Reference
- Kim, W., Seok, J. Simulation acceleration for transmittance of electromagnetic waves in 2D slit arrays using deep learning. Sci Rep 10, 10535 (2020). https://doi.org/10.1038/s41598-020-67545-x
- Kim, Wonsuk, et al. "Inverse design of nanophotonic devices using generative adversarial networks." Engineering Applications of Artificial Intelligence 115 (2022): 105259. https://doi.org/10.1016/j.engappai.2022.105259
- https://www.nature.com/articles/s41598-020-67545-x
- https://github.com/wonderit/maxwellfdfd-ai
- https://github.com/wonderit/maxwellfdfd-controlgan
- https://jovian.ai/aakashns/05-cifar10-cnn
- https://jovian.ai/aakashns/06b-anime-dcgan 
- https://jovian.ai/aakashns/06-mnist-gan
- https://www.kaggle.com/kmldas/gan-in-pytorch-deep-fake-anime-faces/notebook
