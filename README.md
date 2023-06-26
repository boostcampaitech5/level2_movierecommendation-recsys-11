# Movie Recommendation
> Movielens 데이터를 활용하여 사용자의 영화 시청 이력 데이터를 분석하고, 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측  
> (2023-05-30 ~ 2023-06-22)

<br>
<div align="center">
<img src="https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white" alt="Python badge">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=PyTorch&logoColor=white" alt="PyTorch badge">
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas badge">
  <img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="numpy badge">
   <img src="https://img.shields.io/badge/scikit learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikitlearn badge">
  <img src="https://img.shields.io/badge/wandb-FFBE00?logo=weightsandbiases&logoColor=white" alt="weightsandbiases badge">
 <img src="https://img.shields.io/badge/-Sweep-orange" alt="scikitlearn badge">
</div>

 
## Members

**팀 목표**
1. 생산성 높일 수 있는 협업 툴 적극적으로 활용
2. 다양한 AI 모델을 직접 구현 및 데이터 기반의 접근 방법 적용

<br>

<div align="center">
<table>
  <tr>
     <td align="center">
        <a href="https://github.com/gangjoohyeong">
          <img src="https://avatars.githubusercontent.com/u/93419379?v=4" width="100px" alt=""/><br />
          <sub><b>강주형</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/watchstep">
          <img src="https://avatars.githubusercontent.com/u/88659167?v=4" width="100px" alt=""/><br />
          <sub><b>김주의</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/eunjios">
          <img src="https://avatars.githubusercontent.com/u/77034159?v=4" width="100px" alt=""/><br />
          <sub><b>이은지</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/hoyajigi">
          <img src="https://avatars.githubusercontent.com/u/1335881?v=4" width="100px" alt=""/><br />
          <sub><b>조현석</b></sub>
        </a><br/>
    </td>
    <td align="center">
        <a href="https://github.com/MSGitt">
          <img src="https://avatars.githubusercontent.com/u/121923924?v=4" width="100px" alt=""/><br />
          <sub><b>최민수</b></sub><br/>
        </a>
    </td>
  </tr>
</table>

| 공통 | 문제 정의, 계획 수립, EDA, 데이터 전처리, 랩업리포트 작성 |
| :---: | :--- |
| 강주형 |  VAE 계열 모델 PyTorch template 구축 및 튜닝, Ensemble 전략 수립 |
| 김주의 | 모델 구현/실험 (RecVAE, PinSAGE) |
| 이은지 | RecBole dataset 구성 및 template 작성, 모델 구현/실험 (Sequential Models) |
| 조현석 | RecBole template 작성, 모델 구현/실험 (Rule-base Models) |
| 최민수 | RecBole template 구축,  모델 구현/실현 (EASE, DeepFM, NCF) |
</div>

## 모델과 앙상블 성능
<div align="center">

단일 모델 성능

| Model | EASE | ADMMSLIM | MultiVAE | RecVAE | S3Rec | GRU4Rec |
|:---------:|:------:|:----------:|:----------:|:--------:|:-------:|:---------:|
| **Recall@10** | 0.1594 | 0.1546 | 0.1403 | 0.1514 | 0.0898 | 0.051 |

<br><br>

앙상블 성능

![Untitled](https://github.com/boostcampaitech5/level2_movierecommendation-recsys-11/assets/93419379/5e17a671-aedf-4a17-bcc6-57507e86e1fc)

<br><br>

**최종 성능**

||🔒 Private|🔑 Public|
|:---:|:---:|:---:|
|Recall@10|0.1637|0.1648|

구체적인 앙상블 실험은 랩업 리포트 참조


</div>

## 데이터셋 구조

```
data/
├── traing_ratings.csv
├── genres.tsv
├── directors.tsv
├── titles.tsv
├── writers.tsv
└── years.tsv
```
