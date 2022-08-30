# GRU 기반 행동 분석 모델을 이용한 어린이 이상 행동 검출 시스템

## Quick Start Examples
<details>
<summary>Preprocessing</summary>
<div markdown="1">

```
$ python preprocess.py -v ./원본/싸움/*/*/*.mp4 -l ./싸움/*/*/*.xml -s new_datasets/train/2 ../final_datasets/train/2 -f features_train_normal
```
</div>
</details>
<details>
<summary>Train</summary>
<div markdown="1">

```
$ python train.py -d ../datasets
```
</div>
</details>
<details>
<summary>Inference</summary>
<div markdown="1">

```
$ python inference.py 
```
</div>
</details>

