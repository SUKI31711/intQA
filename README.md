# Overview

모두연 슬기로운 챗봇의 코드를 공유하기위한 리포지토리입니다.
기존의 SQuAD 솔루션들에 Reinforcement Learning을 적용하여 Mnemonic Reader에서의 RL 접근법을 일반화 해보는 실험을 하기로 했습니다.

기존의 SQuAD 솔루션 내용은 [obryanlouis/qa](https://github.com/obryanlouis/qa)를 참조했습니다.

# 슬기로운 챗봇 환경 세팅

## pip based setup

### Environment setup

```bash
git clone https://github.com/DeepLearningCollege/qa.git intQA
cd intQA
virtualenv -p python3 envIntQA
source envIntQA/bin/activate
```

### Install spaCy

```bash
pip install -U spacy
python -m spacy download en
```

### Install pytorch

```
https://pytorch.org 에서 맞는 환경을 선택한 후 나오는 커맨드를 사용한다.
```

### Install cove

설치

```bash
mkdir ext
cd ext
git clone https://github.com/salesforce/cove.git
cd cove
pip install -r requirements.txt
python setup.py develop
```

설치 확인

```bash
# python test/example.py #왠지 안됨...
```

### Install Tensorflow

아래 커맨드가 실패할 경우 [텐서플로우1.4 인스톨 가이드 참고](https://www.tensorflow.org/versions/r1.4/install/)

설치

```bash
pip3 install --upgrade tensorflow-gpu 
```

설치 확인

```python
# 파이썬 쉘에서 GPU이름이 나오는지 확인
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

### Install obryanlouis/qa
```
pip install boto3
python3 setup.py
```
