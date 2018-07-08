# Code Walkthroguh
트레이닝 시 코드의 각 부분에서 어떤 기능들을 수행하는지 가이드 하여 코드의 흐름 파악을 용이하게 하고자 함.

### train_local.py 흐름
트레이닝은 [train_local.py](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py) 스크립트를 사용하여 실행.

train_local.py 는 다음 기능들을 수햏한다.
1. [L8](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L8): [flags.py]() 에서 옵션들을 읽어온다.
1. [L9](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L9): 데이터 파일이 필요한 경우 다운로드 한다.
1. [L10](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L10): [Trainer.train()](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L36) 함수를 실행한다.

```python
    # option들 읽어오기
    options = get_options_from_flags()
    
    # 데이터 파일 다운로드 하기
    maybe_download_data_files_from_s3(options)
    
    # `Trainer`의 `train()` 함수를 실행
    Trainer(options).train()
```

### Trainer.train() 흐름
1. [L41](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L41):
cpu를 사용하여 아래의 작업들을 실행함
1. [L45-L61](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L45-L61):
word_embedding, char_embedding, learning_rate, optimizer 에 대한 Tensorflow Varaible 설정
1. [L62-L64](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L62-L64):
모델 초기화
1. [L73-L75](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L73-L75):
그레디언트를 초깃값들을 가져와서, global norm 으로 clipping 후, optimizer를 통해 모델 variable들을 업데이트 하도록 wire한다.
1. [L80-L86](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L80-L86):
로그 파일 정리
1. [L87-L106](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L87-L106):
로그에 summary를 출력하기위한 em, f1, highest_f1, loss, gradients 등의 variable 등을 생성
1. [L121-L131](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L121-L131):
Tensorflow session에서 현재 iteration 횟수, iterations_per_epoch, learning_rate 등을 가져와 출력한다.
1. [L136](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L136):
트레이닝 시작, loop이 한번은 iteration 1번을 실행한다
1. [L139-L144](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L139-L144):
iteration 한번 실행
1. [L150-L175](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L150-L175):
각 iteration관련된 로그 및 요약 (iter #, f1, loss, time per iteration 등) 출력후, iter 종료.
1. [L179-L187](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L179-L187):
한 epoch가 끝나면 training set 과 validation set에 대한 model evaluation 실행
1. [L190-L203](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L190-L203):
evaluation 관련 로그 출력
1. [L204-L222](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L204-L222):
evaluation 결과에서 모델이 bad_checkpoints_tolerance 만큼 향상되지 않았을경우, learning_rate를 더 작게 조정한다.
1. [L224-L232](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L224-L232):
evaluation 결과에서 모델이 향상되었을 경우 모델을 저장하고 향상된 F1을 출력한다.
1. [L233-L237](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L233-L237):
epoch 종료
