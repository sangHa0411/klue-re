# klue-re

1. 목표
    * KLUE Relation Extraction Task에서의 Benchmark 제출을 위한 학습 및 추론 baseline 코드 개발 
    * Relation Extraction Task 관련 모델 성능 고도화
    
2. 방향 
    * transformers 및 datasets 라이브러리 사용을 최대한 자제하고 torch 기반으로 작성

3. 구조
    ```
    ├── dataset
    │   ├── collator.py
    │   └── dataset.py
    ├── models
    │   ├── metrics.py
    │   ├── model.py
    │   └── scheduler.py
    ├── requirements.txt
    ├── train.py
    └── utils
        ├── encoder.py
        └── loader.py
    ``` 

    - `dataset`

      - `collator.py` : input_ids, attention_mask를 batch안에서 가장 큰 길이로 padding을 하며 torch.tensor 구조로 바꾸기 위한 파일
      - `dataset.py` : 주어진 데이터를 torch.utils.data.Dataset 형태로 만들기 위한 파일

    - `models`

      - `model.py` : roberta 기반의 모델을 고도화 하기 위한 파일
      - `scheduler.py` : 학습 과정 주에서 learning rate를 스케쥴링 하기 위한 파일
      - `metrics.py` : dev 데이터 대상으로 evaluation을 진행할 때 f1, accuracy, auprc 점수를 구하기 위한 파일

    - `utils`

      - `encoder.py` : 문장으로 된 데이터를 tokenizer를 통해서 인코딩하기 위한 파일
      - `loader.py` : json 파일로 된 원시 데이터를 불러와서 필요한 데이터를 추출하고 전처리하기 위한 파일
      
    - `train.py`

      - 기존 PLM (klue/roberta-base & klue/roberta-large)를 기반으로 relation extraction task에 맞게 학습하고 모델을 저장하기 위한 파일
      
      
4. 결과
    1. klue/roberta-base
        * Dev Data 기준
            * F1 : 76.001
            * Auprc : 57.66
            * Accuracy : 0.76
        * Test Data 기준
            * 아직 제출하지 못함
      
 5. 진행중
     1. Benchmark에 제출하는 방법 조사 중
     2. 모델 성능이 높아지기 위한 모델 고도화 진행 중
     3. Auprc의 점수가 낮은데 이에 대한 원인과 해결 방안을 조사 중
     
