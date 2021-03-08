트랜스포머 인코더를 세 개 붙이고 해당 출력 벡터를 feed foward 네트워크를 통해 classify하는 모델입니다.
정확도는 기존 BiLSTM과 거의 유사합니다.
dataset.csv과 testset.csv는 Aihub의 데이터 (https://aihub.or.kr/node/274) 를 사용하였습니다.
내용은 다음과 같습니다.

Sentence|Emotion
제가 한가지 고민이 생겼습니다.|공포
짝남이 자기 왕이래요......|공포
...
