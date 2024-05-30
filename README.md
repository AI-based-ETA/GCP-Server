# GCP Server 아키텍처 설명

VDS 데이터셋으로 학습한 인공지능이 예측한 미래 속도 데이터셋을 활용하여 최단 시간 ETA와 경로를 구할 수 있습니다.

- 인공지능 모델은 Graph WaveNet for Deep Spatial-Temporal Graph Modeling을 이용하였고, 한국도로교통공사 공공데이터포탈의 VDS 데이터를 활용하여 학습시켰습니다.

- 인공지능 모델이 추론한 미래 속도 데이터셋을 통해, 시간에 따라 동적으로 가중치가 변화는 A* 알고리즘을 직접 구현하였습니다.

- GCP에 배포된 인공지능 모델은 Flask를 통해 react와 통신합니다.

---

## Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).

<p align="center">
  <img width="350" height="400" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/da88d77f-5ea9-48db-8f45-fb3041fc4f4e">
</p>

위의 인공지능을 통해 교통량 예측을 통한 교통량에 따라 미래의 구간 별 속도 데이터셋을 구할 수 있습니다. 

구간 별 속도 데이터셋과 거리 데이터셋을 통해 구간 별 ETA(Estimated Time of Arrival)를 구할 수 있습니다.



## A* 알고리즘

A* 알고리즘의 가중치가 시간에 따라 동적으로 변화도록 구현하였습니다.

~~~
# 통행 시간 계산
speed = get_speed_data_for_time(current_time, vms_timetable_df, current_node)
distance_km = distance / 1000  # 단위 맞추기
time_a = distance_km / speed  # 시간 = 거리 / 속도(속력)
time_elapsed = time_a * 3600  # 초로 변환

next_time = current_time + timedelta(seconds=time_elapsed)
next_time = round_time_to_nearest_5_minutes(next_time)
~~~

인공지능 모델은 60분 미래 데이터를 추론하기 때문에, Sliding Window 기법을 응용하여 필요할 때마다 인공지능을 추론할 수 있도록 구현하였습니다.

~~~~
if current_time - last_request_time >= timedelta(hours=1) + timedelta(minutes=5):
    last_request_time += timedelta(hours=1)
    vms_timetable_df = request_speed_data(last_request_time)
~~~~


## Flask

요청 받은 출발지, 도착지, 현재 시각을 통해 최단 시간 ETA와 경로를 json 형태로 반환합니다.
~~~
@app.route('/find_path', methods=['POST'])
def find_path():
~~~

요청 받은 노드 정보들을 json 형태로 가져 옵니다.
~~~
@app.route('/get-node-info', methods=['POST'])
def get_node_info():
~~~


# GCP Server 내부 동작

## 서울시청 ~ 인천시청

### AI Model Prediction

인공지능 모델이 구간 별로 horizon 1 ~ 12까지 총 60분 추론하였습니다.

<img width="750" alt="Screen Shot 2024-05-30 at 11 59 30 AM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/c70effde-afeb-433e-963c-01be25981d22">

### 결과

A* 알고리즘을 통해 구한 최단 시간 ETA와 경로는 다음과 같고, ``total_hours: 0, total_minutes: 55, total_seconds: 0``으로 총 55분 거렸습니다.

<img width="750" alt="Screen Shot 2024-05-30 at 11 57 40 AM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/8dfb62bd-c016-4e9b-a639-dd5107372f18">

### 비교

다음은 동일 시각의 ``네이버 길찾기``를 통해 구한 결과입니다.

<img width="750" alt="Screen Shot 2024-05-30 at 11 58 13 AM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/d15c2029-0a18-4f9d-894d-aa8568ea49a2">


## 서울시청 ~ 부산시청

### AI Model Prediction

인공지능 모델은 한번에 구간 별로 horizon 1 ~ 12까지 총 60분 추론까지 추론하는데, ``서울시청 ~ 부산시청``는 4시간 이상 걸리기 때문에 총 5번 추론하였습니다. 

``첫번째 추론``


<img width="750" alt="Screen Shot 2024-05-30 at 12 08 40 PM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/d097beb3-8b82-4831-aed1-cf21fd269378">

``5번째 추론``

<img width="750" alt="Screen Shot 2024-05-30 at 12 08 55 PM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/4f314947-0108-478e-a31f-ef531f3591fe">


첫 번째 추론과 마지막 추론을 비교하면, 추론을 길게 할 수록 정확도가 낮아지는 것을 알 수 있습니다.

### 결과


A* 알고리즘을 통해 구한 최단 시간 ETA와 경로는 다음과 같고, ``total_hours: 4, total_minutes: 10, total_seconds: 0``으로 총 4시간 10분 걸렸습니다.

<img width="750" alt="Screen Shot 2024-05-30 at 12 09 07 PM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/b36c9e0a-6f8e-4bfd-80b7-924bb4fff2f4">

### 비교

다음은 동일 시각의 ``네이버 길찾기``를 통해 구한 결과입니다.

<img width="750" alt="Screen Shot 2024-05-30 at 12 09 32 PM" src="https://github.com/daydream-er/pretrained-AI-Model/assets/65798779/9ad1b33e-76e1-49d8-9112-eca8c9587036">
