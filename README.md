## 목차
- [GCP Server 아키텍처 설명](#gcp-server-아키텍처-설명)
- [GCP Server 내부 동작](#gcp-server-내부-동작)

# GCP Server 아키텍처 설명

## GCP Server 동작 순서

1. 사용자의 요청을 GCP(클라우드)의 Flask로 받고, Pretrained AI Model로 구간별 미래 속도 데이터를 추론합니다.
2. 동적 환경에서 최단 경로를 찾을 수 있도록 구현한 A* 알고리즘과 구간별 미래 속도 데이터를 활용하여 ``최단 시간 경로``와 ``최단 ETA``를 구할 수 있습니다.
3. ``최단 시간 경로``와 ``최단 ETA``를 json 형태로 사용자에게 응답을 보냅니다.


## Graph WaveNet for Deep Spatial Temporal Graph Modeling

다음의 인공지능 모델은 추론 요청 전 1시간 동안의 구간별 속도 데이터를 통해 미래의 1시간 속도 데이터를 구할 수 있습니다.

해당 인공지능 모델을 ``AI-Based-ETA`` 캡스톤 프로젝트에 사용한 이유는 다음과 같습니다.

1. 32-layer의 ``TCN 모듈``, ``GCN 모듈`` 그리고 ``Residual connection``을 사용하여 빠른 학습과 빠른 추론이 가능합니다.
2. 시간적 의존관계와 공간적 의존관계를 포착하기 때문에 다른 인공지능 모델에 비해 높은 성능을 갖고 있습니다.
3. 인기있는 모델이기 때문에 참고할 수 있는 자료가 많아서 트러블슈팅에 유리합니다.

---

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).

<p align="center">
  <img width="350" height="400" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/da88d77f-5ea9-48db-8f45-fb3041fc4f4e">
</p>



## A* 알고리즘

구간별 속도는 시간에 따라 동적으로 변화합니다. 따라서 동적 환경에서 최단 시간 경로와 ETA를 구하기 위해 A* 알고리즘을 직접 구현하였습니다.

A* 알고리즘의 핵심 로직의 다음 2가지 입니다.

첫 번째는 시간에 따라 변화는 속도를 ``current_time``을 기준으로 ``vms_timetable_df``에서 찾고, 구간별 거리에 나눠서 경과 시간(동적 가중치)를 구합니다.

~~~
# 통행 시간 계산
speed = get_speed_data_for_time(current_time, vms_timetable_df, current_node)
distance_km = distance / 1000  # 단위 맞추기
time_a = distance_km / speed  # 시간 = 거리 / 속도(속력)
time_elapsed = time_a * 3600  # 초로 변환

next_time = current_time + timedelta(seconds=time_elapsed)
next_time = round_time_to_nearest_5_minutes(next_time)
~~~

두 번째는 Sliding Window 기법을 활용하여, 미래 속도 데이터가 추가로 필요할 때마다 인공지능으로 구하였습니다.

~~~~
if current_time - last_request_time >= timedelta(hours=1) + timedelta(minutes=5):
    last_request_time += timedelta(hours=1)
    vms_timetable_df = request_speed_data(last_request_time)
~~~~


## Flask

사용자의 요청인 ``출발지``, ``도착지``, ``시각``을 받고, 인공지능과 A* 알고리즘으로 구한 최단 시간 경로와 ETA를 json 형태로 응답을 반환합니다.
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

## (1) 출발지: 서울시청, 도착지: 인천시청

### AI Model Prediction

인공지능 모델이 구간 별로 horizon(5분 단위) 1 ~ 12까지 총 60분 추론하였습니다.

<img width="750" alt="model eta1" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/5c3e8101-f972-413b-88d8-0c1f6188641e">

### 결과

A* 알고리즘을 통해 구한 최단 시간 ETA와 경로는 다음과 같고, ``total_hours: 0, total_minutes: 55, total_seconds: 0``으로 총 55분 거렸습니다.

<img width="750" alt="path1" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/3ab903d4-a696-4c5f-b8cc-b786c5b8c4ee">

### 비교

다음은 동일 시각의 ``네이버 길찾기``를 통해 구한 결과입니다.

<img width="650" alt="naver eta1" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/f3fcdc21-a13e-42b8-b30f-d503ed077375">

---

## (2) 출발지: 서울시청, 도착지: 부산시청

### AI Model Prediction

인공지능 모델은 한번에 구간 별로 horizon(5분 단위) 1 ~ 12까지 총 60분 추론까지 추론하는데, ``서울시청 ~ 부산시청``는 4시간 이상 걸리기 때문에 총 5번 추론하였습니다. 

``1번째 추론``


<img width="750" alt="predict1" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/8d7350dd-6586-40ec-8325-66b0dde8d978">

``5번째 추론``

<img width="750" alt="predict5" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/ef80ba4b-9c62-40e5-879f-bee1c67d186e">

첫 번째 추론과 마지막 추론을 비교하면, 추론을 길게 할 수록 정확도가 낮아지는 것을 알 수 있습니다.

### 결과


A* 알고리즘을 통해 구한 최단 시간 ETA와 경로는 다음과 같고, ``total_hours: 4, total_minutes: 10, total_seconds: 0``으로 총 4시간 10분 걸렸습니다.

<img width="750" alt="path2" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/df472c6d-1e06-4303-9484-83c7ea8098e9">

### 비교

다음은 동일 시각의 ``네이버 길찾기``를 통해 구한 결과입니다.


<img width="650" alt="naver path2" src="https://github.com/AI-based-ETA/GCP-Server/assets/65798779/24bed4ad-193b-4255-a0ff-4cf8c93178bb">
