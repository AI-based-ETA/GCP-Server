from flask import Flask, request, jsonify
import pandas as pd
import heapq
from datetime import datetime, timedelta
import subprocess
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 파일 경로
MODIFY_FROM_TO_COST_PATH = 'modify_from_to_cost(cityhall).csv'

# 데이터 로드
modify_from_to_cost_df = pd.read_csv(MODIFY_FROM_TO_COST_PATH)

# 노드 번호와 이름 매핑 생성
node_to_name = {row['from']: row['ST_ND_NM'] for _, row in modify_from_to_cost_df.iterrows()}
node_to_name.update({row['to']: row['END_ND_NM'] for _, row in modify_from_to_cost_df.iterrows()})

# 간선 정보를 딕셔너리로 변환
edges = {}
for _, row in modify_from_to_cost_df.iterrows():
    edges.setdefault(row['from'], []).append((row['to'], row['cost']))

# 추가된 시청 노드 목록
cityhall_nodes = {
    '서울특별시청': 9990,
    '인천광역시청': 9991,
    '대전광역시청': 9992,
    '광주광역시청': 9993,
    '대구광역시청': 9994,
    '부산광역시청': 9995,
    '울산광역시청': 9996
}


def round_time_to_nearest_5_minutes(dt):
    # 초 이하 값은 0으로 설정
    dt = dt.replace(second=0, microsecond=0)
    # round 5 minutes
    discard = timedelta(minutes=dt.minute % 5)
    dt -= discard
    if discard >= timedelta(minutes=2.5):
        dt += timedelta(minutes=5)
    return dt


def request_speed_data(start_time, sampled=True):
    if sampled:
        sampled_path = "vms_timetable.h5" # use last sampled path
    else:
        sampled_path = "data/vms_valid_data.h5"
    try:
        # 파이썬 스크립트 실행
        print("start_time: ", start_time.strftime('%Y-%m-%d %H:%M'))
        # script_path = 'generating_train_data.py'
        subprocess.run(['python', 'generating_train_data.py', "--traffic_df_filename=" + sampled_path, "--date=" + start_time.strftime('%Y-%m-%d %H:%M')], check=True)
        subprocess.run(["python", "generating_test_data.py", "--traffic_df_filename_x=data/vms_test_data.h5"])
        subprocess.run(["python", "prediction_test.py", "--gcn_bool", "--adjtype", "doubletransition", "--addaptadj", "--num_nodes=249", "--device=cuda:0",
            "--checkpoint=garage/metr_exp1_best_2.03.pth"])

        # 데이터 로드
        new_vms_timetable_path = "vms_timetable.h5"  # 새로 생성된 파일의 경로
        new_vms_timetable_df = pd.read_hdf(new_vms_timetable_path)
        new_vms_timetable_df.index = pd.to_datetime(new_vms_timetable_df.index)
        return new_vms_timetable_df
    except subprocess.CalledProcessError as e:
        print(f"Error requesting speed data: {e}")
        return None


def get_speed_data_for_time(current_time, vms_timetable_df, node):
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M')
    try:
        speed = vms_timetable_df.loc[formatted_time, str(node)]
        return speed
    except KeyError:
        print(formatted_time, str(node), "Key Error")
        return 20  # 도시에서 속도 default값


@app.route('/find_path', methods=['POST'])
def find_path():
    data = request.json
    start_point_name = data['start_point']
    end_point_name = data['end_point']
    start_time_str = data['start_time']
    current_date_str = '2024-04-20'  # 예제 데이터와 일치하도록 설정
    start_time = datetime.strptime(current_date_str + ' ' + start_time_str, '%Y-%m-%d %H:%M')
    start_time = round_time_to_nearest_5_minutes(start_time)

    # 노드 번호 찾기
    try:
        start_node = cityhall_nodes[start_point_name]
        end_node = cityhall_nodes[end_point_name]
    except KeyError:
        return jsonify({'error': '유효하지 않은 출발지 또는 목적지 이름'}), 400

    # 속도 데이터셋 요청 및 로드
    vms_timetable_df = request_speed_data(start_time, False)
    if vms_timetable_df is None:
        return jsonify({'error': '속도 데이터 요청 실패'}), 500

    # 경로 탐색
    # path_result = find_shortest_path(start_node, end_node, start_time, vms_timetable_df)
    path_result = a_ster(start_node, end_node, start_time, vms_timetable_df)
    if path_result['error']:
        return jsonify({'error': path_result['error']}), 500

    response = {
        'path': path_result['path_names'],
        'total_hours': path_result['total_hours'],
        'total_minutes': path_result['total_minutes'],
        'total_seconds': path_result['total_seconds'],
        'eta': path_result['eta_str']
    }
    print(response)
    return jsonify(response)

def a_ster(start_node, end_node, start_time, vms_timetable_df):
    vertex = {int(node): datetime.max for node in vms_timetable_df.columns.tolist()} # vertex[node] = 시작시각 (cost)
    for key, value in cityhall_nodes.items():
        vertex[value] = datetime.max
    parent_node = {node: None for node in vms_timetable_df.columns.tolist()} # parent_node[node] = node의 부모 (최단)
    last_request_time = start_time
  
    vertex[start_node] = start_time
    queue = []
    heapq.heappush(queue, (vertex[start_node], start_node))

    while queue:
        (current_time, current_node) = heapq.heappop(queue) # (1) current_time가 작은 순서대로
        if vertex[current_node] < current_time:
            continue
        
        if current_node == end_node:
            break
        
        # (2) 1시간이 지날 때 마다 인공지능에 request 
        # (3) 이때 timetable_df의 이전 시각에 대한 요구는 (1) 조건에 의해 존재할 수 없다.
        # if current_time - last_request_time >= timedelta(hours=1) + timedelta(minutes=5):
        #     last_request_time += timedelta(hours=1)
        #     vms_timetable_df = request_speed_data(last_request_time)
        if round_time_to_nearest_5_minutes(current_time) - last_request_time >= timedelta(hours=1) + timedelta(minutes=5):
            last_request_time += timedelta(hours=1)
            vms_timetable_df = request_speed_data(last_request_time)
        
        for next_node, distance in edges.get(current_node, []):
            # 통행 시간 계산
            # speed = get_speed_data_for_time(current_time, vms_timetable_df, current_node)
            speed = get_speed_data_for_time(round_time_to_nearest_5_minutes(current_time), vms_timetable_df, current_node)
            distance_km = distance / 1000  # 단위 맞추기
            time_a = distance_km / speed  # 시간 = 거리 / 속도(속력)
            time_elapsed = time_a * 3600  # 초로 변환

            next_time = current_time + timedelta(seconds=time_elapsed)
            # next_time = round_time_to_nearest_5_minutes(next_time)
            if next_time < vertex[next_node]:
                vertex[next_node] = next_time
                # vertex[next_node] = round_time_to_nearest_5_minutes(next_time)
                parent_node[next_node] = current_node # next_node 부모는 current_dest
                heapq.heappush(queue, (vertex[next_node], next_node))  # 다음 인접 거리를 계산 하기 위해 큐에 삽입

    # 최종 시간 계산
    end_time = vertex[end_node]  # new_cost를 총 시간(초)으로 사용
    eta_str = end_time.strftime("%H시 %M분")
    time_difference = end_time - start_time
    total_hours = time_difference.seconds // 3600
    total_minutes = (time_difference.seconds % 3600) // 60
    total_seconds = time_difference.seconds % 60
    
    path = []
    current_node = end_node
    path.append(current_node)
    print("current_node:", current_node)
    while current_node is not start_node:
        current_node = parent_node[current_node]
        path.append(current_node)
        print("current_node:", current_node, " current_time", vertex[current_node])
    path = path[::-1]

    path_names = [node_to_name.get(node, f"Node {node}") for node in path]

    return {
        'path_names': path_names,
        'total_hours': total_hours,
        'total_minutes': total_minutes,
        'total_seconds': total_seconds,
        'eta_str': eta_str,
        'error': None,
        'debug_output': []
    }


def find_shortest_path(start_node, end_node, start_time, vms_timetable_df):
    pq = [(0, start_node, start_time)]  # (누적 비용, 현재 노드, 현재 시간)
    visited = set()
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    path = []
    time_list = []
    total_time_spent = {start_node: 0}
    last_request_time = start_time
    input_time = start_time
    debug_output = []

    while pq:
        current_cost, current_node, current_time = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)
        path.append(current_node)

        if current_node == end_node:
            break

        # 1시간이 지날 때 마다 인공지능에 request
        if current_time - last_request_time >= timedelta(hours=1):
            last_request_time += timedelta(hours=1)
            vms_timetable_df = request_speed_data(last_request_time)

        for next_node, distance in edges.get(current_node, []):
            if next_node in visited:
                continue

            # 통행 시간 계산
            speed = get_speed_data_for_time(current_time, vms_timetable_df, current_node)
            distance_km = distance / 1000  # 단위 맞추기
            time_a = distance_km / speed  # 시간 = 거리 / 속도(속력)
            time_elapsed = time_a * 3600  # 초로 변환

            new_cost = current_cost + time_elapsed
            next_time = input_time + timedelta(seconds=new_cost)
            next_time = round_time_to_nearest_5_minutes(next_time)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                total_time_spent[next_node] = time_elapsed
                heapq.heappush(pq, (new_cost, next_node, next_time))
                came_from[next_node] = current_node

                # current_time 업데이트
                current_time = next_time

                # 디버깅
                debug_output.append(
                    f"현재 노드: {current_node}, 다음 노드: {next_node}, 거리: {distance_km} km, 속도: {speed} km/h, 소요 시간: {time_elapsed} 초, 총 비용: {new_cost}, 현재 시간: {current_time}")

                break  # 경로에 저장됐을 때 반복문 종료

    # 최종 시간 계산
    total_seconds = new_cost  # new_cost를 총 시간(초)으로 사용
    total_hours = int(total_seconds / 3600)
    total_minutes = int((total_seconds % 3600) / 60)
    total_seconds = int(total_seconds % 60)

    eta = start_time + timedelta(hours=total_hours, minutes=total_minutes, seconds=total_seconds)
    eta_str = eta.strftime("%H시 %M분")

    path_names = [node_to_name.get(node, f"Node {node}") for node in path]

    return {
        'path_names': path_names,
        'total_hours': total_hours,
        'total_minutes': total_minutes,
        'total_seconds': total_seconds,
        'eta_str': eta_str,
        'error': None,
        'debug_output': debug_output
    }



@app.route('/get-node-info', methods=['POST'])
def get_node_info():
    data = request.json
    node_names = data.get('nodeNames')

    print("where 1")
    if not node_names:
        return jsonify({'error': 'No node names provided'}), 400

    print("where 2")
    csv_file_path = os.path.join(os.path.dirname(__file__), 'nodelatlng.csv')
    df = pd.read_csv(csv_file_path)
    node_info = []

    print("where 3")
    for name in node_names:
        print(name)
        node_row = df[df['Name'] == name]
        if not node_row.empty:
            info = {
                'name': name,
                'lat': float(node_row.iloc[0]['Latitude']),
                'lng': float(node_row.iloc[0]['Longitude'])
            }
            node_info.append(info)
    print("where 4")
    return jsonify(node_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
