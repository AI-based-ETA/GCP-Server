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
        return 20  # default speed in km/h for cityhall nodes


@app.route('/find_path', methods=['POST'])
def find_path():
    data = request.json
    start_point_name = data['start_point']
    end_point_name = data['end_point']
    start_time_str = data['start_time']
    current_date_str = '2024-04-20'  # 예제 데이터와 일치하도록 설정
    start_time = datetime.strptime(current_date_str + ' ' + start_time_str, '%Y-%m-%d %H:%M')
    start_time = round_time_to_nearest_5_minutes(start_time)
    print("formatted_time:", start_time)

    # 노드 번호 찾기
    try:
        start_node = cityhall_nodes[start_point_name]
        end_node = cityhall_nodes[end_point_name]
    except KeyError:
        return jsonify({'error': 'Invalid start or end point name'}), 400

    # 속도 데이터셋 요청 및 로드
    vms_timetable_df = request_speed_data(start_time, False)
    if vms_timetable_df is None:
        return jsonify({'error': 'Failed to request speed data'}), 500

    # 경로 탐색
    path_result = find_shortest_path(start_node, end_node, start_time, vms_timetable_df)
    if path_result['error']:
        return jsonify({'error': path_result['error']}), 500

    response = {
        'path': path_result['path_names'],
        'total_hours': path_result['total_hours'],
        'total_minutes': path_result['total_minutes'],
        'total_seconds': path_result['total_seconds'],
        'eta': path_result['eta_str']
    }
    print("response", response)
    return jsonify(response)


def find_shortest_path(start_node, end_node, start_time, vms_timetable_df):
    pq = [(0, start_node, start_time)]  # (누적 비용, 현재 노드, 현재 시간)
    visited = set()
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    total_time_spent = {start_node: 0}
    last_request_time = start_time
    debug_output = []

    while pq:
        current_cost, current_node, time_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end_node:
            break

        # 마지막 데이터 요청 후 1시간이 지났는지 확인
        if time_node - last_request_time >= timedelta(hours=1):
            vms_timetable_df = request_speed_data(time_node)
            last_request_time = time_node

        for next_node, distance in edges.get(current_node, []):
            if next_node in visited:
                continue

            # 현재 속도 데이터를 사용하여 통행 시간 계산
            speed = get_speed_data_for_time(time_node, vms_timetable_df, current_node)
            distance_km = distance / 1000  # 미터에서 킬로미터로 변환
            time_a = distance_km / speed  # 시간 (시간 단위)
            time_elapsed = time_a * 3600  # 시간 (초 단위)
            next_time = time_node + timedelta(seconds=time_elapsed)
            next_time = round_time_to_nearest_5_minutes(next_time)
            new_cost = current_cost + time_elapsed

            if next_node in cost_so_far and new_cost < cost_so_far[next_node]:
                # 이전 경로에서 소요된 시간을 빼고 총 시간 업데이트
                total_time_spent[next_node] -= cost_so_far[next_node]
                cost_so_far[next_node] = new_cost
                total_time_spent[next_node] += time_elapsed
            elif next_node not in cost_so_far:
                cost_so_far[next_node] = new_cost
                total_time_spent[next_node] = time_elapsed

            if next_node not in came_from or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost
                heapq.heappush(pq, (priority, next_node, next_time))
                came_from[next_node] = current_node

            debug_output.append(
                f"현재 노드: {current_node}, 다음 노드: {next_node}, 거리: {distance_km} km, 속도: {speed} km/h, 소요 시간: {time_elapsed} 초, 총 비용: {new_cost}")

    path = []
    current = end_node
    while current:
        path.append(current)
        if current not in came_from:
            break
        current = came_from[current]
    path.reverse()

    if end_node in cost_so_far:
        total_seconds = sum(total_time_spent.values())
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
    else:
        return {'error': '목적지 도달 못했을 경우.', 'debug_output': debug_output}


@app.route('/get-node-info', methods=['POST'])
def get_node_info():
    data = request.json
    node_names = data.get('nodeNames')

    if not node_names:
        return jsonify({'error': 'No node names provided'}), 400

    csv_file_path = os.path.join(os.path.dirname(__file__), 'nodelatlng.csv')
    df = pd.read_csv(csv_file_path)
    node_info = []

    for name in node_names:
        node_row = df[df['Name'] == name]
        if not node_row.empty:
            info = {
                'name': name,
                'lat': float(node_row.iloc[0]['Latitude']),
                'lng': float(node_row.iloc[0]['Longitude'])
            }
            node_info.append(info)
    return jsonify(node_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
