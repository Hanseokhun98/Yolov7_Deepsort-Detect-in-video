# %%
import subprocess
import csv
import os
import pickle

def create_or_append_csv_file(file_path, columns, data):
    """
    CSV 파일을 생성하거나 기존 파일에 데이터를 추가합니다.
    Parameters:
        file_path (str): CSV 파일의 경로 및 파일 이름
        columns (list): CSV 파일의 칼럼 이름들을 담은 리스트
        data (list of lists): CSV 파일에 추가할 데이터를 담은 2차원 리스트
    """
    if os.path.exists(file_path):
        # 파일이 이미 존재하는 경우, 파일을 열어 데이터를 추가 모드로 쓰기
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
    
    else:
        # 기존 파일이 존재하지 않는 경우, 새로운 파일을 생성하여 데이터 쓰기
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 칼럼 이름들을 CSV 파일에 쓰기
            writer.writerow(columns)
            # 데이터를 CSV 파일에 쓰기
            writer.writerows(data)

    print(f"CSV 파일 생성 또는 데이터 추가 완료: {file_path}")
# %%
# 정상 데이터 테스트
folder_path_normal = './테스트/0'

# 다른 파이썬 파일 실행
for filename in os.listdir(folder_path_normal):

    columns = ['File_name', 'Double_plate_predict', 'Double_plate_real']
    data = []
    
    # 파일 이름 가져오기
    file_name = os.path.basename(filename)
    
    # 'Double_plate_real'에 폴더명 가져오기
    folder_name = int(os.path.basename(folder_path_normal))
    
    # python original_test.py --weights nice.pt --no-trace --view-img --nosave --source test5.avi --device 0
    command = ["python", "original_test.py", "--weights", "nice.pt", "--no-trace", "--view-img", "--nosave", "--source", folder_path_normal+'/'+ filename, "--device", "0"]
    subprocess.run(command)
    
    # double_plating_count 값을 읽어오기
    if os.path.exists('double_plating_count.pkl'):
        with open('double_plating_count.pkl', 'rb') as f:
            double_plating_count = pickle.load(f)
        print("double_plating_count:", double_plating_count)
    else:
        print("double_plating_count 파일이 존재하지 않습니다.")
        
    # data에 추가
    data.append([file_name, double_plating_count, folder_name])  # 두번째 열은 일단 0으로 놓고, 추후에 값 설정 가능

    create_or_append_csv_file('./테스트결과/test_result.csv', columns, data)


# %%
# 불량 데이터 테스트
folder_path_strange = './테스트/1'

# 다른 파이썬 파일 실행
for filename in os.listdir(folder_path_strange):

    columns = ['File_name', 'Double_plate_predict', 'Double_plate_real']
    data = []
    
    # 파일 이름 가져오기
    file_name = os.path.basename(filename)
    
    # 'Double_plate_real'에 폴더명 가져오기
    folder_name = int(os.path.basename(folder_path_strange))
    
    # python original_test.py --weights nice.pt --no-trace --view-img --nosave --source test5.avi --device 0
    command = ["python", "original_test.py", "--weights", "nice.pt", "--no-trace", "--view-img", "--nosave", "--source", folder_path_strange+'/'+ filename, "--device", "0"]
    subprocess.run(command)
    
    # double_plating_count 값을 읽어오기
    if os.path.exists('double_plating_count.pkl'):
        with open('double_plating_count.pkl', 'rb') as f:
            double_plating_count = pickle.load(f)
        print("double_plating_count:", double_plating_count)
    else:
        print("double_plating_count 파일이 존재하지 않습니다.")
        
    # data에 추가
    data.append([file_name, double_plating_count, folder_name])  # 두번째 열은 일단 0으로 놓고, 추후에 값 설정 가능

    create_or_append_csv_file('./테스트결과/test_result.csv', columns, data)