#%%
import os
import glob

def update_text_files(folder_path):
    # 폴더 내의 모든 .txt 파일을 찾습니다.
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # 각 파일에 대해 작업을 수행합니다.
    for file_path in file_paths:
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)  # 파일 포인터를 파일의 맨 앞으로 이동시킵니다.
            
            # 각 줄에 대해 작업을 수행합니다.
            for line in lines:
                words = line.split()
                
                if len(words) > 0:
                    # 첫 번째 단어를 확인하여 작업을 수행합니다.
                    first_word = words[0]
                    if first_word == '2':
                        words[0] = '0'
                    # elif first_word == '0':
                    #     words[0] = '1'
                
                # 변경된 줄을 파일에 쓰고 개행 문자를 추가합니다.
                file.write(' '.join(words) + '\n')
            
            file.truncate()  # 파일의 나머지 내용을 삭제합니다.

# 폴더 경로를 지정하고 함수를 호출합니다.
folder_path = 'c:/Users/202207/Documents/10-2'  # 실제 폴더 경로로 변경해야 합니다.
update_text_files(folder_path)
# %%
