import zipfile36
import os

data_path = '[USER_PATH]\\ArtificialVision\\data'
input_path = os.path.join(data_path, 'Middle_Resolution_177') # 177 facial classes from the middle resolution data
output_path = os.path.join(data_path, 'Middle_Resolution_177_unzipped')
os.makedirs(output_path, exist_ok=True)

for folder in os.listdir(input_path):
    print(folder)
    os.mkdir(output_path +'\\'+folder)
    if folder.endswith('.zip'):
     x = os.path.join(input_path, folder)
     one_folder = zipfile36.ZipFile(x)
     one_folder.extractall(output_path +'\\'+folder)