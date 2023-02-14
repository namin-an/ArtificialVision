import zipfile36
import os

input_path = 'C:\\Users\\Na Min An\\Desktop\\Middle_Resolution_177' 
output_path = 'C:\\Users\\Na Min An\\Desktop\\Middle_Resolution_177_unzipped'
os.mkdir(output_path)

for folder in os.listdir(input_path):
    print(folder)
    os.mkdir(output_path +'\\'+folder)
    if folder.endswith('.zip'):
     x = os.path.join(input_path, folder)
     one_folder = zipfile36.ZipFile(x)
     one_folder.extractall(output_path +'\\'+folder)