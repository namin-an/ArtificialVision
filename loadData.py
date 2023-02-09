import os
import time
import random
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 


def loadData(com_list_n, which_data_mode, train_path, test_path, ft_path, is_finetune, test_type, accessory_list, light_list, expression_list, camera_list, seed, r, model_type2):

    if which_data_mode == 'train':
        path = train_path
    elif which_data_mode == 'test':
        path = test_path

    file_path_list, label_list = list(), list()
    start_time = time.time()

    if is_finetune == 'ft' and which_data_mode == 'train':
        tot_num_trains = len(os.listdir(os.path.join(path, str(com_list_n[0]))))
        zeros_and_ones = [0]*500 + [1]*500 + [2]*(tot_num_trains-1000) 
        random.seed(random)
        rand_indexes = random.shuffle(zeros_and_ones)
        
        filePaths = []
        for i in range(1000): 
            ran_val = zeros_and_ones[i]
            if ran_val == 0: # high-resolution
                for face in com_list_n:
                    one_folder = os.path.join(path, str(face))
                    file_name = os.listdir(one_folder)[i] # same index i for all files of faces
                    
                    sl_split = file_name.split('L')
                    file_name2 = sl_split[-1]
                    e_split = file_name2.split('E')
                    file_name3 = e_split[-1]
                    c_split = file_name3.split('C')
                    file_name4 = c_split[-1]
                    dot_split = file_name4.split('.')
                    
                    if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                        file_path_list.append(os.path.join(one_folder, file_name))
                        label_list.append(str(face))

            elif ran_val == 1: # args.finetune
                for face in com_list_n:
                    one_folder = os.path.join(ft_path, str(face))
                    file_name = os.listdir(one_folder)[i]
                    
                    sl_split = file_name.split('L')
                    file_name2 = sl_split[-1]
                    e_split = file_name2.split('E')
                    file_name3 = e_split[-1]
                    c_split = file_name3.split('C')
                    file_name4 = c_split[-1]
                    dot_split = file_name4.split('.')
                    
                    if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                        file_path_list.append(os.path.join(one_folder, file_name))
                        label_list.append(str(face))
                
    else:
        folderPaths = [os.path.join(path, str(folder_name)) for folder_name in com_list_n]
        for folderPath in folderPaths: # for all participants
            
            folder_name = os.path.basename(folderPath) # real person id (com_list_n[i], i=0,1,...,r)
            filePaths = [os.path.join(folderPath, file_name) for file_name in os.listdir(folderPath)]

            if which_data_mode == 'train':
                random.seed(seed)
                train_int_list = random.sample(range(len(filePaths)), 1000) # ~ 1K images from over 4,000 images (same for all ppl (folders))

            for i, filePath in enumerate(filePaths): # for all files per participant
                file_name = os.path.basename(filePath)
            
                sl_split = file_name.split('L')
                file_name2 = sl_split[-1]
                e_split = file_name2.split('E')
                file_name3 = e_split[-1]
                c_split = file_name3.split('C')
                file_name4 = c_split[-1]
                dot_split = file_name4.split('.')

                if which_data_mode == 'train': 
                    if i in train_int_list: # take only 1,000 images from 4,872 images
                        if (str(sl_split[0][-1]) not in accessory_list) or (str(e_split[0]) not in light_list) or (str(c_split[0][-1]) not in expression_list) or (str(dot_split[0]) not in camera_list):                       
                            file_path_list.append(filePath)
                            label_list.append(folder_name)

                elif which_data_mode == 'test':
                    if (str(sl_split[0][-1]) in accessory_list) and (str(e_split[0]) in light_list) and (str(c_split[0][-1]) in expression_list) and (str(dot_split[0]) in camera_list):
                        file_path_list.append(filePath)
                        label_list.append(folder_name)

    if which_data_mode == 'train': 
        assert len(file_path_list) <= 1000*r 
    elif which_data_mode == 'test':
        if test_type == 'normal':
            assert len(file_path_list) == 9*r 


    print("Files found in {:.2f} sec.".format((time.time()-start_time))) 

  
    if model_type2 == 'CNN_ResNet' or model_type2 == 'CNN_AlexNet': 
        X = np.array([np.array(Image.open(filePath).convert('RGB')) for filePath in file_path_list])
    else:
        X = np.array([np.array(Image.open(filePath).convert('L')) for filePath in file_path_list])
    y = np.array(label_list)

    print("Data shape: ", X.shape, y.shape)

    X, y = X.astype(np.uint8), y.astype(np.float64) 
    print("Data type: ", type(X[0][0][0]))

    old_uniq_labels = list(set(y))
    key_list = list(range(r))
    dictionary = dict(zip(old_uniq_labels, key_list))
    for i, face_num in enumerate(y):
        y[i] = dictionary[face_num]
    new_uniq_labels = list(set(y))
    assert len(old_uniq_labels) == r
    assert len(new_uniq_labels) == r
    print(f'Old unique labels: {old_uniq_labels}')
    print(f'New unique labels: {new_uniq_labels}')
    y = np.array(y)

    return X, y, file_path_list, old_uniq_labels, new_uniq_labels


def visualizeData(Xtrain, ytrain, Xtest, ytest):
    random_idx = random.randint(0, len(Xtrain))
    plt.imshow(Xtrain[random_idx], cmap='gray')
    plt.title(f"Training example #{random_idx} (Class: {ytrain[random_idx]})")
    #plt.axis('off')
    plt.show()

    random_idx = random.randint(0, len(Xtest))
    plt.imshow(Xtest[random_idx], cmap='gray')
    plt.title(f"Testing example #{random_idx} (Class: {ytest[random_idx]})")
    #plt.axis('off')
    plt.show()