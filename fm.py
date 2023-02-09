import os
import random
import itertools
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import torch
from torch.optim import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import models, layers
    

if __name__ == 'main':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str, default='opt') # 'opt' or 'elec'
    parser.add_argument('--human_data', type=str, default='E:\\ANNA_INTERN\\Human_Exp\\211202') # 2, 4, 16
    parser.add_argument('--machine_data', type=str, default='E:\\ANNA_INTERN\\Human_Exp\\211105_QAs_for_Set0_CNN_SVC_4classes_partial.csv') 
    parser.add_argument('--meta_data', type=str, default='E:\\ANNA_INTERN\\210827_ANNA_Removing_uncontaminated_data.csv')
    parser.add_argument('--qb_path', type=str, default='E:\\ANNA_INTERN\\Question Banks AI Hub_final')
    parser.add_argument('--cp_path', type=str, default='Checkpoints_FB/Checkpoint_4.h5')
    parser.add_argument('--r', type=int, default=4) # 2, 4, 16
    args = parser.parse_args()

    """ Human data """
    if args.test_type == 'opt':
        sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] # 18 participants
    elif args.test_type == 'elec': 
        sel_ppl = [499, 500, 502] + list(range(503, 509)) + list(range(602, 607)) + list(range(608, 612)) # 18 participants

    human_df = pd.DataFrame()
    n = 9
    for i in range(1, 80*n+1, 80):
        try:
            j = i+79
            temp_df = pd.read_csv(os.path.join(args.human_data, f'main_test({i}_{j}).xls.csv'))
            if i == 1:
                pass
            else:
                temp_df = temp_df.rename(columns = {'유저식별아이디':'useless', 'MC구분':'useless', '성별':'useless', '나이':'useless', '학력':'useless'})
            human_df = pd.concat([human_df, temp_df], axis=1)
        except:
            pass

    human_df = human_df[human_df['유저식별아이디'].isin(sel_ppl)]
    orig_human_df = human_df
    human_df = human_df.fillna(0)

    sel_col = []
    for j in range(1, 80*n+1):
        temp_str = f'선택_A_{j}'
        sel_col.append(temp_str)

    human_df = human_df[sel_col]
    human_df.index = sel_ppl 
    human_df.columns = list(range(80*n))

    # To check outliers (zero: no answer)
    plt.hist(human_df.values, density=True)
    plt.show()

    """ Machine data """
    answer_df = pd.read_csv(args.machine_data)

    act_per_list, pix_list, gs_list, par_list = [], [], [], []
    for answer in answer_df['Answer']:
        img, _ = answer.split('.jpg')
        act_per, pix, gs, par = img.split('_')
        
        act_per_list.append(act_per)
        pix_list.append(pix)
        gs_list.append(gs)
        par_list.append(par)

    answer_df['act_per'] = act_per_list
    answer_df['PIX'] = pix_list
    answer_df['GS'] = gs_list
    answer_df['par'] = par_list

    answer_df = answer_df[:80*n]
    answer_df = answer_df.T

    orig_answer_df = answer_df

    """ Human and machine data """
    mer_df = pd.concat([human_df, answer_df], axis=0)
    mer_df = mer_df.T
    mer_df = mer_df.fillna(0)
    orig_mer_df = mer_df


    """ Getting ready... """
    human_df = orig_human_df.copy()
    human_df = human_df.fillna(0)
    human_df['유저식별아이디'] = human_df['유저식별아이디'].astype(int)
    human_df.index = human_df['유저식별아이디']

    sel_col = []
    for (i, col) in enumerate(human_df.columns):
        split = col.split('_')
        if split[0] == '선택' and split[1] == 'A':
            sel_col.append(col)
        
    acc_df = human_df[sel_col]
    acc_df.columns = [n for n in range(acc_df.shape[1])]
    acc_df = acc_df.astype(int)

    for q in range(acc_df.shape[1]):
        act_per = answer_df.loc['act_per'][q]
        for s in range(acc_df.shape[0]):
            pred_per = acc_df.iloc[s, q]
            try:
                if str(int(pred_per)) == str(int(act_per)):
                    acc_df.iloc[s, q] = 1 
                else:
                    acc_df.iloc[s, q] = 0
            except:
                acc_df.iloc[s, q] = 0

    acc_df.columns = answer_df.T['Answer']

    test_type_list = [args.test_type] #['opt', 'elec']
    model_type1_list = [''] #['', ''] #['PCA', 'PCA', '', '']
    model_type2_list = ['CNN_SVC'] #['CNN_SVC', 'CNN_SVCft'] #['SVC', 'LR', 'CNN_LR', 'CNN_SVC']
    seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # different 1,000 training images (total of 10 permutations)
    pix_order_list = ['16PIX', '24PIX', '32PIX', '64PIX', '128PIX']
    gs_order_list = ['2GS', '4GS', '6GS', '8GS', '16GS']
    class_list = [args.r] # [2, 4, 16]

    df = pd.read_csv(args.meta_path)
    l = list(range(df.shape[0]))
    n = 16
    random.seed(22)
    set_1 = random.sample(l, n)
    sets = [set_1]
    r = class_list[0]

    for test_type in test_type_list:
        mac_df = pd.DataFrame()
        for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
            model_type = model_type1 + model_type2
            for c in class_list:
                for m in range(1):
                    input_folder = [df.iloc[i, 0] for i in sets[m]] 
                    assert len(input_folder) == 16
                    com_obj = itertools.combinations(input_folder, r)
                    com_list = list(com_obj)

                # mac_df = pd.DataFrame()
                    for seed in seed_list:
                        data_path = os.path.join(args.qb_path, '{c}classes\\set{m}\\seed{seed}')

                        for n in range(len(os.listdir(data_path))): # len(com_list)

                            preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # only 1 comb. for 16 classes
                            high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                            
                            add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                            add_high_df['Hit Rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                            add_high_df = add_high_df[['file_name', 'actual_person', 'Hit Rate']]
                            add_high_df['actual_person'] = add_high_df['actual_person'].astype(int)
                            
                            img_list, hyperpar_list, par_list = [], [], []
                            for i in range(add_high_df.shape[0]):
                                file_name = add_high_df['file_name'][i].split('.')[0]
                                pix, gs, par = file_name.split('_')
                                per = add_high_df['actual_person'][i]
                                img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                                img_list.append(img)
                                hyperpar_list.append(str(pix) + '_' + str(gs))
                                par_list.append(str(par))
                            add_high_df['img'] = img_list
                            add_high_df['hyperpar'] = hyperpar_list
                            add_high_df['par'] = par_list
                            
                            add_high_df = add_high_df[['img', 'hyperpar', 'par', 'Hit Rate']]
                        
                        add_high_df['Seed'] = [seed] * add_high_df.shape[0] #[f'{model_type}_{seed}'] * add_high_df.shape[0]
                        
                        mac_df = pd.concat([mac_df, add_high_df], axis=0)

    new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS', 
                            '32PIX_2GS', '32PIX_4GS', '32PIX_8GS',  
                            '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
    par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
                'S001L1E02C7',
                'S001L1E03C7']
    mac_df = mac_df[mac_df['hyperpar'].isin(new_hyperpar_name_list)]
    mac_df = mac_df[mac_df['par'].isin(par_list)]
    mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

    mer_df3 = pd.concat([acc_df, mac_df], join='inner')
    mer_df3 = mer_df3[mer_df3.index != 'img']
    mer_df3 = mer_df3.T.astype(float)


    """ FM """
    for test_type in test_type_list:
        for (model_type1, model_type2) in zip(model_type1_list, model_type2_list):
            model_type = model_type1 + model_type2
            for c in class_list:
                for m in range(1): 
                    input_folder = [df.iloc[i, 0] for i in sets[m]] 
                    assert len(input_folder) == 16
                    com_obj = itertools.combinations(input_folder, r)
                    com_list = list(com_obj)

                    mac_df = pd.DataFrame()
                    for seed in seed_list:
                        data_path = os.path.join(args.qb_path, '{c}classes\\set{m}\\seed{seed}')

                        for n in range(len(os.listdir(data_path))): # len(com_list)

                            preprocessed_data_path =  os.path.join(data_path, f'comb{n}') # 16 classes 는 1 comb 밖에 없음.
                            high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{test_type}')
                            
                            add_high_df = pd.read_csv(os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{model_type1}_{model_type2}.csv'))
                            add_high_df['Hit Rate'] = add_high_df['correctness'].replace(['correct', 'wrong'], [1, 0]) 
                            add_high_df = add_high_df[['file_name', 'actual_person', 'Hit Rate']]
                            add_high_df['actual_person'] = add_high_df['actual_person'].astype(int)
                            
                            img_list, hyperpar_list, par_list = [], [], []
                            for i in range(add_high_df.shape[0]):
                                file_name = add_high_df['file_name'][i].split('.')[0]
                                pix, gs, par = file_name.split('_')
                                per = add_high_df['actual_person'][i]
                                img = str(per) + '_' + str(pix) + '_' + str(gs) + '_' + str(par) + '.jpg'
                                img_list.append(img)
                                hyperpar_list.append(str(pix) + '_' + str(gs))
                                par_list.append(str(par))
                            add_high_df['img'] = img_list
                            add_high_df['hyperpar'] = hyperpar_list
                            add_high_df['par'] = par_list
                            
                            add_high_df = add_high_df[['img', 'hyperpar', 'par', 'Hit Rate']]
                        
                        add_high_df['Seed'] = [seed] * add_high_df.shape[0]
                        
                        mac_df = pd.concat([mac_df, add_high_df], axis=0)

    new_hyperpar_name_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_6GS', '16PIX_8GS', '16PIX_16GS',
                             '24PIX_2GS', '24PIX_4GS', '24PIX_6GS', '24PIX_8GS', '24PIX_16GS',
                             '32PIX_2GS', '32PIX_4GS', '32PIX_6GS', '32PIX_8GS', '32PIX_16GS',
                             '64PIX_2GS', '64PIX_4GS', '64PIX_6GS', '64PIX_8GS', '64PIX_16GS',
                             '128PIX_2GS', '128PIX_4GS', '128PIX_6GS', '128PIX_8GS', '128PIX_16GS']
    par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
                'S001L1E02C7',
                'S001L1E03C7']
    mac_df = mac_df[mac_df['hyperpar'].isin(new_hyperpar_name_list)]
    mac_df = mac_df[mac_df['par'].isin(par_list)]
    # mac_df = mac_df.groupby('img').mean().reset_index()

    mac_df = mac_df.pivot(index='Seed', columns='img', values='Hit Rate')

    mac_df

    mac_df_copy = mac_df.copy()

    mac_df_T = mac_df_copy.T

    hyper_par_list = []
    for img in mac_df_T.index:
        _, pix, gs, _ = img.split('_')
        hyper_par_list.append(f'{pix}_{gs}')
    mac_df_T['hyperpar'] = hyper_par_list
    #mac_df_T_par = mac_df_T.iloc[:, 2]
    mac_df_T_gp = mac_df_T.groupby('hyperpar').mean()

    mac_df_T_gp.mean(axis=1)


    """ FM algorithm """
    fb_df = mer_df3.copy()

    par_list = []
    for file_name in fb_df.index:
        _, pix, gs, par = file_name.split('_')
        par_list.append(f'{pix}_{gs}_{par}')
    fb_df.index = par_list

    unique_pars = list(set(fb_df.index.values))
    df = pd.DataFrame()
    for par in unique_pars:
        df[par] = fb_df.loc[[par], :].mean(axis=0)
    old_df = df.T

    # Creating a new type of dataframe
    hyperpar_list = ['16PIX_2GS', '16PIX_4GS', '16PIX_8GS',
                     '32PIX_2GS', '32PIX_4GS', '32PIX_8GS', 
                     '64PIX_2GS', '64PIX_4GS', '64PIX_8GS']
    par_list = ['S001L1E01C4', 'S001L1E01C7', 'S001L1E01C10',
                'S001L1E02C7',
                'S001L1E03C7']
    temp_list1, temp_list2 = [], []
    for ind in old_df.index:
        pix, gs, par = ind.split('_')
        temp_list1.append(f'{pix}_{gs}')
        temp_list2.append(par.split('.')[0])
    old_df['hyperpar'] = temp_list1
    old_df['par'] = temp_list2
    cor_type = 'pear' # 'adj_pear', 'pear'

    df = pd.DataFrame()
    for i in sel_ppl:
        for j in seed_list:
            for l, hyperpar in enumerate(hyperpar_list): # for all resolutions
                par_old_df = old_df[old_df['hyperpar'] == hyperpar]
                # for p, par in enumerate(par_list):
                #    par_old_df2 = par_old_df[par_old_df['par'] == par]
                j = int(j)
                par_df = par_old_df[[i, j]]
                hum, mac = par_old_df[i], par_old_df[j] # for all hyperpars (9)
                
                if cor_type == 'pear':
                    cor_val = stats.pearsonr(old_df[i], old_df[j]) # 모든 hyperpar, par에 대해서 같음

                elif cor_type == 'adj_pear':
                    indexes = list(range(par_df.shape[0]))
                    avg_l = []
                    for p in range(10):
                        split_half_ran_ind1 = random.sample(indexes, par_df.shape[0]//2)
                        split_half_ran_ind2 = [ind for ind in indexes if ind not in split_half_ran_ind1][1:]
                        
                        hum_par1 = hum.to_frame().iloc[split_half_ran_ind1, :].squeeze()
                        hum_par2 = hum.to_frame().iloc[split_half_ran_ind2, :].squeeze()
                        mac_par1 = mac.to_frame().iloc[split_half_ran_ind1, :].squeeze()
                        mac_par2 = mac.to_frame().iloc[split_half_ran_ind2, :].squeeze()

                        num = (stats.pearsonr(hum_par1, mac_par1)[0] + stats.pearsonr(hum_par1, mac_par2)[0] + stats.pearsonr(hum_par2, mac_par1)[0] + stats.pearsonr(hum_par2, mac_par2)[0]) / 4
                        den = math.sqrt(abs(stats.pearsonr(hum_par1, hum_par2)[0] * stats.pearsonr(mac_par1, mac_par2)[0]))
                        cor_val = num / den

                        avg_l.append(cor_val)

                    cor_val = np.mean(avg_l)

                temp_df = pd.concat([hum, mac], axis=0).to_frame().T # for one hyperpar and one resol
                
                par_list = list(range(1, 6))
                temp_df.columns = [f'Human_{j}' for j in par_list] + [f'ANN_{i}' for i in par_list]
                temp_df['Avg of ANN'] = temp_df[[f'ANN_{i}' for i in par_list]].mean(axis=1)
                temp_df['Cor'] = cor_val[0]
                temp_df['Avg of Hum'] = temp_df[[f'Human_{i}' for i in par_list]].mean(axis=1)

                # temp_df = temp_df[['Avg of ANN', 'Cor', 'Avg of Hum']]
                temp_df = temp_df[[f'ANN_{j}' for j in list(range(1, 6))] + ['Cor'] + [f'Human_{i}' for i in list(range(1, 6))]]

                df = pd.concat([df, temp_df], axis=0)

    # print(np.mean(df['Cor'].values))

    # X_train = df[['Avg of Hum', 'Cor']]
    X_train = df[[f'ANN_{j}' for j in par_list] + ['Cor']]
    # y_train = df['Avg of Hum']
    y_train = df[[f'Human_{i}' for i in par_list]]

    model = models.Sequential()
    model.add(keras.Input(shape=(6, )))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='relu'))

    #if os.path.isfile(args.checkpoint_filepath):
    #    model = keras.models.load_model(args.checkpoint_filepath)
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mean_absolute_error'])
    callback_list = [ModelCheckpoint(filepath=args.checkpoint_filepath, monitor='val_loss', save_best_only=True)]
                    # TensorBoard(log_dir=f'logs_{time.asctime}')]

    history = model.fit(X_train, y_train, batch_size=16, epochs=3000, validation_split=0.1, callbacks=callback_list)

