import os
import random
import itertools
import argparse
import pandas as pd
import torch

from loadData import loadData
from expModels import beginModeling


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(description='The Alternative Models to Human Psychophysical Tests of Artificial Vision')

parser.add_argument('--data_type', type=str, default='opt') # 'opt', 'elec', 'normal'
parser.add_argument('--model_type1', type=str, default='') # '', '', 'PCA', 'PCA', '', '', '', '', '', '', '', ''
parser.add_argument('--model_type2', type=str, default='') # 'PIXEL_SVC', 'PIXEL_LR', 'SVC', 'LR', 'CNN_SVC', 'CNN_LR','CNN_AlexNet2_SVC', 'CNN_AlexNet2', 'CNN_VggNet2_SVC', 'CNN_VggNet2', 'CNN_ResNet2_SVC', 'CNN_ResNet2', 
parser.add_argument('--r', type=int, default=4) # 2, 4, 16
parser.add_argument('--finetune', type=str, default='') # '', 'ft'
parser.add_argument('--xai', type=bool, default=False)
parser.add_argument('--base_path', type=str, default='C:\\Users\\Na Min An\\Desktop\\Development\\ArtificialVision')


def main():
    global args
    args = parser.parse_args()

    data_path = os.path.join(args.base_path, 'data')
    meta_path = os.path.join(data_path, '210827_ANNA_Removing_uncontaminated_data.csv')

    train_path = os.path.join(data_path, 'Middle_Resolution_137_unzipped_parcropped_128_removed_train')
    tr_camera_list = list(map(str, [4,5,6,7,8,9,10,14,15,16,17,18,19,20])) #4(C)
    tr_light_list = list(map(str, [1,2,3,4,5,6, # 7: all black
                                   8,9,10,11,12,13,14,15,16,17,18,
                                   19,20,21,22,23,24,25,26,27,28,29,30])) #2(L)
    tr_accessory_list =['1'] #1(S)
    tr_expression_list = ['1','2','3'] #3(E)

    test_path = os.path.join(data_path, 'Middle_Resolution_137_unzipped_parcropped_128')
    if args.data_type == 'opt' or args.data_type == 'elec':
        test_path =  f'{args.test_path}_{args.data_type}'
    elif args.data_type == 'normal':
        test_path = f'{train_path}'
    camera_list = ['4','7','10'] #4(C)
    light_list = ['1'] #2(L)
    accessory_list =['1'] #1(S)
    expression_list = ['1','2','3'] #3(E)

    if args.finetune:
        ft_path = os.path.join(data_path, 'Middle_Resolution_137_unzipped_parcropped_128_removed_train_finetune')

    df = pd.read_csv(meta_path)

    l = list(range(df.shape[0]))
    l_2030_f = list(df.loc[(df['연령대'].isin(['20대','30대'])) & (df['성별']=='여')].index)
    l_4050_f = list(df.loc[(df['연령대'].isin(['40대','50대'])) & (df['성별']=='여')].index)
    l_2030_m = list(df.loc[(df['연령대'].isin(['20대','30대'])) & (df['성별']=='남')].index)
    l_4050_m = list(df.loc[(df['연령대'].isin(['40대','50대'])) & (df['성별']=='남')].index)
    l_20304050_f = list(df.loc[df['성별'] == '여'].index)
    l_20304050_m = list(df.loc[df['성별'] == '남'].index)

    n = 16

    random.seed(22)
    set_1 = random.sample(l, n)
    set_2 = random.sample(l_2030_f, n)
    set_3 = random.sample(l_4050_f, n)
    set_4 = random.sample(l_2030_m, n)
    set_5 = random.sample(l_4050_m, n)
    set_6 = random.sample(l_20304050_f, n)
    set_7 = random.sample(l_20304050_m, n)

    random.seed(116)
    set_8 = random.sample(l, n)
    set_9 = random.sample(l_2030_f, n)
    set_10 = random.sample(l_4050_f, n)
    set_11 = random.sample(l_2030_m, n)
    set_12 = random.sample(l_4050_m, n)
    set_13 = random.sample(l_20304050_f, n)
    set_14 = random.sample(l_20304050_m, n)

    sets = [set_1, set_2, set_3, set_4, set_5, set_6, set_7, set_8, set_9, set_10, set_11, set_12, set_13, set_14] # 14 independent sets 
    seed_list = [22, 77, 2, 100, 81, 42, 7, 1, 55, 50] # 10 permutations

    for m in range(14): 
        input_folder = [df.iloc[i, 0] for i in sets[m]] 
        assert len(input_folder) == 16
        com_obj = itertools.combinations(input_folder, args.r)
        com_list = list(com_obj)
        
        for seed in seed_list:
            for n in range(len(com_list)): # for every combination of question set
                qbank_path = os.path.join(data_path, 'Question Banks AI Hub_final')
                data_path = os.path.join(qbank_path, f'{args.r}classes\\set{m}\\seed{seed}')
                preprocessed_data_path =  os.path.join(data_path, f'comb{n}')
                model_path = os.path.join(preprocessed_data_path, 'Saved_Models')
                high_analysis_path = os.path.join(preprocessed_data_path, f'High_Analysis_{args.data_type}')
                low_analysis_path = os.path.join(preprocessed_data_path, f'Low_Analysis_{args.data_type}')

                os.makedirs(data_path, exist_ok=True)
                os.makedirs(preprocessed_data_path, exist_ok=True) 
                os.makedirs(model_path, exist_ok=True)
                os.makedirs(high_analysis_path, exist_ok=True)

                model_file1 = os.path.join(model_path, f'Model_{args.model_type1}{args.finetune}')
                model_file2 = os.path.join(model_path, f'Model_{args.model_type2}{args.finetune}.pt')
                checkpoint_file = os.path.join(model_path, f'Checkpoint_{args.model_type2}{args.finetune}.pt') 
                earlystop_file = os.path.join(model_path, f'Early_Stop_{args.model_type2}{args.finetune}.pt') 
                high_csv_file = os.path.join(high_analysis_path, f'High_Level_Data_Analysis_{args.model_type1}_{args.model_type2}{args.finetune}.csv')
                
                low_csv_file = os.path.join(low_analysis_path, f'Classification Report_{args.model_type1}_{args.model_type2}.csv') 
                auc_info_file = os.path.join(low_analysis_path, f'TP_FP_AUC_Dictionary_{args.model_type1}_{args.model_type2}.csv') 
                roc_file = os.path.join(low_analysis_path, f'ROC_{args.model_type1}_{args.model_type2}.png') 
                check_file = high_csv_file

                if os.path.isfile(check_file) and not args.xai:
                    pass
                else:
                    print(f'\n START {m}th set {n}th comb (seed{seed}): comb{com_list[n]} \n')

                    """1. Load processed data"""
                    Xtrain, ytrain, _, old_uniq_labels, unique_items = loadData(com_list[n], 'train', train_path, test_path, ft_path, args.finetune, args.data_type, tr_accessory_list, tr_light_list, tr_expression_list, tr_camera_list, seed, args.r, args.model_type2)
                    Xtest, ytest, file_path_list, old_uniq_labels2, test_unique_items = loadData(com_list[n], 'test', test_path, test_path, ft_path, args.finetune, args.data_type, accessory_list, light_list, expression_list, camera_list, seed, args.r, args.model_type2)
                    assert set(old_uniq_labels) == set(old_uniq_labels2)
                    assert set(unique_items) == set(test_unique_items)
                    unique_labels = unique_items

                    """2. Train and test ML models"""
                    instance = beginModeling(device, args.model_type1, args.model_type2, Xtrain, ytrain, Xtest, ytest, unique_labels, model_file1, model_file2, high_csv_file, low_csv_file, auc_info_file, checkpoint_file, earlystop_file, roc_file) # INITIALIZATION

                    model, Xtrain, Xtest = instance.loadOrSaveModel1() # FEATURE EXTRACTION (PART 1)
                    train_loader, val_loader, test_loader = instance.convertAndVisualData(model, Xtrain, Xtest, ytrain, ytest) # (OPTIONAL) VISUALIZATION 1|
                    ytest, yfit, yprob, _, y_test_oh= instance.loadOrSaveModel2andEval(train_loader, val_loader, test_loader, Xtrain, Xtest, old_uniq_labels2, file_path_list) # CLASSIFICATION (PART 2)
                    instance.ready4Visualization(ytest, yfit, yprob, file_path_list, old_uniq_labels2, unique_labels, y_test_oh) # VISUALIZATION 3


if __name__ == 'main':
    main()