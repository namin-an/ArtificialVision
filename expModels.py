import os
import time
import random
import itertools
import argparse
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame
import joblib
import gzip
import collections
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.nn import Linear, ReLU, CrossEntropyLoss, MultiMarginLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
from torch.optim import Adam

print(torch.__version__)
print(torch.cuda.get_self.device_name(0)) # my GPU
print(torch.cuda.is_available()) 
print(torch.version.cuda)

from mypackages import pytorchtools


class beginModeling():
    # class variables
    n_components = 128
    val_percent = 0.15
    batch_size = 16
    num_epoch = 1000 #1000
    learning_rate = 1e-4 # 1e-4 
    n_class = 4


    # INITIALIZATION
    def __init__(self, device, model_type1, model_type2, Xtrain, ytrain, Xtest, ytest, unique_labels, model_file1, model_file2, high_csv_file, checkpoint_file, earlystop_file):

        self.device = device

        self.model_type1 = model_type1
        self.model_type2 = model_type2
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.unique_labels = unique_labels
        
        self.model_file1 = model_file1
        self.model_file2 = model_file2
        self.high_csv_file = high_csv_file
        self.checkpoint_file = checkpoint_file
        self.earlystop_file = earlystop_file


    # FEATURE EXTRACTION (PART 1)
    def loadOrSaveModel1(self):
            
        if os.path.isfile(self.model_file1): 
            print(f'Loading a model 1 ({self.model_type1})...')            

            if self.model_type1 == 'PCA':
                with gzip.GzipFile(self.model_file1, 'rb') as f:
                    model = joblib.load(f)
                Xtrain = self.Xtrain.reshape(self.Xtrain.shape[0], self.Xtrain.shape[1]*self.Xtrain.shape[2])
                Xtest = self.Xtest.reshape(self.Xtest.shape[0], self.Xtest.shape[1]*self.Xtest.shape[2])
                print("Original data shape: ", Xtrain.shape, Xtest.shape)
                Xtrain, Xtest = model.transform(Xtrain), model.transform(Xtest) 
                print("PCA transformed data shape: ", Xtrain.shape, Xtest.shape)

            elif self.model_type1 == '':
                model, Xtrain, Xtest = None, self.Xtrain, self.Xtest

                inputs, targets, testinputs, testtargets = torch.from_numpy(Xtrain).float(), torch.from_numpy(self.ytrain).long(), torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long() # long: integers
                dataset, testdataset = TensorDataset(inputs, targets), TensorDataset(testinputs, testtargets)
                train_loader, test_loader = DataLoader(dataset), DataLoader(testdataset)

                ytrain, ytrain_fit = np.array([]), np.array([])
                for i, (inputs, targets) in enumerate(train_loader):
                    y, yhat = targets, model(inputs)
                    ytrain, ytrain_fit = np.append(ytrain, y), np.append(ytrain_fit, yhat)

                ytest, ytest_fit = np.array([]), np.array([])
                for i, (inputs, targets) in enumerate(test_loader):
                    y, yhat = targets, model(inputs)
                    ytest, ytest_fit = np.append(ytest, y), np.append(ytest_fit, yhat)
                Xtrain, Xtest = ytrain_fit, ytest_fit
                print('CNN transformed data shape: ', Xtrain.shape, Xtest.shape) # (batch, 1, beginModeling.n_components)


        else: 
            print(f'Making a model 1 ({self.model_type1})...')

            if self.model_type1 == 'PCA':
                Xtrain = self.Xtrain.reshape(self.Xtrain.shape[0], self.Xtrain.shape[1]*self.Xtrain.shape[2])
                Xtest = self.Xtest.reshape(self.Xtest.shape[0], self.Xtest.shape[1]*self.Xtest.shape[2])
                model = PCA(beginModeling.n_components, whiten=True, random_state=42)
                model.fit(Xtrain)
                plt.plot(np.cumsum(model.explained_variance_ratio_))
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')
                plt.show()
                print(f'Saving a model 1 ({self.model_type1})...')
                with gzip.GzipFile(self.model_file1, 'wb', compresslevel=3) as f:
                    joblib.dump(model, f) 
                print("Original data shape: ", Xtrain.shape, Xtest.shape)
                Xtrain, Xtest = model.transform(Xtrain), model.transform(Xtest)
                print("PCA transformed data shape: ", Xtrain.shape, Xtest.shape)

            elif self.model_type1 == '':
                model, Xtrain, Xtest = None, self.Xtrain, self.Xtest

        return model, Xtrain, Xtest
    
    
    # (OPTIONAL) VISUALIZATION (PART 1)
    def convertAndVisualData(self, model, Xtrain, Xtest, ytrain, ytest):
       
        inputs, targets, testinputs, testtargets = torch.from_numpy(Xtrain).float(), torch.from_numpy(self.ytrain).long(), torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long() # long: integers
        dataset, testdataset = TensorDataset(inputs, targets), TensorDataset(testinputs, testtargets)

        if self.model_type2 != 'SVC2':
            val_size = int(self.Xtrain.shape[0]*beginModeling.val_percent)
            train_size = self.Xtrain.shape[0] - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            train_loader, val_loader = DataLoader(train_ds, beginModeling.batch_size, shuffle=True), DataLoader(val_ds, beginModeling.batch_size*2, shuffle=True)
            test_loader = DataLoader(testdataset, shuffle=False) 
        else:
            train_loader, val_loader, test_loader = None, None, None
        
        return train_loader, val_loader, test_loader


    ## CLASSIFICATION (PART 2)
    def loadOrSaveModel2andEval(self, train_loader, val_loader, test_loader, Xtrain, Xtest, old_uniq_labels2, file_path_list):

        def flatten(L):
            for item in L:
                try:
                    yield from flatten(item)
                except TypeError:
                    yield item

        def gpu2cpu(L):
            temp_list = list()
            for item in L:
                try:
                    item = item.cpu().numpy()
                except:
                    pass
                temp_list.append(item)
            temp_list2 = list(map(float, temp_list))
            return temp_list2

        #Load or save a model.#
        if os.path.isfile(self.model_file2): 
            print(f'Loading a model 2 ({self.model_type2})...')
            if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_ResNet2_SVC' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_AlexNet2_SVC' or self.model_type2 == 'CNN_VggNet2' or self.model_type2 == 'CNN_VggNet2_SVC':
                model = torch.load(self.model_file2)
                if os.path.isfile(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                if os.path.isfile(self.earlystop_file):
                    os.remove(self.earlystop_file)

            elif self.model_type2 == 'SVC2':
                with gzip.GzipFile(self.model_file2, 'rb') as f:
                    model = joblib.load(f)

        else:
            print(f'Making a model 2 ({self.model_type2})...')

            m1, m2 = self.model_type1, self.model_type2
            model = self.LR_Model(m1, m2) 

            model, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs = model.fit(beginModeling.num_epoch, beginModeling.learning_rate, model, train_loader, val_loader, self.checkpoint_file, self.earlystop_file) 
        
            print(f'Saving a model 2 ({self.model_type2})...')
            torch.save(model, self.model_file2) 
        

        #Fit the model.#
        print(f'Evaluating a model 2 ({self.model_type2})...')
        if self.model_type2 == 'SVC2':
            res_model = model.best_estimator_ 
            print('Best Estimator: ', res_model)

            testinputs, testtargets = torch.from_numpy(Xtest).float(), torch.from_numpy(self.ytest).long()
            testdataset = TensorDataset(testinputs, testtargets)
            test_loader = DataLoader(testdataset, shuffle=False)

            yfit, ytest = np.array([]), np.array([])
            yprob = np.array([])
            for i, (inputs, targets) in enumerate(test_loader):
                yhat, y = res_model.predict(inputs), targets
                ytest, yfit = np.append(ytest, y), np.append(yfit, yhat)

        elif self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_ResNet2_SVC' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_AlexNet2_SVC' or self.model_type2 == 'CNN_VggNet2' or self.model_type2 == 'CNN_VggNet2_SVC':
            ytest, yfit, yprob = np.array([]), np.array([]), np.array([])
            model = model.to(self.device)
            model.eval()

            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs.requires_grad_()
                y = targets

                model = model.to(self.device)
                yhat = model(inputs) 
                pred_probs = yhat.detach() 
                _, yhat = torch.max(yhat, dim=1) 
                y, yhat = y.cpu().numpy(), yhat.cpu().numpy()
                pred_probs = gpu2cpu(flatten(pred_probs))
                ytest, yfit, yprob = np.append(ytest, y), np.append(yfit, yhat), np.append(yprob, pred_probs)

                _, _, _, f1, f2 = file_path_list[i].split('\\')

            yprob = yprob.reshape(yfit.shape[0], beginModeling.n_class)

        print("Real ppl.: ", ytest.shape, "Predicted ppl.: ", yfit.shape, "Predicted prob.: ", yprob.shape) # predicted face-class / actual face-class / predicted probability 

        #Getting ready to evaluate the model.#
        y_test_oh = tf.keras.utils.to_categorical(ytest)
        for idx in range(ytest.shape[0]):
            temp_test, temp_fit = old_uniq_labels2[int(ytest[idx])], old_uniq_labels2[int(yfit[idx])]
            ytest[idx], yfit[idx] = temp_test, temp_fit

        return ytest, yfit, yprob, model, y_test_oh


    class LR_Model(torch.nn.Module):

        # 1) Initialize the class.
        def __init__(self, m1, m2):
            super().__init__() 

            self.model_type1, self.model_type2 = m1, m2

            # For the logistic regression model, or when model_type1 == 'PCA'
            self.linear = Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class).to(self.device) # Logistic Regression, # of input nodes & # of output nodes

            # For the 'lite' CNN, or when model_type == 'CNN_LR' or model_type == 'CNN_SVC'
            # Oliveira[2017], 23 classes, 99% end-to-end acc., 25,000 test images
            self.cnn_num_block = nn.Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0).to(self.device),
                ReLU(inplace=True).to(self.device), # perform th eoperation w/ using any additional memory
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(self.device),

                Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(self.device),

                Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False).to(self.device),

                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0).to(self.device),
                ReLU(inplace=True).to(self.device))
             
            self.linear_num_block = nn.Sequential(
                 Linear(in_features=256*11*11, out_features=beginModeling.n_components, bias=True).to(self.device), 
                 ReLU(inplace=True).to(self.device),
                 Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True)).to(self.device)

            # For the pixel baseline model, or when model_type == 'PIXEL_LR' or model_type == 'PIXEL_SVC'
            self.pixel = Linear(in_features=128*128, out_features=beginModeling.n_class).to(self.device)

            # Reference: source code for torchvision.models.alexnet
            self.alexnet_features = nn.Sequential(
                Conv2d(1, 64, kernel_size=11, stride=4, padding=2).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=3, stride=2).to(self.device),

                Conv2d(64, 192, kernel_size=5, padding=2).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=3, stride=2).to(self.device),

                Conv2d(192, 384, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),

                Conv2d(384, 256, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),

                Conv2d(256, 256, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=3, stride=2).to(self.device)
            )
            self.alexnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(self.device)
            self.alexnet_classifier = nn.Sequential(
                Linear(in_features=1*1*256, out_features=beginModeling.n_components, bias=True).to(self.device),
                ReLU(inplace=True).to(self.device),
                Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True).to(self.device)
            )

            # Reference: source code for torchvision.models.vggnet
            self.vggnet_features = nn.Sequential(
                Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2).to(self.device),

                Conv2d(64, 128, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2).to(self.device),

                Conv2d(128, 256, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                Conv2d(256, 256, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2).to(self.device),

                Conv2d(256, 512, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                Conv2d(512, 512, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2).to(self.device),

                Conv2d(512, 512, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                Conv2d(512, 512, kernel_size=3, padding=1).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=2, stride=2).to(self.device),
            )
            self.vggnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(self.device)
            self.vggnet_classifier = nn.Sequential(
                Linear(in_features=1*1*512, out_features=beginModeling.n_components, bias=True).to(self.device),
                ReLU(inplace=True).to(self.device),
                Linear(in_features=beginModeling.n_components, out_features=beginModeling.n_class, bias=True).to(self.device)
            )

            # Reference: source code for torchvision.models.resnet
            # 2*padding = kernel_size - 1
            # Ex.1: padding=1, kernel_size=3
            # Ex.2: padding=0, kernel_size=1
            self.resnet_filters = [64, 128, 256, 512]

            self.resnet_conv = nn.Sequential(
                Conv2d(1, self.resnet_filters[0], kernel_size=7, stride=2, padding=3, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[0]).to(self.device),
                ReLU(inplace=True).to(self.device),
                MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.device)
            )
            
            # BLOCK 1 (no change in image size, but more channels)
            self.resnet_conv1 = nn.Sequential(
                Conv2d(self.resnet_filters[0], self.resnet_filters[0], kernel_size=3, stride=1, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[0]).to(self.device),
                ReLU().to(self.device),

                Conv2d(self.resnet_filters[0], self.resnet_filters[0] , kernel_size=3, stride=1, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[0] ).to(self.device)
            )
            self.resnet_conv1_id = nn.Sequential(
                        Conv2d(self.resnet_filters[0], self.resnet_filters[0] , kernel_size=1, stride=1, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[0] ).to(self.device)
                    )
        
            # BLOCK 2
            self.resnet_conv2 = nn.Sequential(
                Conv2d(self.resnet_filters[0] , self.resnet_filters[1], kernel_size=3, stride=2, padding=1, bias=False).to(self.device), # ((h or w) + 2*padding(=1) - (kernel_size(=3)-1)) // stride(=2) = (h or w) // stride(=2)
                BatchNorm2d(self.resnet_filters[1]).to(self.device),
                ReLU().to(self.device),

                Conv2d(self.resnet_filters[1], self.resnet_filters[1], kernel_size=3, stride=1, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[1]).to(self.device)
            )
            self.resnet_conv2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[0] , self.resnet_filters[1] , kernel_size=1, stride=2, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[1] ).to(self.device)
                    )
            self.resnet_conv2_2 = nn.Sequential(
                Conv2d(self.resnet_filters[1], self.resnet_filters[1] , kernel_size=1, stride=1, padding=0, bias=False).to(self.device), # ((h or w) + 2*padding(=1) - (kernel_size(=3)-1)) // stride(=2) = (h or w) // stride(=2)
                BatchNorm2d(self.resnet_filters[1] ).to(self.device),
                ReLU().to(self.device),

                Conv2d(self.resnet_filters[1] , self.resnet_filters[1], kernel_size=1, stride=1, padding=0, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[1] ).to(self.device)
            )
            self.resnet_conv2_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[1], self.resnet_filters[1] , kernel_size=1, stride=1, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[1] ).to(self.device)
                    )

            # BLOCK 3
            self.resnet_conv3 = nn.Sequential(
                Conv2d(self.resnet_filters[1] , self.resnet_filters[2], kernel_size=3, stride=2, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[2]).to(self.device),
                ReLU().to(self.device),

                Conv2d(self.resnet_filters[2], self.resnet_filters[2], kernel_size=3, stride=1, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[2]).to(self.device)
            )
            self.resnet_conv3_id = nn.Sequential(
                        Conv2d(self.resnet_filters[1] , self.resnet_filters[2], kernel_size=1, stride=2, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[2]).to(self.device)
                    )
            self.resnet_conv3_2 = nn.Sequential(
                Conv2d(self.resnet_filters[2], self.resnet_filters[2], kernel_size=1, stride=1, padding=0, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[2]).to(self.device),
                ReLU().to(self.device),

                Conv2d(self.resnet_filters[2], self.resnet_filters[2] , kernel_size=1, stride=1, padding=0, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[2]).to(self.device)
            )
            self.resnet_conv3_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[2], self.resnet_filters[2] , kernel_size=1, stride=1, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[2] ).to(self.device)
                    )

            # BLOCK 4
            self.resnet_conv4 = nn.Sequential(
                Conv2d(self.resnet_filters[2] , self.resnet_filters[3], kernel_size=3, stride=2, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[3]).to(self.device),
                ReLU().to(self.device),
            
                Conv2d(self.resnet_filters[3], self.resnet_filters[3], kernel_size=3, stride=1, padding=1, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[3]).to(self.device)
            )
            self.resnet_conv4_id = nn.Sequential(
                        Conv2d(self.resnet_filters[2] , self.resnet_filters[3], kernel_size=1, stride=2, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[3]).to(self.device)
                    )
            self.resnet_conv4_2 = nn.Sequential(
                Conv2d(self.resnet_filters[3], self.resnet_filters[3], kernel_size=1, stride=1, padding=0, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[3]).to(self.device),
                ReLU().to(self.device),
            
                Conv2d(self.resnet_filters[3], self.resnet_filters[3] , kernel_size=1, stride=1, padding=0, bias=False).to(self.device),
                BatchNorm2d(self.resnet_filters[3]).to(self.device)
            )
            self.resnet_conv4_2_id = nn.Sequential(
                        Conv2d(self.resnet_filters[3], self.resnet_filters[3] , kernel_size=1, stride=1, bias=False).to(self.device),
                        BatchNorm2d(self.resnet_filters[3] ).to(self.device)
                    )

            self.resnet_avgpool = AdaptiveAvgPool2d((1, 1)).to(self.device)
            self.resnet_classifier = nn.Sequential(
                Linear(in_features=self.resnet_filters[-1], out_features=beginModeling.n_components, bias=True).to(self.device),
                ReLU(inplace=True).to(self.device),
                Linear(beginModeling.n_components, beginModeling.n_class).to(self.device)
            )
            

        def forward(self, xb):
            if (self.model_type1 == 'PCA' and self.model_type2 == 'LR') or (self.model_type1 == 'PCA' and self.model_type2 == 'SVC'):
                xb = xb.reshape(-1, beginModeling.n_components).to(self.device)
                xb = self.linear(xb)
                out = xb.to(self.device) 

            elif self.model_type2 == 'CNN_LR' or self.model_type2 == 'CNN_SVC':
                xb = torch.unsqueeze(xb, 1).to(self.device) 
                xb = self.cnn_num_block(xb)
                xb = xb.view(xb.size(0), -1).to(self.device)
                xb = self.linear_num_block(xb)
                out = xb.to(self.device) 

            elif self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'PIXEL_SVC':
                xb = xb.reshape(-1, 128*128).to(self.device)
                xb = self.pixel(xb)
                out = xb.to(self.device) 

            elif self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_VGGNet':
                xb = xb.permute(0, 3, 1, 2) # (batch_size, w, h, c) -> (batch_size, c, w, h)
                if self.model_type2 == 'CNN_ResNet':
                    model = torchvision.models.resnet18(pretrained=False).to(self.device)
                    # for param in model.parameters():
                    #     param.requires_grad = False # Freezing weights.
                    num_infeat = model.fc.in_features
                    model.fc = Linear(num_infeat, beginModeling.n_class).to(self.device)
                elif self.model_type2 == 'CNN_AlexNet':
                    model = torchvision.models.alexnet(pretrained=False).to(self.device)
                    num_infeat = model.classifier[-1].in_features
                    model.classifier[-1] = Linear(num_infeat, beginModeling.n_class).to(self.device)
                elif self.model_type2 == 'CNN_VGGNet':
                    model = torchvision.models.vgg16(pretrained=False).to(self.device)               
                    num_infeat = model.classifier[-1].in_features
                    model.classifier[-1] = Linear(num_infeat, beginModeling.n_class).to(self.device)
                xb = model(xb)
                out = xb.to(self.device)
            
            elif self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_AlexNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(self.device)
                xb = self.alexnet_features(xb)
                xb = self.alexnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(self.device) 
                xb = self.alexnet_classifier(xb)
                out = xb.to(self.device)
                return out
            
            elif self.model_type2 == 'CNN_VggNet2' or self.model_type2 == 'CNN_VggNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(self.device)
                xb = self.vggnet_features(xb)
                xb = self.vggnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(self.device) 
                xb = self.vggnet_classifier(xb)
                out = xb.to(self.device)
                return out

            elif self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_ResNet2_SVC':
                xb = torch.unsqueeze(xb, 1).to(self.device)
                xb = self.resnet_conv(xb)

                # BLOCK 1
                xb1_1 = self.resnet_conv1(xb)
                xb1_2 = self.resnet_conv1_id(xb)
                assert xb1_1.shape == xb1_2.shape
                xb = xb1_1 + xb1_2
                xb = ReLU(inplace=True)(xb).to(self.device)

                xb1_3 = self.resnet_conv1(xb)
                xb1_4 = self.resnet_conv1_id(xb)
                assert xb1_3.shape == xb1_4.shape
                xb = xb1_3 + xb1_4
                xb = ReLU(inplace=True)(xb).to(self.device)

                # BLOCK 2
                xb2_1 = self.resnet_conv2(xb)
                xb2_2 = self.resnet_conv2_id(xb)
                assert xb2_1.shape == xb2_2.shape
                xb = xb2_1 + xb2_2
                xb = ReLU(inplace=True)(xb).to(self.device)

                xb2_3 = self.resnet_conv2_2(xb)
                xb2_4 = self.resnet_conv2_2_id(xb)
                assert xb2_3.shape == xb2_4.shape
                xb = xb2_3 + xb2_4
                xb = ReLU(inplace=True)(xb).to(self.device)

                # BLOCK 3
                xb3_1 = self.resnet_conv3(xb)
                xb3_2 = self.resnet_conv3_id(xb)
                assert xb3_1.shape == xb3_2.shape
                xb = xb3_1 + xb3_2
                xb = ReLU(inplace=True)(xb).to(self.device)

                xb3_3 = self.resnet_conv3_2(xb)
                xb3_4 = self.resnet_conv3_2_id(xb)
                assert xb3_3.shape == xb3_4.shape
                xb = xb3_3 + xb3_4
                xb = ReLU(inplace=True)(xb).to(self.device)

                # BLOCK 4
                xb4_1 = self.resnet_conv4(xb)
                xb4_2 = self.resnet_conv4_id(xb)
                assert xb4_1.shape == xb4_2.shape
                xb = xb4_1 + xb4_2
                xb = ReLU(inplace=True)(xb).to(self.device)

                xb4_3 = self.resnet_conv4_2(xb)
                xb4_4 = self.resnet_conv4_2_id(xb)
                assert xb4_3.shape == xb4_4.shape
                xb = xb4_3 + xb4_4
                xb = ReLU(inplace=True)(xb).to(self.device)

                xb = self.resnet_avgpool(xb)
                xb = torch.flatten(xb, 1).to(self.device)
                xb = self.resnet_classifier(xb)
                out = xb.to(self.device)
                return out

            return out
        

        # 2) Fit the model.
        def fit(self, epochs, lr, model, train_loader, val_loader, checkpoint_file, earlystop_file, opt_func=torch.optim.Adam):

            train_losses, valid_losses, avg_train_losses, avg_valid_losses = [], [], [], []
            train_accs, valid_accs, avg_train_accs, avg_valid_accs = [], [], [], []  

            optimizer = opt_func(model.parameters(), lr=lr, weight_decay=1e-4) # weight_decay is used to prevent overfitting (It keeps the weights small and avoid exploding gradients.)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=False)
            early_stopping = pytorchtools.EarlyStopping(patience=20, verbose=True, path=earlystop_file)                                                                                                       

            if os.path.isfile(checkpoint_file):
                checkpoint = torch.load(checkpoint_file)
                model.load_state_dict(checkpoint['model_state_dict']) # update the model where it left off
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # update the optimizer where it left off
                past_epoch = checkpoint['epoch']
                avg_train_losses.append(checkpoint['avg_train_losses'])
                avg_valid_losses.append(checkpoint['avg_valid_losses'])
                avg_train_accs.append(checkpoint['avg_train_accs'])
                avg_valid_accs.append(checkpoint['avg_valid_accs'])
            else:
                past_epoch = 0

            if (past_epoch+1) < epochs+1:
                for epoch in range(past_epoch+1, epochs+1): # whew what a relief (that we do not have to start all over again every time we train the model)
                    # Training Phase
                    for (images, labels) in train_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        out = self(images) 
                        if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_VggNet2':
                            loss = F.cross_entropy(out, labels) 
                        elif self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_AlexNet2_SVC' or self.model_type2 == 'CNN_ResNet2_SVC' or self.model_type2 == 'CNN_VggNet2_SVC':
                            loss = F.multi_margin_loss(out, labels) 
                        _, preds = torch.max(out, dim=1) 
                        acc = torch.tensor(torch.sum(preds==labels).item() / len(preds))

                        loss.backward() # back-propagation (compute gradients of the loss)
                        optimizer.step() # perform a single optimization step (parameter update)
                        optimizer.zero_grad() # clear the gradients of all optimized variables

                        train_losses.append(loss.detach()) # record training loss
                        train_accs.append(acc.detach())

                    for (images, labels) in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        out = self(images)
                        if self.model_type2 == 'LR' or self.model_type2 == 'SVC' or self.model_type2 == 'CNN_LR' or self.model_type2 == 'PIXEL_LR' or self.model_type2 == 'CNN_ResNet' or self.model_type2 == 'CNN_ResNet2' or self.model_type2 == 'CNN_AlexNet' or self.model_type2 == 'CNN_AlexNet2' or self.model_type2 == 'CNN_VggNet2':
                            loss = F.cross_entropy(out, labels) 
                        elif self.model_type2 == 'CNN_SVC' or self.model_type2 == 'PIXEL_SVC' or self.model_type2 == 'CNN_AlexNet2_SVC' or self.model_type2 == 'CNN_ResNet2_SVC' or self.model_type2 == 'CNN_VggNet2_SVC':
                            loss = F.multi_margin_loss(out, labels) 
                        _, preds = torch.max(out, dim=1) # 
                        acc = torch.tensor(torch.sum(preds==labels).item() / len(preds))

                        valid_losses.append(loss.item())
                        valid_accs.append(acc.detach())
                    
                    scheduler.step(loss) # should be called after validation.

                    train_loss = sum(train_losses) / len(train_losses)
                    valid_loss = sum(valid_losses) / len(valid_losses)
                    avg_train_losses.append(train_loss)
                    avg_valid_losses.append(valid_loss)

                    train_acc = sum(train_accs) / len(train_accs)
                    valid_acc = sum(valid_accs) / len(valid_accs)
                    avg_train_accs.append(train_acc)
                    avg_valid_accs.append(valid_acc)

                    epoch_len = len(str(epochs))

                    msg = (f'{epoch}/{epochs} train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f}')
                    print(msg)

                    torch.save({'epoch' : epoch,
                                'model_state_dict' : model.state_dict(),
                                'optimizer_state_dict' : optimizer.state_dict(),
                                'avg_train_losses' : avg_train_losses,
                                'avg_valid_losses' : avg_valid_losses,
                                'avg_train_accs' : avg_train_accs,
                                'avg_valid_accs' : avg_valid_accs}, checkpoint_file)

                    train_losses, valid_losses = [], [] # clear lists to track next epoch

                    early_stopping(valid_loss, model)
                    if early_stopping.early_stop:
                        print('Early stopped.')
                        break 
      
            else:
                final_checkpoint = torch.load(earlystop_file) # WARNING: not the checkpoint_file
                model.load_state_dict(final_checkpoint)       

            return model, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs

        # 3) Visualize the log.
        def visualize(self, avg_train_losses, avg_valid_losses, avg_train_accs, avg_valid_accs): 
            # to get values from list containing sub-lists
            def flatten(x):
                if isinstance(x, collections.Iterable):
                    return [a for i in x for a in flatten(i)]
                else:
                    return [x]
                    

            avg_train_losses = flatten(avg_train_losses)
            avg_valid_losses = flatten(avg_valid_losses)
            avg_train_accs = flatten(avg_train_accs)
            avg_valid_accs = flatten(avg_valid_accs)

            fig, acc_ax = plt.subplots()
            
            acc_ax.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
            acc_ax.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Valid Loss')
            acc_ax.set_ylabel('Loss')
            acc_ax.legend(loc='lower left')

            acc_ax2 = acc_ax.twinx()
            acc_ax2.plot(range(1, len(avg_train_accs)+1), avg_train_accs, label='Training Accuracy', color='green')
            acc_ax2.plot(range(1, len(avg_valid_accs)+1), avg_valid_accs, label='Valid Accuracy', color='red')
            acc_ax2.set_ylabel('Accuracy')
            acc_ax2.legend(loc='upper left')

            minpos = avg_valid_losses.index(min(avg_valid_losses)) + 1 
            plt.axvline(minpos, linestyle='--', color='r', label='Early Stopping Checkpoint')
            plt.legend(loc='center left')

            plt.xlabel('Epochs')
            plt.xlim(0, len(avg_train_losses)+1) 
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # VISUALIZATION (PART 3)
    def ready4Visualization(self, ytest, yfit, yprob, file_path_list, old_uniq_labels):
        # High Analysis #
        if os.path.isfile(self.high_csv_file): 
            pass
        else:
            df_high = pd.DataFrame()
            for i in range(yfit.shape[0]):
                idx = old_uniq_labels.index(int(ytest[i])) # (0, 1, ..., beginModeling.n_class-1)
                actual_prob = yprob[i][idx]
                pred_prob = np.max(yprob[i], axis=-1)
                if yfit[i] == ytest[i]:
                    ans = 'correct'
                else:
                    ans = 'wrong'
                chance_prob = 1 / beginModeling.n_class
                if ((chance_prob+1)/2 < pred_prob) and (pred_prob <= 1):
                    conf_level = 1
                elif (0 < pred_prob) and (pred_prob <= (chance_prob+1)/2):
                    conf_level = 0
                else:
                    conf_level = None
                _, _, _, _, file_name_list = file_path_list[i].split('\\')
                new_data = {
                            'file_name' : str(file_name_list), # save 3
                            'correctness' : ans, # save 4
                            'pred_prob_vector' : yprob[i], # save 5
                            'actual_person' : ytest[i], # save 6
                            'actual_prob' : actual_prob, # save 7
                            'pred_person' : yfit[i], # save 8
                            'pred_prob' : pred_prob, # save 9
                            'conf_level' : conf_level # save 10
                            }
                df_high = df_high.append(new_data, ignore_index=True)  
            df_high = df_high[['file_name', 'correctness', 'pred_prob_vector', 'actual_person', 'actual_prob', 'pred_person', 'pred_prob', 'conf_level']] # rearrange columns of the dataframe    


            print(df_high.groupby(df_high['correctness']).count()) 
            df_high.to_csv(self.high_csv_file, index=False) # save 1, 2