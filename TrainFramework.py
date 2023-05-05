from sklearn import svm
from sklearn.model_selection import KFold
from DataGenerator import DataGenerator
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter("ignore", category=UserWarning)

class TrainFramework(object):
    def __init__(self,data_size,random,po1, po2,dim = 3,ws = (0.5,0.5)):
        self.data_maker = DataGenerator(data_size)
        self.true_data_map = {}
        if random:
            noise_free_data,self.n1,self.n2 = self.data_maker.random_data(dim,ws)    # weight and dimension are features of data distribution
            self.set_random = True
        else:
            noise_free_data = self.data_maker.Data_logger()      
            self.n1 = self.n2 = self.data_maker.data_size/2
            self.set_random = False
        self.init_true_data_map(noise_free_data)
        noised_data = self.data_maker.add_noise(noise_free_data,po1, po2)
        self.noised_train_set, self.noised_test_set = self.data_maker.dividedtt_data(noised_data)
        self.noisy_test_map={(x,y):label for label,x,y in self.noised_test_set}
        self.unbiased_loss_pred_map = {}

    def init_true_data_map(self,data):
        for i in data:
            self.true_data_map[(i[1],i[2])]=i[0]

    def train_SVM(self, train_set):             #SVM classifier for the noisy labels for linear, nonlinear kernel multiclassification
     train_X, train_y = [], []
     for label, x, y in train_set:
        train_X.append((x, y))
        train_y.append(label)
     clf = svm.SVC()                            #Training the SVM classifier for small data high accuracy
     clf.fit(train_X,train_y)
     test_X = [(x,y) for _, x, y in self.noised_test_set]
     pred_y = clf.predict(test_X)
     self.unbiased_loss_pred_map = {}
     for (x, y), label in zip(test_X,pred_y):
        self.unbiased_loss_pred_map[(x, y)] = int(label)
     print("Trained Classifier")
     return clf
    
    def crossVldt(self, po1, po2):                             #Cross validation
     min_loss_cal_data,target_dataset = float('inf'),None
     data = np.array(self.noised_train_set)
     kf = KFold(n_splits=2)                                    #Kfold cross validation for the performance 
     for train, test in kf.split(data):
        size,trained_data = len(train),data[train]
        summation = 1.0*sum(1 for j in trained_data if j[0]==-1)/size  #counting -1 labeled data
        summation= 1.0*sum(1 for j in trained_data if j[0]==1)/size    # counting +1 labeled data
        loss_cal_data = []
        for m in trained_data:
            Indexing = self.true_data_map[m[1], m[2]]               #Asisgining true_data_map at index
            y = m[0]
            loss = 0 if Indexing==y else 1
            p1,p2 = int(round(float(po1)*10)),int(round(float(po2)*10))
            est_loss = ((1-summation)*loss-summation*loss)/(1-p1-p2)
            loss_cal_data.append(est_loss)
        if np.mean(loss_cal_data)<min_loss_cal_data:
            min_loss_cal_data = np.mean(loss_cal_data)
            target_dataset = trained_data
     print("Cross-validation")
     return self.train_SVM(target_dataset)

    def accuracy(self, pred, rst):                                        #Accuracy calculation
     match_cnt,i = 0,0
     all_cnt = len(pred)
     while i<all_cnt:
        if pred[i] == rst[i]:
            match_cnt += 1
        i += 1
     acc = round(match_cnt/all_cnt, 4)
     print("Accuracy Prediction:",acc)
     return acc
    

    def all_figures(self, clf, po1, po2):                                      #Images syntax and code
     f, *axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
     f.suptitle('x axis', y=0.04, ha='center')
     f.text(0.06, 0.5, 'y axis', va='center', rotation='vertical')
     ax1, ax2, ax3 = axes[0]
     x_o = [d[1] for d in self.noised_test_set]
     y_o = [d[2] for d in self.noised_test_set]
     label_o = [self.true_data_map[(x, y)] for x, y in zip(x_o, y_o)]
     COLOR = {-1: "r", 1: "b"}
     label_o, color_o = zip(*[(self.true_data_map[(x, y)], COLOR[self.true_data_map[(x, y)]]) for x, y in zip(x_o, y_o)])
     ax1.scatter(x_o, y_o, marker='+', c=color_o, s=20, edgecolor='b')
     ax1.set_title('Noise-free')
     x_n, y_n = x_o, y_o
     label_n = [d[0] for d in self.noised_test_set]
     color_n = [COLOR[d] for d in label_n]
     ax2.scatter(x_n, y_n, marker='+', c=color_n, s=20, edgecolor='y')         
     ax2.set_title(f'Noise rate: {po1} and {po2}')
     x_t1, y_t1 = x_o, y_o
     noisy_test_X1 = np.array((x_t1, y_t1)).T
     print(noisy_test_X1)
     print(clf.predict([[-0.8, -1]]))
     pred_label1 = clf.predict(noisy_test_X1)
     label_p1 = [COLOR[d] for d in pred_label1]
     Acc = self.accuracy(label_o, pred_label1)
     ax3.scatter(x_t1, y_t1, marker='+',c=label_p1,s=20,edgecolor='y')
     ax3.set_title(f'Accuracy: {Acc}')
     fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(8, 8),gridspec_kw={'hspace': 0})
     for ax in f.axes[:-1]:
        ax.xaxis.set_visible(False)
     table = [((x, y), self.true_data_map[(x, y)], self.noisy_test_map[(x, y)], self.unbiased_loss_pred_map[(x, y)]) for x, y in self.noisy_test_map.keys()]
     headers = ["Co-ordinates", "Noise-free data", "Noisy Data", "Unbiased_estiamtor"]    #Unbiased_loss_predition
     df = pd.DataFrame(table, columns=headers)#.set_index("Co-ordinates")
     df.set_index("Co-ordinates")
     filename = f'{"random" if self.set_random else "linearly"}_test_data_{self.data_maker.data_size}_{po1}_{po2}'
     df.to_csv(f'./Output/{filename}.csv',index=False)
     print("CSV data saved")
     f.savefig(f'./Figures/{filename}.jpg')
     print("Figure saved")
     return Acc

def main_part():
    cur_path = os.curdir                            
    n, random, po1, po2 = 10000, True, 0.2, 0.4
    run = TrainFramework(n,random, po1, po2)
    clf = run.crossVldt(po1, po2)
    run.all_figures(clf, po1, po2)
     
if __name__ == "__main__":
    main_part()


