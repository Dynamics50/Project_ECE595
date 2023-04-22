import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import collections
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class DataGenerator():                                    
    def __init__(self,n):
        self.data_size = n   
        self.label_map = collections.defaultdict(set)
        self.labels =[]                                   
        for i in range(self.data_size):
            x = random.randrange(-250,250)
            y = random.randrange(-250,250)
            if x-(0.65*250)>=y and random.random()<0.62:
               self.label_map[1].add((x, y))
               self.labels.append(1)
            else:
               self.label_map[-1].add((x, y))
               self.labels.append(-1)
    def generate_data(self,n):
        data_gen = DataGenerator(n)
        return data_gen.random_data()
    
    def random_data(self, dim=3, ws=(0.5, 0.5)):
        point, label = make_classification(n_samples=self.data_size, n_features=dim, n_redundant=1, n_informative=2, n_clusters_per_class=2, flip_y=0.0001, weights=ws)
        n1, n2, data = 0, 0, []
        for i, label_value in enumerate(label):
          point_value = point[i]
          if label_value == 0:
            data.append([-1, point_value[0], point_value[1]])
            n2 += 1
          else:
            data.append([label_value, point_value[0], point_value[1]])
            n1 += 1
        print("1. Noise Free Data")
        return data, n1, n2

    def add_noise(self,data, po1, po2):
      noise_data = []
      for j in data:
        noise_data.append(list(j))
      po1=float(po1)
      po2=float(po2)
      swap1, swap_1 = 0, 0
      for j in noise_data:
        if j[0] == 1: 
            swap1 += 1
        elif j[0] == -1:
            swap_1 += 1
      swap1 *= po2
      swap_1 *= po1
      for j in noise_data:
        if swap1 != 0 and j[0] == 1:
            if random.random() < swap1:
                j[0] = -1
                swap1 -= 1
        elif swap_1 != 0 and j[0] == -1:
            if random.random()< swap_1:
                j[0] = 1
                swap_1 -= 1
      print("Noised Data:")
      return noise_data

    def dividedtt_data(self,data):
        train, test = train_test_split(data,shuffle=True)
        print ("3. Splitted data into TrainSize and TestSize:")
        print("Train Size:",len(train))
        print("Test Size:",len(test))
        return train, test

def main_part():                                                   # Main Function
    current_path = os.curdir
    data_size, is_random, po1, po2 = 5000,True,0.25,0.30           #earlier Po1=0.2, Po2=0.2
    generate = DataGenerator(data_size)
    if is_random:
           noise_free_data,_,_ = generate.generate_data(1000)
           data_size, is_random, po1, po2 = 5000,True,0.25,0.30    #earlier Po1=0.2, Po2=0.2
    else:
        noise_free_data = generate.random_data()

    # Noise free data Generation
    noise_free_data_imprint = pd.DataFrame.from_records([(x, y, l) for l, x, y in noise_free_data], columns=['x', 'y', 'l'])
    noise_free_data_imprint.to_csv(r".\DataSet\Sample_Noise_free_data.csv", header=False, index=False)
    
    # Noised data Generation
    noised_data = generate.add_noise(noise_free_data, po1, po2)
    noised_data_imprint = pd.DataFrame.from_records([(x, y, l) for l, x, y in noised_data], columns=['x', 'y', 'l'])
    noised_data_imprint.to_csv(r".\DataSet\Sample_Noised_data.csv", header=False, index=False)

    # Noised train and test dataset generation
    generate.noised_train_set, generate.noised_test_set = generate.dividedtt_data(noised_data)
    data_map3 = {(x, y): l for l, x, y in generate.noised_train_set}
    noised_train = pd.DataFrame.from_dict(data_map3, "index")
    noised_train.to_csv(".\DataSet\Sample_Noised_Train_data.csv", header=False)
    data_map4 = {(x, y): l for l, x, y in generate.noised_test_set}
    noised_test = pd.DataFrame.from_dict(data_map4, "index")
    noised_test.to_csv(".\DataSet\Sample_Noised_Test_data.csv", header=False)

if __name__ == "__main__":
    main_part()