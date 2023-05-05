from TrainFramework import TrainFramework
import pandas as pd
import numpy as np
import os

def operation():                          #Running the program
    cur_path = os.curdir
    while True:
        run_type = input("Single vs Group\nType 1 for single or Type 2:")
        if run_type == 1:
            single_run(True)
        else:
            random_Table(True) if input("Table 1 or 2 for data storage?\n1 for balanced class or type 2:") == '1' else random_Table(True)
        if input("Try more? Y/N ")=='N': break           

def single_run(call = False):
    if call:
        cur_path = os.curdir                           
    type = input("Linear data (1) or Random data(2)? Type 1 or 2: ")          #Linear or non-linear data
    if type =="1":
        random = False
    else:
        random = True
    n = 10000
    po1 = input("Noise rate for pos label_1: ")                               #Assigning noise rate
    po2 = input("Noise rate for neg label1: ")
    run = TrainFramework(n,random,po1, po2,n)
    clf = run.crossVldt(po1, po2)
    run.all_figures(clf, po1, po2)
    
def random_Table(random_data=False):                                      # Table creation
    rates = [[0.23, 0.23], [0.35, 0.13], [0.45, 0.45]]                    # values of different features
    dimension = [10, 9, 6, 30, 14]
    ns = [2200, 7800, 2800, 9600, 1800]
    ws = [[0.32,0.68],[0.45,0.58],[0.27,0.77],[0.45,0.55],[0.58,0.42]]
    table,i = [],0
    headers = ["D", "N+", "N-", "P+1", "P-1", "Accuracy"]
    while i<len(ns):
        n = ns[i]
        d = dimension[i]
        w = ws[i]
        j = 0
        while j<len(rates):
            r = rates[j]
            Assembled_run = TrainFramework(n, random_data, r[0], r[1],d,w)
            clf = Assembled_run.crossVldt(r[0], r[1])
            x_o = [item[1] for item in Assembled_run.noised_test_set]
            y_o = [item[2] for item in Assembled_run.noised_test_set]
            nosiy_test_X1 =zip(x_o,y_o)
            int_test_X1 = [(x_o[size],y_o[size]) for size in range(len(x_o))]
            label_o = [Assembled_run.true_data_map[(x, y)] for x, y in nosiy_test_X1]
            pred_label1 = clf.predict(int_test_X1)
            accuracy = Assembled_run.accuracy(label_o, pred_label1)
            table.append((d, Assembled_run.n1, Assembled_run.n2, r[0], r[1], accuracy))
            j += 1
        i += 1
    df = pd.DataFrame(table,columns=headers)
    df.to_csv("./Tables/Table.csv")
    print ("Table")

if __name__ == "__main__":
    operation()