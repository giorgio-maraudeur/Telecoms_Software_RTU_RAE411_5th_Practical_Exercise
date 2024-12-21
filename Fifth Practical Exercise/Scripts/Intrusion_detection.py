from Traffic_classification_with_decision_tree import DecisionTreeClassifierManual

import os 
import pandas as pd
import numpy as np
import matplotlib

path ='C:\\Users\\NZUZI MANIEMA\\Documents\\AERO 4\\Riga Semestre 7\\Cours\\Telecomunications Software (RAE 411)\\Lab Works\\Mes cuts\\Fifth Practical Exercise\\labeled_flows_xml'

X_Normal = []
Y_Normal = []
X_Attack = []
Y_Attack = []

for file in os.listdir(path):
    df = pd.read_xml(path + '\\' + file)
    print(df.info())
    
    AppCount = pd.value_counts(df['appName']) 
    AttackCount = pd.value_counts(df['Tag'])
    AttackDataframe = pd.DataFrame(df.loc[df['Tag'] == 'Attack'])
    AttackCount2 = pd.value_counts(AttackDataframe['appName'])
    NormalDataframe = pd.DataFrame(df.loc[df['Tag'] == 'Normal'])
    NormalDataframeY = NormalDataframe['[Tag]']
    AttackDataframeY = AttackDataframe['[Tag]']
    AttackDataframe = AttackDataframe[['totalSourcesBytes', 'totalDestinationBytes', 'totalDestinationPackets', 'totalSourcePackets', 'sourcePort', 'destinationPort']]
    NormalDataframe = NormalDataframe[['totalSourcesBytes', 'totalDestinationBytes', 'totalDestinationPackets', 'totalSourcePackets', 'sourcePort', 'destinationPort']]
    X_Normal.append(NormalDataframe)
    Y_Normal.append(NormalDataframeY)
    X_Attack.append(AttackDataframe)
    Y_Attack.append(AttackDataframeY)

X_Normal = pd.concat(X_Normal, ignore_index=True)
Y_Normal = pd.concat(Y_Normal, ignore_index=True)
X_Attack = pd.concat(X_Attack, ignore_index=True)
Y_Attack = pd.concat(Y_Attack, ignore_index=True)

# Split the dataset
from sklearn.model_selection import train_test_split
X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(X_Normal, Y_Normal, random_state=0, test_size=0.3)
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(X_Attack, Y_Attack, random_state=0, test_size=0.3)

X_train = pd.concat([X_train_N, X_train_A]) # Concatenate the normal and attack data
X_train = X_train.sample(frac=1, random_state= 42) # Shuffle the data
X_test = pd.concat([X_test_N, X_test_A])   
X_test = X_test.sample(frac=1, random_state= 42)
Y_train = np.concatenate([Y_train_N, Y_train_A])
Y_train = pd.DataFrame(Y_train) # Convert the numpy array to a pandas dataframe
Y_train = Y_train.sample(frac=1, random_state= 42)
Y_test = np.concatenate([Y_test_N, Y_test_A])
Y_test = pd.DataFrame(Y_test)
Y_test = Y_test.sample(frac=1, random_state= 42)

