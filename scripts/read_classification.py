import os
import numpy as np
from tqdm import tqdm


class DataStruct(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.accuracy = []
        self.configs = []

    def add(self, accuracy, configs):
        self.accuracy.append(accuracy)
        self.configs.append(configs)

    def best(self):
        acc_list = np.array(self.accuracy)
        best_index = np.argmax(acc_list)
        best_configs = self.configs[best_index]
        return acc_list[best_index], self.dataset_name, best_configs


data_path = "../results/"

path_list = []
for folder_path in os.listdir(data_path):
    # print(folder_path)
    if "classification" in folder_path:
        path_list.append(data_path + folder_path)


results = {}

for folder_path in tqdm(path_list):
    info = folder_path.split("_")
    dataset = info[1]
    with open(data_path + folder_path + "/" + "result_classification.txt") as f:
        accuracy = float(f.readlines()[1][:-1].split(":")[1])

    results.setdefault(dataset, DataStruct(dataset))
    results[dataset].add(accuracy, folder_path)

for dataset in results.keys():
    print(results[dataset].best()[1], results[dataset].best()[0])
