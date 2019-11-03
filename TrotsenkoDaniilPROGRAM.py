import torch
import torch.nn as nn
import csv
import numpy as np
from collections import Counter

class PredictionAbonent(torch.nn.Module):
    def __init__(self, hidden_neurons=400):
        super(PredictionAbonent, self).__init__()
        self.fc1 = nn.Linear(44, hidden_neurons)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons-200)
        self.ac2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_neurons-200, 1)
        self.act3 = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)


def learning(wine_net, dataset, target):
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)
    loss = torch.nn.MSELoss()
    """
    ! Batch-normalization
    """
    batch_size = 50
    for epoch in range(501):
        order = np.random.permutation(len(dataset))
        for start_index in range(0, len(dataset), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index+batch_size]
            dataset_batch = dataset[batch_indexes]
            target_batch = target[batch_indexes]

            preds = wine_net.forward(dataset_batch) 

            loss_value = loss(preds, target_batch)
            loss_value.sum().backward()

            optimizer.step()
        
        if epoch % 500 == 0:
            data = ["Predict,Target,Error".split(',')]
            for i in range(len(dataset)):
                test_preds = wine_net.forward(dataset[i])
                loss_value = loss(test_preds, target[i])
                data.append("{},{},{}".format(test_preds.max(), target[i].max(), loss_value).split(','))
            torch.save(wine_net.state_dict(), "./model.pth")
            path = "OUT/epoch{}.csv".format(epoch)
            csv_writer(data, path)
            print("Epoch: {}".format(epoch))
    print("Learning end")



def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
    
                    
        

def load_target():
    train_data = []
    with open("tests/train_target.csv", "r") as f_obj:
        csv_reader(f_obj, train_data)
    return [ list(map(float, i)) for i in train_data ]


def csv_reader(file_obj, var):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    for row in reader:
        if row[0] in ["PERIOD", "ID"]:
            continue
        var.append( list(map(lambda y: 0 if y == "" else y, list(row))))

def load_end():
    test_target = []
    with open("tests/test_target.csv", "r") as f_obj:
        csv_reader(f_obj, test_target)
    return test_target


def load_dataset():
    tabular_data = []
    hash_data = []

    with open("tests/tabular_data.csv", "r") as f_obj:
        csv_reader(f_obj, tabular_data)

    with open("tests/hashed_data.csv", 'r') as f_obj:
        csv_reader(f_obj, hash_data)
    
    hash_data = Counter([int(i[0]) for i in hash_data ])
    tabular_data = np.array([list( map(float, i) ) for i in tabular_data])
    dataset = {}
    j = 0
    tmp = np.zeros(43)

    for i in tabular_data:
        tmp += i[2:]
        if j % 3 == 2:
            dataset[int(i[1])] = list(map(lambda y: y/3, tmp[:])) + [hash_data.get(int(i[1]), 0)]
            tmp = np.zeros(43)
        j += 1
    return dataset


if __name__ == "__main__":
    dataset = load_dataset()
    train_data = load_target()

    full_train_dataset = ([], [])
    for i in range(len(train_data)):
        full_train_dataset[0].append(dataset[train_data[i][0]])
        full_train_dataset[1].append([train_data[i][1]])

    myNet = PredictionAbonent()
    
    datasets = torch.tensor(full_train_dataset[0])
    targets = torch.tensor(full_train_dataset[1])
    learning(myNet,datasets,targets)
    test_target = load_end()
    test_target = [int(i[0]) for i in test_target ]
    data = ['ID,SCORE'.split(',')]

    for i in range(len(test_target)):
        test_preds = myNet.forward( torch.tensor(dataset.get(test_target[i],dataset[test_target[1]]) ))
        data.append("{},{}".format(test_target[i],test_preds.max()).split(','))
        

    path = "TrotsenkoDaniil_test.txt"
    csv_writer(data, path)
    
    