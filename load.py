import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def train():
    datas, labels = unpickle('data/cifar-10-batches-py/data_batch_1')
    for i in range(2,6):
        data, label = unpickle('data/cifar-10-batches-py/data_batch_' + str(i))
        datas = np.vstack([datas, data])
        labels = np.hstack([labels, label])
    datas = datas.reshape((len(datas),3, 32, 32))
    datas = datas.transpose((0,2,3,1))
    return datas, labels

def test():
    datas, labels = unpickle('data/cifar-10-batches-py/test_batch')
    datas = datas.reshape((len(datas), 3, 32, 32))
    datas = datas.transpose((0,2,3,1))
    return datas, labels

