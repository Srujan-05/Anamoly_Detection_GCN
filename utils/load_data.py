import os


def load_data(path="../data/training"):
    # loading the anamolous point clouds
    data = []
    labels = []
    data += [path+"/anamolous/"+i for i in os.listdir(path+"/anamolous")]
    labels += [1 for i in range(len(data))]

    # loading the good point clods
    data += [path + "/good/" + i for i in os.listdir(path + "/good")]
    n = len(labels)
    labels += [-1 for i in range(len(data)-n)]

    return data, labels

