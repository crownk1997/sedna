
from sedna.datasources import PyTorchDataSource
from sedna.algorithms.trainer import BasicTrainer
from torch.utils.data import Dataset
# use default trainer, the user can also overwrite some functions
class Trainer(BasicTrainer):
    def __init__(self):
        super().__init__()

class CustomDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Data(PyTorchDataSource):
    def __init__(self, path):
        super().__init__()
        # build training set and testing set
        training_file = 'training_data.pt'
        training_label_file = 'training_label.pt'
        test_file = 'test_data.pt'
        test_label_file = 'test_label.pt'
        
        train_data = torch.load(os.path.join(path, training_file))
        train_labels = torch.load(os.path.join(path, training_label_file))
        test_data = torch.load(os.path.join(path, test_file))
        test_labels = torch.load(os.path.join(path, test_label_file))
        
        self.trainset = CustomDataSet(train_data, train_labels)
        self.testset = CustomDataSet(test_data, test_labels)