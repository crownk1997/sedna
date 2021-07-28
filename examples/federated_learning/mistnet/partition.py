import warnings
from PIL import Image
import os
import os.path
import sys
import numpy as np
import torch
import codecs
import string
from typing import Any, Callable, Dict, List, Optional, Tuple
import shutil

# from torchvision.datasets.mnist
SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=True)).view(*s)

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()

def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def load_data(path, train=True):
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    data = read_image_file(os.path.join(path, image_file))
    
    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    labels = read_label_file(os.path.join(path, label_file))
    
    return data, labels

import torch
from torchvision.datasets.utils import extract_archive

# download data
# trainset = MNIST(path,
#                  train=True,
#                  download=True,
#                  transform=ToTensor())

# testset = MNIST(path,
#                 train=False,
#                 download=True,
#                 transform=ToTensor())

# train_data, train_labels = trainset.data, trainset.targets
# test_data, test_labels = testset.data, testset.targets

path = sys.argv[1]
num_partition = int(sys.argv[2])

for gzip_file in os.listdir(path):
    if gzip_file.endswith('.gz'):
        extract_archive(os.path.join(path, gzip_file), path)
        os.remove(os.path.join(path, gzip_file))

train_data, train_labels = load_data(path, True)
test_data, test_labels = load_data(path, False)

training_file = 'training_data.pt'
training_label_file = 'training_label.pt'
test_file = 'test_data.pt'
test_label_file = 'test_label.pt'

idx = torch.randperm(train_data.shape[0])
train_data = train_data[idx]
train_labels = train_labels[idx]
idx = torch.randperm(test_data.shape[0])
test_data = test_data[idx]
test_labels = test_labels[idx]

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

num_train_samples = int(train_data.shape[0]/num_partition)
num_test_samples = int(test_data.shape[0]/num_partition)

for i in range(num_partition-1):
    sub_path = os.path.join(path, str(i))
    if os.path.exists(sub_path) == False:
        os.mkdir(sub_path)
    train_start = i*num_train_samples
    train_end = (i+1)*num_train_samples
    test_start = i*num_test_samples
    test_end = (i+1)*num_test_samples
    torch.save(train_data[train_start:train_end], os.path.join(sub_path, training_file))
    torch.save(train_labels[train_start:train_end], os.path.join(sub_path, training_label_file))
    torch.save(test_data[test_start:test_end], os.path.join(sub_path, test_file))
    torch.save(test_labels[test_start:test_end], os.path.join(sub_path, test_label_file))

sub_path = os.path.join(path, str(num_partition-1))
if os.path.exists(sub_path) == False:
    os.mkdir(sub_path)
train_start = (num_partition-1)*num_train_samples
test_start = (num_partition-1)*num_test_samples
torch.save(train_data[train_start:], os.path.join(sub_path, training_file))
torch.save(train_labels[train_start:], os.path.join(sub_path, training_label_file))
torch.save(test_data[test_start:], os.path.join(sub_path, test_file))
torch.save(test_labels[test_start:], os.path.join(sub_path, test_label_file))

# validation
# for i in range(num_partition):
#     sub_path = os.path.join(path, str(i))
#     train_data = torch.load(os.path.join(sub_path, training_file))
#     train_labels = torch.load(os.path.join(sub_path, training_label_file))
#     test_data = torch.load(os.path.join(sub_path, test_file))
#     test_labels = torch.load(os.path.join(sub_path, test_label_file))
    
#     print(train_data.shape)
#     print(train_labels.shape)
#     print(test_data.shape)
#     print(test_labels.shape)
