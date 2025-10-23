import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch, random, os, cv2, argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# from data_loader import getLoaders

class Wafer_receptive_field_v1(Dataset):
    def __init__(self, data, param_data, label):
        self.data = data
        self.param_data = param_data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.param_data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class Wafer_receptive_field(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)



#############################################################
np.random.seed(42)
dir_path = os.getcwd()+"/"+"Wafer_3x3_25_27_Exp1"

X_test_dies_v1 =  np.load(dir_path+"/X_test_dies_v1.npy")
X_train_dies_v1 = np.load(dir_path+"/X_train_dies_v1.npy")
X_test_dies_param_v1 =  np.load(dir_path+"/X_test_dies_param_v1.npy")
X_train_dies_param_v1 = np.load(dir_path+"/X_train_dies_param_v1.npy")

labels = []
X_train_dies_v2 = []

# mu = 0.5965451703930418
# std = 0.3482061850943174

mu = 0
std = 1

# mu = 0.5424
# std = 0.30885
for x in X_train_dies_v1:
  labels.append(x[x.shape[0]//2, x.shape[1]//2].item())
  x1 = x.copy()
  x1 = (x1/x1.max() - mu)/std
  x1[x1.shape[0]//2, x1.shape[1]//2]=0
  X_train_dies_v2.append(x1)

wafer_tr_data = np.array(X_train_dies_v2)
labels_v1 = np.array(labels) - 1
y_train = labels_v1
print("Max Value wafer training data:", wafer_tr_data.max(), "Min Value wafer training data:",wafer_tr_data.min())


# Test set
y_test = []
X_test_dies_v2 = []
for x in X_test_dies_v1:
  y_test.append(x[x.shape[0]//2, x.shape[1]//2].item())
  x1 = x.copy()
  x1 = (x1/x1.max() - mu)/std
  x1[x1.shape[0]//2, x1.shape[1]//2]=0
  X_test_dies_v2.append(x1)

wafer_test_data = np.array(X_test_dies_v2)
wafer_test_label = np.array(y_test) - 1
wafer_test_label = torch.tensor(wafer_test_label).float()
print("Max Value wafer test data:", wafer_test_data.max(), "Min Value wafer test data:",wafer_test_data.min())
print()


# Test parametric data
# mus = [0.7762742501586967, 0.3104178934695622, 0.588879825610195, 0.5687342162150539, 0.4490443586054258, 0.37651138269966045, 0.4250408335662345]
# stds = [0.3490586951934936, 0.4072947386844294, 0.3516854901948171, 0.351697729393864, 0.3611352022022382, 0.3535069737222751, 0.3469982613766242]

mus = [0]*7
stds = [1]*7

# mus = [0.7342, 0.2377, 0.4723, 0.4366, 0.4012, 0.2977, 0.3469]
# stds = [0.31431, 0.37259, 0.27551, 0.26583, 0.32972, 0.30334, 0.29529]
X_train_dies_param_v2 = []
for test_param in X_train_dies_param_v1:
    test_params = []
    for ix,x in enumerate(test_param):
        x1 = x.copy()
        x1 = (x1/x1.max() - mus[ix])/stds[ix]
        x1[x1.shape[0]//2, x1.shape[1]//2]=0
        test_params.append(x1)
    X_train_dies_param_v2.append(test_params)
wafer_tr_param_data = np.array(X_train_dies_param_v2)

print("Max Value training parametric data:", wafer_tr_param_data.max(), "Min Value training parametric data:",wafer_tr_param_data.min())

# Test set
X_test_dies_param_v2 = []
for test_param in X_test_dies_param_v1:
    test_params = []
    for x in test_param:
        x1 = x.copy()
        x1 = (x1/x1.max() - mus[ix])/stds[ix]
        x1[x1.shape[0]//2, x1.shape[1]//2]=0
        test_params.append(x1)
    X_test_dies_param_v2.append(test_params)

wafer_test_param_data = np.array(X_test_dies_param_v2)
print("Max Value test parametric data:", wafer_test_param_data.max(), "Min Value test parametric data:",wafer_test_param_data.min())
print()

y_train_ran_index = np.random.permutation(len(y_train))
wafer_tr_data = wafer_tr_data[y_train_ran_index]
wafer_tr_param_data = wafer_tr_param_data[y_train_ran_index]
wafer_tr_label = np.array(y_train)[y_train_ran_index]
wafer_tr_label = torch.from_numpy(wafer_tr_label).float()


wafer_tr_param_data, wafer_val_param_data, _, _ = train_test_split(wafer_tr_param_data, wafer_tr_label, test_size=0.10, stratify=wafer_tr_label, random_state=42)
wafer_tr_data, wafer_val_data, wafer_tr_label, wafer_val_label = train_test_split(wafer_tr_data, wafer_tr_label, test_size=0.10, stratify=wafer_tr_label, random_state=42)

wafer_val_label = wafer_val_label.float()
wafer_val_data = torch.from_numpy(wafer_val_data)
wafer_val_data = wafer_val_data[:, None, :, :]

wafer_val_param_data = torch.from_numpy(wafer_val_param_data)
print("Validation data :", wafer_val_data.shape, "Validation test parametric data :", wafer_val_param_data.shape)

wafer_tr_data = torch.from_numpy(wafer_tr_data)
wafer_test_data = torch.from_numpy(wafer_test_data)
wafer_tr_data = wafer_tr_data[:, None, :, :]
wafer_test_data = wafer_test_data[:, None, :, :]

wafer_tr_param_data = torch.from_numpy(wafer_tr_param_data)
wafer_test_param_data = torch.from_numpy(wafer_test_param_data)

print("Training data :", wafer_tr_data.shape, "Test data :", wafer_test_data.shape)
print("Training parametric data :", wafer_tr_param_data.shape, "Test parametric data :", wafer_test_param_data.shape)
print()

np.unique(y_train, return_counts=True), len(y_train)

np.unique(wafer_tr_label, return_counts=True), len(wafer_tr_label)

np.unique(wafer_test_label, return_counts=True), len(wafer_test_label)

cls_cnts = np.unique(wafer_tr_label, return_counts=True)[1]
cls_cnts

# train_loader, val_loader, test_loader = getLoaders(augmentation=False, batch_size=256, multiMode=True, data_path="Wafer_3x3_25_27_Exp1")

def test_accuracy(network, test_loader, criterion, device, set_name, sl_thresh=0.5):
    """
    args:
    network: Wafer2Spike, an SNN model
    test_loader: Data loader for test set
    criterion: Crossentropy Loss
    device: CPU or CUDA (GPU)
    set_name: Val or Test set

    """

    test_loss = 0
    correct = 0
    network.eval()
    for batch_id, (data, param_data, target) in enumerate(test_loader):
        data, param_data, target = data.float().to(device), param_data.float().to(device), target.to(device)
        output = network(data, param_data)
        loss = criterion(output, target)
        # loss = ( (-target*torch.log(output + 1e-6))*pos_weight + (target-1)*torch.log(1-output + 1e-6) ).mean()
        test_loss += loss.to('cpu').item()
        correct += sum(np.where(output.data.cpu().numpy() >= sl_thresh, 1, 0) == target.data.cpu().numpy() )

    print("{} loss: {:.6f} | {} accuracy: {:.6f}".format(set_name, test_loss / len(test_loader), set_name, correct / len(test_loader.dataset)))

def training(model, optimizer, criterion, device, sl_thresh=0.5, epochs=10, lr=0.0001, dataloaders=None, dropout_fc=0.3, hidden_neurons=128):

    """
    args:
    network: NN model
    epochs: Number of epochs for training
    lr: Learning rate
    dataloaders: Loading and preprocessing of data from a dataset into the training, validation, or testing pipelines
    numClasses: Number of classes
    dropout_fc: Dropout percentage for spiking-based fully connected layers

    """

    if len(dataloaders)==3:
        train_loader, val_loader, test_loader = dataloaders

    # Instantiating an optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)


    # Training
    for e in range(epochs):
        loss_per_epoch = 0
        correct = 0
        model.train()
        for i, data in enumerate(train_loader):
            wafer_data, param_data, label = data
            wafer_data, param_data, label = wafer_data.float().to(device), param_data.float().to(device), label.to(device)
            optimizer.zero_grad()
            output = model(wafer_data, param_data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.to('cpu').item()
            correct += sum(np.where(output.data.cpu().numpy() >= sl_thresh, 1, 0) == label.data.cpu().numpy() )

        # for param in decay_params:
        #     param.data = param.data.clamp(min=1e-7)

        print(f"Epoch: {e} | Training Loss: {loss_per_epoch / len(train_loader.dataset)} | Training accuracy : {correct / len(train_loader.dataset)}")

        if len(dataloaders)==3:
            test_accuracy(model, val_loader, criterion, device, "Validation", sl_thresh)
        elif len(dataloaders)==2:
            test_accuracy(model, test_loader, criterion, device, "Test", sl_thresh)


    pred = []
    y = []
    model.eval()
    for batch_id, (data, param_data, target) in enumerate(test_loader):
        data, param_data, target = data.float().to(device), param_data.float().to(device), target.to(device)
        pred += np.where(model(data, param_data).data.cpu().numpy() >= sl_thresh, 1, 0).tolist()
        y += target.data.cpu().numpy().tolist()

    pred = np.array(pred)
    y = np.array(y)
    # Confusion Matrix
    print()
    print("Confusion Matrix :")
    cf = confusion_matrix(y, pred)
    print(cf)
    print()
    print("Classification Report :")
    print(classification_report(y, pred))
    print()
    print("*"*25)

    # Defect_coverage: Ratio of defects covered by testing
    defect_coverage = 99.9/100 # Ratio of defects covered by testing
    # Number of negatives (negative dice): Dice testing bad
    N = np.where(y == 1)[0].shape[0]
    P = np.where(y == 0)[0].shape[0]
    test_escapes = ((1-defect_coverage)/defect_coverage) * N
    # NN: The number of the net-negative dice, i.e., the negative dice captured by Model with with SL > threshold
    index_1_y = np.where(np.array(y) == 1)[0]
    NN = (pred[index_1_y] == y[index_1_y]).sum()
    # Number of captured/covered test escapes
    CTE = (test_escapes/N)*NN
    # Yield Loss
    YL = cf[0][1].item()/cf.sum().item()

    """
    Rate for Return Merchandise Authorization (RMA) represents how many times of a single die's selling price equals the penalty of receiving a customer
    return. It implies the vendor would trade losing how many times of the die's test cost for removing one test escape. A GDBN model with a higher Gain is
    regarded as a better method.

    LOSS represents the total number of good dice discarded by the GDBN method.

    GAIN / Cost Reduction (CR) is the penalty to be paid minus the payment to be received if those test escapes are shipped to a customer, in terms of the
    times of a die's selling price.

    """

    RMA = 500
    LOSS = cf[0][1].item()
    GAIN = RMA * CTE - LOSS # CR
    DPPM = ((test_escapes - CTE)/ (P - LOSS - CTE)) * 10**6

    print("RMA: ", RMA)
    print("Defect_coverage :", defect_coverage*100, "%")
    print("Total number of negative dies :", N)
    print("Test Escapes :", np.round(test_escapes, 1))
    print("Covered Test Escapes :", CTE)
    print("Yield Loss :", YL)
    print("LOSS: ", LOSS)
    print("GAIN/CR: ", GAIN)
    print("DPPM: ", DPPM)
    print("*"*25)

class AttentionBasedCNN(nn.Module):
  def __init__(self, in_channels=1, numHeads= 2, numCls=2, out_channels=32, dropout_fc=0.3, hidden_neurons=128):
    super(AttentionBasedCNN, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.numCls = numCls
    self.dropout_fc = dropout_fc
    self.inter_channels = max(1, numHeads // 8)
    self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
    self.key_conv   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
    self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
    # self.gamma = nn.Parameter(torch.ones(1))

    self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)
    self.bn1 = nn.BatchNorm2d(self.out_channels)
    self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(self.out_channels)

    # To deal with Test parametric data
    self.query_conv_param = nn.Conv2d(7, self.inter_channels, kernel_size=1, bias=False)
    self.key_conv_param   = nn.Conv2d(7, self.inter_channels, kernel_size=1, bias=False)
    self.value_conv_param = nn.Conv2d(7, 7, kernel_size=1, bias=False)

    self.conv1_param = nn.Conv2d(7, self.out_channels, kernel_size=1, stride=1, groups=1)
    self.bn1_param = nn.BatchNorm2d(self.out_channels)
    self.conv2_param = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1)
    self.bn2_param = nn.BatchNorm2d(self.out_channels)

    self.fully_layers = nn.Sequential(nn.Linear(self.out_channels*3*3*2, hidden_neurons), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout_fc), \
                                      nn.Linear(hidden_neurons, numCls), nn.Sigmoid())

  def forward(self, x, x_param):
    batch_size, C, height, width = x.size()
    _, C_param, _, _ = x_param.size()
    # Generate query, key, and value projections
    proj_query = self.query_conv(x).view(batch_size, self.inter_channels, -1)  # (B, inter_channels, H*W)
    proj_query = proj_query.permute(0, 2, 1)  # (B, H*W, inter_channels) (64,9,4)
    proj_key   = self.key_conv(x).view(batch_size, self.inter_channels, -1)    # (B, inter_channels, H*W) (64,4,9)

    # Compute the dot-product attention map 36 pixel values
    energy = torch.bmm(proj_query, proj_key)  # (B, H*W, H*W) attention weight (64, 9, 9) pairwise relation
    attention = F.softmax(energy, dim=-1)     # Normalize along the last dimension

    # Project the values
    proj_value = self.value_conv(x).view(batch_size, C, -1)  # (B, C, H*W) (64, 1, 9 )

    # Apply attention: weighted sum of values
    out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, H*W) (9,9 . 9,1 => 9,1)
    out = out.view(batch_size, C, height, width)

    # Combine with the input (residual connection)
    # out = self.gamma * out + x
    out = F.relu(self.bn1(self.conv1(out)))
    out = F.relu(self.bn2(self.conv2(out)))


    # Test Paramteric Data
    proj_query_param = self.query_conv_param(x_param).view(batch_size, self.inter_channels, -1)
    proj_query_param = proj_query_param.permute(0, 2, 1)
    proj_key_param = self.key_conv_param(x_param).view(batch_size, self.inter_channels, -1)

    # Compute the dot-product attention map
    energy_param = torch.bmm(proj_query_param, proj_key_param)
    attention_param = F.softmax(energy_param, dim=-1)

    # Project the values
    proj_value_param = self.value_conv_param(x_param).view(batch_size, C_param, -1)

    # Apply attention: weighted sum of values
    out_param = torch.bmm(proj_value_param, attention_param.permute(0, 2, 1))  # (B, C, H*W)
    out_param = out_param.view(batch_size, C_param, height, width)

    out_param = F.relu(self.bn1_param(self.conv1_param(out_param)))
    out_param = F.relu(self.bn2_param(self.conv2_param(out_param)))

    out_param = out_param.view(-1, self.out_channels*3*3)


    # print("before", out.shape)
    out = out.view(-1, self.out_channels*3*3)
    # print("after", out.shape)
    out_concat = torch.cat((out, out_param), dim=1)
    out = self.fully_layers(out_concat)
    return out.flatten()

def seed_worker(worker_id):
    np.random.seed(seed_num)
    random.seed(seed_num)

batch_size = 256
dropout_fc = 0.35 # 0.25
hidden_neurons = 256
out_channels = 24
epochs = 14, 20, 23
lr = 0.001 # 0.001
sl_thresh = 0.5 # 0.5
model = AttentionBasedCNN(in_channels=1, numHeads=32, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

for w in model.state_dict().keys():
  print(w, model.state_dict()[w].shape)

# params = 0
# for p in model.parameters():
#   params += p.numel()
# params

# # Save the entire model
# torch.save(model, 'attention_based_multi_model.pth')



if __name__ == "__main__":

  for e in range(41, 7, -1):
    batch_size = 256
    dropout_fc = 0.35 # 0.25
    hidden_neurons = 256
    out_channels = 24
    epochs = e # 14, 20, 23
    lr = 0.001 # 0.001
    sl_thresh = 0.5 # 0.5

    # For reproducibility
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed_num)

    train_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_tr_data, wafer_tr_param_data, wafer_tr_label), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_val_data, wafer_val_param_data, wafer_val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_test_data, wafer_test_param_data, wafer_test_label), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([(cls_cnts[0]/cls_cnts[1]).item()]).to(device)  # Actual Training ratio of classes after splitting (764978/311269)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = AttentionBasedCNN(in_channels=1, numHeads=32, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training(model, optimizer, criterion, device, sl_thresh=sl_thresh, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

# dropout_fc = 0.28

if __name__ == "__main__":

  for e in range(41, 7, -1):
    batch_size = 256
    dropout_fc = 0.27 # 0.25
    hidden_neurons = 256
    out_channels = 24
    epochs = e # 14, 20, 23
    lr = 0.001 # 0.001
    sl_thresh = 0.5 # 0.5

    # For reproducibility
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed_num)

    train_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_tr_data, wafer_tr_param_data, wafer_tr_label), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_val_data, wafer_val_param_data, wafer_val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_test_data, wafer_test_param_data, wafer_test_label), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([(cls_cnts[0]/cls_cnts[1]).item()]).to(device)  # Actual Training ratio of classes after splitting (764978/311269)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = AttentionBasedCNN(in_channels=1, numHeads=32, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training(model, optimizer, criterion, device, sl_thresh=sl_thresh, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

if __name__ == "__main__":

  for e in range(35, 7, -1):
    batch_size = 256
    dropout_fc = 0.25 # 0.25
    hidden_neurons = 256
    out_channels = 24
    epochs = e # 14, 20, 23
    lr = 0.001 # 0.001
    sl_thresh = 0.5 # 0.5

    # For reproducibility
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed_num)

    train_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_tr_data, wafer_tr_param_data, wafer_tr_label), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_val_data, wafer_val_param_data, wafer_val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_test_data, wafer_test_param_data, wafer_test_label), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([(cls_cnts[0]/cls_cnts[1]).item()]).to(device)  # Actual Training ratio of classes after splitting (764978/311269)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = AttentionBasedCNN(in_channels=1, numHeads=32, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training(model, optimizer, criterion, device, sl_thresh=sl_thresh, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)


Epochs=12
dropout_fc = 0.1
hidden_neurons = 256
out_channels = 32
epochs = 20
lr = 0.001
network = AttentionBasedCNN(in_channels=1, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)
training(network, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

# Augmentation with applied 850231/345599
dropout_fc = 0.1
hidden_neurons = 256
out_channels = 32
epochs = 20
lr = 0.001
network = AttentionBasedCNN(in_channels=1, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)
training(network, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

42380/(211853+85930)

# Augmentation
dropout_fc = 0.1
hidden_neurons = 256
out_channels = 32
epochs = 20
lr = 0.001
network = AttentionBasedCNN(in_channels=1, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)
training(network, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)



if __name__ == "__main__":

  # for e in range(41, 7, -1):
    batch_size = 256
    dropout_fc = 0.35 # 0.25
    hidden_neurons = 256
    out_channels = 24
    epochs = 24 # 14, 20, 23
    lr = 0.001 # 0.001
    sl_thresh = 0.5 # 0.5

    # For reproducibility
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed_num)

    train_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_tr_data, wafer_tr_param_data, wafer_tr_label), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_val_data, wafer_val_param_data, wafer_val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset = Wafer_receptive_field_v1(wafer_test_data, wafer_test_param_data, wafer_test_label), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([(cls_cnts[0]/cls_cnts[1]).item()]).to(device)  # Actual Training ratio of classes after splitting (764978/311269)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = AttentionBasedCNN(in_channels=1, numHeads=32, numCls=1, out_channels=out_channels, dropout_fc=dropout_fc, hidden_neurons=hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training(model, optimizer, criterion, device, sl_thresh=sl_thresh, epochs=epochs, lr=lr, dataloaders=(train_loader, val_loader, test_loader), dropout_fc=dropout_fc, hidden_neurons=hidden_neurons)

