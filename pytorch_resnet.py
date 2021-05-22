##############このファイルの不安点##############################

#v2 に対して SEBlock を適応
#Dataset:FashionMnist なので自作のデータセットを利用

################################################################

##############パラメータ##########################################

#SEBlock をしようするなら "ON" 不使用なら "OFF"
SEBlock = "ON"

import torch.nn as nn
loss_function = nn.CrossEntropyLoss()

epochs = 100
fig_name = "SEBlock_ResNet50_Cifar100_CrossEntropyLoss.jpg"
out_dim = 100
learning_rate = 0.001

#NLLoss:1Dの際に利用

################################################################

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#VisualCordで実行する際、torch,torchvision,sklearnをpipでインストール
#Plus+α make_dotを利用することでモデルの可視化が可能

print("No Import Error")

class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        
        #Block1
        #ModuleListについて調べる
        self.block1_1 = self._building_block(256, channel_in=64)
        self.block1_2 = nn.ModuleList([
            self._building_block(256, channel_in=64, channel_ex_in=256) for _ in range(2)
        ])
        
        #Block2
        self.block2_1 = self._building_block(512, channel_in=128, channel_ex_in=256)
        self.block2_2 = nn.ModuleList([
            self._building_block(512, channel_in=128, channel_ex_in=512) for _ in range(3)
        ])
        
        #Block3
        self.block3_1 = self._building_block(1024, channel_in=256, channel_ex_in=512)
        self.block3_2 = nn.ModuleList([
            self._building_block(1024, channel_in=256, channel_ex_in=1024) for _ in range(5)
        ])
        
        #Block4
        self.block4_1 = self._building_block(2048, channel_in=512, channel_ex_in=1024)
        self.block4_2 = nn.ModuleList([
            self._building_block(2048, channel_in=512, channel_ex_in=2048) for _ in range(2)
        ])
        self.avg_pool = GlobalAvgPool2d()   #TODO: GlobalAvgPool2d
        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block1_1(h)
        for block in self.block1_2:
            h = block(h)
        h = self.block2_1(h)
        for block in self.block2_2:
            h = block(h)
        h = self.block3_1(h)
        for block in self.block3_2:
            h = block(h)
        h = self.block4_1(h)
        for block in self.block4_2:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)
        return y
        
    def _building_block(self, channel_out, channel_in, channel_ex_in=None):
        if channel_ex_in is None:
            channel_ex_in = channel_in
        return Block(channel_in, channel_out, channel_ex_in)
        

class Block(nn.Module):
    def __init__(self, channel_in, channel_out, channel_ex_in):
        super().__init__()
        
        #ResNet50なので 1x1 -> 3x3 -> 1x1 の畳み込みとなっている
        # 1x1 の畳み込み
        self.conv1 = nn.Conv2d(channel_ex_in, channel_in, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu1 = nn.ReLU()
        
        # 3x3 の畳み込み
        self.conv2 = nn.Conv2d(channel_in, channel_in, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel_in)
        self.relu2 = nn.ReLU()
                
        # 1x1 の畳み込み
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=(1,1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)

        #SEBlock
        if SEBlock is "ON":
            self.se = SELayer(channel_out)
        
        #SkipConnection用のチャネル数調整
        self.shortcut = self._shortcut(channel_ex_in, channel_out)
        
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)

        #SEBlock
        if SEBlock is "ON":
            h = self.se(h)

        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)    #SkipConnection
        return y
    
    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x
    
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1,1), padding=0)

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()

        self.sq = GlobalAvgPool2d()
        self.fc1 = nn.Linear(channel, channel//16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(channel//16, channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        h = self.sq(x).view(b, c)
        h = self.fc1(h)
        h = self.relu1(h)
        h = self.fc2(h)
        h = self.sig(h)
        y = h.view(b, c, 1, 1)

        return x * y.expand_as(x)
        
class GlobalAvgPool2d(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))

print("No Class Error")

if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    print("Start main")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:{}".format(device))
    
    #データセットの読み込み
    print("Start read Dataset")
    root = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_mnist")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = \
        torchvision.datasets.CIFAR100(root=root, download=True, train=True, transform=transform)
    mnist_test = \
        torchvision.datasets.CIFAR100(root=root, download=True, train=False, transform=transform)
    train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    #モデルの構築
    print("Start make model")
    print("SEBlock is {}".format(SEBlock))
    #最後の出力結果の数を代入
    model = ResNet50(out_dim).to(device)
    print(model)
    
    #モデルの学習・評価
    def compute_loss(label, pred):
        #criterionについて調べる
        return criterion(pred, label)
        
    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        #optimizerについて調べる
        optimizer.zero_grad()
        #lossについて調べる
        loss.backward()
        optimizer.step()
        return loss, preds
    
    def test_step(x, t):
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)
        return loss, preds
    
    print("Start learning & evaluate")
    criterion = loss_function
    optimizer = optimizers.Adam(model.parameters(lr=learning_rate))
    
    #Set Table
    fig, (Gloss, Gacc) = plt.subplots(ncols=2, figsize=(10, 4))
    Gacc.set_title("Accuracy")
    Gloss.set_title("Loss")

    graph_ON_x = []
    graph_ON_acc = []
    graph_ON_loss = []
    graph_OFF_x = []
    graph_OFF_acc = []
    graph_OFF_loss = []

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        test_loss = 0.
        test_acc = 0.

        graph_ON_x.append(epoch+1)

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = train_step(x, t)
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        for (x, t) in test_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t)
            test_loss += loss.item()
            test_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        graph_ON_loss.append(test_loss)
        graph_ON_acc.append(test_acc)

        #formatは{0}のようにしなくてもいい
        #プログレスバーを表示するようにすべき
        print("Epoch: {},Training Cost: {:.3f}, Training Acc: {:.3f}, Valid Cost: {:.3f}, Valid Acc: {:.3f}".format(epoch+1, train_loss, train_acc, test_loss, test_acc))

    #モデルの構築
    SEBlock = "OFF"
    print("Start make model")
    print("SEBlock is {}".format(SEBlock))
    #最後の出力結果の数を代入
    model = ResNet50(out_dim).to(device)
    print(model)

    print("Start learning & evaluate")
    criterion = loss_function
    optimizer = optimizers.Adam(model.parameters(lr=learning_rate))

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        test_loss = 0.
        test_acc = 0.

        graph_OFF_x.append(epoch+1)

        for (x, t) in train_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = train_step(x, t)
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        
        for (x, t) in test_dataloader:
            x, t = x.to(device), t.to(device)
            loss, preds = test_step(x, t)
            test_loss += loss.item()
            test_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        graph_OFF_loss.append(test_loss)
        graph_OFF_acc.append(test_acc)

        #formatは{0}のようにしなくてもいい
        print("Epoch: {},Training Cost: {:.3f}, Training Acc: {:.3f}, Valid Cost: {:.3f}, Valid Acc: {:.3f}".format(epoch+1, train_loss, train_acc, test_loss, test_acc))

    #plot
    Gacc.plot(graph_ON_x, graph_ON_acc, color="red", label="Use SEBlock")
    Gloss.plot(graph_ON_x, graph_ON_loss, color="red", label="Use SEBlock")

    Gacc.plot(graph_OFF_x, graph_OFF_acc, color="blue", label="Not use SEBlock")
    Gloss.plot(graph_OFF_x, graph_OFF_loss, color="blue", label="Not use SEBlock")

    Gacc.legend(loc=0)
    Gloss.legend(loc=0)

    fig.savefig(fig_name)
