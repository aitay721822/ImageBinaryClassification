import argparse
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageDataset, transform_resize
from sklearn.metrics import f1_score   
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, default='dataset')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-r', '--resize', type=int, default=224)
parser.add_argument('-s', '--save', type=str, default='model.pth')

def test(data_dir, device, model, batch_size, resize):
    dataset = ImageDataset(pathlib.Path(data_dir) / 'test' / 'test.csv', transform=transform_resize(resize))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        true, pred = [], []
        
        for data in tqdm(dataloader, desc="test"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true.extend(labels.tolist())
            pred.extend(predicted.tolist())
            
        print(f'[f1score: {f1_score(true, pred, average="binary")}][accuracy: {100 * correct / total}]')
        return f1_score(true, pred, average="binary")

def train(data_dir, learning_rate, epochs, batch_size, resize, save):
    dataset = ImageDataset(pathlib.Path(data_dir) / 'train' / 'train.csv', transform=transform_resize(resize))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 設置訓練裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 初始化模型、損失函數和優化器
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 將模型放到訓練裝置上
    model.to(device)

    # 訓練模型
    max_f1, stor_model = 0, None
    for epoch in range(epochs):
        # 進入訓練模式
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 9:    # 每10個batch輸出一次訓練狀態
                print(f'[epoch: {epoch}][batch: {i}][current loss: {loss.item()}][running loss: {running_loss / 10}]')
                running_loss = 0.0
                
        # 進入測試模式(評估模式)
        f1 = test(data_dir, device, model, batch_size, resize)
        if f1 > max_f1:
            max_f1 = f1
            stor_model = model.state_dict()

    torch.save(stor_model, save)
    print('Finished Training')
    

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.data_dir, args.learning_rate, args.epochs, args.batch_size, args.resize, args.save)