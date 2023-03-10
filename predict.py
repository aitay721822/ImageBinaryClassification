import argparse
import os
import torch
import imghdr
import shutil
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from dataset import transform_resize
from model import Model
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type=str, required=True)
parser.add_argument('-r', '--resize', type=int, default=224)
parser.add_argument('-l', '--load_path', type=str, default='model.pth')
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-s', '--save_dir', type=str, default='predict_result')

class OnlyImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = self.load_data(data_dir)
        self.transform = transform
        
    def load_data(self, data_dir):
        res = []
        for i, filename in enumerate(os.listdir(data_dir)):
            if imghdr.what(os.path.join(data_dir, filename)) is not None:
                res.append([i, os.path.join(data_dir, filename), 0])
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index][1]
        label = self.data[index][2]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        return img_path, img, label

def predict(data_dir, load_path, batch_size, resize, save_dir):
    # 載入圖片，並轉換為tensor
    dataset = OnlyImageDataset(data_dir, transform=transform_resize(resize))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 設置訓練裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型，載入模型參數
    model = Model()
    model.to(device)
    model.load_state_dict(torch.load(load_path))
    model.eval()
        
    # 測試模型
    with torch.no_grad():
        for img_path, img, label in tqdm(dataloader):
            img = img.to(device)
            
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            #print(f'Image Path: {img_path} Predicted: {"Like" if predicted[0] == 1 else "Dislike"}')
            if predicted[0] == 1:
                os.makedirs(save_dir, exist_ok=True)
                copy_path = os.path.join(save_dir, os.path.basename(img_path[0]))
                if not os.path.exists(copy_path):
                    shutil.copy(img_path[0], copy_path)

if __name__ == '__main__':
    args = parser.parse_args()
    predict(args.data_dir, args.load_path, args.batch_size, args.resize, args.save_dir)