from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def transform_resize(pixels):
    return transforms.Compose([
        transforms.Resize((pixels, pixels)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = self.load_data(data_dir)
        self.transform = transform
        
    def load_data(self, data_dir):
        data = []
        with open(data_dir, 'r', encoding='utf8') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                line = line.strip('\n').split(',')
                data.append([line[0], line[1], int(line[2])])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index][1]
        label = self.data[index][2]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

if __name__ == '__main__':
    dataset = ImageDataset('dataset/test/test.csv', transform_resize(512))
    for img, label in dataset:
        try:
            print(img.shape, label)
        except Exception as e:
            print(e)
            print(img, label)