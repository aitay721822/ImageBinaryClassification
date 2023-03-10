import os
import random
import sys
import pandas as pd
import imghdr
import argparse
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

parser = argparse.ArgumentParser(description='Build dataset')
parser.add_argument('-d', '--image_dir', type=str, default='images', help='Directory containing images')
parser.add_argument('-n', '--num_images', type=int, default=100, help='Number of images to select')
parser.add_argument('-s', '--storage_path', type=str, default='dataset.csv', help='Path to store dataset')

class ImageLabel(QWidget):
    def __init__(self, image_path, num_images, storage_path):
        super().__init__()
        
        # Initialize variables
        self.storage_path = storage_path
        self.image_paths = []
        self.current_image_path = ""
        self.current_index = -1
        self.num_images = num_images # Number of images to select
        self.likes = []

        # Get image paths from directory
        self.image_dir = image_path
        self.image_paths = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir) if imghdr.what(os.path.join(self.image_dir, filename)) is not None]

        # Shuffle image paths
        random.shuffle(self.image_paths)

        # Create GUI elements
        self.index_label = QLabel('0/0') 
        self.image_label = QLabel()
        self.like_button = QPushButton("Like")
        self.dislike_button = QPushButton("Dislike")

        # Connect button signals to slots
        self.like_button.clicked.connect(self.like_button_clicked)
        self.dislike_button.clicked.connect(self.dislike_button_clicked)

        # Create layout and add GUI elements
        layout = QVBoxLayout()
        layout.addWidget(self.index_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.like_button)
        layout.addWidget(self.dislike_button)

        # Set layout
        self.setLayout(layout)

        # Show first image
        self.show_next_image()

    def show_next_image(self):
        # Check if there are more images
        if self.current_index + 1 >= len(self.image_paths) or len(self.likes) >= self.num_images:
            self.save_csv()
            self.close()
            return

        # Increment index and get next image path
        self.index_label.setText(f"{self.current_index + 2}/{len(self.image_paths)}")
        self.current_index += 1
        self.current_image_path = self.image_paths[self.current_index]

        # Load image and display in label
        pixmap = QPixmap(self.current_image_path)
        self.image_label.setPixmap(pixmap.scaled(512, 512, Qt.KeepAspectRatio))

    def like_button_clicked(self):
        self.likes.append([self.current_index, self.current_image_path, 1])
        self.show_next_image()

    def dislike_button_clicked(self):
        self.likes.append([self.current_index, self.current_image_path, 0])
        self.show_next_image()

    def save_csv(self):
        # Create dataframe and save to csv
        if os.path.exists(self.storage_path):
            df = pd.read_csv(self.storage_path)
            df = df.append(pd.DataFrame(self.likes, columns=["id", "image_path", "like"]))
        else:
            df = pd.DataFrame(self.likes, columns=["id", "image_path", "like"])
        df.to_csv(self.storage_path, index=False, encoding='utf8')

if __name__ == "__main__":
    arg = parser.parse_args()
    
    if not os.path.exists(arg.image_dir):
        print("Image directory does not exist")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    label = ImageLabel(arg.image_dir, arg.num_images, arg.storage_path)
    label.show()
    app.exec_()
