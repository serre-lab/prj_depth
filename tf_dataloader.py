import pickle
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

class DepthDataset(tf.data.Dataset):
    def __init__(self, image_dir, depth_detail_path):
        super(DepthDataset, self).__init__()

        self.image_dir = image_dir
        
        # Load depth details
        with open(depth_detail_path, 'rb') as f:
            depth_details = pickle.load(f)
            
        self.depth_diff_arr = depth_details['depth_diff_arr']
        self.lower_depth_range = depth_details['lower_depth_range']
        self.upper_depth_range = depth_details['upper_depth_range']

        print("Upper Depth Range:", self.upper_depth_range[0], "to", self.upper_depth_range[1])
        print("Lower Depth Range:", self.lower_depth_range[0], "to", self.lower_depth_range[1])
        
        # Filter image indices based on depth_diff range
        self.valid_indices = [i for i, diff in enumerate(self.depth_diff_arr) 
                              if self.lower_depth_range[0] <= diff <= self.lower_depth_range[1] or 
                                 self.upper_depth_range[0] <= diff <= self.upper_depth_range[1]]
        
        self.transform = layers.experimental.preprocessing.Rescaling(1./255)
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        img_path = f"{self.image_dir}/{img_idx}.png"
        img = tf.image.decode_image(tf.io.read_file(img_path))
        img = self.transform(img)
        
        depth_diff = self.depth_diff_arr[img_idx]
        if self.lower_depth_range[0] <= depth_diff <= self.lower_depth_range[1]:
            label = tf.constant(0, dtype=tf.int64)
        else:
            label = tf.constant(1, dtype=tf.int64)
        
        return img, label
