from __future__ import annotations
#postpones evaluation of type hints so annotations are stored as strings.

import torch
from torch.utils.data import Dataset
import os

class ModelNetCustomDataset(Dataset):
    def __init__(self,train:bool,classes=None):
        if train:
            x = torch.load("../data/ModelNet_subset/train_x.pt")
            y = torch.load("../data/ModelNet_subset/train_y.pt")
        else:
            x = torch.load("../data/ModelNet_subset/test_x.pt")
            y = torch.load("../data/ModelNet_subset/test_y.pt")

        # Load class names
        raw_dir = r"C:\Users\snevi\OneDrive\Documents\coding\AI\deep_learning\data\ModelNet\raw"
        
        self.class_names = sorted([
            name for name in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, name))
        ])

        # Filter classes FIRST, then standardize the filtered data
        if classes is not None:
            x, y = self.filter_classes(x, y, classes)
            # print(f"Filtered to {len(classes)} classes: {[self.class_names[c] for c in classes]}")
            # print(f"Dataset size after filtering: {x.shape[0]} objects")

        # Apply standardization to the (potentially filtered) data
        self.x = self.standardise(x)
        self.y = y
        

    def standardise(self,x):
        standardised_data = torch.zeros_like(x)

        for i in range(x.shape[0]):
            obj=x[i]  # (200, 3) - single object point cloud

            # Per-object mean centering (per coordinate)
            mean_x = obj.mean(dim=0)  # (3,) - centroid [mean_x, mean_y, mean_z]

            # Per-object scalar std (single value for all coordinates)
            std_x = obj.std()  # scalar - overall spread (single number)

            # Apply transformation
            standardised_data[i] = (obj - mean_x) / (std_x + 1e-8)

        return standardised_data

    def filter_classes(self,x,y,classes):
        indices_to_keep=[]
        for idx, cloud in enumerate(x):
            if y[idx].item() in classes:
                indices_to_keep.append(idx)

        x_selected = torch.index_select(x,dim=0,index=torch.tensor(indices_to_keep))
        y_selected = torch.index_select(y,dim=0,index=torch.tensor(indices_to_keep))
        return x_selected,y_selected
            

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        point_cloud = self.x[idx]
        class_id = self.y[idx]
        label = self.class_names[class_id.item()]
        
        return point_cloud, class_id, label