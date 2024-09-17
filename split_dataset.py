from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import torch
from skimage import io as skio
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def seed_all(seed):
    '''
    sets the initial seed for numpy and pytorch to get reproducible results.
    One still need to restart the kernel to get reproducible results, as discussed in:
    https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def stratify(df):
    Path("./data/splits").mkdir(exist_ok=True, parents=True)
    trainval, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[["image_type"]])
    train, val = train_test_split(trainval, test_size=0.25, random_state=42, stratify=trainval[["image_type"]])
    train = train.sort_values(by="file_name")
    val = val.sort_values(by="file_name")
    test = test.sort_values(by="file_name")
    print(f"training samples: {len(train)}")
    print(f"validation samples: {len(val)}")
    print(f"test samples: {len(test)}")
    train["file_name"].to_csv(f"./data/splits/train.csv", index=False)
    val["file_name"].to_csv(f"./data/splits/val.csv", index=False)
    test["file_name"].to_csv(f"./data/splits/test.csv", index=False)
    return

class COCOImage:
    def __init__(self, image_id, width, height, file_name):
        self.id = image_id
        self.width = width
        self.height = height
        self.file_name = file_name

class COCOImages:
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)
    
    def remove_files_without_annotations(self, annotations):
        
        ann_img_ids = [ann.image_id for ann in annotations]
        ann_img_ids = set(ann_img_ids)
        all_img_ids = set([ci.id for ci in self.images])
        
        missing_ids = all_img_ids - ann_img_ids
        print(missing_ids)
        return
        

    def get_id_by_file_name(self, file_name):
        candidates = [image.id for image in self.images if image.file_name == file_name]
        assert len(candidates) == 1 , f"Found {len(candidates)} for filename {file_name}"
        return candidates[0]

    def get_info(self):
        """
        get as list of dicts
        :return: 
        """
        info = []
        for image in self.images:
            info_image = {"id": image.id, "width": image.width, "height": image.height, "file_name": image.file_name}
            info.append(info_image)
            
        return info

class COCOCategories:
    def __init__(self, cat:list):
        cats = []
        for idx, category in enumerate(cat,start=1):
            cat_dict = {"id": idx, "name": category}
            cats.append(cat_dict)
        self.cats = cats

    def get_category_id(self, label):
        ids = [cat["id"] for cat in self.cats if cat["name"]==label]
        assert len(ids) == 1, f"Indices greater than 1: {ids}"
        return ids[0]
    
    def remove_category_ids(self, labels: list):
        removed_cat_ids = []
        idx_to_remove = []
        for label in labels:
            for idx, cat in enumerate(self.cats):
                if cat["name"] == label:
                    removed_cat_id = self.cats[idx]["id"]
                    idx_to_remove.append(idx)
                    removed_cat_ids.append(removed_cat_id)
                    
        self.cats = [cat for cat in self.cats if cat["id"] not in removed_cat_ids]
        return removed_cat_ids
                

class COCOAnnotation:
    def __init__(self, id, image_id, category_id, area, bbox, iscrowd):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd

class COCOAnnotations:
    def __init__(self, annotations):
        self.annotations = annotations
    
    def remove_annotations_by_category_id(self, category_id):
        idx_to_remove = []
        for idx, annotation in enumerate(self.annotations):
            if annotation.category_id == category_id:
                idx_to_remove.append(annotation.id)
        self.annotations = [ann for ann in self.annotations if ann.id not in idx_to_remove]
        return

    def get_info(self):
        """
        get as list of dicts
        :return: 
        """
        info = []
        for annotation in self.annotations:
            info_annotation = {"id": annotation.id, "image_id": annotation.image_id, "category_id": annotation.category_id, "area": annotation.area, "bbox": annotation.bbox, "iscrowd": annotation.iscrowd}
            info.append(info_annotation)

        return info
        
        
def load_COCO_annotations(annotation_path, ci):
    annotations = []
    annotation_id = 1
    for file_name in [info["file_name"] for info in ci.get_info()]:
        df = pd.read_csv(f"{annotation_path}/{file_name.replace('.png', '.csv')}")
        image_id = ci.get_id_by_file_name(file_name)
        for idx, row in df.iterrows():
            width = row["xmax"] - row["xmin"]
            height = row["ymax"] - row["ymin"]
            bbox = [row["xmin"], row["ymin"], width, height]
            area = width*height
            category_id = 1
            annotation = COCOAnnotation(id=annotation_id, image_id=image_id, category_id=category_id, area=area, bbox=bbox, iscrowd=0)
            annotations.append(annotation)
            annotation_id +=1
    ca = COCOAnnotations(annotations)
    return ca

def load_COCO_images(image_file_path, df, suffix):
    cocoimages = []
    for image_id, file_path in enumerate(sorted(list(image_file_path.glob(f"*.{suffix}")))):
        if file_path.stem in list(df['file_name']):  # check if the file is actually in the list
            ann = skio.imread(file_path, as_gray=True)
            cocoimage = COCOImage(image_id=image_id+1, width= ann.shape[1], height=ann.shape[0], file_name=f"{file_path.stem}.{suffix}")
            cocoimages.append(cocoimage)
    ci = COCOImages(cocoimages)
    return ci

def load_COCO_labels():
    cl = COCOCategories(["Stoma"])
    return cl

def save_COCO(save_path, cl, ci, ca):
    a = {"categories":cl.cats, "images":ci.get_info(), "annotations": ca.get_info()}
    json_object = json.dumps(a, cls=NpEncoder)
    with open(f"{save_path}", "w") as outfile:
        outfile.write(json_object)
        
        






if __name__ == "__main__":
    image_file_path = Path("./data/images")
    annotation_path = Path("./data/annotations")
    seed_all(42)
    strat_file = Path("./data/dataset.csv", dtype=str)
    df = pd.read_csv(strat_file)
    stratify(df)
    
    print("Saving splits in coco format...")
    for split_name in ["train", "val", "test"]:
        split = pd.read_csv(f"./data/splits/{split_name}.csv", dtype=str)
        ci = load_COCO_images(image_file_path, split, "png")
        cl = load_COCO_labels()
        ca = load_COCO_annotations(annotation_path, ci)
        print(f"{split_name}: {len(ca.annotations)} annotations found")
        save_path = Path("./data") / "splits" / f"{split_name}.json" 
        save_path.parent.mkdir(exist_ok=True)
        save_COCO(save_path, cl, ci, ca)



