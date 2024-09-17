import pandas as pd
import numpy as np
from pathlib import Path
import skimage.draw as skdraw
from skimage import io as skio
import json
import skimage.morphology as morph
from torch.utils.data import DataLoader, Dataset
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity


class COCODetection(Dataset):
    def __init__(self, img_folder, annotate_file):
        self.img_folder = img_folder
        self.annotate_file = annotate_file
        self.gt = []
        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"],img["width"])
            if img_id in self.images: raise Exception("duplicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))
            img_name = self.images[img_id][0]
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2]
            height = bbox[3]
            x_max = x_min + width
            y_max = y_min + height
            row = [img_name,x_min, y_min, x_max, y_max]
            self.gt.append(row)

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.gt = pd.DataFrame(self.gt, columns=["file_path","xmin","ymin","xmax","ymax"])


def filter_stoma_summary(df, confidence_score: int):
    df = df.loc[df.conf_score_per_cent > confidence_score]
    filtered_grps = []
    for file_id, grp in df.groupby("file_path"):  # going through image one by one
        filtered_grps.append(grp)
    details = pd.concat(filtered_grps)
    return details


def draw_preds_and_gt(img, preds, gt, initial_image_shape, fname, dilation_radius=0):
    init_image = np.zeros((img.shape[0] + 100, img.shape[1] + 100, 3), dtype="uint8")
    init_image[:img.shape[0], :img.shape[1], :] = img.copy()
    blank = np.zeros((img.shape[0] + 100, img.shape[1] + 100), dtype="uint8")
    for idx, pred in gt.iterrows():
        rr, cc = skdraw.rectangle_perimeter(start=(pred.xmin, pred.ymin),
                                            end=(pred.xmax - 2 - dilation_radius, pred.ymax - 2 - dilation_radius))
        blank[cc, rr] = 128

    dilated = morph.dilation(blank, morph.disk(radius=dilation_radius))
    init_image[dilated == 128] = (255, 0, 234)  # GT: pink

    for idx, pred in preds.iterrows():
        rr, cc = skdraw.rectangle_perimeter(start=(pred.xmin, pred.ymin),
                                                end=(pred.xmax - 2 - dilation_radius, pred.ymax - 2 - dilation_radius))
        blank[cc, rr] = 255
    dilated = morph.dilation(blank, morph.disk(radius=dilation_radius))
    init_image[dilated == 255] = (234, 255, 0)  # PRED: yellow
    img2 = init_image[:initial_image_shape[0], :initial_image_shape[1], :]
    skio.imsave(fname, img2, check_contrast=False)

    return


if __name__ == "__main__":
    annotations_file = Path("./data/splits/test.json")
    file_root = Path("./data/images")
    details_csv_path = Path("./data/predictions/test_set_details.csv")
    substring = "only_test_Set"
    save_path = Path("./output/predictions")

    stoma = COCODetection(file_root, annotations_file)
    dets = pd.read_csv(details_csv_path,
                       dtype={'area': np.uint16, 'xmin': np.uint16, 'ymin': np.uint16, 'xmax': np.uint16,
                              'ymax': np.uint16, 'conf_score_per_cent': np.uint8})
    dets = dets.loc[dets.file_path.str.contains(f"{substring}")]
    print(f"loaded {len(pd.unique(stoma.gt.file_path))} images.")
    print(f"Saving to {save_path}...")
    for file_name in file_root.glob("*.png"):

        file_name = file_name.stem
        filtered_grps = filter_stoma_summary(dets, 50)
        all_preds_per_image = filtered_grps.loc[filtered_grps.file_path.str.contains(str(file_name))]
        if len(all_preds_per_image) > 0:
            print(f"Processing {file_name}...")
            img = skio.imread(f"{str(file_root)}/{file_name}.png", as_gray=True)
            img = img_as_ubyte(img)
            img = rescale_intensity(img)
            img = np.dstack((img, img, img))
            save_sub_path = save_path
            save_sub_path.mkdir(exist_ok=True, parents=True)
            gt = stoma.gt.loc[stoma.gt.file_path.str.contains(file_name)]
            draw_preds_and_gt(img, all_preds_per_image, gt, img.shape, fname=f"{save_sub_path}/pred_{file_name}.png", dilation_radius=4)