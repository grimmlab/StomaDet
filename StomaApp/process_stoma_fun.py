from pathlib import Path
import zipfile
import skimage.io as skio
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from sql_fun import select_stoma_summary_details, update_stoma_summary, insert_processed_zip_entry, \
    insert_stoma_details_entry, insert_stoma_summary_entry
import argparse
from skimage import color
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
import numpy as np


def create_app_parser():
    # TODO fix argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_db", type=bool, default=False, help="Delete the db and create a new one")
    return parser.parse_known_args()

class Metadata:
    def get(self, _):
        return ['stoma']


def run_inference(model_path, image):
    cfg = get_cfg()
    cfg.SEED = 42
    cfg.merge_from_file("./models/stoma_detector.yaml")
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.PIXEL_MEAN = [0.8020,0.8020,0.8020]
    cfg.MODEL.PIXEL_STD = [0.1936, 0.1936, 0.1936]
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.SOLVER.AMP.ENABLED = True
    cfg.CUDNN_BENCHMARK = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image[:, :, :3])
    outputs = outputs["instances"].to("cpu")
    detail_entries = []
    for idx, (bbox, score) in enumerate(zip(outputs.pred_boxes, outputs.scores)):
        # xmin,ymin,xmax,ymax
        area = (bbox.numpy()[2] - bbox.numpy()[0]) * (bbox.numpy()[3] - bbox.numpy()[1])
        score = int(score.numpy().item() * 100)
        detail_entry = (
            idx + 1, int(area), int(bbox.numpy()[0]), int(bbox.numpy()[1]), int(bbox.numpy()[2]), int(bbox.numpy()[3]),
            score)
        detail_entries.append(detail_entry)
    return detail_entries


def process_zip_file(zip_file_p: Path, db_file_path: Path, progress):
    model_path = Path("./models/stoma_detector.pth")
    with zipfile.ZipFile(zip_file_p, mode="r") as archive:
        images_in_zip = [a for a in archive.infolist() if a.filename.endswith(".png")]
        last_row_zip = insert_processed_zip_entry(db_file_path, str(zip_file_p))
        for file in progress.tqdm(images_in_zip, total=len(images_in_zip), desc="Processing zip file"):
            if file.filename.endswith("png"):
                print(f"Processing {file.filename}")
                img_name = archive.open(file.filename)
                img = skio.imread(img_name)
                if len(img.shape) == 3:
                    img = color.rgb2gray(img[:, :, :3])
                img = img_as_ubyte(img)
                img = rescale_intensity(img)
                img = np.dstack((img, img, img))
                stoma_details = run_inference(model_path, img)
                last_row_summary = insert_stoma_summary_entry(db_file_path, last_row_zip, (file.filename, len(stoma_details)))
                last_row_details = insert_stoma_details_entry(db_file_path, last_row_summary, stoma_details)
    return last_row_details


def filter_stoma_summary(db_file_path: Path, confidence_score=50):
    def remove_small_stoma(df, min_area_to_keep):
        df.drop(df[df.area < min_area_to_keep].index, inplace=True)
        return

    df = select_stoma_summary_details(db_file_path)
    if len(df) > 0:
        df = df.loc[df.conf_score_per_cent > confidence_score]
        stoma_summary_update_values = []
        filtered_grps = []
        for file_id, grp in df.groupby("file_id"):  # going through image one by one
            remove_small_stoma(grp, min_area_to_keep=grp.area.mean() // 2)  # remove stomata that are smaller than half of mean area
            filtered_grps.append(grp)
            stoma_summary_update_values.append((len(grp), file_id))
        update_stoma_summary(db_file_path, stoma_summary_update_values)
    return df


def save_all_data_as_csv(db_file_path: Path, save_file_path: Path, select_function, suffix):
    df = select_function(db_file_path)
    if len(df) > 0:
        print(f"Saving to {str(save_file_path)}...")
        df.to_csv(f"{str(save_file_path)}/{suffix}.csv", index=False)
    else:
        print(f"No data found to save...")
    return f"{str(save_file_path)}/{suffix}.csv"
