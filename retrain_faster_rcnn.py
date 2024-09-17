from pathlib import Path
import argparse
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluators
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder:
        Path(output_folder).mkdir(exist_ok=True, parents=True)
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class StomaTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)


def create_train_parser():
    my_parser = argparse.ArgumentParser(
        description='Script used for training a model different hyperparameter sets')
    my_parser.add_argument('--bs', type=int, help='Number patches per batch')
    my_parser.add_argument('--roi', type=int, help='Number of ROI heads')
    my_parser.add_argument('--lr', type=float, help='Learning rate')
    my_parser.add_argument('--max_iter', type=int,
                           help='Maximal number of iterations')
    args = my_parser.parse_args()
    return args


if __name__ == "__main__":
    register_coco_instances(f"Stoma_train", {}, f"./data/splits/train.json", f"./data/images/")
    register_coco_instances(f"Stoma_valid", {}, f"./data/splits/val.json", f"./data/images/")
    register_coco_instances(f"Stoma_test", {}, f"./data/splits/test.json", f"./data/images/")
    args = create_train_parser()
    cfg = get_cfg()
    cfg.SEED = 42
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("Stoma_train", "Stoma_valid")
    cfg.DATASETS.TEST = ("Stoma_test",)
    cfg.MODEL.PIXEL_MEAN=[0.8020,0.8020,0.8020]
    cfg.MODEL.PIXEL_STD=[0.1936, 0.1936, 0.1936]
    cfg.TEST.EVAL_PERIOD = 200
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = args.bs
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.AMP.ENABLED = True
    cfg.CUDNN_BENCHMARK = True
    cfg.VIS_PERIOD = 200

    run_name = f"trainval_{cfg.SOLVER.BASE_LR}_{cfg.SOLVER.IMS_PER_BATCH}_{cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}_{cfg.SOLVER.MAX_ITER}"
    cfg.OUTPUT_DIR = f"./output/{run_name}/"
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    with open(f"{cfg.OUTPUT_DIR}mycfg.yaml", "w") as f:
        f.write(cfg.dump())
    trainer = StomaTrainer(cfg)
    trainer.build_evaluator(cfg, f"Stoma_test", output_folder=f"./output/{run_name}")
    trainer.resume_or_load(resume=False)
    trainer.train()
