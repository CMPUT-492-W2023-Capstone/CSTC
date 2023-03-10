from enum import Enum
from pathlib import Path

import torch

from models.models import Darknet
from utils.datasets import LoadImages, LoadStreams
from utils.torch_utils import load_classifier, select_device

from yolov8.ultralytics.yolo.data.utils import VID_FORMATS
from yolov8.ultralytics.yolo.utils.checks import check_file
from yolov8.ultralytics.yolo.utils.files import increment_path

APP_PATH = Path(__file__).resolve().parents[0].parents[0]
MODELS_PATH = APP_PATH / 'models'
TRACKING_CONFIG = APP_PATH / 'tracking_configs'
VALID_URLs = ('rtsp://', 'rtmp://', 'https://')


def _is_valid_file(source) -> bool:
    return Path(source).suffix[1:] in VID_FORMATS


def _is_valid_url(source) -> bool:
    return source.lower().startswith(VALID_URLs)


def _is_valid_webcam(source) -> bool:
    return source.isnumeric() or source.endswith('.txt')


class ComputingDevice(Enum):
    CUDA_0 = 0
    CUDA_1 = 1
    CUDA_2 = 2
    CUDA_3 = 3
    CPU = 'CPU'


class TrackingMethod(Enum):
    BYTETRACK = 'bytetrack'
    OCSORT = 'ocsort'
    STRONGSORT = 'strongsort'


class InputConfig:

    def __init__(
            self,
            device: str | int = ComputingDevice.CPU.value,
            reid_models: Path = MODELS_PATH / 'osnet_x0_25_msmt17.pt',
            media_source: str = '0',
            yolo_config: str = 'cfg/yolor_p6.cfg',
            yolo_models: Path = MODELS_PATH / 'yolor_p6.pt'
    ):
        self.device = select_device(device)

        self.media_source = check_file(media_source) \
            if _is_valid_file(media_source) and _is_valid_url(media_source) else media_source

        self.reid_models = reid_models
        self.yolo_config = yolo_config
        self.yolo_models = yolo_models

        self.webcam_enable = _is_valid_webcam(media_source) or \
            (_is_valid_url(media_source) and not _is_valid_file(media_source))
        self.segmentation = self.yolo_models.name.endswith('-seg')

    def load_dataset_model(self, inference_img_size, fp16=False):
        model = Darknet(self.yolo_config, inference_img_size).cuda()
        model.load_state_dict(torch.load(self.yolo_models, map_location=self.device)['model'])
        model.to(self.device).eval()
        if fp16:
            model.half()

        media_dataset = LoadStreams(
            self.media_source, img_size=inference_img_size
        ) if self.webcam_enable else LoadImages(
            self.media_source,
            img_size=inference_img_size,
            auto_size=64
        )

        dataset_size = len(media_dataset) if self.webcam_enable else 1

        return media_dataset, dataset_size, model

    def load_classifier(self):
        model = load_classifier(name='resnet101', n=2)
        model.load_state_dict(
            torch.load(
                str(MODELS_PATH / 'resnet101.pt'),
                map_location=self.device)['model']
        )
        model.to(self.device).eval()

        return model


class OutputVideoConfig:

    def __init__(
            self,
            line_thickness: int = 2,
            hide_conf: bool = False,
            hide_class: bool = False,
            hide_labels: bool = False,
            show_video: bool = False,
            vid_frame_stride: int = 1,  # number of frame will skip per second
            retina_masks: bool = False,
            class_names: str = 'data/coco.names'
    ):
        self.line_thickness = line_thickness
        self.hide_conf = hide_conf
        self.hide_class = hide_class
        self.hide_labels = hide_labels
        self.show_video = show_video
        self.vid_frame_stride = vid_frame_stride
        self.retina_masks = retina_masks

        # Loads *.names file at 'path'
        with open(class_names, 'r') as f:
            names = f.read().split('\n')
            self.class_names = list(filter(None, names))  # filter removes empty strings (such as last line)


class OutputResultConfig:

    def __init__(
            self,
            no_save: bool = False,
            save_confid: bool = False,
            save_crop: bool = False,
            save_directory: Path = APP_PATH / 'output',
            save_existed: bool = False,
            save_name: str = 'exp',
            save_project_exist: bool = False,
            save_project_path: Path = APP_PATH / 'runs' / 'track',
            save_text: bool = False,
            save_traj: bool = False,
            save_video: bool = False,
            visualization: bool = False
    ):
        self.no_save = no_save
        self.save_confid = save_confid
        self.save_crop = save_crop
        self.save_directory = save_directory
        self.save_existed = save_existed
        self.save_name = save_name
        self.save_project_exist = save_project_exist
        self.save_project_path = save_project_path
        self.save_text = save_text
        self.save_traj = save_traj
        self.save_video = save_video
        self.visualization = visualization
        self.export_name = 'ensemble'

    def configure_save_location(self, yolo_models: Path | list[Path]):
        if type(yolo_models) is not list:
            self.export_name = yolo_models.stem
        elif len(yolo_models) == 1:
            self.export_name = Path(yolo_models[0]).stem

        save_dir = increment_path(
            Path(self.save_project_path) / self.export_name,
            exist_ok=self.save_project_exist
        )

        return save_dir


class AlgorithmConfig:

    def __init__(
            self,
            agnostic_nms: bool = False,
            augment: bool = False,
            class_filter: list[int] = None,
            conf_thres: float = 0.25,
            device: ComputingDevice = ComputingDevice.CPU,
            dnn: bool = False,
            fp16: bool = False,
            inference_img_size: list[int] = None,
            iou_thres: float = 0.5,
            max_det: int = 1000,
            tracking_method: TrackingMethod = TrackingMethod.BYTETRACK,
            tracking_config: Path = TRACKING_CONFIG / 'bytetrack.yaml',
    ):
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.class_filter = class_filter
        self.conf_thres = conf_thres
        self.device = device
        self.dnn = dnn
        self.fp16 = fp16
        self.iou_thres = iou_thres
        self.max_det = max_det

        if inference_img_size is None or len(inference_img_size) > 2 or len(inference_img_size) < 1:
            self.inference_img_size = [640, 640]
        else:
            self.inference_img_size = inference_img_size if len(inference_img_size) == 2 else 2 * inference_img_size

        self.tracking_method = tracking_method.value
        self.tracking_config = tracking_config
