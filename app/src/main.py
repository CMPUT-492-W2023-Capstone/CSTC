import platform
from jsonargparse import CLI
from pathlib import Path

import cv2
import torch

from utils.general import scale_coords

from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors

from algorithm import DetectionTask, TrackingTask
from configuration import InputConfig, OutputVideoConfig, OutputResultConfig, AlgorithmConfig
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class TrackedObject:

    @staticmethod
    def format(t_id, vehicle_type, confi, bbox):
        coord = f'x1: {bbox[0]}, x2: {bbox[2]}, y1: {bbox[1]}, y2: {bbox[3]}'

        return f'Vehicle id {t_id} : type: {vehicle_type} | confidence: ' \
               f'{round(confi)} | ' f'{coord}'

    def __init__(self, t_id: int = -1, vehicle_type: str = '', confidence: float = -1, bbox: list = None):
        self.t_id = t_id
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.bbox = bbox

    def update(self, vehicle_type, bbox):
        self.vehicle_type = vehicle_type
        self.bbox = bbox

    def __hash__(self):
        return self.t_id

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and getattr(other, 't_id', None) == self.t_id \
               and getattr(other, 'vehicle_type') == self.vehicle_type

    def __str__(self):
        return self.format(self.t_id, self.vehicle_type, self.confidence, self.bbox)

    def __repr__(self):
        return self.__str__()


def stream_result(box_annotator: Annotator, im0, source_path: Path,
                  stream_windows: list[Path]):
    im0 = box_annotator.result()

    if platform.system() == 'Linux' and source_path not in stream_windows:
        stream_windows.append(source_path)

        cv2.namedWindow(str(source_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(source_path), im0.shape[1], im0.shape[0])

    cv2.imshow(str(source_path), im0)

    if cv2.waitKey(1) == ord('q'):
        exit()


def label_annotator(t_id, vehicle_type, confidence, bbox, box_annotator: Annotator,
                    config: OutputVideoConfig):
    label = None
    if not config.hide_labels:
        class_label = None if config.hide_class else f'{vehicle_type}'
        confidence_label = None if config.hide_conf else f'{confidence:.2f}'

        label = f'{t_id} {class_label} {confidence_label}'

    box_annotator.box_label(bbox, label, color=colors(1, True))

    return box_annotator


def main(
        input_config: InputConfig,
        output_video_config: OutputVideoConfig,
        output_result_config: OutputResultConfig,
        algorithm_config: AlgorithmConfig
):
    # save_dir = save_options.configure_save(input_options.yolo_models)

    # load video frames (media dataset) and AI model
    media_dataset, media_dataset_size, model = input_config.load_dataset_model(
        algorithm_config.inference_img_size,
        fp16=algorithm_config.fp16
    )

    # names of the classification defined in the AI model
    class_names = output_video_config.class_names

    # save_video_paths, video_writes, save_txt_paths = [[None] * dataset_size for i in range(3)]

    seen_obj: int = 0
    streaming_windows: list = []

    current_frames, prev_frames = [[None] * media_dataset_size for i in range(2)]

    detection_task = DetectionTask(input_config, algorithm_config, output_result_config)
    tracking_task = TrackingTask(
        media_dataset_size,
        algorithm_config.tracking_method,
        algorithm_config.tracking_config,
        input_config.reid_models,
        input_config.device,
        algorithm_config.fp16
    )

    results = [[]] * media_dataset_size

    vehicles: dict[int, TrackedObject] = dict()

    # OpenCV convention : im means image after modify, im0 means copy
    # of the image before modification (i.e. original image)
    im = torch.zeros((1, 3, *algorithm_config.inference_img_size), device=input_config.device)
    model(im.half() if algorithm_config.fp16 else im) if input_config.device.type != 'cpu' else None
    for frame_index, batch in enumerate(media_dataset):
        source_paths, im, im0s, video_capture = batch

        im, detection_objs = detection_task.get_detection_objs(im, model)

        for i, detection in enumerate(detection_objs):
            seen_obj += 1

            im0 = im0s[i].copy() if input_config.webcam_enable else im0s.copy()
            source_path = Path(source_paths[i] if input_config.webcam_enable else source_paths)

            current_frames[i] = im0

            box_annotator = Annotator(
                im0,
                line_width=output_video_config.line_thickness,
                example=str(class_names)
            )

            tracking_task.motion_compensation(i, current_frames[i], prev_frames[i])

            if detection is None or not len(detection):
                continue

            detection[:, :4] = scale_coords(
                im.shape[2:],
                detection[:, :4],
                im0.shape).round()

            results[i] = tracking_task.tracker_hosts[i].update(detection.cpu(), im0)

            if len(results[i]) < 0:
                continue

            for result in results[i]:
                if len(result) == 7:
                    t_id = int(result[4])
                    vehicle_type = class_names[int(result[5])]
                    confi = result[6]
                    bbox = result[0:4]

                    print(TrackedObject.format(t_id, vehicle_type, confi, bbox))

                    box_annotator = label_annotator(t_id, vehicle_type, confi, bbox,
                                                    box_annotator, output_video_config)

                    if t_id not in vehicles:
                        vehicles[t_id] = TrackedObject(t_id, vehicle_type, confi, bbox)
                    else:
                        vehicles[t_id].update(vehicle_type, bbox)

                box_annotator = label_annotator(box_annotator, output_video_config)

                # TODO: Update to the database -- Rucha

                # Application Default credentials are automatically created.
                app = firebase_admin.initialize_app()
                db = firestore.client()

                # only uploads the id and vehicle type, not time yet
                doc_ref = db.collection(u'camera').document(u'vehicle_data')
                doc_ref.set({
                    u'id': u't_id',
                    u'vehicle_type': u'vehicle_type'
                })

            stream_result(box_annotator, im0, source_path, streaming_windows)

            prev_frames[i] = current_frames[i]


if __name__ == '__main__':
    with torch.no_grad():  # Use with yolor
        CLI(main, as_positional=False)
