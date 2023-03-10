import platform
from jsonargparse import CLI
from pathlib import Path

import cv2
import torch

from utils.general import scale_coords

from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors

from algorithm import DetectionTask, TrackingTask
from configuration import InputConfig, OutputVideoConfig, OutputResultConfig, AlgorithmConfig


class TrackedObject:

    def __init__(self, t_id: int = -1, vehicle_type: str = '', confidence: int = -1, bbox: list = None):
        self.t_id = t_id
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.bbox = bbox

    def label_annotator(self, box_annotator: Annotator, config: OutputVideoConfig):
        label = None
        if not config.hide_labels:
            class_label = None if config.hide_class else f'{self.vehicle_type}'
            confidence_label = None if config.hide_conf else f'{self.confidence:.2f}'

            label = f'{self.t_id} {class_label} {confidence_label}'

        box_annotator.box_label(self.bbox, label, color=colors(1, True))

        return box_annotator


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
                tracked_object: TrackedObject = TrackedObject(
                    result[4],
                    class_names[int(result[5])],
                    result[6],
                    result[0:4]
                )

                # TODO: Update to the database -- Rucha

                box_annotator = tracked_object.label_annotator(box_annotator, output_video_config)

            stream_result(box_annotator, im0, source_path, streaming_windows)

            prev_frames[i] = current_frames[i]


if __name__ == '__main__':
    with torch.no_grad():  # Use with yolor
        CLI(main, as_positional=False)
