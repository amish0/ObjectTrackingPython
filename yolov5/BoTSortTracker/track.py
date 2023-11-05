# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import numpy as np

from .bot_sort import BoTSORT
from .loadyaml import load_yaml
# TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

TRACKER_MAP = {'botsort': BoTSORT}
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

class Tracker:
    """Tracker class for multi object tracking with some modifications to the original Sort class"""

    def __init__(self, tracker_type: str = 'sort') -> None:
        """@brief Tracker class
           @details will initialize the tracker with the given tracker_type and tracker parameters from corresponding yaml file
           @param tracker_type type of tracker to be used
        """
        if isinstance(tracker_type, str) and tracker_type in TRACKER_MAP.keys():
            self.parm = load_yaml(ROOT / "cfg/botsort.yaml")
            self.tracker = TRACKER_MAP[tracker_type](**self.parm)
        else:
            print("tracker_type must be string and in {}".format(TRACKER_MAP.keys()))

    def update(self, detections: dict, img: np.ndarray)->any:
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None
        """
        if hasattr(self, 'tracker'):
            return self.tracker.update(detections, img)
        else:
            print("tracker not init")
            return None

    def __call__(self, detections, img):
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None"""
        return self.update(detections, img)

    def check_parameters(self):
        print(self.parm)

# def on_predict_start(predictor, persist=False):
#     """
#     Initialize trackers for object tracking during prediction.

#     Args:
#         predictor (object): The predictor object to initialize trackers for.
#         persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

#     Raises:
#         AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
#     """
#     if hasattr(predictor, 'trackers') and persist:
#         return
#     tracker = check_yaml(predictor.args.tracker)
#     cfg = IterableSimpleNamespace(**yaml_load(tracker))
#     assert cfg.tracker_type in ['bytetrack', 'botsort'], \
#         f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
#     trackers = []
#     for _ in range(predictor.dataset.bs):
#         tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
#         trackers.append(tracker)
#     predictor.trackers = trackers


# def on_predict_postprocess_end(predictor):
#     """Postprocess detected boxes and update with object tracking."""
#     bs = predictor.dataset.bs
#     im0s = predictor.batch[1]
#     for i in range(bs):
#         det = predictor.results[i].boxes.cpu().numpy()
#         if len(det) == 0:
#             continue
#         tracks = predictor.trackers[i].update(det, im0s[i])
#         if len(tracks) == 0:
#             continue
#         idx = tracks[:, -1].astype(int)
#         predictor.results[i] = predictor.results[i][idx]
#         predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))


# def register_tracker(model, persist):
#     """
#     Register tracking callbacks to the model for object tracking during prediction.

#     Args:
#         model (object): The model object to register tracking callbacks for.
#         persist (bool): Whether to persist the trackers if they already exist.

#     """
#     model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
#     model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)
