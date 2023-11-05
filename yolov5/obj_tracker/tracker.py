
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
from .Sort.Sort_cls import Sort
from .loadyaml import load_yaml

tracker_map = {
    'sort': Sort
}


class Tracker:
    """Tracker class for multi object tracking with some modifications to the original Sort class"""

    def __init__(self, tracker_type: str = 'sort') -> None:
        """@brief Tracker class
           @details will initialize the tracker with the given tracker_type and tracker parameters from corresponding yaml file
           @param tracker_type type of tracker to be used
        """
        if isinstance(tracker_type, str) and tracker_type in tracker_map.keys():
            self.parm = load_yaml(ROOT / "sort.yaml")
            self.tracker = tracker_map[tracker_type](**self.parm)
        else:
            print("tracker_type must be string and in {}".format(tracker_map.keys()))

    def update(self, detections: (list, np.ndarray))->any:
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None
        """
        if isinstance(detections, list):
            detections = np.array(detections)
        if hasattr(self, 'tracker'):
            return self.tracker.update(detections)
        else:
            print("tracker not init")
            return None

    def __call__(self, detections):
        """@brief update the tracker with new detections
           @details update the tracker with new detections
           @param detections new detections in the format [[x1, y1, x2, y2, score, class_id], ...]
           @return updated bounding boxes in the format [[x1, y1, x2, y2, score, class_id, track_id], ...] if tracker is not initalized it will return None"""
        return self.update(detections)

    def check_parameters(self):
        print(self.parm)


class MuliCameraTracker:
    """@brief MultiCameraTracker class
       @details This class is used to create and configure logger instances with different settings."""

    def __init__(self, tracker_type, tracker_id: set = None) -> None:
        if isinstance(tracker_type, str) and tracker_type in tracker_map.keys():
            self.tracker_type = tracker_type
            self.tracker_dict = dict()
            if isinstance(tracker_id, set) and len(tracker_id) > 0 and all(isinstance(x, int) for x in tracker_id):
                for i in tracker_id:
                    self.add_tracker(i)

    def add_tracker(self, tracker_id: int, tracker_type: str = 'sort'):
        """@brief add tracker with given tracker_id
           @param tracker_id tracker_id to be added
           @param tracker_type type of tracker to be used
           @return updated tracker_id list
        """
        if tracker_id not in self.tracker_dict.keys():
            self.tracker_dict[tracker_id] = Tracker(tracker_type)
        else:
            print("tracker_id {} already exist".format(tracker_id))

    def remove_tracker(self, tracker_id: int):
        """@brief remove tracker with given tracker_id
           @param tracker_id tracker_id to be removed
           @return updated tracker_id list"""
        if tracker_id in self.tracker_dict.keys():
            self.tracker_dict.pop(tracker_id)
        else:
            print("tracker_id {} not exist".format(tracker_id))
        return self.tracker_dict.keys()

    def get_tracker_id(self):
        """@brief get tracker_id list
           @return tracker_id list
        """
        return self.tracker_dict.keys()

    def update(self, detections: dict, tracker_id: set = None):
        """
        @brief update tracker with new detections
        @details It will update the tracker with new detections, before update it will go with validation
        @param detections detections in the format {tracker_id: [[x1, y1, x2, y2, score, class_id], ...], ...}
        @param tracker_id tracker_id to be updated
        @return updated detections in the format {tracker_id: [[x1, y1, x2, y2, score, class_id, track_id], ...], ...} if tracker is not initalized or incase of any kissmatch it will return None
        """
        # print("detections: ", detections)
        # print("tracker_id: ", tracker_id)
        # print("length detections: ", len(detections))
        # print("length tracker_id: ", len(tracker_id))
        if tracker_id is not None and len(detections) == len(tracker_id):
            for i in tracker_id:
                # print("i: ", i)
                # print("self.tracker_dict.keys(): ", self.tracker_dict.keys())
                # print("len(detections[i]): ", len(detections[i]))
                # print("detections[i]: ", detections[i])
                if i in self.tracker_dict.keys() and len(detections[i]) > 0:
                    # print("detections[i]: ", detections[i])
                    detections[i] = self.tracker_dict[i](
                        detections[i][:, :6])
                else:
                    detections[i] = None
        else:
            return None
        return detections

    def __call__(self, detections: dict, tracker_id: set = None):
        """ 
        @brief update tracker with new detections
        @details for more details check update function documentation
        """
        return self.update(detections, tracker_id)

if __name__ == "__main__":
    object_tracker = Tracker('sort')
    pass
    # print(object_tracker([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6 ]]))
