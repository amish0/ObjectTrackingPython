# import argparse
# import csv
# import os
# import platform
# import sys
# from pathlib import Path


from obj_tracker.tracker import Tracker
import torch
track_obj = Tracker('sort')
print(track_obj([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6 ]]))