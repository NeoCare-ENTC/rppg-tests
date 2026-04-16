"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
import math
from multiprocessing import Pool, Process, Value, Array, Manager
import sys

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from ultralytics import YOLO
sys.path.append("/home/ddew0188/ASK/yoloface")
from face_detector import YoloDetector

EYE_CLASS_IDS = (0,)


class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data
        self.yolo_detector = None
        self.region_detector = None  # For YOLO-based neonatal region detection
        self.eye_detector = None  # For eye region detection
        self.face_detector = None  # For privacy-aware face masking
        self.face_region_dual_detector = None  # For unified face+region segmentation
        self.detection_stats = None  # Will be initialized in preprocess_dataset

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and its corresponding signals(T)."""

        max_attempts = len(self.inputs)  # Avoid infinite loops
        attempts = 0

        while attempts < max_attempts:
            item_path = self.inputs[index]
            label_path = self.labels[index]

            if os.path.exists(item_path) and os.path.exists(label_path):
                try:
                    data = np.load(item_path)
                    label = np.load(label_path)
                    
                    # Try to load mean_hr file if it exists (for NBHR dataset)
                    mean_hr_path = item_path.replace('_input', '_meanhr')
                    mean_hr = None
                    if os.path.exists(mean_hr_path):
                        mean_hr = np.load(mean_hr_path)[0]  # Extract scalar value
                    
                    if self.data_format == 'NDCHW':
                        data = np.transpose(data, (0, 3, 1, 2))
                    elif self.data_format == 'NCDHW':
                        data = np.transpose(data, (3, 0, 1, 2))
                    elif self.data_format == 'NDHWC':
                        pass
                    else:
                        raise ValueError('Unsupported Data Format!')

                    data = np.float32(data)
                    label = np.float32(label)

                    item_path_filename = item_path.split(os.sep)[-1]
                    split_idx = item_path_filename.rindex('_')
                    filename = item_path_filename[:split_idx]
                    chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

                    if mean_hr is not None:
                        return data, label, filename, chunk_id, mean_hr
                    else:
                        return data, label, filename, chunk_id

                except Exception as e:
                    print(f"Error loading file {item_path} or {label_path}: {e}")

            index = (index + 1) % len(self.inputs) 
            attempts += 1

        raise FileNotFoundError("No valid files found in dataset.")

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '\
            + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    
    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset 
        
        # Initialize shared detection statistics using Manager for multiprocessing
        manager = Manager()
        self.detection_stats = manager.dict()
        self.detection_stats['detected_first_frame'] = manager.Value('i', 0)
        self.detection_stats['detected_dynamic'] = manager.Value('i', 0)
        self.detection_stats['failed_detection'] = manager.Value('i', 0)
        self.detection_stats['failed_videos'] = manager.list()
        
        # send data directories to be processed
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess) 
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        
        # Print detection statistics
        print("\n" + "="*70)
        print("REGION DETECTION STATISTICS")
        print("="*70)
        total = len(data_dirs_split)
        detected_first = self.detection_stats['detected_first_frame'].value
        detected_dynamic = self.detection_stats['detected_dynamic'].value
        failed = self.detection_stats['failed_detection'].value
        success_total = detected_first + detected_dynamic
        
        print(f"Total Videos Processed: {total}")
        print(f"  ✓ Detected in First Frame: {detected_first} ({detected_first/total*100:.1f}%)")
        print(f"  ✓ Detected via Dynamic Detection: {detected_dynamic} ({detected_dynamic/total*100:.1f}%)")
        print(f"  ✗ Failed Detection (using full frame): {failed} ({failed/total*100:.1f}%)")
        print(f"\nOverall Detection Success Rate: {success_total}/{total} ({success_total/total*100:.1f}%)")
        
        if failed > 0 and len(self.detection_stats['failed_videos']) > 0:
            print(f"\nVideos with Failed Detection:")
            failed_list = list(self.detection_stats['failed_videos'])
            for vid in failed_list[:10]:  # Show first 10
                print(f"  - {vid}")
            if len(failed_list) > 10:
                print(f"  ... and {len(failed_list) - 10} more")
        
        print("="*70)
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess(self, frames, hr_bvps, spo2_bvps, config_preprocess, filename):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """
        # resize frames and crop for face region
        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H, filename)
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            elif data_type == "Normalized":
                data.append(BaseLoader.per_channel_normalize(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        
        # Initialize hr_bvps_standard (will be used if LABEL_TYPE requires it)
        hr_bvps_standard = hr_bvps  # Default to raw hr_bvps
        
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            hr_bvps = BaseLoader.diff_normalize_label(hr_bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            hr_bvps_standard = BaseLoader.standardized_label(hr_bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, hr_bvps_clips, spo2_bvps_clips = self.chunk(
                data, hr_bvps_standard, hr_bvps, spo2_bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            hr_bvps_clips = np.array([hr_bvps_standard])
            spo2_bvps_clips = np.array([spo2_bvps])

        return frames_clips, hr_bvps_clips, spo2_bvps_clips

    def face_detection(self, frame, backend, use_larger_box, larger_box_coef, filename, return_mask=False):
      """Region detection on a single frame using YOLO segmentation.

      Args:
          frame(np.array): a single frame.
          backend(str): backend to utilize for region detection (supports 'YOLO-Region', 'YOLO-RegionBlur',
                       'YOLO-RegionBlurEyeZero', 'YOLO-RegionBlurEyeDual', 'YOLO-RegionFaceDualBlur', 'YOLO-RegionFaceDualBlurNew', 'YOLO-RegionFaceDualZero',
                       'YOLOv5', 'HC', 'RF').
          use_larger_box(bool): whether to use a larger bounding box on region detection.
          larger_box_coef(float): Coef. of larger box.
          return_mask(bool): If True, also return the segmentation mask from YOLO.
      Returns:
          face_box_coor(List[int]): coordinates of region bounding box.
          mask(np.array): segmentation mask (only if return_mask=True and backend='YOLO-Region').
                         For YOLO-RegionBlurEyeZero, returns (region_mask, combined_mask) tuple.
                         For YOLO-RegionFaceDualBlur / YOLO-RegionFaceDualBlurNew, returns (region_mask, face_mask).
      """
      # YOLO-based neonatal region detection using trained segmentation model
      if backend == "YOLO-Region":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        # YOLO models don't serialize well across process boundaries
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        # Convert RGB to BGR for detection
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25  # Use standard confidence threshold
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # [x, y, width, height]
            
            # Extract segmentation mask if requested
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()  # Get first mask
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask
      
      elif backend == "YOLO-RegionBlur":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False,)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Extract segmentation mask if requested for blurring
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask
      
      elif backend == "YOLO-RegionFeather":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Extract segmentation mask if requested for feathering
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask
      
      elif backend == "YOLO-RegionBilateral":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Extract segmentation mask if requested for bilateral filtering
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask
      
      elif backend == "YOLO-RegionInpaint":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Extract segmentation mask if requested for inpainting
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask
      
      elif backend == "YOLO-RegionMean":
        
        model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"WARNING: Region detection model not found at {model_path}")
            print("Falling back to full frame")
            return [0, 0, frame.shape[1], frame.shape[0]]
        
        # Always reinitialize model in each call to ensure it works across multiprocessing
        if self.region_detector is None:
            self.region_detector = YOLO(model_path)
        
        # YOLO expects BGR format, but frames are stored as RGB in preprocessing pipeline
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run inference with confidence threshold
        conf_threshold = 0.25
        results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
        
        # Check if any regions detected
        if results.boxes is None or len(results.boxes) == 0:
            # Try rotations if no detection
            right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    return 0
                else:
                    # Adjust coordinates for left rotation
                    frame_height, frame_width = frame.shape[:2]
                    box = results.boxes.xyxy[0].cpu().numpy()
                    face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
            else:
                # Adjust coordinates for right rotation
                frame_height, frame_width = frame.shape[:2]
                box = results.boxes.xyxy[0].cpu().numpy()
                face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
        else:
            # Use the detected bounding box (xyxy format -> xywh format)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Extract segmentation mask if requested for mean color replacement
            if return_mask and results.masks is not None:
                mask = results.masks.data[0].cpu().numpy()
                # Resize mask to match frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
                return face_box_coor, mask

      elif backend in ["YOLO-RegionBlurEyeZero", "YOLO-RegionBlurEyeDual"]:

          # Use separate models: region model from YOLO-RegionBlur + dedicated eye model
          region_model_path = "/srv/data/YU/Neonatal-Facial-Region-Extraction/runs/segment/train/weights/best.pt"
          eye_model_path = "/srv/data/YU/eye-region-detection/runs/segment/eye_region_yolov11/weights/best.pt"

          # Check both models exist
          if not os.path.exists(region_model_path):
              print(f"WARNING: Region detection model not found at {region_model_path}")
              print("Falling back to full frame")
              return [0, 0, frame.shape[1], frame.shape[0]]

          if not os.path.exists(eye_model_path):
              print(f"WARNING: Eye detection model not found at {eye_model_path}")
              print("Will use region detection only")

          # Initialize region detector (same as YOLO-RegionBlur)
          if self.region_detector is None:
              self.region_detector = YOLO(region_model_path)

          # Initialize eye detector
          if not hasattr(self, 'eye_detector') or self.eye_detector is None:
              if os.path.exists(eye_model_path):
                  self.eye_detector = YOLO(eye_model_path)
              else:
                  self.eye_detector = None

          # Convert RGB to BGR for detection
          frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

          # Step 1: Run region detection (same logic as YOLO-RegionBlur)
          conf_threshold = 0.25
          region_results = self.region_detector.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]

          # Check if any regions detected
          if region_results.boxes is None or len(region_results.boxes) == 0:
              # Try rotations if no detection
              right_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
              region_results = self.region_detector.predict(right_rotated_frame, conf=0.15, verbose=False)[0]

              if region_results.boxes is None or len(region_results.boxes) == 0:
                  left_rotated_frame = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  region_results = self.region_detector.predict(left_rotated_frame, conf=0.15, verbose=False)[0]

                  if region_results.boxes is None or len(region_results.boxes) == 0:
                      return 0
                  else:
                      # Adjust coordinates for left rotation
                      frame_height, frame_width = frame.shape[:2]
                      box = region_results.boxes.xyxy[0].cpu().numpy()
                      face_box_coor = [box[1], frame_height-box[2], box[3], frame_height-box[0]]
              else:
                  # Adjust coordinates for right rotation
                  frame_height, frame_width = frame.shape[:2]
                  box = region_results.boxes.xyxy[0].cpu().numpy()
                  face_box_coor = [frame_width-box[3], box[0], frame_width-box[1], box[2]]
          else:
              # Use the detected region bounding box (xyxy format -> xywh format)
              box = region_results.boxes.xyxy[0].cpu().numpy()
              x1, y1, x2, y2 = box
              face_box_coor = [int(x1), int(y1), int(x2-x1), int(y2-y1)]

          # Step 2: Run eye detection if model is available and masks are requested
          if return_mask and region_results.masks is not None:
              # Initialize region mask from region detection
              region_mask = region_results.masks.data[0].cpu().numpy()
              region_mask = cv2.resize(region_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
              region_mask = (region_mask > 0.5).astype(np.uint8)

              eye_keep_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
              eye_detection_mask = np.zeros_like(eye_keep_mask)

              if self.eye_detector is not None:
                  eye_results = self.eye_detector.predict(frame_bgr, conf=0.3, verbose=False)[0]

                  if eye_results.boxes is not None and len(eye_results.boxes) > 0 and eye_results.masks is not None:
                      cls_ids = eye_results.boxes.cls.int().cpu().numpy() if hasattr(eye_results.boxes, 'cls') else np.zeros(len(eye_results.boxes), dtype=int)
                      allowed_ids = getattr(self, 'eye_class_ids', None) or EYE_CLASS_IDS
                      allowed_set = set(int(cid) for cid in allowed_ids)

                      for i in range(len(eye_results.boxes)):
                          if cls_ids[i] not in allowed_set:
                              continue
                          eye_mask_data = eye_results.masks.data[i].cpu().numpy()
                          eye_mask_resized = cv2.resize(eye_mask_data, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                          eye_mask_binary = (eye_mask_resized > 0.5).astype(np.uint8)
                          eye_keep_mask = eye_keep_mask & (1 - eye_mask_binary)
                          eye_detection_mask = np.maximum(eye_detection_mask, eye_mask_binary)

              combined_mask = (region_mask.astype(np.float32) * eye_keep_mask.astype(np.float32)).astype(np.uint8)

              return face_box_coor, (region_mask, combined_mask, eye_detection_mask)

      elif backend == "YOLO-RegionFaceDualBlurNew":
          model_path = "/srv/data/YU/face-roi-detection/runs/segment/eye_region_yolov11_seg/weights/best.pt"

          if not os.path.exists(model_path):
              print(f"WARNING: Unified face-region detector not found at {model_path}")
              print("Falling back to legacy dual-detector pipeline")
              return self.face_detection(
                  frame,
                  "YOLO-RegionFaceDualBlur",
                  use_larger_box,
                  larger_box_coef,
                  filename,
                  return_mask,
              )

          if self.face_region_dual_detector is None:
              self.face_region_dual_detector = YOLO(model_path)

          frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
          results = self.face_region_dual_detector.predict(frame_bgr, conf=0.3, verbose=False)[0]

          if results.boxes is None or len(results.boxes) == 0:
              return self.face_detection(
                  frame,
                  "YOLO-RegionFaceDualBlur",
                  use_larger_box,
                  larger_box_coef,
                  filename,
                  return_mask,
              )

          detector_names = getattr(self.face_region_dual_detector, "names", {})

          def collect_ids(names_obj, target_label):
              ids = []
              if isinstance(names_obj, dict):
                  iterable = names_obj.items()
              elif isinstance(names_obj, (list, tuple)):
                  iterable = enumerate(names_obj)
              else:
                  iterable = []
              for idx, name in iterable:
                  if str(name).lower() == target_label:
                      ids.append(int(idx))
              return ids

          face_class_ids = collect_ids(detector_names, "face") or [0]
          region_class_ids = collect_ids(detector_names, "region") or [1]

          frame_height, frame_width = frame.shape[:2]
          boxes_xyxy = results.boxes.xyxy.cpu().numpy()
          cls_tensor = getattr(results.boxes, "cls", None)
          conf_tensor = getattr(results.boxes, "conf", None)
          cls_ids = cls_tensor.int().cpu().numpy() if cls_tensor is not None else np.zeros(boxes_xyxy.shape[0], dtype=int)
          confs = conf_tensor.cpu().numpy() if conf_tensor is not None else None
          has_masks = results.masks is not None and getattr(results.masks, "data", None) is not None

          def box_to_mask(box_xywh):
              mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
              x, y, w, h = [int(v) for v in box_xywh]
              x2 = min(frame_width, x + w)
              y2 = min(frame_height, y + h)
              x = max(0, x)
              y = max(0, y)
              if x2 > x and y2 > y:
                  mask[y:y2, x:x2] = 1
              return mask

          best_region = None
          best_face = None

          for det_idx in range(boxes_xyxy.shape[0]):
              cls_id = int(cls_ids[det_idx])
              conf_score = float(confs[det_idx]) if confs is not None else 0.0
              x1, y1, x2, y2 = boxes_xyxy[det_idx]
              box_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

              mask_binary = None
              if has_masks:
                  mask_data = results.masks.data[det_idx].cpu().numpy()
                  mask_resized = cv2.resize(mask_data, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                  mask_binary = (mask_resized > 0.5).astype(np.uint8)

              entry = {
                  "box": box_xywh,
                  "mask": mask_binary,
                  "conf": conf_score,
              }

              if cls_id in region_class_ids:
                  if best_region is None or conf_score > best_region["conf"]:
                      best_region = entry
              elif cls_id in face_class_ids:
                  if best_face is None or conf_score > best_face["conf"]:
                      best_face = entry

          if best_region is None:
              return self.face_detection(
                  frame,
                  "YOLO-RegionFaceDualBlur",
                  use_larger_box,
                  larger_box_coef,
                  filename,
                  return_mask,
              )

          face_box_coor = best_region["box"]
          region_mask = best_region["mask"] if best_region["mask"] is not None else box_to_mask(face_box_coor)

          if return_mask:
              if best_face is not None and best_face["mask"] is not None:
                  face_mask = best_face["mask"]
              elif best_face is not None:
                  face_mask = box_to_mask(best_face["box"])
              else:
                  face_mask = box_to_mask(face_box_coor)
              return face_box_coor, (region_mask, face_mask)

          return face_box_coor

      elif backend in ["YOLO-RegionFaceDualBlur", "YOLO-RegionFaceDualZero"]:

          project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
          face_model_path = os.path.join(project_root, "yolo_weights", "face_detector", "best.pt")

          face_mask = None
          face_box_candidate = None

          frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

          if os.path.exists(face_model_path):
              if self.face_detector is None:
                  self.face_detector = YOLO(face_model_path)

              face_results = self.face_detector.predict(frame_bgr, conf=0.35, verbose=False)[0]
              if face_results.boxes is not None and len(face_results.boxes) > 0:
                  if hasattr(face_results.boxes, "conf") and face_results.boxes.conf is not None:
                      confs = face_results.boxes.conf.cpu().numpy()
                      best_idx = int(np.argmax(confs))
                  else:
                      best_idx = 0

                  box = face_results.boxes.xyxy[best_idx].cpu().numpy()
                  x1, y1, x2, y2 = box
                  face_box_candidate = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                  if face_results.masks is not None:
                      face_mask_data = face_results.masks.data[best_idx].cpu().numpy()
                      face_mask_resized = cv2.resize(
                          face_mask_data,
                          (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_LINEAR,
                      )
                      face_mask = (face_mask_resized > 0.5).astype(np.uint8)
                  else:
                      face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                      x1c = max(0, int(np.floor(x1)))
                      y1c = max(0, int(np.floor(y1)))
                      x2c = min(frame.shape[1], int(np.ceil(x2)))
                      y2c = min(frame.shape[0], int(np.ceil(y2)))
                      face_mask[y1c:y2c, x1c:x2c] = 1
          else:
              print(f"WARNING: Face detector model not found at {face_model_path}")

          # Reuse YOLO-RegionBlur for the ROI detection
          region_detection = self.face_detection(
              frame,
              "YOLO-RegionBlur",
              False,
              1.0,
              filename,
              return_mask,
          )

          region_mask = None
          if return_mask and isinstance(region_detection, tuple):
              face_box_coor, region_mask = region_detection
          else:
              face_box_coor = region_detection

          if (face_box_coor is None or (isinstance(face_box_coor, int) and face_box_coor == 0)) and face_box_candidate is not None:
              face_box_coor = face_box_candidate

          if return_mask:
              if face_mask is None and isinstance(face_box_coor, (list, tuple)):
                  x, y, w, h = [int(v) for v in face_box_coor]
                  face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                  x2 = min(frame.shape[1], x + w)
                  y2 = min(frame.shape[0], y + h)
                  x1c = max(0, x)
                  y1c = max(0, y)
                  if x2 > x1c and y2 > y1c:
                      face_mask[y1c:y2, x1c:x2] = 1
              return face_box_coor, (region_mask, face_mask)

      elif backend == "YOLOv5":
          frame_height, frame_width = frame.shape[:2]
          model = YoloDetector(target_size=None, device='cpu', min_face=80)
          bboxes, points = model.predict(frame)

          if len(bboxes[0]) == 0:
              right_rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
              bboxes, points = model.predict(right_rotated_frame)
              if len(bboxes[0]) == 0:
                  left_rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  bboxes, points = model.predict(left_rotated_frame)
                  if len(bboxes[0]) == 0:
                      return 0
                  else:
                      face_box_coor = [bboxes[0][0][1], frame_height-bboxes[0][0][2], bboxes[0][0][3], frame_height-bboxes[0][0][0]]
              else:
                  face_box_coor = [frame_width-bboxes[0][0][3], bboxes[0][0][0], frame_width-bboxes[0][0][1], bboxes[0][0][2]]
          else:
              face_box_coor = bboxes[0][0]

      else:
          raise ValueError("Unsupported face detection backend!")

      if use_larger_box and isinstance(face_box_coor, (list, tuple, np.ndarray)):
          # Ensure we operate on a mutable sequence when expanding the box
          face_box_list = list(face_box_coor)
          face_box_list[0] = max(0, face_box_list[0] - (larger_box_coef - 1.0) / 2 * face_box_list[2])
          face_box_list[1] = max(0, face_box_list[1] - (larger_box_coef - 1.0) / 2 * face_box_list[3])
          face_box_list[2] = larger_box_coef * face_box_list[2]
          face_box_list[3] = larger_box_coef * face_box_list[3]
          face_box_coor = face_box_list
      return face_box_coor


    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection, 
                         detection_freq, use_median_box, width, height, filename):
        """Crop neonatal region and resize frames.

        Args:
            frames(np.array): Video frames.
            use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the region
                                         and resizing.
                                         If True, it performs region detection every "detection_freq" frames.
            detection_freq(int): The frequency of dynamic region detection e.g., every detection_freq frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from region detection.
            use_face_detection(bool):  Whether crop the region (kept for backward compatibility).
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """
        # If not detected, turn on dynamic detection
        box_coor = self.face_detection(frames[0], backend, use_larger_box, larger_box_coef, filename)
        if box_coor == 0:
            print(f"Using Dynamic detection for {filename}")
            use_dynamic_detection = True
        else:
            # Successfully detected in first frame
            if self.detection_stats is not None:
                self.detection_stats['detected_first_frame'].value += 1
        # Region Cropping
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []

        # Perform region detection by num_dynamic_det times.
        for idx in range(num_dynamic_det):
            if use_face_detection:
                frame_idx = min(detection_freq * idx, frames.shape[0] - 1)
                box_coor = self.face_detection(frames[frame_idx], backend, use_larger_box, larger_box_coef, filename)
                if box_coor != 0:
                    face_region_all.append(box_coor)
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        
        # If using dynamic detection and we found at least one valid region, use only the first valid one
        if use_dynamic_detection and face_region_all:
            face_region_all = [face_region_all[0]]
        if not face_region_all:
            print(f"ERROR: Region not Detected in {filename}")
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
            if self.detection_stats is not None:
                self.detection_stats['failed_detection'].value += 1
                self.detection_stats['failed_videos'].append(filename)
        elif use_dynamic_detection and len(face_region_all) > 0:
            # Dynamic detection succeeded
            if self.detection_stats is not None:
                self.detection_stats['detected_dynamic'].value += 1
        
        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            # Generate a median bounding box based on all detected face regions
            face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Frame Resizing with optional masking or blurring
        resized_frames = np.zeros((frames.shape[0], height, width, 3))
        
        # Get segmentation mask from first frame if using YOLO-based backends that support masking
        segmentation_mask = None
        combined_mask = None  # For YOLO-RegionBlurEyeZero backend
        eye_detection_mask = None  # For YOLO-RegionBlurEyeZero and Dual backends
        face_mask_full = None  # For YOLO-RegionFaceDualBlur variants
        if use_face_detection and backend in ["YOLO-Region", "YOLO-RegionBlur", "YOLO-RegionFeather", "YOLO-RegionBilateral", "YOLO-RegionInpaint", "YOLO-RegionMean", "YOLO-RegionBlurEyeZero", "YOLO-RegionBlurEyeDual", "YOLO-RegionFaceDualBlur", "YOLO-RegionFaceDualBlurNew", "YOLO-RegionFaceDualZero"]:
            detection_result = self.face_detection(frames[0], backend, use_larger_box, larger_box_coef, filename, return_mask=True)
            if isinstance(detection_result, tuple):
                if backend in ["YOLO-RegionBlurEyeZero", "YOLO-RegionBlurEyeDual"]:
                    # These backends return (box_coor, (region_mask, combined_mask, eye_detection_mask))
                    _, mask_tuple = detection_result
                    if isinstance(mask_tuple, tuple):
                        tuple_len = len(mask_tuple)
                        if tuple_len >= 1:
                            segmentation_mask = mask_tuple[0]
                        if tuple_len >= 2:
                            combined_mask = mask_tuple[1]
                        if tuple_len >= 3:
                            eye_detection_mask = mask_tuple[2]
                    else:
                        segmentation_mask = mask_tuple
                elif backend in ["YOLO-RegionFaceDualBlur", "YOLO-RegionFaceDualBlurNew", "YOLO-RegionFaceDualZero"]:
                    _, mask_tuple = detection_result
                    if isinstance(mask_tuple, tuple):
                        if len(mask_tuple) >= 1:
                            segmentation_mask = mask_tuple[0]
                        if len(mask_tuple) >= 2:
                            face_mask_full = mask_tuple[1]
                    else:
                        segmentation_mask = mask_tuple
                else:
                    _, segmentation_mask = detection_result
        
        for i in range(0, frames.shape[0]):
            frame = frames[i].copy()
            if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
                reference_index = i // detection_freq
            else:  # use the first region obtrained from the first frame.
                reference_index = 0
            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    # Clamp reference_index to available face regions
                    reference_index = min(reference_index, len(face_region_all) - 1)
                    face_region = face_region_all[reference_index]
                
                # Step 1: Crop the frame to the detected region
                x, y, w, h = face_region
                cropped_frame = frame[max(y, 0):min(y + h, frame.shape[0]),
                                     max(x, 0):min(x + w, frame.shape[1])]
                
                # Cache the original crop for any restoration needs
                original_cropped = frame[max(y, 0):min(y + h, frame.shape[0]),
                                        max(x, 0):min(x + w, frame.shape[1])]

                # Step 2: Apply mask or blur to cropped frame if available
                if segmentation_mask is not None:
                    # Crop the mask to the same region
                    cropped_mask = segmentation_mask[max(y, 0):min(y + h, frame.shape[0]),
                                                    max(x, 0):min(x + w, frame.shape[1])]
                    
                    if backend == "YOLO-RegionBlur":
                        # Apply blur to non-ROI regions instead of zeroing them
                        # Expand mask to 3 channels
                        cropped_mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                        # Apply Gaussian blur to the entire frame
                        blurred_frame = cv2.blur(cropped_frame, (53, 53), 0)
                        # Combine: use original where mask=1, use blurred where mask=0
                        cropped_frame = cropped_frame * cropped_mask_3ch + blurred_frame * (1 - cropped_mask_3ch)
                    elif backend == "YOLO-RegionFeather":
                        # Apply soft feathering with gradual fade-out at mask edges
                        feather_distance = 20  # Distance in pixels for feathering transition
                        # Apply Gaussian blur to the mask to create smooth transitions
                        mask_float = cv2.GaussianBlur(cropped_mask.astype(float), (21, 21), feather_distance/2)
                        # Expand to 3 channels
                        mask_float_3ch = np.stack([mask_float] * 3, axis=-1)
                        # Apply soft feathering: gradually fade out from ROI to background
                        cropped_frame = (cropped_frame * mask_float_3ch).astype(np.uint8)
                    elif backend == "YOLO-RegionBilateral":
                        # Apply bilateral filter to non-ROI regions (edge-preserving blur)
                        # Preserves important edges while smoothing non-ROI areas
                        cropped_mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                        # Apply bilateral filter: preserves edges, removes noise
                        # Parameters: d=9 (diameter), sigmaColor=75, sigmaSpace=75
                        bilateral_filtered = cv2.bilateralFilter(cropped_frame, 9, 75, 75)
                        # Combine: use original where mask=1, use bilateral filtered where mask=0
                        cropped_frame = cropped_frame * cropped_mask_3ch + bilateral_filtered * (1 - cropped_mask_3ch)
                    elif backend == "YOLO-RegionInpaint":
                        # Apply content-aware inpainting to non-ROI regions
                        # Removes identifiable features while maintaining structural coherence
                        # Create inpaint mask: 255 where mask=0 (non-ROI), 0 where mask=1 (ROI)
                        inpaint_mask = (1 - cropped_mask).astype(np.uint8) * 255
                        # Apply inpainting using Telea algorithm (fast marching)
                        # inpaintRadius=3: smaller radius for fine details
                        cropped_frame = cv2.inpaint(cropped_frame, inpaint_mask, 3, cv2.INPAINT_TELEA)
                    elif backend == "YOLO-RegionMean":
                        # Replace non-ROI regions with the mean color of the ROI
                        # Maintains color consistency and reduces edge artifacts
                        if np.sum(cropped_mask) > 0:
                            # Calculate mean color of ROI pixels
                            roi_mean_color = cropped_frame[cropped_mask > 0].mean(axis=0).astype(np.uint8)
                            # Create a frame filled with the mean color
                            mean_color_frame = np.full_like(cropped_frame, roi_mean_color)
                            # Expand mask to 3 channels
                            cropped_mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                            # Combine: use original where mask=1, use mean color where mask=0
                            cropped_frame = cropped_frame * cropped_mask_3ch + mean_color_frame * (1 - cropped_mask_3ch)
                    elif backend == "YOLO-RegionBlurEyeZero":
                        # Effect pipeline per frame:
                        # 1) Blur the full frame first
                        # 2) Paste original pixels back for detected 'region' areas
                        # 3) Zero out detected 'eye' areas to black
                        
                        # Step 1: Blur the full frame
                        blurred_frame = cv2.GaussianBlur(cropped_frame, (21, 21), 0)
                        cropped_frame = blurred_frame.copy()
                        
                        # Step 2: Paste original pixels back for detected 'region' areas
                        cropped_mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                        # Get original frame region for this crop
                        # Restore region pixels from original frame
                        cropped_frame[cropped_mask > 0] = original_cropped[cropped_mask > 0]
                        
                        # Step 3: Replace detected eye areas with their mean color
                        if combined_mask is not None:
                            cropped_combined_mask = combined_mask[max(y, 0):min(y + h, frame.shape[0]),
                                                                   max(x, 0):min(x + w, frame.shape[1])]
                            # Eyes exist where combined mask removed region pixels
                            eye_mask = (cropped_combined_mask == 0) & (cropped_mask > 0)
                            if np.any(eye_mask):
                                eye_pixels = original_cropped[eye_mask]
                                mean_color = np.clip(eye_pixels.mean(axis=0), 0, 255).astype(cropped_frame.dtype)
                                cropped_frame[eye_mask] = mean_color
                    elif backend == "YOLO-RegionBlurEyeDual":
                        # Similar pipeline with dual blur levels:
                        # 1) Blur everything lightly (non-ROI)
                        # 2) Restore region pixels
                        # 3) Apply heavy blur to detected eye regions

                        low_kernel = 21 | 1
                        high_kernel = 53 | 1
                        blurred_frame = cv2.GaussianBlur(cropped_frame, (low_kernel, low_kernel), 0)
                        cropped_frame = blurred_frame.copy()

                        cropped_mask_bool = cropped_mask.astype(bool)
                        cropped_frame[cropped_mask_bool] = original_cropped[cropped_mask_bool]

                        if eye_detection_mask is not None:
                            cropped_eye_mask = eye_detection_mask[max(y, 0):min(y + h, frame.shape[0]),
                                                                  max(x, 0):min(x + w, frame.shape[1])]
                            eye_mask_bool = (cropped_eye_mask > 0) & cropped_mask_bool
                        else:
                            eye_mask_bool = np.zeros_like(cropped_mask_bool, dtype=bool)

                        if np.any(eye_mask_bool):
                            heavy_blur = cv2.GaussianBlur(original_cropped, (high_kernel, high_kernel), 0)
                            cropped_frame[eye_mask_bool] = heavy_blur[eye_mask_bool]
                    elif backend in ["YOLO-RegionFaceDualBlur", "YOLO-RegionFaceDualBlurNew"]:
                        # Two-level blur: heavy blur for face-only areas, base blur elsewhere
                        cropped_mask_bool = cropped_mask.astype(bool)
                        base_blurred = cv2.GaussianBlur(cropped_frame, (43, 43), 0)
                        cropped_frame = base_blurred.copy()
                        cropped_frame[cropped_mask_bool] = original_cropped[cropped_mask_bool]

                        if face_mask_full is not None:
                            cropped_face_mask = face_mask_full[max(y, 0):min(y + h, frame.shape[0]),
                                                               max(x, 0):min(x + w, frame.shape[1])]
                            face_mask_bool = cropped_face_mask.astype(bool)
                            face_only_mask = face_mask_bool & (~cropped_mask_bool)
                            if np.any(face_only_mask):
                                heavy_blur = cv2.GaussianBlur(original_cropped, (83, 83), 0)
                                cropped_frame[face_only_mask] = heavy_blur[face_only_mask]
                    elif backend == "YOLO-RegionFaceDualZero":
                        # Keep ROI sharp, blur background, zero face-only areas outside ROI
                        cropped_mask_bool = cropped_mask.astype(bool)
                        base_blurred = cv2.GaussianBlur(cropped_frame, (53, 53), 0)
                        cropped_frame = base_blurred.copy()
                        cropped_frame[cropped_mask_bool] = original_cropped[cropped_mask_bool]

                        if face_mask_full is not None:
                            cropped_face_mask = face_mask_full[max(y, 0):min(y + h, frame.shape[0]),
                                                               max(x, 0):min(x + w, frame.shape[1])]
                            face_mask_bool = cropped_face_mask.astype(bool)
                            face_only_mask = face_mask_bool & (~cropped_mask_bool)
                            if np.any(face_only_mask):
                                cropped_frame[face_only_mask] = 0
                    else:
                        # YOLO-Region: Apply mask to zero out non-ROI regions
                        cropped_mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                        cropped_frame = cropped_frame * cropped_mask_3ch
                
                # Step 3: Resize the cropped (and masked) frame
                resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
                resized_frames[i] = resized_frame
            else:
                resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, hr_bvps_standard, hr_bvps, spo2_bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        hr_bvps_clips = []
        spo2_bvps_clips = []
        frames_clips = []
        # frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        # bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        for i in range(clip_num):
            hr_clip = np.array(hr_bvps[i * chunk_length:(i + 1) * chunk_length])
            if np.count_nonzero(hr_clip == 127) > 10:
                continue
            elif np.count_nonzero(hr_clip == 0) > 10:
                continue
            hr_bvp_clip = hr_bvps_standard[i * chunk_length:(i + 1) * chunk_length]
            spo2_bvp_clip = spo2_bvps[i * chunk_length:(i + 1) * chunk_length]
            hr_bvps_clips.append(hr_bvp_clip)
            spo2_bvps_clips.append(spo2_bvp_clip)
            frames_clips.append(frames[i * chunk_length:(i + 1) * chunk_length])
        return np.array(frames_clips), np.array(hr_bvps_clips), np.array(spo2_bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            count: count of preprocessed data
        """

        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, hr_bvps_clips, spo2_bvps_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(hr_bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, np.array([hr_bvps_clips[i], spo2_bvps_clips[i]]))
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=4):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        p_list = []  # list of processes
        running_num = 0  # number of running processes

        # in range of number of files to process
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < multi_process_quota:  # in case of too many processes
                    # send data to be preprocessing task
                    p = Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs,config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Build a list of files used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """
        file_list = []
        # iterate through processes and add all processed file paths
        for process_num, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def per_channel_normalize(data):
        """
        Normalize RGB video data per channel along the time-axis.

        Args:
            data (numpy.ndarray): Video data with shape (n, h, w, c), where
                                  n = number of frames,
                                  h = height of each frame,
                                  w = width of each frame,
                                  c = number of channels (should be 3 for RGB).

        Returns:
            numpy.ndarray: Per-channel normalized data of the same shape.
        """
        n, h, w, c = data.shape
        assert c == 3, "The input data must have 3 channels (RGB)."

        normalized_data = np.zeros_like(data, dtype=np.float32)

        for channel in range(c):
            channel_data = data[:, :, :, channel]

            mean = np.mean(channel_data)
            std = np.std(channel_data)

            std = std if std > 1e-7 else 1e-7

            normalized_data[:, :, :, channel] = (channel_data - mean) / std

        normalized_data[np.isnan(normalized_data)] = 0

        return normalized_data


    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)