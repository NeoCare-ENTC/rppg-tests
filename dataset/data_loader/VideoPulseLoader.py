"""The dataloader for the NBHR dataset.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class VideoPulseLoader(BaseLoader):
    """The data loader for the VideoPulse dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an VideoPulse dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                -----------------
                     |-- PPG/
                     |    |-- 000000000.csv/
                     |    |-- 000000001.csv
                     |-- Video/
                     |    |-- 000000000.avi/
                     |    |-- 000000001.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For NBHR dataset)."""
        # Check the 'Video' folder name
        video_dir = os.path.join(data_path, "Video")
        ppg_dir = os.path.join(data_path, "PPG")
        
        data_dirs = glob.glob(video_dir + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty! Looking in: " + video_dir)
        
        dirs = []
        skipped_count = 0
        for data_dir in data_dirs:
            # Extract filename without extension (e.g., "20250221_120948")
            filename = os.path.split(data_dir)[-1].replace(".avi", "")
            
            # Check if corresponding PPG file exists
            ppg_file = os.path.join(ppg_dir, "{0}_GT.csv".format(filename))
            if not os.path.exists(ppg_file):
                print(f"Skipping {filename}: No corresponding PPG file found")
                skipped_count += 1
                continue
            
            # Use the filename string as the index for matching with PPG files
            dirs.append({"index": filename, "path": data_dir})
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} videos without PPG files")
        
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        frames = self.read_video(
            os.path.join(data_dirs[i]['path']))

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            # Get the root data directory (parent of Video/video folder)
            data_dir = os.path.abspath(os.path.join(data_dirs[i]['path'], "..", ".."))
            
            # Check for PPG folder name
            ppg_file = os.path.join(data_dir, "PPG", "{0}_GT.csv".format(saved_filename))
            
            # Skip if PPG file doesn't exist (safety check)
            if not os.path.exists(ppg_file):
                print(f"Warning: PPG file not found for {saved_filename}, skipping")
                file_list_dict[i] = []
                return
            
            hr_bvps, spo2_bvps, mean_hr = self.read_wave(ppg_file)

        hr_bvps = BaseLoader.resample_ppg(hr_bvps, frames.shape[0])
        spo2_bvps = BaseLoader.resample_ppg(spo2_bvps, frames.shape[0])
            
        frames_clips, hr_bvps_clips, spo2_bvps_clips = self.preprocess(frames, hr_bvps, spo2_bvps, config_preprocess, filename)
        input_name_list, label_name_list = self.save_multi_process_with_hr(frames_clips, hr_bvps_clips, spo2_bvps_clips, saved_filename, mean_hr)
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    def preprocess(self, frames, hr_bvps, spo2_bvps, config_preprocess, filename):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            hr_bvps(np.array): HR PPG signal labels for a video.
            spo2_bvps(np.array): SpO2 PPG signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            filename: name of the file being processed.
        Returns:
            frame_clips(np.array): processed video data by frames
            hr_bvps_clips(np.array): processed hr bvp (ppg) labels by frames  
            spo2_bvps_clips(np.array): processed spo2 bvp labels by frames
        """
        # Replicate BaseLoader's preprocessing logic without modifying config
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

        # Chunk the data
        if config_preprocess.DO_CHUNK:
            frames_clips, hr_bvps_clips, spo2_bvps_clips = self.chunk_data(
                data, hr_bvps_standard, hr_bvps, spo2_bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            hr_bvps_clips = np.array([hr_bvps_standard])
            spo2_bvps_clips = np.array([spo2_bvps])

        return frames_clips, hr_bvps_clips, spo2_bvps_clips

    def chunk_data(self, frames, hr_bvps_standard, hr_bvps, spo2_bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            hr_bvps_standard(np.array): standardized HR BVP labels.
            hr_bvps(np.array): raw HR BVP labels (used for filtering).
            spo2_bvps(np.array): SpO2 BVP labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            hr_bvps_clips: all chunks of hr bvp frames
            spo2_bvps_clips: all chunks of spo2 bvp frames
        """
        clip_num = frames.shape[0] // chunk_length
        hr_bvps_clips = []
        spo2_bvps_clips = []
        frames_clips = []
        
        for i in range(clip_num):
            start_idx = i * chunk_length
            end_idx = (i + 1) * chunk_length
            
            # Add this chunk without quality checks
            hr_bvp_clip = hr_bvps_standard[start_idx:end_idx]
            spo2_bvp_clip = spo2_bvps[start_idx:end_idx]
            hr_bvps_clips.append(hr_bvp_clip)
            spo2_bvps_clips.append(spo2_bvp_clip)
            frames_clips.append(frames[start_idx:end_idx])
            
        return np.array(frames_clips), np.array(hr_bvps_clips), np.array(spo2_bvps_clips)

    def save_multi_process_with_hr(self, frames_clips, hr_bvps_clips, spo2_bvps_clips, filename, mean_hr):
        """Save all the chunked data with mean_hr metadata.

        Args:
            frames_clips(np.array): video frames clips
            hr_bvps_clips(np.array): hr bvp signal clips
            spo2_bvps_clips(np.array): spo2 bvp signal clips
            filename: name the filename
            mean_hr: mean heart rate from original CSV
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
            mean_hr_path_name = self.cached_path + os.sep + "{0}_meanhr{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, np.array([hr_bvps_clips[i], spo2_bvps_clips[i]]))
            np.save(mean_hr_path_name, np.array([mean_hr]))  # Save mean HR for this video
            count += 1
        return input_path_name_list, label_path_name_list

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        df = pd.read_csv(bvp_file, header=None)
        rppg = df.iloc[:, 1].astype(float).values
        spo2_bvp = df.iloc[:, 3].astype(float).values
        mean_hr = np.mean(df.iloc[:, 2].astype(float).values)
        return rppg, spo2_bvp, mean_hr