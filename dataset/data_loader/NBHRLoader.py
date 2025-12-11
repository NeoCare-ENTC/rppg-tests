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

class NBHRLoader(BaseLoader):
    """The data loader for the NBHR dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an NBHR dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                -----------------
                     |-- PPG/
                     |     |-- 000000000.csv/
                     |     |-- 000000001.csv
                     |-- video/
                     |     |-- 000000000.avi/
                     |     |-- 000000001.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For NBHR dataset)."""
        # Support both 'video' and 'Video' folder names
        video_dir = os.path.join(data_path, "video")
        data_dirs = glob.glob(video_dir + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty! Looking in: " + video_dir)
        
        dirs = []
        for data_dir in data_dirs:
            # Extract filename without extension (e.g., "20250221_120948")
            filename = os.path.split(data_dir)[-1].replace(".avi", "")
            # Use the filename string as the index for matching with PPG files
            dirs.append({"index": filename, "path": data_dir})
        
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
            ppg_file = os.path.join(data_dir, "PPG", "{0}.csv".format(saved_filename))
            
            if not os.path.exists(ppg_file):
                raise FileNotFoundError(f"PPG file not found: {ppg_file}")
            
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