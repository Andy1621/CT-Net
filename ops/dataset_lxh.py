# -*- coding: utf-8 -*-
#  @Author: KunchangLi
#  @Date: 2020-02-07 17:32:18
#  @LastEditor: KunchangLi
#  @LastEditTime: 2020-03-06 08:47:29
#  @Oringin: https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/dataset.py

import gc
import math
import os

import numpy as np
from numpy.random import randint
import torch.utils.data as data
from PIL import Image

__all__ = ['TSNDataSet']


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, sampler=None, twice_sample=False, seed = 0):

        self.root_path = root_path
        self.sampler = sampler
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense new 80 sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self.rng = np.random.RandomState(seed)
        self.speed = [1.0, 1.0]

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

        del tmp
        gc.collect()

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            range_max = record.num_frames
            assert range_max > 0, \
                ValueError("range_max = {}".format(range_max))
            interval = self.rng.choice([80 // self.num_segments])
            if self.num_segments == 1:
                return [self.rng.choice(range(1, range_max))]
            # sampling
            speed_min = self.speed[0]
            speed_max = min(self.speed[1], (range_max - 1) / ((self.num_segments - 1) * interval))
            if speed_max < speed_min:
                average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif record.num_frames > self.num_segments:
                    offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
                else:
                    offsets = np.zeros((self.num_segments,))
                return offsets + 1
            random_interval = self.rng.uniform(speed_min, speed_max) * interval
            frame_range = (self.num_segments - 1) * random_interval
            if (range_max - 1) - frame_range == 0:
                clip_start = 1
            else:
                clip_start = self.rng.uniform(1, (range_max - 1) - frame_range)

            clip_end = clip_start + frame_range
            return np.linspace(clip_start, clip_end, self.num_segments).astype(dtype=np.int).tolist()

        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            interval = 80 // self.num_segments
            valid_start_range = record.num_frames - (self.num_segments - 1) * interval
            start_idx = int(valid_start_range / 2.0)
            if start_idx <= 0:
                if record.num_frames > self.num_segments + self.new_length - 1:
                    tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                else:
                    offsets = np.zeros((self.num_segments,))
                return offsets + 1

            offsets = []
            for i in range(self.num_segments):
                offsets.append(start_idx + i * interval + 1)
            return np.array(offsets)
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            offsets = []
            range_max = record.num_frames
            num_times = 10
            interval = 80 // self.num_segments
            frame_range = (self.num_segments - 1) * interval + 1

            if frame_range > range_max:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                if tick < 1:
                    for i in range(10):
                        offsets += [int(0) for x in range(self.num_segments)]
                    return np.array(offsets) + 1
                else:
                    start_list = np.linspace(0, tick - 1, num=10, dtype=int)
                    offsets = []
                    for start_idx in start_list.tolist():
                        offsets += [int(x * tick + start_idx) for x in range(self.num_segments)]
                    return np.array(offsets) + 1

            if range_max - frame_range * num_times == 0:
                clips = [x * frame_range for x in range(0, num_times)]
            elif range_max - frame_range * num_times > 0:
                step_size = (range_max - frame_range * num_times) / float(num_times + 1) + frame_range
                clips = [math.ceil(x * step_size - frame_range) for x in range(1, num_times + 1)]
            else:
                step_size = (range_max - frame_range * num_times) / float(num_times - 1) + frame_range
                clips = [int(x * step_size) for x in range(0, num_times)]
            for cursor in range(num_times):
                idx = range(clips[cursor] + 1, clips[cursor] + frame_range + 1, interval)
                offsets += list(idx)

            return np.array(offsets)

        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])
            return offsets + 1

        else:
            # interval = 64 // self.num_segments
            # valid_start_range = record.num_frames - (self.num_segments - 1) * interval
            # start_idx = int(valid_start_range / 2.0)
            # if start_idx < 0:
            #     start_idx = 0
            # offsets = []
            # for i in range(self.num_segments):
            #     offsets.append(start_idx + i * interval + 1)
            # return np.array(offsets)
            # interval = 80 // self.num_segments
            # valid_start_range = record.num_frames - (self.num_segments - 1) * interval
            # start_idx = int(valid_start_range / 2.0)
            # if start_idx <= 0:
            #     if record.num_frames > self.num_segments + self.new_length - 1:
            #         tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            #         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            #     else:
            #         offsets = np.zeros((self.num_segments,))
            #     return offsets + 1

            # offsets = []
            # for i in range(self.num_segments):
            #     offsets.append(start_idx + i * interval + 1)
            # return np.array(offsets)
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video fold
        # er

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            #if p > record.num_frames:
            #    p = record.num_frames
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        # return process_data, record.label, record.path
        return process_data, record.label


    def __len__(self):
        return len(self.video_list)
