import cv2
import numpy as np

class VideoProcess(object):
    def __init__(self, h_ratio = 0.33, w_ratio = 0.33):
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.video_capture = None
        self.num_frames = 0
        self.cur_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.no_frames = 0

        self.video_file_types = ['mp4']

    def open_video(self, video_dir):
        self.video_capture = cv2.VideoCapture(video_dir)
        self.cur_frames = 0
        self.weight, self.height = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.no_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        file_type = video_dir.split('.')[-1]
        if (type(self.video_capture) != list
            or not self.cap.isOpened()) and (file_type not in self.video_file_types):
            raise IOError('Cannot open video file at dir:', video_dir)
        elif type(self.video_capture) != list and self.video_capture.isOpened() and file_type in self.video_file_types:
            self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def next_frame(self, subsample_rate = 0.5):
        if self.cur_frames == self.num_frames:
            return (None, None, 'max_len')

        ret, input_img = self.video_capture.read()
        if ret is False:
            return (None, None, False)

        self.cur_frames += 1

        if self.cur_frames % 2 != 0:
            return (None, None, 'frame_skip')

        input_image = cv2.resize(input_img,
                                 (int(input_img.shape[1] * subsample_rate),
                                  int(input_img.shape[0] * subsample_rate)))
        image_shape = input_image.shape[:2]
        patch_shape = np.asarray([int(input_image.shape[0] * self.h_ratio),
                                  int(input_image.shape[1] * self.w_ratio)])
        border_remove = 5

        center_image_anchor = (image_shape - patch_shape) // 2
        center_image = input_image[
            center_image_anchor[0]:center_image_anchor[0] + patch_shape[0],
            center_image_anchor[1]:center_image_anchor[1] + patch_shape[1],
        ].copy()
        center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2GRAY)

        left_up_image = input_image[
            border_remove:patch_shape[0] + border_remove,
            border_remove:patch_shape[1] + border_remove
        ].copy()
        left_up_image = cv2.cvtColor(left_up_image, cv2.COLOR_RGB2GRAY)

        right_up_image_anchor = [border_remove,
                                 image_shape[1] - patch_shape[1] - border_remove]
        right_up_image = input_image[
            right_up_image_anchor[0]:right_up_image_anchor[0] + patch_shape[0],
            right_up_image_anchor[1]:right_up_image_anchor[1] + patch_shape[1]
        ].copy()
        right_up_image = cv2.cvtColor(right_up_image, cv2.COLOR_RGB2GRAY)

        left_down_image_anchor = [image_shape[0] - patch_shape[0] - border_remove,
                                  border_remove]
        left_down_image = input_image[
            left_down_image_anchor[0]:left_down_image_anchor[0] + patch_shape[0],
            left_down_image_anchor[1]:left_down_image_anchor[1] + patch_shape[1]
        ].copy()
        left_down_image = cv2.cvtColor(left_down_image, cv2.COLOR_RGB2GRAY)

        right_down_image_anchor = [image_shape[0] - patch_shape[0] - border_remove,
                                   image_shape[1] - patch_shape[1] - border_remove]
        right_down_image = input_image[
            right_down_image_anchor[0]:right_down_image_anchor[0] + patch_shape[0],
            right_down_image_anchor[1]:right_down_image_anchor[1] + patch_shape[1]
        ].copy()
        right_down_image = cv2.cvtColor(right_down_image, cv2.COLOR_RGB2GRAY)

        patches = np.array([center_image,
                            left_up_image,
                            right_up_image,
                            left_down_image,
                            right_down_image]).astype('float32') / 255.0

        return (input_image, patches, True)