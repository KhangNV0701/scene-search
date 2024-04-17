import numpy as np

from src.module.scene_search.image_process import ImageProcess
from src.module.scene_search.video_process import VideoProcess
from src.utils.logger import logger

class KeyframeExtractionModule:
    def __init__(self, weight_path, video_path):
        self.weight_path = weight_path
        self.video_path = video_path

    def nearest_neighbor_match_two_way(self, desc_1, desc_2, nn_thresh):
        if desc_1.shape[1] == 0 or desc_2.shape[1] == 0:
            return np.zeros((3, 0))

        l2_matrix = np.dot(desc_1.T, desc_2)
        l2_matrix = 2 - 2 * np.clip(l2_matrix, -1, 1)

        nn_idx_1 = np.argmin(l2_matrix, axis = 1)
        scores = l2_matrix[np.arange(l2_matrix.shape[0]), nn_idx_1]

        keep_scores = scores < nn_thresh

        nn_idx_2 = np.argmin(l2_matrix, axis = 0)
        keep_both_dir = np.arange(len(nn_idx_1)) == nn_idx_2[nn_idx_1]
        keep = np.logical_and(keep_scores, keep_both_dir)

        idx = nn_idx_1[keep]
        scores = scores[keep]

        match_idx_1 = np.arange(desc_1.shape[1])[keep]
        match_idx_2 = idx

        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = match_idx_1
        matches[1, :] = match_idx_2
        matches[2, :] = scores

        return matches

    def extract_keyframe(self):
        logger.info("Scene search - Start extracting keyframes")

        video_process = VideoProcess()
        video_process.open_video(self.video_path)
        image_process = ImageProcess(self.weight_path, cuda=False)

        frames = []
        keyframes = []
        frame_pos = []
        point_list = []
        desc_list = []
        tmp_pos = []

        nearest_neighbor_thresh = 0.7
        min_matches = 10
        extract_dist = 60
        extracted_frame_id = 1

        while True:
            img, patches, status = video_process.next_frame()

            if status is False:
                print("Failed to read")
                continue

            if status == 'max_len': #end of video
                break

            if status == 'frame_skip':
                continue

            results = image_process.process(patches)
            points = [point for point in results[0] if point is not None]
            descs = [desc for desc in results[1] if desc is not None]

            frames.append(img)
            tmp_pos.append(video_process.cur_frames)
            point_list.append(np.concatenate(points, axis = 1))
            desc_list.append(np.concatenate(descs, axis = 1))

            if len(frames) == 2 and len(point_list) == 2 and len(desc_list) == 2:
                matches = self.nearest_neighbor_match_two_way(desc_list[0],
                                                        desc_list[1],
                                                        nearest_neighbor_thresh)

                if matches.shape[1] < min_matches:
                    frames.pop(0)
                    point_list.pop(0)
                    desc_list.pop(0)
                    tmp_pos.pop(0)
                    continue

                key_points_1 = point_list[0][:2, matches[0].astype(int)].transpose((1, 0))
                key_points_2 = point_list[1][:2, matches[1].astype(int)].transpose((1, 0))

                distance = np.mean(np.sqrt(np.sum(np.square(key_points_1 - key_points_2), axis = 1)))

                if distance >= extract_dist:
                    keyframes.append(frames[-1])
                    frame_pos.append(tmp_pos[-1])
                    frames.pop(0)
                    point_list.pop(0)
                    desc_list.pop(0)
                    tmp_pos.pop(0)
                else:
                    frames.pop(1)
                    point_list.pop(1)
                    desc_list.pop(1)
                    tmp_pos.pop(1)
        if len(keyframes) == 0 and len(frames) > 0:
            keyframes.extend(frames)
            frame_pos.extend(tmp_pos)
        logger.info("Scene search - Completed extracting keyframes")
        return keyframes, frame_pos, video_process.fps

    
