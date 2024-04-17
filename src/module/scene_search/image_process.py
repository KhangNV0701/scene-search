import numpy as np
import torch

from src.module.scene_search.superpoint import SuperPoint

class ImageProcess(object):
    def __init__(self, weight_path, cuda = True):
        self.name = 'SuperPoint'
        self.cuda = cuda

        self.nms_dist = 4
        self.conf_thresh = 0.015
        self.nn_thresh = 0.7

        self.cell = 8
        self.border_dist = 4

        self.image_net = SuperPoint()

        if cuda:
            self.image_net.load_state_dict(torch.load(weight_path))
            self.image_net = self.image_net.cuda()
        else:
            self.image_net.load_state_dict(
                torch.load(weight_path,
                           map_location = lambda storage, loc: storage)
            )

        self.image_net.eval()

    def non_maximum_suppression(self, corners, h, w, dist_thresh):
        grid = np.zeros((h, w)).astype(int)
        inds = np.zeros((h, w)).astype(int)

        sorted_corners_inds = np.argsort(-corners[2, :])
        sorted_corners = corners[:, sorted_corners_inds]
        rounded_corners = sorted_corners[:2, :].round().astype(int)

        if rounded_corners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

        if rounded_corners.shape[1] == 1:
            return np.vstack((rounded_corners, corners[2])).reshape(3, 1), np.zeros((1)).astype(int) # MEM: put up
        # print('rounded corner: ', rounded_corners)
        for i in range(rounded_corners.shape[1]):
            grid[rounded_corners[1, i], rounded_corners[0, i]] = 1
            inds[rounded_corners[1, i], rounded_corners[0, i]] = i

        padding_value = dist_thresh
        grid = np.pad(grid, ((padding_value, padding_value),
                             (padding_value, padding_value)), mode = 'constant')

        for i, rounded_corner in enumerate(rounded_corners.T):
            point = (rounded_corner[0] + padding_value,
                     rounded_corner[1] + padding_value)
            if grid[point[1], point[0]] == 1:
                grid[point[1] - padding_value:point[1] + padding_value + 1,
                     point[0] - padding_value:point[0] + padding_value + 1] = 0
                grid[point[1], point[0]] = -1

        keep_xy = np.where(grid == -1)
        keep_x, keep_y = keep_xy[0] - padding_value, keep_xy[1] - padding_value

        inds_keep = inds[keep_x, keep_y]
        output = sorted_corners[:, inds_keep]
        output_values = output[2, :]
        sorted_output_inds = np.argsort(-output_values)
        output = output[:, sorted_output_inds]
        output_inds = sorted_corners_inds[inds_keep[sorted_output_inds]]

        return output, output_inds

    def process(self, imgs):
        n, h, w = imgs.shape

        input_imgs = imgs.copy()
        input_imgs = np.expand_dims(input_imgs, 1)

        input_tensor = torch.from_numpy(input_imgs)
        input_tensor = torch.autograd.Variable(input_tensor).view(n, 1, h, w) # n batch with 1-channel

        if self.cuda:
            input_tensor = input_tensor.cuda()

        points, descs = self.image_net.forward(input_tensor)
        point_list, desc_list, heatmap_list = [], [], []

        for i in range(points.shape[0]):
            point = points[i].unsqueeze(0)
            point = point.data.cpu().numpy().squeeze() #numpy is faster on cpu

            probs = np.exp(point)
            probs /= np.sum(probs, axis = 0) + 0.0001

            no_dustbin_probs = probs[:64, :, :]

            h_cell = int(h / self.cell)
            w_cell = int(w / self.cell)

            no_dustbin_probs = no_dustbin_probs.transpose(1, 2, 0)

            heatmap = np.reshape(no_dustbin_probs, [h_cell, w_cell, self.cell, self.cell])
            heatmap = np.transpose(heatmap, [0, 2, 1, 3])
            heatmap = np.reshape(heatmap, [h_cell * self.cell, w_cell * self.cell])

            x_choose, y_choose = np.where(heatmap >= self.conf_thresh)

            if len(x_choose) == 0:
                point_list.append(np.zeros((3, 0)))
                desc_list.append(None)
                heatmap_list.append(None)

            point_data = np.zeros((3, len(x_choose)))
            point_data[0, :] = y_choose
            point_data[1, :] = x_choose
            point_data[2, :] = heatmap[x_choose, y_choose]

            point_data, _ = self.non_maximum_suppression(point_data, h, w, self.nms_dist)

            w_choose_coors = np.logical_or(point_data[0, :] < self.border_dist,
                                           point_data[0, :] >= (w - self.border_dist))

            h_choose_coors = np.logical_or(point_data[1, :] < self.border_dist,
                                           point_data[1, :] >= (h - self.border_dist))

            choose_coors = np.logical_or(w_choose_coors, h_choose_coors)
            point_data = point_data[:, ~choose_coors]

            desc = descs[i].unsqueeze(0)
            d = desc.shape[1]

            if point_data.shape[1] == 0:
                refined_desc = np.zeros((d, 0))
            else:
                sample_points = torch.from_numpy(point_data[:2, :].copy())
                sample_points[0, :] = (sample_points[0, :] / (float(w) / 2.)) - 1.
                sample_points[1, :] = (sample_points[1, :] / (float(h) / 2.)) - 1.

                sample_points = sample_points.transpose(0, 1).contiguous().view(1, 1, -1, 2).float()

                if self.cuda:
                    sample_points = sample_points.cuda()

                refined_desc = torch.nn.functional.grid_sample(desc, sample_points, align_corners=True)
                refined_desc = refined_desc.data.cpu().numpy().reshape(d, -1)
                refined_desc /= np.linalg.norm(refined_desc, axis = 0)[np.newaxis, :]

            point_list.append(point_data)
            desc_list.append(refined_desc)
            heatmap_list.append(heatmap)

        return point_list, desc_list, heatmap_list