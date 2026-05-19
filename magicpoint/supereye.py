#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch

# Optional imports — only needed for the standalone demo (__main__ block).
# The VO pipeline (SuperPointFrontend.run) does NOT need these.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from pyapriltags import Detector as _AprilDetector
except ImportError:
    _AprilDetector = None
# from height_simulator import AltitudeSimulator  # Uncomment if available

# ----------------------------
# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

# ----------------------------
# SuperPoint Network and Frontend.
class SuperPointNet(torch.nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc

class SuperPointFrontend(object):
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8
        self.border_remove = 4
        self.net = SuperPointNet()
        if cuda:
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W), dtype=int)
        inds = np.zeros((H, W), dtype=int)
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = np.round(corners[:2, :]).astype(int)
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0), dtype=int), np.zeros(0, dtype=int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1), dtype=int)
        for i, rc in enumerate(rcorners.T):
            grid[rc[1], rc[0]] = 1
            inds[rc[1], rc[0]] = i
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        # img is expected to be float32 in [0,1] at neural input resolution.
        assert img.ndim == 2 and img.dtype == np.float32
        H, W = img.shape
        inp = img.reshape(1, H, W)
        inp = torch.from_numpy(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        with torch.no_grad():
            outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        semi = semi.data.cpu().numpy().squeeze()
        dense = np.exp(semi)
        dense = dense / (np.sum(dense, axis=0) + 1e-5)
        nodust = dense[:-1, :, :]
        Hc, Wc = int(H / self.cell), int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]
        bord = self.border_remove
        toremove = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremove |= np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        pts = pts[:, ~toremove]
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (W / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (H / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous().view(1, 1, -1, 2).float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap

# ----------------------------
# PointTracker (using neural displacement from SuperPoint).
class PointTracker(object):
    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be >= 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = [np.zeros((2, 0)) for _ in range(self.maxl)]
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        keep = scores < nn_thresh
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = keep & keep_bi
        idx = idx[keep]
        scores = scores[keep]
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        offsets = [0]
        for i in range(len(self.all_pts) - 1):
            offsets.append(self.all_pts[i].shape[1])
        return np.cumsum(offsets)

    def update(self, pts, desc):
        if pts is None or desc is None or pts.shape[1] == 0:
            remove_size = self.all_pts[0].shape[1]
            self.all_pts.pop(0)
            self.all_pts.append(pts)
            return
        if self.last_desc is None or self.last_desc.shape[1] == 0:
            self.last_desc = desc.copy()
            offsets = self.get_offsets()
            new_ids = np.arange(pts.shape[1]) + offsets[-1]
            new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
            new_tracks[:, -1] = new_ids
            new_tracks[:, 0] = self.track_count + np.arange(new_ids.shape[0])
            new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
            self.tracks = np.vstack((self.tracks, new_tracks))
            self.track_count += new_ids.shape[0]
            return
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        self.tracks = np.delete(self.tracks, 2, axis=1)
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        matched = np.zeros((pts.shape[1]), dtype=bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found[0, 0])
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    self.tracks[row, 1] = match[2]
                else:
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1 - frac) * self.tracks[row, 1] + frac * match[2]
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        if min_length < 1:
            raise ValueError('min_length too small.')
        valid = np.ones((self.tracks.shape[0]), dtype=bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        not_headless = (self.tracks[:, -1] != -1)
        keepers = valid & good_len & not_headless
        return self.tracks[keepers, :].copy()

    def draw_tracks(self, out, tracks):
        pts_mem = self.all_pts
        N = len(pts_mem)
        offsets = self.get_offsets()
        stroke = 1
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                if pts_mem[i].shape[1] == 0 or pts_mem[i + 1].shape[1] == 0:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                if idx1 < 0 or idx1 >= pts_mem[i].shape[1]:
                    continue
                if idx2 < 0 or idx2 >= pts_mem[i + 1].shape[1]:
                    continue
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                if i == N - 2:
                    cv2.circle(out, p2, stroke, (255, 0, 0), -1, lineType=16)

    def compute_mean_displacement(self):
        pts_mem = self.all_pts
        if len(pts_mem) < 2:
            return 0.0, 0.0
        offsets = self.get_offsets()
        displacements = []
        for track in self.tracks:
            if track[-2] == -1 or track[-1] == -1:
                continue
            offset_prev = offsets[-2]
            offset_last = offsets[-1]
            idx_prev = int(track[-2] - offset_prev)
            idx_last = int(track[-1] - offset_last)
            if idx_prev < 0 or idx_prev >= pts_mem[-2].shape[1]:
                continue
            if idx_last < 0 or idx_last >= pts_mem[-1].shape[1]:
                continue
            pt_prev = pts_mem[-2][:2, idx_prev]
            pt_last = pts_mem[-1][:2, idx_last]
            displacements.append(pt_last - pt_prev)
        if len(displacements) == 0:
            return 0.0, 0.0
        mean_disp = np.mean(np.array(displacements), axis=0)
        return mean_disp[0], mean_disp[1]

# ----------------------------
# VideoStreamer.
class VideoStreamer(object):
    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]  # Raw image size.
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> Processing Image Directory Input.')
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError('No images found (check img_glob parameter)')

    def read_image(self, impath, img_size):
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
        return (grayim.astype('float32') / 255.)

    def next_frame(self):
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if not ret:
                print('VideoStreamer: Cannot get image from camera.')
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            # Get raw image at full resolution.
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]), interpolation=cv2.INTER_AREA)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float') / 255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        self.i += 1
        return (input_image.astype('float32'), True)

# ----------------------------
# NeuralHeightDisplacement Module.
class NeuralHeightDisplacement:
    def __init__(self, focal_length, alpha=0.7):
        self.focal_length = focal_length
        self.alpha = alpha
        self.smoothed_pixel_disp = np.array([0.0, 0.0], dtype=np.float32)
        self.smoothed_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.net_disp = np.array([0.0, 0.0], dtype=np.float32)

    def update(self, pixel_disp, dt, altitude):
        self.smoothed_pixel_disp = self.alpha * self.smoothed_pixel_disp + (1 - self.alpha) * pixel_disp
        if dt > 0:
            raw_vel = self.smoothed_pixel_disp / dt
        else:
            raw_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.smoothed_vel = self.alpha * self.smoothed_vel + (1 - self.alpha) * raw_vel
        real_vel = (self.smoothed_vel * altitude) / self.focal_length
        self.net_disp = self.net_disp + real_vel * dt
        return real_vel, self.net_disp

# ----------------------------
# AprilTagDetector.
class AprilTagDetector:
    def __init__(self, camera_matrix, dist_coeffs, tag_size=0.05, tag_family="tag25h9"):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)
        if _AprilDetector is None:
            raise ImportError("pyapriltags is not installed. Install it or don't use AprilTagDetector.")
        self.detector = _AprilDetector(families=tag_family)

    def detect(self, gray):
        results = self.detector.detect(gray)
        if results:
            tag = results[0]
            corners = tag.corners.astype(np.float32)
            reordered = np.array([corners[3], corners[2], corners[1], corners[0]], dtype=np.float32)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.object_points, reordered, self.camera_matrix, self.dist_coeffs,
                reprojectionError=8.0, iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
            if retval and np.isfinite(rvec).all() and np.isfinite(tvec).all():
                return rvec, tvec, reordered
        return None, None, None

# ----------------------------
# Main Function.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory, video file, or "camera"')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file')
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob pattern for images if directory is used')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if video or directory')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs')
    parser.add_argument('--H', type=int, default=480,
                        help='Raw image height')
    parser.add_argument('--W', type=int, default=640,
                        help='Raw image width')
    parser.add_argument('--nn_scale', type=float, default=0.5,
                        help='Scale factor for neural network input (e.g., 0.5 for half resolution)')
    parser.add_argument('--display_scale', type=int, default=1,
                        help='Scaling factor for display window (applied to raw image)')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum track length')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum track length')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='NMS distance')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold')
    parser.add_argument('--camid', type=int, default=0,
                        help='Webcam capture ID')
    parser.add_argument('--waitkey', type=int, default=30,
                        help='OpenCV waitKey time (ms)')  # Set to 30ms to allow key press
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for network')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images')
    parser.add_argument('--write', action='store_true',
                        help='Write output frames to directory')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Output directory for frames')
    opt = parser.parse_args()
    print(opt)

    # Raw image resolution.
    raw_H = opt.H
    raw_W = opt.W
    # Neural network input resolution.
    nn_scale = opt.nn_scale
    nn_H = int(raw_H * nn_scale)
    nn_W = int(raw_W * nn_scale)

    # Camera calibration (example values)
    camera_matrix = np.array([[1422.26, 0, 932.47],
                              [0, 1423.41, 548.13],
                              [0, 0, 1]])
    dist_coeffs = np.array([0.065173, -0.349857, 0.00016448, -0.00102321, 0.421465])
    tag_size = 0.05  # 5 cm
    focal_length = camera_matrix[0, 0]

    april_detector = AprilTagDetector(camera_matrix, dist_coeffs, tag_size)
    vs = VideoStreamer(opt.input, opt.camid, raw_H, raw_W, opt.skip, opt.img_glob)
    print('==> Loading pre-trained network.')
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda)
    print('==> Successfully loaded pre-trained network.')
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)
    neural_disp_estimator = NeuralHeightDisplacement(focal_length, alpha=0.7)

    if not opt.no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    else:
        print('Skipping display.')

    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    if opt.write:
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)

    # Histories for plotting.
    frame_indices = []
    april_disp_history = []
    neural_disp_history = []
    april_frame_disp_history = []
    neural_frame_disp_history = []

    prev_april_pose = None
    prev_april_time = time.time()
    alpha_april = 0.5
    smoothed_april_disp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    smoothed_april_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    april_disp = np.array([0.0, 0.0], dtype=np.float32)
    prev_altitude = None

    prev_time_nn = time.time()

    print('==> Running Demo.')
    while True:
        start = time.time()
        raw_img, status = vs.next_frame()
        if not status:
            break

        # ----- AprilTag processing on raw image -----
        raw_img_uint8 = (raw_img * 255).astype(np.uint8)
        current_time_april = time.time()
        rvec, tvec, _ = april_detector.detect(raw_img_uint8)
        if rvec is not None and tvec is not None:
            current_pose_cm = tvec.flatten() * 100  # in cm
            current_altitude = current_pose_cm[2] if current_pose_cm[2] > 0 else None
            if current_altitude is not None:
                if prev_altitude is None:
                    avg_altitude = current_altitude
                else:
                    avg_altitude = (prev_altitude + current_altitude) / 2.0
                prev_altitude = current_altitude
            else:
                avg_altitude = 100.0
            if prev_april_pose is not None:
                dt_april = current_time_april - prev_april_time
                raw_disp = current_pose_cm - prev_april_pose  # frame-to-frame raw displacement (3D)
                frame_disp_april = raw_disp[:2]  # use x,y for instantaneous displacement
                if np.linalg.norm(raw_disp[:2]) > 10.0:
                    raw_disp = np.array([0, 0, 0], dtype=np.float32)
                    frame_disp_april = np.array([0, 0], dtype=np.float32)
                smoothed_april_disp = alpha_april * smoothed_april_disp + (1 - alpha_april) * raw_disp
                if dt_april > 0:
                    raw_april_vel = smoothed_april_disp / dt_april
                    smoothed_april_vel = alpha_april * smoothed_april_vel + (1 - alpha_april) * raw_april_vel
                    april_disp = april_disp + smoothed_april_vel[:2] * dt_april
            else:
                frame_disp_april = np.array([0, 0], dtype=np.float32)
                smoothed_april_disp[:] = 0
                smoothed_april_vel[:] = 0
            prev_april_pose = current_pose_cm.copy()
            prev_april_time = current_time_april
            april_text = f"AprilCumDisp: {april_disp[0]:.2f}, {april_disp[1]:.2f} cm"
            april_frame_text = f"AprilFrameDisp: {frame_disp_april[0]:.2f}, {frame_disp_april[1]:.2f} cm"
        else:
            april_text = "April: No detection"
            april_frame_text = "AprilFrameDisp: 0.00, 0.00 cm"
            avg_altitude = prev_altitude if prev_altitude is not None else 100.0

        # ----- Neural processing on resized image -----
        neural_img = cv2.resize(raw_img, (nn_W, nn_H), interpolation=cv2.INTER_AREA)
        pts, desc, heatmap = fe.run(neural_img)
        # Upscale neural keypoints to raw image coordinates.
        if pts is not None and pts.shape[1] > 0:
            pts[0, :] = pts[0, :] / nn_scale
            pts[1, :] = pts[1, :] / nn_scale
        tracker.update(pts, desc)
        tracks = tracker.get_tracks(opt.min_length)
        out_img = cv2.cvtColor((raw_img * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        if tracks.size > 0:
            tracks[:, 1] /= float(fe.nn_thresh)
            tracker.draw_tracks(out_img, tracks)
        disp_pixels = np.array(tracker.compute_mean_displacement())
        # Compute frame-by-frame neural displacement (in cm)
        if disp_pixels.size > 0:
            neural_frame_disp = (disp_pixels * avg_altitude) / focal_length
        else:
            neural_frame_disp = np.array([0.0, 0.0])
        current_time_nn = time.time()
        dt_nn = current_time_nn - prev_time_nn
        prev_time_nn = current_time_nn
        real_vel, net_disp = neural_disp_estimator.update(disp_pixels, dt_nn, avg_altitude)
        neural_text = f"NeuralCumDisp: {net_disp[0]:.2f}, {net_disp[1]:.2f} cm"
        neural_frame_text = f"NeuralFrameDisp: {neural_frame_disp[0]:.2f}, {neural_frame_disp[1]:.2f} cm"

        # ----- Overlay text on display image -----
        cv2.putText(out_img, april_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(out_img, april_frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(out_img, neural_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out_img, neural_frame_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Resize display image.
        display_img = cv2.resize(out_img, (out_img.shape[1] * opt.display_scale, out_img.shape[0] * opt.display_scale), interpolation=cv2.INTER_LINEAR)
        if not opt.no_display:
            cv2.imshow(win, display_img)
            key = cv2.waitKey(opt.waitkey) & 0xFF  # waitkey is now set to 30 ms by default
            if key == ord('q'):
                break
        if opt.write:
            out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
            cv2.imwrite(out_file, display_img)

        frame_indices.append(vs.i)
        april_disp_history.append((april_disp[0], april_disp[1]))
        neural_disp_history.append((net_disp[0], net_disp[1]))
        april_frame_disp_history.append((frame_disp_april[0], frame_disp_april[1]))
        neural_frame_disp_history.append((neural_frame_disp[0], neural_frame_disp[1]))

    cv2.destroyAllWindows()
    print('==> Finished Demo.')

    # ----- Plotting -----
    frame_indices = np.array(frame_indices)
    april_disp_history = np.array(april_disp_history)
    neural_disp_history = np.array(neural_disp_history)
    april_frame_disp_history = np.array(april_frame_disp_history)
    neural_frame_disp_history = np.array(neural_frame_disp_history)

    # Plot integrated (cumulative) displacement norms.
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, np.linalg.norm(april_disp_history, axis=1), 'r-', label='AprilTag CumDisp Norm (cm)')
    plt.plot(frame_indices, np.linalg.norm(neural_disp_history, axis=1), 'g-', label='Neural CumDisp Norm (cm)')
    plt.xlabel('Frame Index')
    plt.ylabel('Integrated Displacement Norm (cm)')
    plt.title('Integrated Displacement Norms Over Time')
    plt.legend()
    plt.grid(True)

    # Plot frame-by-frame displacement norms.
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, np.linalg.norm(april_frame_disp_history, axis=1), 'r-', label='AprilTag FrameDisp Norm (cm)')
    plt.plot(frame_indices, np.linalg.norm(neural_frame_disp_history, axis=1), 'g-', label='Neural FrameDisp Norm (cm)')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame-by-Frame Displacement Norm (cm)')
    plt.title('Frame-by-Frame Displacement Norms Over Time')
    plt.legend()
    plt.grid(True)

    # Plot individual X and Y components for integrated displacement.
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(frame_indices, april_disp_history[:, 0], 'r-', label='AprilTag CumDisp X (cm)')
    plt.plot(frame_indices, neural_disp_history[:, 0], 'g-', label='Neural CumDisp X (cm)')
    plt.xlabel('Frame Index')
    plt.ylabel('X Displacement (cm)')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(frame_indices, april_disp_history[:, 1], 'r--', label='AprilTag CumDisp Y (cm)')
    plt.plot(frame_indices, neural_disp_history[:, 1], 'g--', label='Neural CumDisp Y (cm)')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Displacement (cm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
