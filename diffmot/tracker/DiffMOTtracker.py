import itertools
import os
import os.path as osp
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffmot.models import *
from diffmot.tracker import matching
from diffmot.tracking_utils.kalman_filter import KalmanFilter
from diffmot.tracking_utils.log import logger
from diffmot.tracking_utils.utils import *

from .basetrack import BaseTrack, TrackState
from .cmc import CMCComputer
from .embedding import EmbeddingComputer
from .gmc import GMC


class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, label, frame_id, temp_feat=None, buffer_size=30):

        self._tlwh = tlwh
        self.emb = temp_feat
        self.buffer_size = buffer_size
        
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = 10
        self.score_thresh = 15

        self.frame_id = frame_id
        self.start_frame = frame_id

        self.tracklet_len = 0

        self.xywh_omemory = deque([self.xywh.copy()], maxlen=buffer_size)
        self.xywh_pmemory = deque([self.xywh.copy()], maxlen=buffer_size)
        self.xywh_amemory = deque([self.xywh.copy()], maxlen=buffer_size)

        self.label_memory = deque([label])
        self.label_counter = Counter()
        self.label_counter[label] += 1

        self.conds = deque([], maxlen=5)
        tmp_conds = np.concatenate((self.xywh.copy(), self.xywh.copy() - self.xywh.copy()))
        self.conds.append(tmp_conds)

        self.features = deque([], maxlen=buffer_size)

    def update_features(self, feat, alpha=0.95):
        self.curr_feat = feat
        self.emb = alpha * self.emb + (1 - alpha) * feat
        self.emb /= np.linalg.norm(self.emb)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_diff(stracks, model, img_w, img_h):
        if len(stracks) > 0:
            dets = np.asarray([st.xywh.copy() for st in stracks]).reshape(-1, 4)

            dets[:, 0::2] = dets[:, 0::2] / img_w
            dets[:, 1::2] = dets[:, 1::2] / img_h

            conds = [st.conds for st in stracks]

            multi_track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            track_pred = multi_track_pred.mean(0)

            track_pred = track_pred + dets

            track_pred[:, 0::2] = track_pred[:, 0::2] * img_w
            track_pred[:, 1::2] = track_pred[:, 1::2] * img_h
            track_pred[:, 0] = track_pred[:, 0] - track_pred[:, 2] / 2
            track_pred[:, 1] = track_pred[:, 1] - track_pred[:, 3] / 2

            for i, st in enumerate(stracks):
                st._tlwh = track_pred[i]
                st.xywh_pmemory.append(st.xywh.copy())
                st.xywh_amemory.append(st.xywh.copy())

                tmp_delta_bbox = st.xywh.copy() - st.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((st.xywh.copy(), tmp_delta_bbox))
                st.conds.append(tmp_conds)

    def re_activate(self, new_track, frame_id, new_id=False):
        new_tlwh = new_track.tlwh
        self._tlwh = new_tlwh
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
        tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
        self.conds[-1] = tmp_conds

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, det, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.score = min(self.score + 1, 30)
        if self.score >= 15 and self.state == TrackState.New:
            self.track_id = self.next_id()
            self.state = TrackState.Tracked

        self._tlwh = det.tlwh
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_amemory[-1] = self.xywh.copy()

        if self.is_activated == True:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_amemory[-2].copy()
            tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds
        else:
            tmp_delta_bbox = self.xywh.copy() - self.xywh_omemory[-2].copy()
            tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds

        self.label_memory.append(det.label)
        self.label_counter[det.label] += 1

        if len(self.label_memory) > self.buffer_size:
            oldest_value = self.label_memory.popleft()
            self.label_counter[oldest_value] -= 1
            if self.label_counter[oldest_value] == 0:
                del self.label_counter[oldest_value]

    @property
    def label(self):
        return self.label_counter.most_common(1)[0][0] if len(self.label_counter) != 0 else self._label

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        # ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class diffmottracker(object):
    def __init__(self, config, model, frame_rate=30):
        self.model = model

        self.config = config

        self.tracks = []

        self.frame_id = 0
        self.det_thresh = self.config.high_thres

        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size

        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

        self.embedder = EmbeddingComputer(self.config, "sports", False, True)
        self.alpha_fixed_emb = 0.95

    def dump_cache(self):
        # self.cmc.dump_cache()
        self.embedder.dump_cache()

    def associate(self, detections, dets_embs):
        if len(detections) > 0:
            iou_matrix = 1 - matching.iou_distance(self.tracks, detections)
            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > 0.1).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    track_embs = np.array([st.emb for st in self.tracks])
                    emb_cost = 0 if (track_embs.shape[0] == 0 or dets_embs.shape[0] == 0) else track_embs @ dets_embs.T

                    w_matrix = matching.compute_aw_new_metric(emb_cost, self.config.w_assoc_emb, self.config.aw_param)
                    emb_cost *= w_matrix

                    final_cost = -(iou_matrix + emb_cost)
                    matched_indices = matching.linear_assignment2(final_cost)
            else:
                matched_indices = np.empty(shape=(0, 2))

            unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 1]]
            unmatched_tracks = [t for t in range(len(self.tracks)) if t not in matched_indices[:, 0]]

            # filter out matched with low IOU
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < 0.1:
                    unmatched_dets.append(m[1])
                    unmatched_tracks.append(m[0])
                else:
                    matches.append(m.reshape(1, 2))

            matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        return matches, unmatched_dets, unmatched_tracks

    def update(self, matches, detections, dets_alpha):
        for ind_track, ind_det in matches:
            track = self.tracks[ind_track]
            det = detections[ind_det]
            alp = dets_alpha[ind_det]

            track.update(det, self.frame_id)
            track.update_features(det.emb, alp)

    def next(self, dets, img_w, img_h, tag, img=None):

        self.frame_id += 1

        # Step 1: Propagate tracks
        STrack.multi_predict_diff(self.tracks, self.model, img_w, img_h)

        # Step 2: Associate
        dets = dets[dets[:, 5] > self.det_thresh].cpu().numpy()
        dets_embs = self.embedder.compute_embedding(img, dets[:, :4], tag) if dets.shape[0] != 0 else np.ones((dets.shape[0], 1))

        detections = [
            STrack(
                tlwh=STrack.tlbr_to_tlwh(tlbrs[:4]),
                label=int(tlbrs[4].item()),
                frame_id=self.frame_id,
                temp_feat=f,
                buffer_size=30,
            ) for (tlbrs, f) in zip(dets, dets_embs)]
        
        matches, unmatched_dets, unmatched_tracks = self.associate(detections=detections, dets_embs=dets_embs)

        # Step 3: Update
        trust = (dets[:, 5] - self.det_thresh) / (1 - self.det_thresh)
        dets_alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
        self.update(matches=matches, detections=detections, dets_alpha=dets_alpha)

        # Step 4: Lifetime Management
        for ind_track in unmatched_tracks:
            self.tracks[ind_track].score -= 1
            
        self.tracks = [track for track in self.tracks if track.score > 0]

        # Step 5: Init new tracks
        for ind_det in unmatched_dets:
            self.tracks.append(detections[ind_det])

        return [track for track in self.tracks if track.state == TrackState.Tracked]

