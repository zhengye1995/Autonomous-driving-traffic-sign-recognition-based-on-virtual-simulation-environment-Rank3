import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def box_results_with_nms_and_limit(multi_bboxes, multi_scores, score_thr, nms_cfg, test_cfg, max_num=-1, score_factors=None):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-max suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        cls_inds = multi_scores[:, j] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, j * 4:(j + 1) * 4]
        _scores = multi_scores[cls_inds, j]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets_j = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets_j, **nms_cfg_)
        if test_cfg.bbox_vote.enable:
            cls_dets = box_voting(
                cls_dets,
                cls_dets_j,
                test_cfg.bbox_vote.vote_th
            )
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0],), j - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
    return bboxes, labels


def box_voting(top_dets, all_dets, thresh):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.clone()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]

    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)

    for k in range(top_dets_out.shape[0]):

        mask = top_to_all_overlaps[k].ge(thresh)
        boxes_to_vote = all_boxes[mask, :]
        ws = all_scores[mask]
        # inds_to_vote = torch.where(top_to_all_overlaps[k] >= thresh)[0]

        # boxes_to_vote = all_boxes[inds_to_vote, :]
        # ws = all_scores[inds_to_vote]
        # top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        top_dets_out[k, :4] = torch.sum(torch.mul(boxes_to_vote, torch.unsqueeze(ws, 1)), dim=0) / torch.sum(ws)
        # if scoring_method == 'ID':
        #     # Identity, nothing to do
        #     pass
        # elif scoring_method == 'TEMP_AVG':
        #     # Average probabilities (considered as P(detected class) vs.
        #     # P(not the detected class)) after smoothing with a temperature
        #     # hyperparameter.
        #     P = np.vstack((ws, 1.0 - ws))
        #     P_max = np.max(P, axis=0)
        #     X = np.log(P / P_max)
        #     X_exp = np.exp(X / beta)
        #     P_temp = X_exp / np.sum(X_exp, axis=0)
        #     P_avg = P_temp[0].mean()
        #     top_dets_out[k, 4] = P_avg
        # elif scoring_method == 'AVG':
        #     # Combine new probs from overlapping boxes
        #     top_dets_out[k, 4] = ws.mean()
        # elif scoring_method == 'IOU_AVG':
        #     P = ws
        #     ws = top_to_all_overlaps[k, inds_to_vote]
        #     P_avg = np.average(P, weights=ws)
        #     top_dets_out[k, 4] = P_avg
        # elif scoring_method == 'GENERALIZED_AVG':
        #     P_avg = np.mean(ws**beta)**(1.0 / beta)
        #     top_dets_out[k, 4] = P_avg
        # elif scoring_method == 'QUASI_SUM':
        #     top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        # else:
        #     raise NotImplementedError(
        #         'Unknown scoring method {}'.format(scoring_method)
        #     )

    return top_dets_out


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = torch.zeros((N, K)).cuda()
    area1 = (boxes[:, 2] - boxes[:, 0] + 1) * (
            boxes[:, 3] - boxes[:, 1] + 1)
    area2 = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (
            query_boxes[:, 3] - query_boxes[:, 1] + 1)
    for i in range(boxes.shape[0]):
        x_start = torch.max(boxes[i, 0], query_boxes[:, 0])
        y_start = torch.max(boxes[i, 1], query_boxes[:, 1])
        x_end = torch.min(boxes[i, 2], query_boxes[:, 2])
        y_end = torch.min(boxes[i, 3], query_boxes[:, 3])
        overlap = torch.max(x_end - x_start + 1, torch.tensor(0, dtype=torch.float).cuda()) * \
                  torch.max(y_end - y_start + 1, torch.tensor(0, dtype=torch.float).cuda())
        union = area1[i] + area2 - overlap
        overlaps[i, :] = overlap / union
    # for k in range(K):
    #     box_area = (
    #         (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
    #         (query_boxes[k, 3] - query_boxes[k, 1] + 1)
    #     )
    #     for n in range(N):
    #         iw = (
    #             min(boxes[n, 2], query_boxes[k, 2]) -
    #             max(boxes[n, 0], query_boxes[k, 0]) + 1
    #         )
    #         if iw > 0:
    #             ih = (
    #                 min(boxes[n, 3], query_boxes[k, 3]) -
    #                 max(boxes[n, 1], query_boxes[k, 1]) + 1
    #             )
    #             if ih > 0:
    #                 ua = float(
    #                     (boxes[n, 2] - boxes[n, 0] + 1) *
    #                     (boxes[n, 3] - boxes[n, 1] + 1) +
    #                     box_area - iw * ih
    #                 )
    #                 overlaps[n, k] = iw * ih / ua
    return overlaps