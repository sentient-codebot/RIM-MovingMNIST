import torch
import torch.nn.functional as F
import numpy as np
import motmetrics as mm
from tqdm import tqdm
import json
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from .consistensy_measure import consistency_measure

DEBUG = os.environ.get('DEBUG', False)

@torch.no_grad()
def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.shape == y_pred.shape
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.item()

def exclude_bg_func(dists, gt_ids, pred_ids, n_gt_bg):
    # remove background slots
    gt_idx = -1
    for k in range(n_gt_bg):
        if dists.shape[1] > 0:
            pred_bg_id = np.where(dists[gt_idx] > 0.2)[0]
            dists = np.delete(dists, pred_bg_id, 1)
            pred_ids = [pi for l, pi in enumerate(pred_ids) if not l in pred_bg_id]
        dists = np.delete(dists, gt_idx, 0)  
        del gt_ids[gt_idx]   
    return dists, gt_ids, pred_ids

def binarize_masks(masks):
    ''' Binarize soft masks.
    Args:
        masks: torch.Tensor(CxHxW)
    '''
    n = masks.size(0)
    idc = torch.argmax(masks, axis=0)
    binarized_masks = torch.zeros_like(masks)
    for i in range(n):
        binarized_masks[i] = (idc == i).int()
    return binarized_masks

def binarize_masks_(masks, threshold=0.2):
    """Custom binarization funciton 
    Args:
        masks: torch.Tensor of Shape [K, H, W], ranging from 0. ~ 1.
    """
    K = masks.shape[0]
    binarized_masks = torch.zeros_like(masks)
    for k in range(K):
        binarized_masks[k] = (masks[k] > threshold).int()
    return binarized_masks


def rle_encode(img):
    '''
    from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle(mask_rle, shape):
    '''
    from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def calculate_iou(mask1, mask2):
    ''' Calculate IoU of two segmentation masks.
    Args:
        mask1: HxW
        mask2: HxW
    '''
    eps = np.finfo(float).eps
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)
    union = ((np.sum(mask1) + np.sum(mask2) - np.sum(mask1*mask2)))
    iou = np.sum(mask1*mask2) / (union + eps)
    iou = 1. if union == 0. else iou
    return iou 

def compute_mot_metrics(acc, summary: DataFrame) -> DataFrame:
    ''' Args:
            acc: motmetric accumulator
            summary: pandas dataframe with mometrics summary
    '''
    df = acc.mot_events
    df = df[(df.Type != 'RAW')
            & (df.Type != 'MIGRATE')
            & (df.Type != 'TRANSFER')
            & (df.Type != 'ASCEND')]
    obj_freq = df.OId.value_counts()
    n_objs = len(obj_freq)
    tracked = df[df.Type == 'MATCH']['OId'].value_counts()
    detected = df[df.Type != 'MISS']['OId'].value_counts()

    track_ratios = tracked.div(obj_freq).fillna(0.)
    detect_ratios = detected.div(obj_freq).fillna(0.)

    summary['mostly_tracked'] = track_ratios[track_ratios >= 0.8].count() / n_objs * 100
    summary['mostly_detected'] = detect_ratios[detect_ratios >= 0.8].count() / n_objs * 100

    n = summary['num_objects'][0]
    summary['num_matches']  = (summary['num_matches'][0] / n * 100)
    summary['num_false_positives'] = (summary['num_false_positives'][0] / n * 100)
    summary['num_switches'] = (summary['num_switches'][0] / n * 100)
    summary['num_misses']  = (summary['num_misses'][0] / n * 100)
    
    summary['mota']  = (summary['mota'][0] * 100)
    summary['motp']  = ((1. - summary['motp'][0]) * 100)

    return summary

def gen_masks(batch_size, n_steps, n_slots, id_counter, pred_list, soft_masks):
    for sample_idx in range(batch_size):
        video = []
        obj_ids = np.arange(n_slots) + id_counter
        for t in range(n_steps):
            binarized_masks = binarize_masks_(soft_masks[sample_idx,t]) # [K, H, W]
            binarized_masks = np.array(binarized_masks).astype(np.uint8)

            frame = {}
            masks = []
            ids = []
            for j in range(n_slots):
                # ignore slots with empty masks
                if binarized_masks[j].sum() == 0.:
                    continue
                else:
                    masks.append(rle_encode(binarized_masks[j]))
                    ids.append(int(obj_ids[j]))
            frame['masks'] = masks
            frame['ids'] = ids
            video.append(frame)
        pred_list.append(video)
        id_counter += n_slots
        
    return pred_list

def compute_dists_per_frame(gt_frame, pred_frame, im_size, min_num_pix, exclude_bg, iou_thresh):
    # Compute pairwise distances between gt objects and predictions per frame. 
    s = im_size
    n_pred = len(pred_frame['ids'])
    n_gt = len(gt_frame['ids'])

    # accumulate pred masks for frame
    preds = []
    pred_ids = []
    for j in range(n_pred):
        mask = decode_rle(pred_frame['masks'][j], (s, s))
        if mask.sum() > min_num_pix:
            preds.append(mask)
            pred_ids.append(pred_frame['ids'][j])
    preds = np.array(preds)

    # accumulate gt masks for frame
    gts = []
    gt_ids = []
    for h in range(n_gt):
        mask = decode_rle(gt_frame['masks'][h], (s, s))
        if mask.sum() > min_num_pix:
            gts.append(mask)
            gt_ids.append(gt_frame['ids'][h])
    gts = np.array(gts)

    # compute pairwise distances
    dists = np.ones((len(gts), len(preds)))
    for h in range(len(gts)):
        for j in range(len(preds)): 
            dists[h, j] = calculate_iou(gts[h], preds[j])

    if exclude_bg:
        n_gt_bg = gt_frame['num_bg']
        dists, gt_ids, pred_ids = exclude_bg_func(dists, gt_ids, pred_ids, n_gt_bg)
        
    dists = 1. - dists
    dists[dists > iou_thresh] = np.nan
        
    return dists, gt_ids, pred_ids
        


def accumulate_events(gt_dict, pred_dict, start_step, stop_step, im_size, min_num_pix, exclude_bg, iou_thresh):
    acc = mm.MOTAccumulator()
    count = 0
    for i in tqdm(range(len(pred_dict))):
        for t in range(start_step, stop_step):
            gt_dict_frame = gt_dict[i][t]
            pred_dict_frame = pred_dict[i][t]
            dist, gt_ids, pred_ids = compute_dists_per_frame(
                gt_dict_frame, 
                pred_dict_frame, 
                im_size,
                min_num_pix, 
                exclude_bg, 
                iou_thresh
            )
            acc.update(gt_ids, pred_ids, dist, frameid=count)
            count += 1
    return acc


def get_mot_metrics(pred_file, gt_file, exclude_bg=True, start_step=2, stop_step=10, im_size=64, min_num_pix=5, iou_thresh=0.5) -> dict:
    """get mot metrics
    
    arguments:
        `pred_file`: pred json file
        `gt_file`: ground truth json file
        (options:)
            `exclude_bg`
            `start_step`
            `stop_step`
            `im_size`
            `min_num_pix`
            `iou_thresh`
        """
    with open(pred_file, 'r') as f:
        pred_dict = json.load(f)
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)
    if not DEBUG:
        assert len(pred_dict) == len(gt_dict)
    else:
        if len(pred_dict) != len(gt_dict):
            print('doing incomplete evaluation')
    
    acc = accumulate_events(gt_dict, pred_dict,
                            start_step=start_step,
                            stop_step=stop_step,
                            im_size=im_size,
                            min_num_pix=min_num_pix,
                            exclude_bg=exclude_bg,
                            iou_thresh=iou_thresh
                            )
    mh = mm.metrics.create()
    summary = mh.compute(acc,
                         metrics=[
                            'num_frames', 'mota', 'motp', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses', 'num_objects'
                         ], 
                         name='acc')
    metrics = compute_mot_metrics(acc, summary) # DataFrame of one row
    metric_dict = metrics.to_dict(orient='records')[0]
    
    return metric_dict
    
# def img_to_mask(ind_images: torch.Tensor) -> torch.Tensor:
#     """covert a set of individual image layers to segmentation mask

#     Args:
#         ind_image (torch.Tensor): shape [N, K, C, H, W]

#     Returns:
#         torch.Tensor: shape [N, K, H, W] one-hot in K dim
#     """
#     ind_images = ind_images.norm(dim=2) # Shape [N, K, H, W]
#     bg_layers = ind_images.max(dim=(-1,-2))[0] < 0.1 * ind_images.max() # Shape [N, K]
    

def adjusted_rand_index(true_mask, pred_mask, exclude_bg=True, reduction='mean'):
    """adjusted rand index implemented by https://gist.github.com/vadimkantorov/bd1616a3a9eea89658ea3efb1f9a1d5d
    Adapted for PyTorch from https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py

    ## Args:
        true_mask (tensor): [N, P == num_pixels == H*W, K1 objects], float. layers of images of different objects
        pred_mask (tensor): [N, P == num_pixels == H*W, K2 objects], float. same but predicted
        exclude_bg (bool): 
        reduction (bool): reduction for different samples in one batch

    ## Returns:
        ari (tensor): float. 
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), ("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    if exclude_bg:
        if torch.is_floating_point(true_mask):
            not_bg_px = torch.logical_not(torch.all(true_mask < 0.1 * torch.max(true_mask, dim=1, keepdim=True)[0], dim=2, keepdim=True))
    else:
        not_bg_px = torch.tensor(True, device=true_mask.device).expand(true_mask.shape[0], true_mask.shape[1], 1)
        
    true_group_ids = torch.argmax(true_mask, -1) # Shape [N, P, K1]
    pred_group_ids = torch.argmax(pred_mask, -1) # Shape [N, P, K2]
    true_mask_oh = true_mask.to(torch.float32) 
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32) # Shape [N, P, K2]

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32) # [N, ] == P 

    nij = torch.einsum('bji,bjk->bki', not_bg_px*pred_mask_oh, not_bg_px*true_mask_oh) # [N,P,K2] [N,P,K1] -> [N,K1,K2]. nij = #pixels in group i (of gt) AND group j (of pred)
    a = torch.sum(nij, dim=1) # [N, K2], #pixels in each group of pred
    b = torch.sum(nij, dim=2) # [N, K1], #pixels in each group of gt

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2]) #[N, ]
    aindex = torch.sum(a * (a - 1), dim=1) # [N, ]
    bindex = torch.sum(b * (b - 1), dim=1) # [N, ]
    expected_rindex = aindex * bindex / (n_points*(n_points-1)) # [N,]
    max_rindex = (aindex + bindex) / 2 # [N,]
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex) # [N,]

    # _all_equal = lambda values: torch.all(torch.equal(values, values[..., :1]), dim=-1) # whether a mask pixel is the same across all groups
    # both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    # return torch.where(both_single_cluster, torch.ones_like(ari), ari)
    if reduction=='mean':
        ari = ari.mean()
    elif reduction=='sum':
        ari = ari.sum()
    else:
        ari = ari
    return ari
    
def main():
    # test mot metrics
    # mot_metrics = get_mot_metrics(
    #     'logs/SPRITES_SASBD_447_SA_4_100_3_RIM_7_100_ver_0/mot_json.json',
    #     'data/gt_jsons/spmot_test.json'
    # )
    
    # test consistency measure
    if os.path.exists('./mmnist_sample.pt'):
        mmnist_sample = torch.load('./mmnist_sample.pt')
        labels, input, output, ind_images = *mmnist_sample, # ind_images, shape [K, T, C, H, W]
    else:
        print('No mmnist_sample.pt found')
        return 444
    target = ind_images.unsqueeze(0) # [N, K, T, C, H, W]
    input = ind_images.unsqueeze(0)[:, (1,0), ...] # [N, K, T, C, H, W]
    # add some noise
    rand_idx = torch.randint(0, 2, (input.shape[0], input.shape[2], 1, 1, 1))
    bad_input = input[:, 1, ...]*rand_idx + (1-rand_idx)*input[:, 0, ...]
    input = torch.cat([
        input,
        bad_input.unsqueeze(1)
    ], dim = 1)
    input = torch.maximum(input, torch.rand_like(input) * 0.5)
    avr_len, max_len, IDs = consistency_measure(input, target, output_ids=True)
    print('average consistent length', avr_len)
    print('maximum consistent length', max_len)
    print('bad_input_ids        ', rand_idx.squeeze())
    print('bad_input_pred_ids   ', IDs[0, -1, ...].squeeze())
    ...
    
if __name__ == '__main__':
    main()