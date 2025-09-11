import fileinput
import os
import re
import subprocess
import time

import nibabel as nib
import numpy as np
import torch
from nibabel import load as load_nii
from skimage import transform as skt
from sklearn.utils import class_weight
from torch.autograd import Variable


def deepMask(
    args,
    model,
    id,
    t1w_np,
    t2w_np,
    t1w_fname,
    t2w_fname,
    nifti=True,
    return_paths=False,
):
    dst = args.outdir
    case_id = id

    # probablity segmentation and discrete segmentation output filenames
    probseg = case_id + "_space-MNI152_desc-deepMask_probseg.nii.gz"
    probseg_path = os.path.join(dst, probseg)

    dseg = case_id + "_space-MNI152_desc-deepMask_dseg.nii.gz"
    dseg_path = os.path.join(dst, dseg)

    model.eval()

    start_time = time.time()
    data = normalize_resize_to_tensor(t1w_np, t2w_np, args)
    # load original input with header and affine
    _, header, affine, out_shape = get_nii_hdr_affine(t1w_fname)
    shape = data.size()
    # convert names to batch tensor
    if args.cuda:
        data.pin_memory()
        data = data.cuda().float()
    else:
        data = data.float()
    data = Variable(data, volatile=True)
    output = model(data)
    output = torch.argmax(output, dim=1)
    output = output.view(shape[2:])
    output = output.cpu()
    output = output.data.numpy()

    print(f"save {case_id}")
    if not os.path.exists(os.path.join(dst)):
        os.makedirs(os.path.join(dst), exist_ok=True)

    if nifti:
        affine = header.get_qform()
        output = skt.resize(
            output,
            output_shape=out_shape,
            order=1,
            mode="wrap",
            preserve_range=1,
            anti_aliasing=True,
        )
        nii_out = nib.Nifti1Image(output, affine, header)
        nii_out.to_filename(probseg_path)

    elapsed_time = time.time() - start_time
    print("=" * 70)
    print(f"=> inference time: {round(elapsed_time, 2)} seconds")
    print("=" * 70)

    # config = './utils/dense3dCrf/config_densecrf.txt'
    cwd = os.path.dirname(__file__)
    config = os.path.join(cwd, "dense3dCrf/config_densecrf.txt")

    start_time = time.time()
    denseCRF(
        case_id,
        t1w_fname,
        t2w_fname,
        out_shape,
        config,
        dst,
        os.path.join(probseg_path),
    )
    elapsed_time = time.time() - start_time
    print("=" * 70)
    print(f"=> dense 3D-CRF inference time: {round(elapsed_time, 2)} seconds")
    print("=" * 70)

    seg_map = load_nii(dseg_path).get_fdata()
    if return_paths:
        return seg_map, (probseg_path, dseg_path)
    else:
        return seg_map


def normalize_resize_to_tensor(t1w_np, t2w_np, args):
    t1w_np = (
        t1w_np.astype(dtype=np.float32) - t1w_np[np.nonzero(t1w_np)].mean()
    ) / t1w_np[np.nonzero(t1w_np)].std()
    t2w_np = (
        t2w_np.astype(dtype=np.float32) - t2w_np[np.nonzero(t2w_np)].mean()
    ) / t2w_np[np.nonzero(t2w_np)].std()
    t1w_np = skt.resize(t1w_np, args.resize, mode="constant", preserve_range=1)
    t2w_np = skt.resize(t2w_np, args.resize, mode="constant", preserve_range=1)
    data = torch.unsqueeze(torch.from_numpy(np.stack((t1w_np, t2w_np), axis=0)), 0)
    return data


def denseCRF(id, t1, t2, input_shape, config, out_dir, pred_labels):
    cwd = os.path.dirname(__file__)
    X, Y, Z = input_shape
    config_tmp = "/tmp/" + id + "_config_densecrf.txt"
    subprocess.call(["cp", "-f", config, config_tmp])
    # find and replace placeholder variables in the config file with actual filenames
    find_str = [
        "<ID_PLACEHOLDER>",
        "<T1_FILE_PLACEHOLDER>",
        "<FLAIR_FILE_PLACEHOLDER>",
        "<OUTDIR_PLACEHOLDER>",
        "<PRED_LABELS_PLACEHOLDER>",
        "<X_PLACEHOLDER>",
        "<Y_PLACEHOLDER>",
        "<Z_PLACEHOLDER>",
    ]
    replace_str = [
        str(id),
        str(t1),
        str(t2),
        str(out_dir),
        str(pred_labels),
        str(X),
        str(Y),
        str(Z),
    ]

    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)

    # subprocess.call(["./utils/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])
    subprocess.call(
        [os.path.join(cwd, "dense3dCrf/dense3DCrfInferenceOnNiis"), "-c", config_tmp]
    )


def datestr():
    now = time.gmtime()
    return (
        f"{now.tm_year}{now.tm_mon:02}{now.tm_mday:02}_{now.tm_hour:02}{now.tm_min:02}"
    )


def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup=".bak") as file:
        for line in file:
            print(
                re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE),
                end="\n",
            )


def compute_weights(labels, binary=False):
    if binary:
        labels = (labels > 0).astype(np.int_)

    weights = class_weight.compute_class_weight(
        "balanced", np.unique(labels.flatten()), labels.flatten()
    )
    return weights


def dice_gross(image, label, empty_score=1.0):
    image = (image > 0).astype(np.int_)
    label = (label > 0).astype(np.int_)

    image = np.asarray(image).astype(np.bool)
    label = np.asarray(label).astype(np.bool)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape} and label {label.shape} must have the same shape."
        )

    im_sum = image.sum() + label.sum()
    if im_sum == 0:
        return empty_score

    # compute Dice coefficient
    intersection = np.logical_and(image, label)

    return 2.0 * intersection.sum() / im_sum


def get_nii_hdr_affine(t1w_fname):
    nifti = load_nii(t1w_fname)
    shape = nifti.get_fdata().shape
    header = load_nii(t1w_fname).header
    affine = header.get_qform()
    return nifti, header, affine, shape
