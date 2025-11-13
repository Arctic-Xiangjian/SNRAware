"""Run the denoising model inference."""

import numpy as np
import torch
from skimage.util.shape import view_as_windows

__all__ = [
    "apply_model",
    "image_to_patches",
    "patches_to_image",
    "running_inference",
]

# -----------------------------------------------------------------


def image_to_patches(image, cutout=(64, 64, 16), overlap=(16, 16, 8)):
    """
    Extract patches from an image.

    Args:
        image (np.ndarray): The input image to extract patches from.
        cutout (tuple, optional): The size of the patches to extract. Defaults to (64,64,16).
        overlap (tuple, optional): The overlap between patches. Defaults to (16,16,8).

    Returns:
        np.ndarray: The extracted patches.
    """
    CO, TO, HO, WO = image.shape  # original
    Hc, Wc, Tc = cutout  # cutout
    Ho, Wo, To = overlap  # overlap
    Ts, Hs, Ws = Tc - To, Hc - Ho, Wc - Wo  # sliding window shape

    # padding the image so we have a complete coverup
    # in each dim we pad the left side by overlap
    # and then cover the right side by what remains from the sliding window
    image_pad = np.pad(
        image, ((0, 0), (To, -TO % Ts), (Ho, -HO % Hs), (Wo, -WO % Ws)), "symmetric"
    )

    # breaking the image down into patches
    # and remembering the length in each dimension
    image_patches = view_as_windows(image_pad, (CO, Tc, Hc, Wc), (1, Ts, Hs, Ws))
    _K, _Ntme, Nrow, Ncol, _, _, _, _ = image_patches.shape
    image_patches_shape = image_patches.shape

    is_2d_mode = False
    if Tc == 1:
        image_patches = np.transpose(image_patches, (4, 0, 1, 2, 3, 5, 6, 7))
        Tc = Nrow * Ncol
        is_2d_mode = True

    image_batch = image_patches.reshape(-1, CO, Tc, Hc, Wc)  # shape:(num_patches,C,T,H,W)

    return (
        image_batch,
        is_2d_mode,
        (CO, TO, HO, WO),
        image_patches_shape,
        image_pad.shape,
        (Ts, Hs, Ws),
    )


# -------------------------------------------------------------------------------------------------


def patches_to_image(
    image_batch_pred,
    d_type,
    is_2d_mode,
    image_shape,
    image_patches_shape,
    image_pad_shape,
    sliding_win_shape,
    ratio_H,
    ratio_W,
    cutout=(64, 64, 16),
    overlap=(16, 16, 8),
):
    """
    Reconstruct an image from its patches.

    Averaged values are computed in the overlapped region.

    Args:
        image_batch_pred (np.ndarray): The predicted image patches.
        d_type (np.dtype): The data type of the output image.
        is_2d_mode (bool): Whether the model is 2D or 3D.
        image_shape (tuple): The shape of the original image.
        image_patches_shape (tuple): The shape of the image patches.
        image_pad_shape (tuple): The shape of the padded image.
        sliding_win_shape (tuple): The shape of the sliding window.
        ratio_H (int): The upsampling ratio for height.
        ratio_W (int): The upsampling ratio for width.
        cutout (tuple, optional): The size of the patches to extract. Defaults to (64,64,16).
        overlap (tuple, optional): The overlap between patches. Defaults to (16,16,8).

    Returns:
        np.ndarray: The reconstructed image.
    """
    _CO, TO, HO, WO = image_shape
    Hc, Wc, Tc = cutout
    Ho, Wo, To = overlap
    K, Ntme, Nrow, Ncol, _, _, _, _ = image_patches_shape
    Ts, Hs, Ws = sliding_win_shape

    H_o, W_o = image_batch_pred.shape[-2], image_batch_pred.shape[-1]
    C_out = image_batch_pred.shape[1]

    if is_2d_mode:
        image_batch_pred = np.reshape(
            image_batch_pred, (Ntme * K * Nrow * Ncol, C_out, 1, H_o, W_o)
        )
        Tc = 1

    # set the output image shape, consider the upsampling ratio
    image_patches_ot_shape = (K, Ntme, Nrow, Ncol, C_out, Tc, H_o, W_o)
    image_pad_ot_shape = (
        C_out,
        image_pad_shape[1],
        image_pad_shape[-2] * ratio_H,
        image_pad_shape[-1] * ratio_W,
    )

    # ---------------------------------------------------------------------------------------------
    # setting up the weight matrix
    # matrix_weight defines how much a patch contributes to a pixel
    # image_wgt is the sum of all weights. easier calculation for result

    cutout_output = list(cutout)
    cutout_output[1] *= ratio_H
    cutout_output[2] *= ratio_W

    matrix_weight = np.ones((cutout_output), dtype=d_type)

    Ho *= ratio_H
    Wo *= ratio_W

    for t in range(To):
        matrix_weight[:, :, t] *= (t + 1) / To
        matrix_weight[:, :, -t - 1] *= (t + 1) / To

    for h in range(Ho):
        matrix_weight[h, :, :] *= (h + 1) / Ho
        matrix_weight[-h - 1, :, :] *= (h + 1) / Ho

    for w in range(Wo):
        matrix_weight[:, w, :] *= (w + 1) / Wo
        matrix_weight[:, -w - 1, :] *= (w + 1) / Wo

    matrix_weight = np.transpose(matrix_weight, (2, 0, 1))

    image_wgt = np.zeros(image_pad_ot_shape, dtype=d_type)  # filled in the loop below
    matrix_weight = np.repeat(matrix_weight[np.newaxis, :], C_out, axis=0)
    matrix_rep = np.repeat(matrix_weight[np.newaxis], Ntme * Nrow * Ncol, axis=0)
    matrix_rep = matrix_rep.reshape(image_patches_ot_shape)

    # ---------------------------------------------------------------------------------------------
    # Putting the patches back together
    image_batch_pred = image_batch_pred.reshape(image_patches_ot_shape)
    image_prd = np.zeros(image_pad_ot_shape, dtype=d_type)

    Hc *= ratio_H
    Wc *= ratio_W

    Hs *= ratio_H
    Ws *= ratio_W

    HO *= ratio_H
    WO *= ratio_W

    for nt in range(Ntme):
        for nr in range(Nrow):
            for nc in range(Ncol):
                image_wgt[
                    :, Ts * nt : Ts * nt + Tc, Hs * nr : Hs * nr + Hc, Ws * nc : Ws * nc + Wc
                ] += matrix_rep[0, nt, nr, nc]
                image_prd[
                    :, Ts * nt : Ts * nt + Tc, Hs * nr : Hs * nr + Hc, Ws * nc : Ws * nc + Wc
                ] += matrix_weight * image_batch_pred[0, nt, nr, nc]

    image_prd /= image_wgt

    # remove the extra padding
    image_fin = image_prd[:, To : To + TO, Ho : Ho + HO, Wo : Wo + WO]

    return image_fin


# -------------------------------------------------------------------------------------------------


def running_inference(
    model,
    image,
    cutout=(64, 64, 16),
    overlap=(16, 16, 4),
    batch_size=1,
    device="cpu",
    verbose=False,
):
    """
    Runs inference by breaking image into overlapping patches
    Runs the patches through the model and then stiches them back
    @args:
        - model (torch or onnx model): the model to run inference with
        - image (numpy.ndarray): the image to run inference on
            - requires the image to have [C,T,H,W]
        - cutout (int 3-tuple): the patch shape for each cutout [H,W,T]
        - overlap (int 3-tuple): the number of pixels to overlap [H,W,T]
            - required to be smaller than cutout
        - batch_size (int): number of patches per model call
        - device (torch.device): the device to run inference on
    @rets:
        - output (4D numpy.ndarray): result as numpy array [C,T,H,W].
    """
    assert cutout > overlap, "cutout should be greater than overlap"

    # split input image to patches
    (
        image_batch,
        is_2d_mode,
        image_shape,
        image_patches_shape,
        image_pad_shape,
        sliding_win_shape,
    ) = image_to_patches(image, cutout=cutout, overlap=overlap)

    d_type = image.dtype
    Tc = image_batch.shape[2]

    model.to(device=device)
    model.eval()

    image_batch_pred = None
    with torch.inference_mode():
        # process every patch, respects the batch setting
        for i in range(0, image_batch.shape[0], batch_size):
            x_in = torch.from_numpy(image_batch[i : i + batch_size]).to(
                device=device, dtype=torch.float32
            )
            res = model(x_in)
            res = res.to(torch.float32).cpu().numpy()

            if image_batch_pred is None:
                image_batch_pred = np.empty(
                    (image_batch.shape[0], res.shape[1], Tc, res.shape[-2], res.shape[-1]),
                    dtype=d_type,
                )
                ratio_H = int(res.shape[-2] // x_in.shape[-2])
                ratio_W = int(res.shape[-1] // x_in.shape[-1])

            image_batch_pred[i : i + batch_size] = res

    # put processed patches together
    output = patches_to_image(
        image_batch_pred,
        d_type,
        is_2d_mode,
        image_shape,
        image_patches_shape,
        image_pad_shape,
        sliding_win_shape,
        ratio_H,
        ratio_W,
        cutout=cutout,
        overlap=overlap,
    )

    return output


# -------------------------------------------------------------------------------------------------


def apply_model(
    model,
    data,
    gmap,
    scaling_factor=1.0,
    cutout=(64, 64, 16),
    overlap=(16, 16, 8),
    batch_size=1,
    device="cuda",
    verbose=False,
):
    """
    Apply the inference model to the data x with gmap g
    Input
        data : [H, W, T or 1]
        gmap : [H, W, T or 1]
        scaling_factor : scaling factor to adjust denoising strength, smaller value is for higher strength (e.g. 0.5 is more smoothing than 1.0)
    Output
        res: [H, W, T].
    """
    H, W, T = data.shape
    if gmap.ndim == 2:
        gmap = np.expand_dims(gmap, axis=2)

    if gmap.shape[2] == 1 and T > 1:
        gmap = np.repeat(gmap, T, axis=2)

    assert gmap.shape[0] == H and gmap.shape[1] == W

    if verbose:
        print(f"---> apply_model, data array {data.shape}", flush=True)
        print(f"---> apply_model, gmap array {gmap.shape}, median {np.median(gmap)}", flush=True)
        print(f"---> apply_model, scaling_factor {scaling_factor}", flush=True)
        print(f"---> apply_model, cutout {cutout}", flush=True)
        print(f"---> apply_model, overlap {overlap}", flush=True)
        print(f"---> apply_model, batch_size {batch_size}", flush=True)
        print(f"---> apply_model, device {device}", flush=True)

    res = np.copy(data)

    try:
        x = np.transpose(data, [2, 0, 1]) * scaling_factor
        g = np.transpose(gmap, [2, 0, 1])

        if np.iscomplexobj(x):
            input = np.concatenate(
                (x[np.newaxis, :].real, x[np.newaxis, :].imag, g[np.newaxis, :]), axis=0
            )
        else:
            input = np.concatenate((x[np.newaxis, :], g[np.newaxis, :]), axis=0)

        if verbose:
            print(
                f"---> apply_model, for input array {input.shape}, gmap {g.shape}, cutout {cutout}, overlap {overlap}, batch_size {batch_size}",
                flush=True,
            )

        res = running_inference(
            model=model,
            image=input,
            cutout=cutout,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )

        if np.iscomplexobj(x):
            res = res[0] + 1j * res[1]

        res = np.transpose(res, [1, 2, 0]) / scaling_factor

    except Exception as e:
        print(f"Error happened in apply_model: {e}")

    return res


# ---------------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
