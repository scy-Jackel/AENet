
import numpy as np
from math import exp
import tensorflow as tf

def compute_measure(x, y, pred, sess, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    sess.run(original_ssim)
    sess.run(pred_ssim)
    return (original_psnr, original_ssim.eval(session = sess), original_rmse), (pred_psnr, pred_ssim.eval(session = sess), pred_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    mse_ = compute_MSE(img1, img2)
    return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, max_val=1.0):
    # print("type1:", type(img1), "type2:", type(img2))

    # im1 = tf.image.convert_image_dtype(im1, tf.float32)
    # im2 = tf.image.convert_image_dtype(im2, tf.float32)
    # ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    img1 = tf.expand_dims(img1, axis=0)
    img1 = tf.expand_dims(img1, axis=-1)
    img2 = tf.expand_dims(img2, axis=0)
    img2 = tf.expand_dims(img2, axis=-1)
    res =  tf.reduce_mean(tf.image.ssim(img1, img2, max_val))
    # sess = tf.Session()
    # with sess.as_default():
        # print(res.eval())
    return res


# def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
#     if len(img1.size()) == 2:
#         shape_ = img1.shape[-1]
#         img1 = img1.view(1,1,shape_ ,shape_ )
#         img2 = img2.view(1,1,shape_ ,shape_ )
#     window = create_window(window_size, channel)
#     window = window.type_as(img1)

#     mu1 = F.conv2d(img1, window, padding=window_size//2)
#     mu2 = F.conv2d(img2, window, padding=window_size//2)
#     mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

#     C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
#     #C1, C2 = 0.01**2, 0.03**2

#     ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
#     if size_average:
#         return ssim_map.mean().item()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1).item()


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

