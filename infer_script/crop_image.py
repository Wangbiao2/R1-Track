import math
import cv2 as cv


def sample_target(im, target_bb, search_area_factor, output_sz):
    """
    Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area.
    Also returns the coordinates of target_bb in the cropped (and possibly resized) image.

    Args:
        im: cv image
        target_bb: target box [x, y, w, h]
        search_area_factor: Ratio of crop size to target size
        output_sz: (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    Returns:
        im_crop_padded: Extracted (and resized) crop
        resize_factor: The factor by which the crop has been resized
        target_bb_in_crop: Coordinates of target_bb in the cropped image [x_min, y_min, x_max, y_max]
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    # Calculate the top-left corner of the crop
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    # Calculate padding if the crop goes outside of the image boundaries
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop the image
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad the cropped image if necessary
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    # Compute resize factor and resize the crop if output_sz is specified
    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
    else:
        resize_factor = 1.0

    # Compute the position of target_bb in the cropped (and possibly resized) image

    # Coordinates of target_bb relative to the crop's origin before resizing
    x_in_crop = x - x1
    y_in_crop = y - y1

    # Dimensions of target_bb
    w_in_crop = w
    h_in_crop = h

    # Adjust coordinates and dimensions according to the resize factor
    x_in_resized = x_in_crop * resize_factor
    y_in_resized = y_in_crop * resize_factor
    w_resized = w_in_crop * resize_factor
    h_resized = h_in_crop * resize_factor

    # Compute x_min, y_min, x_max, y_max in the resized crop
    x_min = x_in_resized
    y_min = y_in_resized
    x_max = x_min + w_resized
    y_max = y_min + h_resized

    target_bb_in_crop = [x_min, y_min, x_max, y_max]

    store_x1_y1 = [x1, y1]

    return im_crop_padded, resize_factor, target_bb_in_crop, store_x1_y1

def map_bbox_back(resize_factor, store_x1_y1, bbox_in_new_image): # [x_min, y_min, x_max, y_max]
    """
    Maps bounding box coordinates from the cropped and resized image back to the original image.

    Args:
        search_area_factor: Ratio of crop size to target size used in sample_target.
        output_sz: Size to which the extracted crop was resized in sample_target.
        resize_factor: The factor by which the crop was resized.
        bbox_crop: Bounding box of the crop in the original image [x1, y1, w, h].
        bbox_in_new_image: Bounding box in the new image [x_min, y_min, x_max, y_max].

    Returns:
        List [x_min, y_min, w, h]: Bounding box coordinates in the original image.
    """
    x1, y1 = store_x1_y1

    x_min_new, y_min_new, x_max_new, y_max_new = bbox_in_new_image

    # Map the coordinates back to the original image
    X_min = x_min_new / resize_factor + x1
    Y_min = y_min_new / resize_factor + y1
    X_max = x_max_new / resize_factor + x1
    Y_max = y_max_new / resize_factor + y1

    W = X_max - X_min
    H = Y_max - Y_min

    bbox_in_original_image = [X_min, Y_min, W, H]

    return bbox_in_original_image