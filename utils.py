import torch
from torchvision import transforms

def transform(img):
    """
    Resize image. Temporarily set dim to 576*576.
    No need to do normalization because detection model already does it.
    """
    transform = transforms.Compose([
        # transforms.Resize((576, 576)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img)

def resize_bbox(boxes, width, height, targetdims=(576, 576)):
    """
    Resize bbox. Temporarily set dim to 576*576.
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize bounding boxes
    old_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    new_dims = torch.FloatTensor([targetdims[1], targetdims[0], targetdims[1], targetdims[0]]).unsqueeze(0)
    new_boxes = new_boxes * new_dims   # scaled absolute coordinates

    return new_boxes

"""
ref:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
"""