

def dice_score(pred, mask):
    up = 2 * (pred * mask).sum()
    down = (pred + mask).sum()
    return up / down


def iou_score(pred, mask):
    ...