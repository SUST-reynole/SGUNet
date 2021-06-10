import numpy as np

def Dice(prediction, target):
    prediction = prediction.detach().numpy()
    prediction = np.uint8(prediction)
    target = target.detach().numpy()
    target = np.uint8(target)
    delta = 1e-10
    return (2 * (prediction * target).sum() + delta) / (prediction.sum() + target.sum() + delta)

def comput_Dice(input,target):
    batch,_,_ = input.shape

    acc = 0
    for i in range(batch):
        a = input[i]
        b = target[i]
        iou = Dice(a,b)
        acc = acc + iou
    return acc/(batch)









    