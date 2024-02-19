from typing import List, Tuple
import numpy as np
import torch

class Accuracy(object):
    
    def __str__(self) -> str:
        return 'Accuracy'
    
    def __repr__(self):
        return 'Accuracy scoring metric'
    
    def _accuracy(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)
    
    def __call__(self, prediction, target):
        return self._accuracy(prediction, target)


class IoU(object):
    
    '''
    Calculate the mean Intersection over Union (IoU) between prediction and ground truth(target).

    >>> iou = IoU()
    >>> prediction = [[15, 20, 28, 31],
    >>>               [10, 27, 25, 34]]
    >>> target = [[14, 24, 27, 33], 
    >>>           [12, 24, 26, 33]]
    >>> iou(prediction, target)
    0.5319989106753813
    '''
    
    def __repr__(self):
        return 'Intersection over Union (IoU) scoring metric'
    
    def __str__(self) -> str:
        return 'IoU'
    
    def _iou_interval(self, interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two intervals.

        Args:
            interval1 (tuple): First interval.
            interval2 (tuple): Second interval.

        Returns:
            float: Intersection over Union (IoU) score.
        """
        true_start, true_end = min(interval1), max(interval1)
        pred_start, pred_end = min(interval2), max(interval2)

        if (pred_start < 0) or (pred_end < 0) or (pred_end < pred_start):
            return 0.0

        intersection = max(pred_start, true_start), min(pred_end, true_end)
        union = min(pred_start, true_start), max(pred_end, true_end)

        if intersection[1] - intersection[0] < 0.0:
            return 0.0
        if union[1] - union[0] == 0.0:
            return 1.0

        iou_score = (intersection[1] - intersection[0]) / (union[1] - union[0])
        return iou_score
    
    def _iou_sequence(self, pred_annotations, true_annotations):
        '''calculates the mean iou in a sequence'''
        total_iou = 0.0
        for i in range(len(true_annotations) - 1):
            total_iou += self._iou_interval(pred_annotations[i:i + 2], true_annotations[i:i + 2])
        mean_iou = total_iou / (len(true_annotations) - 1) if len(true_annotations) > 1 else 0.0
        return mean_iou

    def __call__(self, prediction:np.ndarray, target: np.ndarray) -> float:
        '''
        Args:
            predictions (array): numpy array containing the predicted intervals.
            target (array): numpy array containing the ground truth intervals.
        
        Returns:
            float: Mean Intersection over Union (IoU).
        '''
        
        assert len(prediction) == len(target), 'predictions and target arrays should have the same length !'

        mean_iou = 0.0
        for i in range(len(target)):
            true_annotations = target[i]
            pred_annotations = prediction[i]
            mean_iou += self._iou_sequence(pred_annotations, true_annotations)

        mean_iou /= len(target)

        return mean_iou

"""
## Example of usage
iou = IoU()

## generate random values for prediction and target
## replace this with you true values
prediction = np.random.randint(1, 37, size=(100, 4))
target = np.random.randint(1, 37, size=(100, 4))

## calculates the mean iou over sequences
iou_score = iou(prediction, target)
print('mean iou: {}'.format(iou_score))
"""