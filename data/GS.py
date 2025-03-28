import numpy as np
import cv2
from scipy.io import loadmat
from shapely.geometry import Polygon
import pyclipper

class GS:
    '''generate g_s and g_s_mask'''
    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        
    def process(self, data):
        image = data['image'] # (H, W, C)
        polygons = data['polygons'] # list of polygons
        ignore_tags = data['ignore_tags']
        filename = data['filename']

        # (H, W, C)
        h, w, _ = image.shape
        g_s = np.zeros((1, h, w), dtype=np.float32)
        g_s_mask = np.ones((h, w), dtype=np.float32)

        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            if ignore_tags[i] or min(height, width) < self.min_text_size:
                # IGNORE=TRUE
                # fill the pixels inside the polygon with 0 in the mask.
                cv2.fillPoly(g_s_mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                # IGNORE=FALSE (good case)
                # shrink the polygon.
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if not shrinked:
                    # IGNORE=TRUE
                    # when the polygon got too small after shrinking it.
                    cv2.fillPoly(g_s_mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue

                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(g_s[0], [shrinked.astype(np.int32)], 1)

        data.update(image=image, polygons=polygons, g_s=g_s, g_s_mask=g_s_mask, filename=filename)
        return data