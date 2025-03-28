import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
class GD:
    def __init__(self, dil_ratio, thresh_min, thresh_max):
        """
        dil_ratio: ratio for polygon dilation
        thresh_min, thresh_max: normalization range for the distance map
        """
        self.dil_ratio = dil_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def process(self, data):
        """
        Args:
            data['image']: (H, W, C)
            data['polygons']: list of polygons (N, 2) for each text instance
            data['ignore_tags']: list of bool
            data['filename']: optional
        
        Returns:
            data['g_d']: {thresh_min, thresh_max}
            data['g_d_mask']: {0,1}
        """
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]

        # initalize the g_d and g_d_mask
        g_d = np.zeros((h, w), dtype=np.float32)
        g_d_mask = np.zeros((h, w), dtype=np.float32)

        # for each polygon, create a dilated polygon and make g_d and g_d_mask
        for poly, ignored in zip(polygons, ignore_tags):
            # if ignore tag(where text instance is too small) is true, skipt it
            if ignored:
                continue
            self.draw_border_map(poly, g_d, g_d_mask)

        # rescale g_d from [0, 1] to to [thresh_min, thresh_max]
        g_d = g_d * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['g_d'] = g_d
        data['g_d_mask'] = g_d_mask
        
        return data

    def draw_border_map(self, polygon, canvas, mask):
        poly = np.array(polygon, dtype=np.float32)
        if poly.shape[0] < 3:
            return  # degenerate polygon
        polygon_shape = Polygon(poly)
        # The DBNet formula for distance offset:
        distance = polygon_shape.area * (1 - self.dil_ratio**2) / polygon_shape.length
        if distance < 1e-6:
            return

        # dilate the polygon using pyclipper
        subject = [tuple(pt) for pt in poly]
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded = pco.Execute(distance)
        if not padded:
            return

        # only handle the first polygon(usually there is just one polygon for one text instance)
        dilated_poly = np.array(padded[0], dtype=np.int32)
        # fill mask=1 inside the dilated polygon
        cv2.fillPoly(mask, [dilated_poly], 1.0)

        # local bounding box
        xmin = np.clip(np.min(dilated_poly[:,0]), 0, canvas.shape[1]-1)
        xmax = np.clip(np.max(dilated_poly[:,0]), 0, canvas.shape[1]-1)
        ymin = np.clip(np.min(dilated_poly[:,1]), 0, canvas.shape[0]-1)
        ymax = np.clip(np.max(dilated_poly[:,1]), 0, canvas.shape[0]-1)
        if xmin > xmax or ymin > ymax:
            return

        region_w = xmax - xmin + 1
        region_h = ymax - ymin + 1
        if region_w < 2 or region_h < 2:
            return

        # create a local mask for the bounding box
        local_mask = np.zeros((region_h, region_w), dtype=np.uint8)
        # Shift the polygon to the local coordinates
        shifted_poly = dilated_poly.copy()
        shifted_poly[:,0] -= xmin
        shifted_poly[:,1] -= ymin
        cv2.fillPoly(local_mask, [shifted_poly], 255)
        
        # calculate the distance
        dist = cv2.distanceTransform(255 - local_mask, cv2.DIST_L2, 3)
        dist_norm = dist / max(distance, 1e-6)
        dist_norm = np.clip(dist_norm, 0.0, 1.0)
        dist_norm = 1.0 - dist_norm 

        # combine with global canvas
        canvas_region = canvas[ymin:ymax+1, xmin:xmax+1]
        np.maximum(canvas_region, dist_norm, out=canvas_region)