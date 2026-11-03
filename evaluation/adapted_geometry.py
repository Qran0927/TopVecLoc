"""Geometry and failure-aware scoring for UAV-to-rendered-OSM matching."""

import cv2
import numpy as np


def recover_pose(points_image, points_map, image_shape, map_shape, ransac_px=3.0):
    points_image = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
    points_map = np.asarray(points_map, dtype=np.float64).reshape(-1, 2)
    result = dict(status='insufficient_matches', pred_u=np.nan, pred_v=np.nan,
                  pred_yaw=np.nan, match_count=len(points_image), inlier_count=0)
    if len(points_image) != len(points_map):
        raise ValueError('image and map correspondence counts differ')
    finite = np.isfinite(points_image).all(1) & np.isfinite(points_map).all(1)
    points_image, points_map = points_image[finite], points_map[finite]
    if len(points_image) < 4:
        return result
    if any(np.linalg.matrix_rank(p - p.mean(0)) < 2 for p in (points_image, points_map)):
        result['status'] = 'degenerate_matches'
        return result
    cv2.setRNGSeed(0)
    transform, mask = cv2.findHomography(points_image, points_map, cv2.RANSAC,
                                       ransac_px, maxIters=2000, confidence=0.999)
    if transform is None or mask is None or not np.isfinite(transform).all():
        result['status'] = 'ransac_failed'
        return result
    result['inlier_count'] = int(mask.sum())
    if mask.sum() < 4:
        result['status'] = 'insufficient_inliers'
        return result
    height, width = image_shape[:2]
    # Dataset generator rotates/crops around width/2,height/2, not (width-1)/2.
    query = np.array([[[width/2, height/2], [width/2, height/2-1],
                       [width/2+1, height/2]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(query, transform)[0]
    center, north, east = mapped
    direction = north - center
    east_direction = east - center
    jacobian = np.column_stack((east_direction, -direction))
    if not np.isfinite(mapped).all() or np.linalg.det(jacobian) <= 1e-8:
        result['status'] = 'invalid_transform'
        return result
    if not (0 <= center[0] < map_shape[1] and 0 <= center[1] < map_shape[0]):
        result['status'] = 'center_outside_map'
        return result
    result.update(status='ok', pred_u=float(center[0]), pred_v=float(center[1]),
                  pred_yaw=float(np.degrees(np.arctan2(direction[0], -direction[1])) % 360))
    return result


def summarize_results(frame):
    if frame.empty:
        raise ValueError('cannot summarize zero samples')
    success = frame['status'].eq('ok')
    columns = {'position':'position_max_error_m', 'lateral':'lateral_max_error_m',
               'longitudinal':'longitudinal_max_error_m', 'yaw':'yaw_error_deg'}
    metrics = {}
    for family, column in columns.items():
        errors = frame[column].to_numpy(dtype=float)
        if not np.isfinite(errors[success]).all():
            raise ValueError(f'successful samples contain invalid {column}')
        for threshold in (1,3,5,10):
            unit = 'deg' if family == 'yaw' else 'm'
            metrics[f'{family}_recall_{threshold}{unit}'] = float(
                (success.to_numpy() & (errors < threshold)).mean() * 100)
    return dict(sample_count=len(frame), success_count=int(success.sum()),
                failure_count=int((~success).sum()), metrics=metrics,
                failure_reasons=frame.loc[~success,'status'].value_counts().to_dict())
