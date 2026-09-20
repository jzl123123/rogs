import numpy as np
import cv2

def label2mask(label):
    # Bird, Ground Animal, Curb, Fence, Guard Rail,
    # Barrier, Wall, Bike Lane, Crosswalk - Plain, Curb Cut,
    # Parking, Pedestrian Area, Rail Track, Road, Service Lane,
    # Sidewalk, Bridge, Building, Tunnel, Person,
    # Bicyclist, Motorcyclist, Other Rider, Lane Marking - Crosswalk, Lane Marking - General,
    # Mountain, Sand, Sky, Snow, Terrain,
    # Vegetation, Water, Banner, Bench, Bike Rack,
    # Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box,
    # Mailbox, Manhole, Phone Booth, Pothole, Street Light,
    # Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign (Back),
    # Traffic Sign (Front), Trash Can, Bicycle, Boat, Bus,
    # Car, Caravan, Motorcycle, On Rails, Other Vehicle,
    # Trailer, Truck, Wheeled Slow, Car Mount, Ego Vehicle
    mask = np.ones_like(label)
    label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                     | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | (
                             (30 <= label) & (label <= 40)) | (label >= 42)

    # dilate itereation 2 for moving objects
    label_movable = label >= 52
    kernel = np.ones((10, 10), dtype=np.uint8)
    label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

    label_off_road = label_off_road | label_movable
    mask[label_off_road] = 0
    label[~(mask.astype(bool))] = 64
    mask = mask.astype(np.float32)
    return mask, label
