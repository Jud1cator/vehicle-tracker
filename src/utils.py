import cv2
import numpy as np

from openvino.inference_engine import IECore


DEF_MODEL_XML = "/model/vehicle-detection-0200.xml"
DEF_MODEL_BIN = "/model/vehicle-detection-0200.bin"
DEF_DEVICE = "CPU"
DEF_CONFIDENCE_THRESH = 0.8

MODEL_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (640, 480)
IMG_PADDING = (0, 24)


def get_detector(
        model_xml=DEF_MODEL_XML,
        model_bin=DEF_MODEL_BIN,
        device=DEF_DEVICE,
        num_requests=0
):
    ie_core = IECore()
    net = ie_core.read_network(model=model_xml, weights=model_bin)
    exec_net = ie_core.load_network(
        network=net, device_name=device, num_requests=num_requests)
    return exec_net


def detect_vehicles(detector, frame, confidence_thresh=DEF_CONFIDENCE_THRESH):
    frame = frame[IMG_PADDING[1]:, :]
    frame = cv2.resize(frame, MODEL_IMG_SIZE, interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(frame, size=MODEL_IMG_SIZE, ddepth=cv2.CV_8U)
    output = detector.infer({'image': blob})
    out_data = output['detection_out'][0, 0, :, :]
    detected_vehicles = []
    for i in range(out_data.shape[0]):
        if out_data[i, 2] >= confidence_thresh:
            detected_vehicles.append(out_data[i, 3:])
    return detected_vehicles


def get_bbox(model_output, frame_shape):
    width, height = frame_shape
    x_min, y_min, x_max, y_max = model_output
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    return x_min, y_min, x_max, y_max


def compute_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    From: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    assert x11 < x12
    assert y11 < y12
    assert x21 < x22
    assert y21 < y22

    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x12 - x11) * (y12 - y11)
    bb2_area = (x22 - x21) * (y22 - y21)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou
