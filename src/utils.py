import numpy as np
import cv2

from openvino.inference_engine import IECore


DEF_MODEL_XML = "/model/vehicle-detection-0200.xml"
DEF_MODEL_BIN = "/model/vehicle-detection-0200.bin"
DEF_DEVICE = "CPU"
DEF_CONFIDENCE_THRESH = 0.7

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
    # width = MODEL_IMG_SIZE[0]
    # height = int(width / INPUT_IMG_SIZE[0] *
    #              (INPUT_IMG_SIZE[1] - IMG_PADDING[1]))
    # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    # if (MODEL_IMG_SIZE[1] - height) % 2 == 0:
    #     border1 = int((MODEL_IMG_SIZE[1] - height) / 2)
    #     border2 = border1
    # else:
    #     border1 = int((MODEL_IMG_SIZE[1] - height - 1) / 2)
    #     border2 = border1 + 1
    # border = (border1, border2)
    # frame = cv2.copyMakeBorder(
    #     frame, border[0], border[1], 0, 0, cv2.BORDER_CONSTANT)
    # frame = frame.reshape(1, 3, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[1])
    # cv2.imwrite("/output/test.png", frame.reshape(256, 256, 3))
    blob = cv2.dnn.blobFromImage(frame, size=(256, 256), ddepth=cv2.CV_8U)
    output = detector.infer({'image': blob})
    out_data = output['detection_out'][0, 0, :, :]
    detected_vehicles = []
    for i in range(out_data.shape[0]):
        if out_data[i, 2] >= confidence_thresh:
            detected_vehicles.append(out_data[i, 3:])
    return detected_vehicles


def draw_detections(frame, detections):
    height, width, _ = frame.shape
    for det in detections:
        x_min, y_min, x_max, y_max = det
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)
        cv2.rectangle(
            frame,
            (x_min, y_min+IMG_PADDING[1]),
            (x_max, y_max+IMG_PADDING[1]),
            (0, 255, 0), 2
        )
    return frame
