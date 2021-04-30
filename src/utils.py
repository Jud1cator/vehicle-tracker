import cv2

from openvino.inference_engine import IECore


DEF_MODEL_XML = "/model/vehicle-detection-0200.xml"
DEF_MODEL_BIN = "/model/vehicle-detection-0200.bin"
DEF_DEVICE = "CPU"
DEF_CONFIDENCE_THRESH = 0.6

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


def get_bboxes(model_output, frame_shape):
    width, height = frame_shape
    x_min, y_min, x_max, y_max = model_output
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    return x_min, y_min, x_max, y_max


def draw_detections(frame, detections):
    for det in detections:
        x_min, y_min, x_max, y_max = get_bboxes(det, frame.shape)
        cv2.rectangle(
            frame,
            (x_min, y_min+IMG_PADDING[1]),
            (x_max, y_max+IMG_PADDING[1]),
            (0, 255, 0), 2
        )
    return frame
