import numpy as np

from openvino.inference_engine import IECore


DEF_MODEL_XML = "/model/vehicle-detection-0200.xml"
DEF_MODEL_BIN = "/model/vehicle-detection-0200.bin"
DEF_DEVICE = "CPU"
DEF_CONFIDENCE_THRESH = 0.9
IMG_TOP_PADDING = 24


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
    output = detector.infer({'image': frame})
