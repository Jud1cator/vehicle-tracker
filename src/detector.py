import cv2

from openvino.inference_engine import IECore


class Detector:
    def __init__(
            self,
            xml,
            bin,
            device,
            input_shape,
            confidence_threshold=0.8,
            num_requests=1
    ):
        self.ie_core = IECore()
        net = self.ie_core.read_network(model=xml, weights=bin)
        self.exec_net = self.ie_core.load_network(
            network=net,
            device_name=device,
            num_requests=num_requests
        )
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.input_name = list(self.exec_net.requests[0].input_blobs)[0]

    def detect_vehicles(self, frame):
        frame = cv2.resize(
            frame, self.input_shape, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            frame, size=self.input_shape, ddepth=cv2.CV_8U)
        output = self.exec_net.infer({self.input_name: blob})
        out_data = output['detection_out'][0, 0, :, :]
        detected_vehicles = []
        for i in range(out_data.shape[0]):
            if out_data[i, 2] >= self.confidence_threshold:
                detected_vehicles.append(out_data[i, 3:])
        return detected_vehicles
