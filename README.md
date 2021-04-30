# Vehicle Tracker
A simple vehicle tracker which accepts video file as input and outputs video file with drawn bounding boxes of detected vehicles and tracks of bounding boxes centroids for subsequent frames. Solution is using OpenVINO and deployed in a docker container.

Model used for detection: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-detection-0200/description/vehicle-detection-0200.md

Tracker uses inverted IoU metric calculated on detected bounding boxes.

## Usage
1. Clone the repository
2. Make sure you have the updated version of docker-compose.
Installation guide for the latest version is here: https://docs.docker.com/compose/install/
3. Place your video in `video` folder if you want to try custom video and change `src/configs/sample_config.yml` accordingly.
4. Open terminal in `docker` folder and run:

`docker-compose up --build -d`

Wait until container is built. It may take a while. Then run:

`docker exec -it vehicle_tracker bash`

to get inside the container. Once you are in, run

`python3 run.py -c ./configs/sample_config.yml`

You will see progress bar of video processing. Once it is finished, you may exit the container (Ctrl+D). The ouput video will be in `output` folder.
