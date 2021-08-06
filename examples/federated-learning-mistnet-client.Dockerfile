FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx

COPY ./lib/requirements.txt /home

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib
COPY ./plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/mistnet/coco128   /home/work/

ENTRYPOINT ["python", "train_worker.py"]
