

class Dataset:
    def __init__(self) -> None:
        self.parameters = {
            "datasource": "YOLO", 
            "data_params": "./coco128.yaml", 
            # Where the dataset is located
            "data_path": "./data/COCO",
            "train_path": "./data/COCO/coco128/images/train2017/",
            "test_path": "./data/COCO/coco128/images/train2017/",
            # number of training examples
            "num_train_examples": 128,
            # number of testing examples
            "num_test_examples": 128,
            # number of classes
            "num_classes": 80,
            # image size
            "image_size": 640,
            "classes":
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
               "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ], 
            "partition_size": 128,
        }

class Estimator:
    def __init__(self) -> None:
        self.model = None
        self.hyperparameters = {
            "type": "yolov5", 
            "rounds": 1,
            "target_accuracy": 0.99,
            "epochs": 500,
            "batch_size": 16,
            "optimizer": "SGD",
            "linear_lr": False,
            # The machine learning model 
            "model_name": "yolov5",
            "model_config": "./yolov5s.yaml",
            "train_params": "./hyp.scratch.yaml"
            }

class Aggregation:
    def __init__(self) -> None:
        self.parameters = {
            "type": "mistnet",
            "cut_layer": 4,
            "epsilon": 100
        }

class Transmitter:
    def __init__(self) -> None:
        self.parameters = {
            "address": "0.0.0.0",
            "port": 7363,
            "s3_endpoint_url": "https://obs.cn-south-1.myhuaweicloud.com",
            "s3_bucket": "plato",
            "access_key": "",
            "secret_key": ""
        }
