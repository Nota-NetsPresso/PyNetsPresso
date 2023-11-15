from netspresso.trainer import (
    ModelTrainer,
    Task,
    Backbone,
    Head,
    Format,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomCutmix,
)


# 1. Declare Trainer
trainer = ModelTrainer()

# 2. Set Config
# 2-1. Data
dataset = trainer.get_dataset_config(
    task=Task.Object_Detection,
    format=Format.Local,
    name="dataset_example",
    root_path="/root/coco",
    id_mapping=[
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
)

# 2-2. Model
model = trainer.get_model_config(backbone=Backbone.EfficientFormer, head=Head.FC)

# 2-3. Augmentation
augmentation = trainer.get_augmentation_config(
    img_size=256, transforms=[RandomResizedCrop(size=256), RandomHorizontalFlip()], mix_transforms=[RandomCutmix()]
)

# 2-4. Training
training = trainer.get_training_config(epochs=10, batch_size=32)

# 3. Train
gpus = "0, 1"  # multi-gpu
trainer.train(gpus=gpus, data=dataset, model=model, augmentation=augmentation, training=training)
