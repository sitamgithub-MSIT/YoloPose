import sys
import PIL.Image as Image
from ultralytics import YOLO
import gradio as gr

# Local imports
from src.logger import logging
from src.exception import CustomExceptionHandling


def predict_pose(
    img: str,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    model_name: str,
) -> Image.Image:
    """
    Predicts objects in an image using a YOLO model with adjustable confidence and IOU thresholds.

    Args:
        - img (str or numpy.ndarray): The input image or path to the image file.
        - conf_threshold (float): The confidence threshold for object detection.
        - iou_threshold (float): The Intersection Over Union (IOU) threshold for non-max suppression.
        - max_detections (int): The maximum number of detections allowed.
        - model_name (str): The name or path of the YOLO model to be used for prediction.

    Returns:
        PIL.Image.Image: The image with predicted objects plotted on it.
    """
    try:
        # Check if image is None
        if img is None:
            gr.Warning("Please provide an image.")

        # Load the YOLO model
        model = YOLO(model_name)

        # Predict objects in the image
        results = model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            show_labels=True,
            show_conf=True,
            imgsz=640,
            half=True,
            device="cpu",
        )

        # Plot the predicted objects on the image
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])

        # Log the successful prediction
        logging.info("Pose estimated successfully.")

        # Return the image
        return im

    # Handle exceptions that may occur during the process
    except Exception as e:
        # Custom exception handling
        raise CustomExceptionHandling(e, sys) from e
