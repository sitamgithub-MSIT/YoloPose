# Importing the requirements
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from src.yolo.predict_pose import predict_pose


# Image and input parameters
image = gr.Image(type="pil", label="Image")
confidence_threshold = gr.Slider(
    minimum=0, maximum=1, step=0.01, value=0.25, label="Confidence threshold"
)
iou_threshold = gr.Slider(
    minimum=0, maximum=1, step=0.01, value=0.45, label="IoU threshold"
)
max_detections = gr.Slider(
    minimum=1, maximum=300, step=1, value=300, label="Max detections"
)
model_name = gr.Radio(
    choices=[
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt",
        "yolo11l-pose.pt",
        "yolo11x-pose.pt",
    ],
    label="Model name",
    value="yolo11n-pose.pt",
)

# Output image
pose_image = gr.Image(type="pil", label="Output Image")

# Examples for the interface
examples = [
    ["images/posing-sample-image3.jpg", 0.25, 0.45, 300, "yolo11n-pose.pt"],
    ["images/posing-sample-image4.jpg", 0.25, 0.45, 300, "yolo11s-pose.pt"],
    ["images/posing-sample-image5.jpg", 0.25, 0.45, 300, "yolo11m-pose.pt"],
    ["images/posing-sample-image1.jpg", 0.25, 0.45, 300, "yolo11l-pose.pt"],
    ["images/posing-sample-image2.png", 0.25, 0.45, 300, "yolo11x-pose.pt"],
]

# Title, description, and article for the interface
title = "YOLO11 Pose Estimation"
description = "Gradio Demo for the YOLO11 Pose Estimation model. This model can detect and predict the poses of people in images. To use it, upload your image, select associated parameters, or use the default values, click 'Submit', or click one of the examples to load them. You can read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/ultralytics/ultralytics' target='_blank'>Ultralytics GitHub</a> | <a href='https://docs.ultralytics.com/models/yolo11/' target='_blank'>Model Page</a></p>"


# Launch the interface
interface = gr.Interface(
    fn=predict_pose,
    inputs=[image, confidence_threshold, iou_threshold, max_detections, model_name],
    outputs=pose_image,
    examples=examples,
    cache_examples=True,
    cache_mode="lazy",
    title=title,
    description=description,
    article=article,
    theme="Base",
    flagging_mode="never",
)
interface.launch(debug=False)
