import torch
import gradio as gr
model = torch.hub.load("./", "custom",path="runs/train/exp2/weights/best.pt", source='local')

title = "基于Gradio的YOLOv5目标检测"

desc = "这是一个基于Gradio的YOLOv5目标检测，非常简洁方便"

base_conf, base_iou = 0.25, 0.45

def det_image(img, conf_thres, iou_thres):
    model.conf = conf_thres
    model.iou = iou_thres
    return model(img).render()[0]

gr.Interface(inputs=["image",gr.Slider(minimum=0,maximum=1,value = base_conf),gr.Slider(minimum=0,maximum=1,value = base_iou)],
            outputs=["image"],
            fn=det_image,
            title=title,
            description=desc,
            live=True,
            examples=[["./test/images/train/bag_frame_360.jpg",base_conf,base_iou],["./test/images/train/bag_frame_900.jpg",base_conf,base_iou]]).launch(share=True)
