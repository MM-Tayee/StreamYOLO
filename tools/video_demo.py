import argparse
import os
import time
import cv2
import torch
import numpy as np
from yolox.exp import get_exp
from yolox.utils import fuse_model
from torchvision.ops import batched_nms

def parse_args():
    parser = argparse.ArgumentParser("StreamYOLO Video Demo")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--path", default="./assets/dog.mp4", help="path to video")
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Adopting mix precision evaluating")
    parser.add_argument("--fuse", dest="fuse", action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    return parser.parse_args()

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def inference(outputs, conf_thre=0.01, nms_thresh=0.65):
    box_corner = outputs.new(outputs.shape)
    box_corner[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
    box_corner[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
    box_corner[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
    box_corner[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
    outputs[:, :4] = box_corner[:, :4]

    class_conf, class_pred = torch.max(outputs[:, 5:], 1, keepdim=True)
    conf_mask = (outputs[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    detections = torch.cat((outputs[:, :4], outputs[:, 4:5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]

    nms_out_index = batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thresh,
    )

    detections = detections[nms_out_index]
    return detections

def visual(output, img, ratio, cls_conf=0.35):
    bboxes = output[:, 0:4]
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    for i in range(len(bboxes)):
        box = bboxes[i]
        cls_id = int(cls[i])
        score = scores[i]
        if score < cls_conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (0, 255, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, f'{cls_id}: {score:.2f}', (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def main():
    args = parse_args()
    exp = get_exp(args.exp_file, None)
    
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    if args.fuse:
        model = fuse_model(model)
    
    if args.fp16:
        model.half()

    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if args.save_result:
        os.makedirs("./show_results", exist_ok=True)
        save_path = os.path.join("./show_results", os.path.basename(args.path))
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

    buffer = None
    input_size = exp.test_size
    
    # Warmup
    print("Warming up...")
    tmp_img = torch.zeros(1, 3, input_size[0], input_size[1])
    if args.device == "gpu":
        tmp_img = tmp_img.cuda()
    if args.fp16:
        tmp_img = tmp_img.half()
    for _ in range(10):
        _ = model(tmp_img, buffer=None, mode='on_pipe')

    print("Starting inference...")
    while True:
        ret, frame = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.read()[1]
        if frame is None:
            break
        
        img, ratio = preproc(frame, input_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if args.device == "gpu":
            img = img.cuda()
        if args.fp16:
            img = img.half()

        with torch.no_grad():
            outputs, buffer = model(img, buffer=buffer, mode='on_pipe')
            outputs = inference(outputs, exp.test_conf, exp.nmsthre)
        
        if args.save_result:
            result_frame = visual(outputs, frame, ratio, args.conf)
            vid_writer.write(result_frame)
            cv2.imshow("StreamYOLO", result_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
    cap.release()
    if args.save_result:
        vid_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
