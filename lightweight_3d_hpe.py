from argparse import ArgumentParser
import json
import os
import sys

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


mean_time = 0
stride = 8
net = None
base_height = 256
fx=-1
R = None
t = None

started = False

def initialize(device="GPU", model="human-pose-estimation-3d.pth", height_size=256, _fx=-1, extrinsics_path=None, use_openvino=False):

    global fx, net, base_height, R, t, started

    local_path = os.path.dirname(__file__)
    model = f"{local_path}/{model}"

    if use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(model, device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(model, device)

    file_path = extrinsics_path
    if file_path is None:
        file_path = os.path.join(f'{local_path}/data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    base_height = height_size
    fx = _fx

    print("L3DHPE INITIALIZED")
    started = True



def run(frame):
    if not started:
        initialize()
        
    global fx, net, base_height, mean_time, R, t, stride

    current_time = cv2.getTickCount() 

    input_scale = base_height / frame.shape[0]
    scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
    scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
    if fx < 0:  # Focal length is unknown
        fx = np.float32(0.8 * frame.shape[1])

    inference_result = net.infer(scaled_img)
    poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx)
    if len(poses_3d):
        poses_3d = rotate_poses(poses_3d, R, t)
        poses_3d_copy = poses_3d.copy()
        x = poses_3d_copy[:, 0::4]
        y = poses_3d_copy[:, 1::4]
        z = poses_3d_copy[:, 2::4]
        poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

    frame = frame #np.zeros_like(frame)
    draw_poses(frame, poses_2d)
    current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
    if mean_time == 0:
        mean_time = current_time
    else:
        mean_time = mean_time * 0.95 + current_time * 0.05
    cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    return frame