#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
import fastdeploy as fd
import os
import threading

# 全局变量
last_image = None
image_lock = threading.Lock()  # 线程安全锁
start_detect = 0
floor1_img = []
saved_img_dir = 'result'
camera_running = True


def capture_camera():
    global last_image, camera_running
    cap = cv2.VideoCapture(0)  # 使用索引0打开默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面！")
            break

        with image_lock:
            last_image = frame.copy()

            if start_detect == 1:
                floor1_img.append(frame.copy())

    cap.release()


class Detector():
    def __init__(self):
        # 创建推理后端
        self.runtime_option = fd.RuntimeOption()
        # 使用 CPU 推理
        self.runtime_option.use_cpu()

        # 加载YOLOv5模型
        self.yolo_model = fd.vision.detection.YOLOv5(
            '/home/eaibot/yolov5/models/building.onnx',  # 替换为YOLOv5模型路径
            runtime_option=self.runtime_option
        )
        # 设置预处理参数
        self.yolo_model.preprocessor.set_normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True
        )
        self.floor_num = 1
        self.floor = []
        self.flag = 0
        self.house_name = ''

    def predict(self, img):
        img_basename = 'captured_image.jpg'
        # 使用YOLOv5进行预测
        result = self.yolo_model.predict(img)

        # 在图像上绘制检测结果
        vis_img = fd.vision.vis_detection(img, result, score_threshold=0.5)

        # 保存原始图片和可视化结果到指定文件夹
        save_path_orig = os.path.join(saved_img_dir, "orig_" + img_basename)
        save_path_vis = os.path.join(saved_img_dir, "vis_" + img_basename)
        cv2.imwrite(save_path_orig, img)
        cv2.imwrite(save_path_vis, vis_img)

        # 打印检测结果
        print(f"检测到 {len(result.boxes)} 个目标")
        for i, box in enumerate(result.boxes):
            print(f"目标 {i + 1}: 类别={box.class_id}, 置信度={box.score:.2f}, 位置={box.box}")


if __name__ == '__main__':
    # 启动摄像头线程
    thread = threading.Thread(target=capture_camera)
    thread.start()

    print('开始检测，按q键退出')

    # 确保结果目录存在
    if not os.path.exists(saved_img_dir):
        os.makedirs(saved_img_dir)

    start_time = time.time()

    try:
        while True:
            elapsed_time = time.time() - start_time
            print(f"程序运行时间: {elapsed_time:.2f} 秒")

            # 楼宇识别
            if 30 <= elapsed_time < 31:
                start_detect = 1
                print("检测楼宇")
                with image_lock:
                    if last_image is not None:
                        detect = Detector()
                        detect.predict(last_image)
                start_detect = 0
                time.sleep(10)
                continue

            # 显示实时画面
            with image_lock:
                if last_image is not None:
                    cv2.imshow('Camera', last_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera_running = False
        thread.join()
        cv2.destroyAllWindows()