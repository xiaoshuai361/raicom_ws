#!/usr/bin/env python3
import os
import re
import time

import rospy
from sensor_msgs.msg import Image


GARBAGE_LABELS = {
    0: "可回收物_已投放",
    1: "可回收物_未投放",
    2: "有害垃圾_已投放",
    3: "有害垃圾_未投放",
    4: "厨余垃圾_已投放",
    5: "厨余垃圾_未投放",
    6: "其他垃圾_已投放",
    7: "其他垃圾_未投放",
}

HUMAN_LABELS = {
    0: "职业人员",
    1: "普通行人",
}


class OrtYoloDetector:
    def __init__(self, model_path, labels, threshold, nms_threshold, input_size=640, use_gpu=True, layout="auto"):
        import cv2
        import numpy as np
        import onnxruntime as ort

        self.cv2 = cv2
        self.np = np
        self.labels = labels
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.layout = layout
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        providers = []
        if self.use_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image):
        blob, ratio, pad = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return self.postprocess(output, image.shape[:2], ratio, pad)

    def preprocess(self, image):
        height, width = image.shape[:2]
        ratio = min(self.input_size / float(width), self.input_size / float(height))
        new_w = int(round(width * ratio))
        new_h = int(round(height * ratio))
        resized = self.cv2.resize(image, (new_w, new_h), interpolation=self.cv2.INTER_LINEAR)
        canvas = self.np.full((self.input_size, self.input_size, 3), 114, dtype=self.np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        rgb = self.cv2.cvtColor(canvas, self.cv2.COLOR_BGR2RGB).astype(self.np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None, :, :, :]
        return blob, ratio, (pad_x, pad_y)

    def postprocess(self, output, image_shape, ratio, pad):
        pred = output[0]
        if pred.ndim != 2:
            pred = pred.reshape(pred.shape[-2], pred.shape[-1])
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (6, 7, 8, 12):
            pred = pred.T

        detections = []
        if pred.shape[1] >= 6 and pred.shape[1] - 4 == len(self.labels):
            detections = self.decode_yolov8(pred, image_shape, ratio, pad)
        elif pred.shape[1] >= 7:
            detections = self.decode_yolov5(pred, image_shape, ratio, pad)
        return self.nms(detections)

    def decode_yolov8(self, pred, image_shape, ratio, pad):
        boxes = pred[:, :4]
        class_scores = pred[:, 4:4 + len(self.labels)]
        class_ids = self.np.argmax(class_scores, axis=1)
        scores = class_scores[self.np.arange(class_scores.shape[0]), class_ids]
        return self.make_detections(boxes, scores, class_ids, image_shape, ratio, pad)

    def decode_yolov5(self, pred, image_shape, ratio, pad):
        boxes = pred[:, :4]
        objectness = pred[:, 4]
        class_scores = pred[:, 5:]
        if class_scores.shape[1] > len(self.labels):
            class_scores = class_scores[:, :len(self.labels)]
        class_ids = self.np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[self.np.arange(class_scores.shape[0]), class_ids]
        return self.make_detections(boxes, scores, class_ids, image_shape, ratio, pad)

    def make_detections(self, boxes, scores, class_ids, image_shape, ratio, pad):
        pad_x, pad_y = pad
        image_h, image_w = image_shape
        keep = scores >= self.threshold
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            cx, cy, w, h = [float(v) for v in box]
            x1 = (cx - w / 2.0 - pad_x) / ratio
            y1 = (cy - h / 2.0 - pad_y) / ratio
            x2 = (cx + w / 2.0 - pad_x) / ratio
            y2 = (cy + h / 2.0 - pad_y) / ratio
            x1 = max(0.0, min(float(image_w - 1), x1))
            y1 = max(0.0, min(float(image_h - 1), y1))
            x2 = max(0.0, min(float(image_w - 1), x2))
            y2 = max(0.0, min(float(image_h - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": float(score),
                    "label_id": int(class_id),
                    "label": self.labels.get(int(class_id), "未知"),
                }
            )
        return detections

    def nms(self, detections):
        if not detections:
            return []
        boxes = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(float(det["score"]))
        indices = self.cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms_threshold)
        if len(indices) == 0:
            return []
        return [detections[int(i)] for i in self.np.array(indices).flatten()]


class CameraModelRecognitionNode:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        self.model_root = rospy.get_param("~model_root", "/home/zcy/raicom_ws/2025源码")
        self.process_hz = float(rospy.get_param("~process_hz", 1.0))
        self.stable_frames = int(rospy.get_param("~stable_frames", 1))
        self.max_frame_side = int(rospy.get_param("~max_frame_side", 960))
        self.garbage_threshold = float(rospy.get_param("~garbage_threshold", 0.60))
        self.human_threshold = float(rospy.get_param("~human_threshold", 0.50))
        self.building_threshold = float(rospy.get_param("~building_threshold", 0.60))
        self.fire_threshold = float(rospy.get_param("~fire_threshold", 0.55))
        self.enable_ocr = bool(rospy.get_param("~enable_ocr", False))
        self.nms_threshold = float(rospy.get_param("~nms_threshold", 0.45))
        self.use_gpu = bool(rospy.get_param("~use_gpu", True))
        self.reset_after_missed_frames = int(rospy.get_param("~reset_after_missed_frames", 6))
        self.report_cooldown = float(rospy.get_param("~report_cooldown", 8.0))
        self.garbage_min_area_ratio = float(rospy.get_param("~garbage_min_area_ratio", 0.10))
        self.human_min_area_ratio = float(rospy.get_param("~human_min_area_ratio", 0.07))
        self.building_min_area_ratio = float(rospy.get_param("~building_min_area_ratio", 0.10))
        self.use_fire_color_hint = bool(rospy.get_param("~use_fire_color_hint", True))

        try:
            import cv2
            import numpy as np
            import onnxruntime
            from cv_bridge import CvBridge
        except ImportError as exc:
            rospy.logerr("真实模型识别需要 onnxruntime、cv_bridge、opencv、numpy：%s", exc)
            rospy.logerr("可先执行：python3 -m pip install --user onnxruntime")
            raise

        self.cv2 = cv2
        self.fd = None
        self.np = np
        self.ort = onnxruntime
        self.bridge = CvBridge()
        self.last_process_time = 0.0
        self.min_process_interval = 1.0 / self.process_hz if self.process_hz > 0.0 else 0.0
        self.streaks = {
            "garbage": 0,
            "human": 0,
            "building_fire": 0,
        }
        self.missed_frames = {
            "garbage": 0,
            "human": 0,
            "building_fire": 0,
        }
        self.active_targets = {
            "garbage": False,
            "human": False,
            "building_fire": False,
        }
        self.last_messages = {}
        self.last_reported_messages = {}
        self.last_report_times = {
            "garbage": 0.0,
            "human": 0.0,
            "building_fire": 0.0,
        }

        self.load_models()
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 22)
        rospy.loginfo("[视觉识别] 真实模型识别已启动，订阅 %s。", self.image_topic)

    def load_models(self):
        garbage_model = os.path.join(self.model_root, "垃圾桶", "moxing", "best.onnx")
        human_model = os.path.join(self.model_root, "人", "human.onnx")
        building_model = os.path.join(self.model_root, "楼宇火灾识别", "building.onnx")
        fire_model = os.path.join(self.model_root, "楼宇火灾识别", "fire.onnx")

        self.assert_file(garbage_model)
        self.assert_file(human_model)
        self.assert_file(building_model)
        self.assert_file(fire_model)

        rospy.loginfo("[视觉识别] 加载 2025 源码 ONNX 模型：垃圾桶、人群、楼宇、火灾。")
        self.garbage_model = OrtYoloDetector(
            garbage_model, GARBAGE_LABELS, self.garbage_threshold, self.nms_threshold, use_gpu=self.use_gpu
        )
        self.human_model = OrtYoloDetector(
            human_model, HUMAN_LABELS, self.human_threshold, self.nms_threshold, use_gpu=self.use_gpu
        )
        self.building_model = OrtYoloDetector(
            building_model, {0: "楼宇", 1: "其他"}, self.building_threshold, self.nms_threshold, use_gpu=self.use_gpu
        )
        self.fire_model = OrtYoloDetector(
            fire_model, {0: "楼层", 1: "火焰"}, self.fire_threshold, self.nms_threshold, use_gpu=self.use_gpu
        )
        rospy.loginfo("[视觉识别] ONNXRuntime Providers: %s", self.garbage_model.providers)

        self.ocr_model = None
        if self.enable_ocr:
            self.ocr_model = self.try_load_ocr()
        else:
            rospy.loginfo("[视觉识别] 楼宇 OCR 默认关闭，使用通用楼宇名称，避免 FastDeploy OCR 实时回调崩溃。")

    def try_load_ocr(self):
        try:
            import fastdeploy as fd
        except ImportError as exc:
            rospy.logwarn("[视觉识别] 未安装 fastdeploy，楼宇名称将使用默认描述：%s", exc)
            return None

        root = os.path.join(self.model_root, "楼宇火灾识别", "OCR")
        det_model = os.path.join(root, "ch_PP-OCRv3_det_infer", "inference.pdmodel")
        det_params = os.path.join(root, "ch_PP-OCRv3_det_infer", "inference.pdiparams")
        cls_model = os.path.join(root, "ch_ppocr_mobile_v2.0_cls_infer", "inference.pdmodel")
        cls_params = os.path.join(root, "ch_ppocr_mobile_v2.0_cls_infer", "inference.pdiparams")
        rec_model = os.path.join(root, "ch_PP-OCRv3_rec_infer", "inference.pdmodel")
        rec_params = os.path.join(root, "ch_PP-OCRv3_rec_infer", "inference.pdiparams")
        keys = os.path.join(root, "ppocr_keys_v1.txt")

        for path in [det_model, det_params, cls_model, cls_params, rec_model, rec_params, keys]:
            if not os.path.exists(path):
                rospy.logwarn("[视觉识别] OCR 文件缺失，楼宇名称将使用默认描述：%s", path)
                return None

        try:
            self.fd = fd
            det_option = fd.RuntimeOption()
            cls_option = fd.RuntimeOption()
            rec_option = fd.RuntimeOption()
            det_option.use_cpu()
            cls_option.use_cpu()
            rec_option.use_cpu()
            det = fd.vision.ocr.DBDetector(det_model, det_params, runtime_option=det_option)
            cls = fd.vision.ocr.Classifier(cls_model, cls_params, runtime_option=cls_option)
            rec = fd.vision.ocr.Recognizer(rec_model, rec_params, keys, runtime_option=rec_option)
            det.preprocessor.max_side_len = 960
            det.postprocessor.det_db_thresh = 0.3
            det.postprocessor.det_db_box_thresh = 0.6
            det.postprocessor.det_db_unclip_ratio = 1.5
            det.postprocessor.det_db_score_mode = "slow"
            cls.postprocessor.cls_thresh = 0.9
            ocr = fd.vision.ocr.PPOCRv3(det_model=det, cls_model=cls, rec_model=rec)
            ocr.cls_batch_size = 1
            ocr.rec_batch_size = 6
            rospy.loginfo("[视觉识别] 楼宇 OCR 模型加载成功。")
            return ocr
        except Exception as exc:
            rospy.logwarn("[视觉识别] OCR 模型加载失败，楼宇名称将使用默认描述：%s", exc)
            return None

    @staticmethod
    def assert_file(path):
        if not os.path.exists(path):
            raise RuntimeError("模型文件不存在：{}".format(path))

    def image_callback(self, msg):
        now = time.monotonic()
        if now - self.last_process_time < self.min_process_interval:
            return
        self.last_process_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[视觉识别] 图像转换失败：%s", exc)
            return

        frame = self.resize_keep_ratio(frame, self.max_frame_side)
        try:
            detections = {
                "garbage": self.detect_garbage(frame),
                "human": self.detect_humans(frame),
                "building_fire": self.detect_building_fire(frame),
            }
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[视觉识别] 模型推理失败：%s", exc)
            return
        self.update_streaks(detections)

    def resize_keep_ratio(self, image, max_side):
        height, width = image.shape[:2]
        side = max(height, width)
        if side <= max_side:
            return image
        scale = float(max_side) / float(side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return self.cv2.resize(image, new_size, interpolation=self.cv2.INTER_AREA)

    def detect_garbage(self, frame):
        result = self.garbage_model.predict(frame)
        image_area = float(frame.shape[0] * frame.shape[1])
        best = None
        for item in result:
            score = float(item["score"])
            label = item["label"]
            x1, y1, x2, y2 = item["box"]
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area if image_area > 0.0 else 0.0
            if area_ratio < self.garbage_min_area_ratio:
                continue
            if best is None or score > best[0]:
                best = (score, label)

        if best is None:
            return None

        label = best[1]
        garbage_type = label.split("_", 1)[0]
        status = label.split("_", 1)[1] if "_" in label else ""
        if not garbage_type.endswith("垃圾"):
            garbage_type = "{}垃圾".format(garbage_type)
        if status:
            return "识别结果：垃圾桶类型={}桶，状态={}".format(garbage_type, status)
        return "识别结果：垃圾桶类型={}桶".format(garbage_type)

    def detect_humans(self, frame):
        result = self.human_model.predict(frame)
        image_area = float(frame.shape[0] * frame.shape[1])
        counts = {0: 0, 1: 0}
        for item in result:
            x1, y1, x2, y2 = item["box"]
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area if image_area > 0.0 else 0.0
            if area_ratio < self.human_min_area_ratio:
                continue
            label_id = int(item["label_id"])
            if label_id in counts:
                counts[label_id] += 1

        total = sum(counts.values())
        if total == 0:
            return None

        if counts[0] > 0 and counts[1] > 0:
            human_type = "职业人员{}，普通行人{}".format(counts[0], counts[1])
        elif counts[0] > 0:
            human_type = HUMAN_LABELS[0]
        else:
            human_type = HUMAN_LABELS[1]
        return "识别结果：人群类型={}，数量={}".format(human_type, total)

    def detect_building_fire(self, frame):
        building_result = self.building_model.predict(frame)
        image_area = float(frame.shape[0] * frame.shape[1])
        reports = []
        for item in building_result:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area if image_area > 0.0 else 0.0
            if area_ratio < self.building_min_area_ratio:
                continue
            crop = self.safe_crop(frame, x1, y1, x2, y2)
            if crop is None:
                continue
            building_name = self.extract_building_name(crop)
            fire_report = self.detect_fire_in_building(crop)
            if fire_report:
                reports.append("{}{}".format(building_name, fire_report))

        if not reports:
            return None
        return "识别结果：楼宇灾害情况={}".format("；".join(reports))

    def safe_crop(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def extract_building_name(self, crop):
        if self.ocr_model is None:
            return "楼宇"
        try:
            result = self.ocr_model.predict(crop)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[视觉识别] 楼宇 OCR 推理失败：%s", exc)
            return "楼宇"

        texts = getattr(result, "text", []) or []
        for text in texts:
            clean = re.sub(r"\s+", "", text)
            if "超市" in clean or "商场" in clean:
                return clean
        return "楼宇"

    def detect_fire_in_building(self, crop):
        result = self.fire_model.predict(crop)
        floor_boxes = []
        fire_boxes = []
        for item in result:
            label_id = int(item["label_id"])
            box = [int(v) for v in item["box"]]
            center_y = (box[1] + box[3]) / 2.0
            center_x = (box[0] + box[2]) / 2.0
            if label_id == 0:
                floor_boxes.append((box, center_y))
            elif label_id == 1:
                fire_boxes.append((box, center_x, center_y))

        if not fire_boxes:
            fire_boxes = self.detect_fire_color_regions(crop, floor_boxes)

        if not fire_boxes:
            return None

        floor_boxes.sort(key=lambda item: item[1], reverse=True)
        numbered_floors = [(idx + 1, item[0]) for idx, item in enumerate(floor_boxes)]
        fire_floors = []
        for _, fire_x, fire_y in fire_boxes:
            for floor_num, floor_box in numbered_floors:
                if floor_box[0] <= fire_x <= floor_box[2] and floor_box[1] <= fire_y <= floor_box[3]:
                    fire_floors.append("{}楼".format(floor_num))
                    break

        if fire_floors:
            floors = "、".join(sorted(set(fire_floors)))
            return "{}着火了".format(floors)
        return "检测到火灾"

    def detect_fire_color_regions(self, crop, floor_boxes):
        if not self.use_fire_color_hint:
            return []
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        lower_red_1 = self.np.array([0, 80, 120])
        upper_red_1 = self.np.array([25, 255, 255])
        lower_red_2 = self.np.array([170, 70, 100])
        upper_red_2 = self.np.array([179, 255, 255])
        mask = self.cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask = self.cv2.bitwise_or(mask, self.cv2.inRange(hsv, lower_red_2, upper_red_2))
        kernel = self.np.ones((5, 5), self.np.uint8)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)

        fire_boxes = []
        for box, _ in floor_boxes:
            x1, y1, x2, y2 = box
            region = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if region.size == 0:
                continue
            fire_ratio = float(self.cv2.countNonZero(region)) / float(region.size)
            if fire_ratio >= 0.015:
                fire_boxes.append((box, (x1 + x2) / 2.0, (y1 + y2) / 2.0))

        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(crop.shape[0] * crop.shape[1]) * 0.002
        for contour in contours:
            area = self.cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = self.cv2.boundingRect(contour)
            fire_boxes.append(([x, y, x + w, y + h], x + w / 2.0, y + h / 2.0))
        return fire_boxes

    def update_streaks(self, detections):
        now = time.monotonic()
        for key, message in detections.items():
            if message:
                self.streaks[key] += 1
                self.missed_frames[key] = 0
                self.last_messages[key] = message
            else:
                self.streaks[key] = 0
                self.missed_frames[key] += 1
                if self.missed_frames[key] >= self.reset_after_missed_frames:
                    self.active_targets[key] = False
                continue

            if self.streaks[key] < self.stable_frames:
                continue

            last_report_time = self.last_report_times.get(key, 0.0)
            last_reported = self.last_reported_messages.get(key)
            should_report = (
                not self.active_targets[key]
                or message != last_reported
                or (self.report_cooldown > 0.0 and now - last_report_time >= self.report_cooldown)
            )
            if should_report:
                self.active_targets[key] = True
                self.last_report_times[key] = now
                self.last_reported_messages[key] = message
                rospy.loginfo("%s", message)


def main():
    rospy.init_node("camera_model_recognition_node")
    CameraModelRecognitionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
