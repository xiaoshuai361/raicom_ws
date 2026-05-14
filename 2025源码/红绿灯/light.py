import os
import cv2
import glob
import logging
import fastdeploy as fd

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO
)

LABELS = {
    0: ('Red', (0, 0, 255)),
    1: ('Green', (0, 255, 0)),
    2: ('Yellow', (0, 255, 255))
}


class LightClassifier:
    def __init__(self, model_path='moxing/best.onnx'):
        option = fd.RuntimeOption()
        self.model = fd.vision.detection.YOLOv8(model_path, runtime_option=option)

    def detect(self, image):
        return self.model.predict(image)


def draw_boxes(image, detections):
    for idx in range(len(detections.scores)):
        score = detections.scores[idx]
        if score < 0.8:
            continue

        box = detections.boxes[idx]
        cls_id = detections.label_ids[idx]

        label, color = LABELS.get(cls_id, ('Unknown', (255, 255, 255)))
        pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, pt1, pt2, color, 4)  # 增加边界线宽度

        # 增大字体大小到80，并调整位置和厚度
        font_scale = 2.5  # 相当于约80px的字体大小
        text_pos = (pt1[0], pt1[1] - 25 if pt1[1] > 50 else pt1[1] + 50)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 6, cv2.LINE_AA)

    return image


def show_image(image, title="检测结果", wait_key=True):
    """显示图像，支持调整窗口大小和等待按键"""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # 获取屏幕分辨率并调整窗口大小
    screen_width, screen_height = 1920, 1080
    max_scale = min(screen_width / image.shape[1], screen_height / image.shape[0]) * 0.8
    window_width = int(image.shape[1] * max_scale)
    window_height = int(image.shape[0] * max_scale)
    cv2.resizeWindow(title, window_width, window_height)

    cv2.imshow(title, image)

    if wait_key:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key
    return None


def process_image(image_path, detector, save_dir, show_result=True):
    filename = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Failed to read image: {image_path}")
        return filename, []

    result = detector.detect(image)
    detected = []

    if result and len(result.scores) > 0:
        for i, score in enumerate(result.scores):
            if score >= 0.8:
                cls_id = result.label_ids[i]
                label = LABELS.get(cls_id, ('Unknown',))[0]
                detected.append(label)

        image = draw_boxes(image, result)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)
        cv2.imwrite(output_path, image)

        # 显示结果图像
        if show_result:
            key = show_image(image, f"分析中: {filename}")
            # 按ESC键退出，其他键继续
            if key == 27:
                return filename, detected, False

    else:
        logging.info(f"No valid detections in {filename}")

        # 显示未检测到目标的图像
        if show_result:
            key = show_image(image, f"分析中: {filename}")
            if key == 27:
                return filename, detected, False

    return filename, detected, True


def main():
    input_dir = 'images'
    output_dir = 'output_images'
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    detector = LightClassifier()
    logging.info(f"Images found: {len(image_paths)}")
    results = {}

    for path in image_paths:
        name, detections, continue_processing = process_image(path, detector, output_dir)
        results[name] = detections

        if not continue_processing:
            logging.warning("用户中断处理，将生成已处理图片的报告")
            break

    print("\n=== 检测识别结果 ===")
    for name, lights in results.items():
        if lights:
            print(f"{name} -> Detected: {', '.join(lights)}")
        else:
            print(f"{name} -> No lights detected.")
    print("========================")


if __name__ == '__main__':
    main()