import fastdeploy as fd
import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class GarbageDetector:
    def __init__(self):
        opt = fd.RuntimeOption()
        self.model = fd.vision.detection.YOLOv8('moxing/best.onnx', runtime_option=opt)

        # 大幅增大字体大小
        self.font_size = 72  # 进一步增大字体
        self.font = self._get_font()

        self.labels = {
            0: '可回收物_已投放',
            1: '可回收物_未投放',
            2: '有害垃圾_已投放',
            3: '有害垃圾_未投放',
            4: '厨余垃圾_已投放',
            5: '厨余垃圾_未投放',
            6: '其他垃圾_已投放',
            7: '其他垃圾_未投放'
        }

    def _get_font(self):
        """尝试查找系统可用的中文字体"""
        font_candidates = [
            "simhei.ttf",  # Windows
            "WenQuanYi Micro Hei",  # Linux
            "Heiti TC",  # macOS
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux常见路径
        ]

        for font_path in font_candidates:
            try:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, self.font_size)
                else:
                    return ImageFont.truetype(font_path, self.font_size)
            except (IOError, OSError):
                continue

        # 如果找不到任何中文字体，使用默认字体
        try:
            return ImageFont.load_default()
        except Exception:
            # 创建一个临时字体对象
            class DummyFont:
                def getbbox(self, text):
                    return (0, 0, len(text) * 18, self.font_size)  # 调整宽度估计

            return DummyFont()

    def run_inference(self, image_path, save_dir, show=False):
        filename = os.path.basename(image_path)
        bgr_img = cv2.imread(image_path)

        if bgr_img is None:
            print(f"[跳过] 无法加载图像：{image_path}")
            return []

        # 转换为RGB用于PIL处理
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        drawer = ImageDraw.Draw(pil_img)

        # 模型推理
        result = self.model.predict(bgr_img)

        detected_items = []

        if result and len(result.scores) > 0:
            for idx, score in enumerate(result.scores):
                if score < 0.8:
                    continue

                box = result.boxes[idx]
                label = result.label_ids[idx]
                label_name = self.labels.get(label, '未知')

                detected_items.append(label_name)

                x1, y1, x2, y2 = map(int, box)

                # 优化标签位置计算，避免重叠和越界
                label_width, label_height = self.font.getbbox(label_name)[2:4]
                padding = 10  # 增加内边距

                # 智能选择标签位置，优先显示在框上方
                if y1 - label_height - padding > 0:
                    text_position = (x1, y1 - label_height - padding)
                    bg_position = [(x1 - 5, y1 - label_height - padding - 5),
                                   (x1 + label_width + 5, y1)]
                else:
                    text_position = (x1, y2 + padding)
                    bg_position = [(x1 - 5, y2 + padding - 5),
                                   (x1 + label_width + 5, y2 + label_height + padding + 5)]

                # 绘制标签背景（增加不透明度）
                drawer.rectangle(bg_position, fill=(0, 0, 0, 220))
                # 绘制标签文本
                drawer.text(text_position, label_name, fill=(255, 255, 255), font=self.font)
                # 绘制边界框（增加线条粗细）
                drawer.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=5)

            # 保存结果图像
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, filename)
            result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, result_img)

            # 显示图像（如果需要）
            if show:
                self._show_image(result_img, filename, detected_items, bgr_img.shape)
        else:
            print(f"[提示] '{filename}' 中未检测到显著目标。")

            # 显示原始图像（如果需要）
            if show:
                self._show_image(bgr_img, filename, [], bgr_img.shape)

        return detected_items

    def _show_image(self, img, filename, detections, original_shape):
        """显示图像并添加检测结果信息，保持原始比例"""
        # 创建一个副本，避免修改原图
        display_img = img.copy()

        # 添加图像标题
        title = f"正在识别: {filename}"
        cv2.putText(display_img, title, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)  # 增大字体和粗细

        # 添加检测结果信息
        if detections:
            info_text = f"检测到 {len(detections)} 个目标: {', '.join(detections)}"
        else:
            info_text = "未检测到目标"

        cv2.putText(display_img, info_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)  # 增大字体和粗细

        # 添加操作提示
        cv2.putText(display_img, "按任意键继续... (ESC退出)", (10, display_img.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)  # 增大字体和粗细

        # 保持原始比例显示
        cv2.namedWindow("垃圾分类识别", cv2.WINDOW_KEEPRATIO)  # 保持比例
        h, w = original_shape[:2]
        max_size = 1400  # 增大最大显示尺寸
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        cv2.resizeWindow("垃圾分类识别", new_w, new_h)

        cv2.imshow("垃圾分类识别", display_img)
        key = cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()

        # 如果按下ESC键，则终止处理
        if key == 27:  # ESC键
            return False
        return True


def batch_process(show=False):
    src_dir = 'images'
    out_dir = 'output_images'

    detector = GarbageDetector()

    image_files = glob.glob(os.path.join(src_dir, '*.jpg'))
    total = len(image_files)
    print(f"[信息] 总共发现 {total} 张图像，开始处理...")

    results_summary = {}

    for path in image_files:
        name = os.path.basename(path)
        detections = detector.run_inference(path, out_dir, show)
        results_summary[name] = detections

    print("\n[完成] 所有图像处理完成。")
    print("===== 检测统计 =====")
    for img, states in results_summary.items():
        if states:
            print(f"✔ {img} - 检测 {len(states)} 项：{', '.join(states)}")
        else:
            print(f"✘ {img} - 无有效检测结果")
    print("=====================")


if __name__ == '__main__':
    # 设置为True以显示识别过程中的图像
    batch_process(show=True)