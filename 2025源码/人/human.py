import fastdeploy as fd
import cv2
import random
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Config:
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    SHOW_FULLSCREEN = False
    CONFIDENCE_THRESHOLD = 0.5
    FONT_PATH = "ziti.ttf"
    ONNX_MODEL_PATH = "human.onnx"
    DATA_DIR = "images"


# 确保结果目录存在
os.makedirs('result', exist_ok=True)


def init_model():
    """初始化FastDeploy模型"""
    option = fd.RuntimeOption()
    option.use_cpu()
    option.use_ort_backend()

    logging.info(f"检查模型文件: {Config.ONNX_MODEL_PATH}")
    if not os.path.exists(Config.ONNX_MODEL_PATH):
        logging.error(f"模型文件不存在: {Config.ONNX_MODEL_PATH}")
        return None

    try:
        logging.info("正在加载YOLOv8模型...")
        model = fd.vision.detection.YOLOv5(
            Config.ONNX_MODEL_PATH,
            runtime_option=option
        )
        model.postprocessor.conf_threshold = Config.CONFIDENCE_THRESHOLD
        logging.info("模型加载成功!")
        return model
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return None


# 类别名称和颜色
class_names = ['职业人员', '普通人员']
class_colors = [(0, 255, 0), (0, 0, 255)]


def get_font(font_size=80):  # 增大默认字体大小
    """尝试查找系统可用的中文字体"""
    font_candidates = [
        Config.FONT_PATH,
        "simhei.ttf",  # Windows
        "WenQuanYi Micro Hei",  # Linux
        "Heiti TC",  # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux常见路径
    ]

    for font_path in font_candidates:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    # 如果找不到任何中文字体，使用默认字体
    try:
        return ImageFont.load_default()
    except Exception:
        # 创建一个临时字体对象
        class DummyFont:
            def getbbox(self, text):
                return (0, 0, len(text) * 16, font_size)  # 调整宽度估计

        return DummyFont()


def cv2_add_chinese_text(img, text, position, font_size=32, color=(255, 255, 255)):
    """增强版添加中文文本函数，支持更好的字体显示"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = get_font(font_size)
        draw.text(position, text, font=font, fill=tuple(reversed(color)))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        if 'font_error_logged' not in globals():
            logging.error(f"添加中文时出错: {e}。回退到默认方法。")
            globals()['font_error_logged'] = True
        # 回退到默认方法
        return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                           font_size / 30, color, 2)


def detect_and_visualize_people(img_path, model, show=False):
    """检测并可视化人员，优化标签显示"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"无法读取图片: {img_path}")
            return None, None, None

        result = model.predict(img)

        class_counts = {0: 0, 1: 0}
        vis_img = img.copy()

        if len(result.boxes) > 0:
            for i in range(len(result.scores)):
                class_id = int(result.label_ids[i])
                conf = float(result.scores[i])

                if conf >= Config.CONFIDENCE_THRESHOLD and class_id in class_counts:
                    class_counts[class_id] += 1
                    box = result.boxes[i]
                    x1, y1, x2, y2 = map(int, box)

                    color = class_colors[class_id]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)  # 增加边框粗细

                    # 优化标签位置和样式
                    label = f"{class_names[class_id]}: {conf:.2f}"

                    # 使用PIL精确计算文本大小
                    img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font = get_font(32)  # 使用相同的字体大小计算
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    padding = 8  # 文本周围的内边距

                    # 智能选择标签位置，优先显示在框上方
                    if y1 - text_height - padding > 0:
                        # 显示在框上方
                        text_position = (x1, y1 - text_height - padding)
                        bg_position = [
                            (x1 - padding, y1 - text_height - padding * 2),
                            (x1 + text_width + padding, y1)
                        ]
                    else:
                        # 显示在框下方
                        text_position = (x1, y2 + padding)
                        bg_position = [
                            (x1 - padding, y2),
                            (x1 + text_width + padding, y2 + text_height + padding * 2)
                        ]

                    # 添加标签背景
                    draw.rectangle(bg_position, fill=color + (200,))  # 添加透明度
                    draw.text(text_position, label, font=font, fill=(255, 255, 255))
                    vis_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        total_number = sum(class_counts.values())
        stats_text = f"总人数: {total_number} | 职业: {class_counts[0]} | 普通: {class_counts[1]}"
        vis_img = cv2_add_chinese_text(vis_img, stats_text, (10, 40), 32, (0, 255, 255))  # 增大字体

        # 显示图像（如果需要）
        if show:
            show_image(vis_img, f"正在检测: {os.path.basename(img_path)}", original_shape=img.shape)

        return vis_img, total_number, class_counts

    except Exception as e:
        logging.error(f"处理图片 {img_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def show_image(img, title, original_shape=None, wait_key=True):
    """显示图像，保持原始比例"""
    display_img = img.copy()

    # 添加操作提示
    cv2.putText(display_img, "按任意键继续... (ESC退出)", (10, display_img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 保持原始比例显示
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    if original_shape:
        h, w = original_shape[:2]
        max_size = 1400  # 增大最大显示尺寸
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        cv2.resizeWindow(title, new_w, new_h)

    cv2.imshow(title, display_img)

    if wait_key:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 如果按下ESC键，则终止处理
        if key == 27:
            return False
    return True


def create_text_summary_image(results, width=800):
    """创建检测结果汇总图像，优化显示效果"""
    if not results:
        return None

    # 优化界面设计
    bg_color = (45, 45, 45)  # 深灰色背景
    title_color = (52, 199, 235)  # 亮蓝色标题
    text_color = (230, 230, 230)  # 浅灰色文本
    header_color = (255, 199, 0)  # 黄色标题
    font_size_title = 36  # 增大标题字体
    font_size_text = 24  # 增大内容字体
    line_height = 50  # 增加行高
    padding = 30

    height = padding * 3 + line_height * (len(results) + 3)  # 增加总高度
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # 添加标题
    title_text = "检 测 结 果 汇 总"
    canvas = cv2_add_chinese_text(canvas, title_text, (padding, padding + 10), font_size_title, title_color)

    # 添加副标题
    subtitle = f"共检测 {len(results)} 张图片"
    canvas = cv2_add_chinese_text(canvas, subtitle, (padding, padding + 60), 28, header_color)

    start_y = padding * 2 + 60
    cv2.line(canvas, (padding, start_y), (width - padding, start_y), (80, 80, 80), 2)

    # 添加表头
    header_text = "序号 | 图片名称 | 总人数 | 职业人员 | 普通人员"
    canvas = cv2_add_chinese_text(canvas, header_text, (padding, start_y + line_height), font_size_text, header_color)
    cv2.line(canvas, (padding, start_y + line_height + 10),
             (width - padding, start_y + line_height + 10), (80, 80, 80), 1)

    # 添加内容
    current_y = start_y + line_height * 2
    for i, (_, img_name, total, counts) in enumerate(results):
        text = f"{i + 1:2d}. {img_name[:20]:<20s}  总数: {total:2d} 人    职业: {counts[0]:2d}    普通: {counts[1]:2d}"
        canvas = cv2_add_chinese_text(canvas, text, (padding, current_y), font_size_text, text_color)
        current_y += line_height

    # 添加统计摘要
    cv2.line(canvas, (padding, current_y - 10), (width - padding, current_y - 10), (80, 80, 80), 1)
    total_people = sum([r[2] for r in results])
    total_professionals = sum([r[3][0] for r in results])
    total_normal = sum([r[3][1] for r in results])

    summary_text = f"总计: {len(results)}张图片 | 总人数: {total_people} | 职业人员: {total_professionals} | 普通人员: {total_normal}"
    canvas = cv2_add_chinese_text(canvas, summary_text, (padding, current_y + 10), 26, (255, 100, 100))

    return canvas


def main(show=False):
    """主函数，增加显示选项"""
    print("=" * 60)
    print(" " * 20 + "人员检测与统计")
    print("=" * 60)

    model = init_model()
    if model is None:
        logging.error("程序终止：模型初始化失败。")
        return

    print("-" * 60)
    img_paths = glob.glob(os.path.join(Config.DATA_DIR, '*.*'))
    if not img_paths:
        logging.error(f"程序终止：在目录 '{Config.DATA_DIR}' 中未找到任何图片文件。")
        return

    logging.info(f"在 '{Config.DATA_DIR}' 中找到 {len(img_paths)} 张图片，将随机选择最多10张进行处理。")
    selected_images = random.sample(img_paths, min(10, len(img_paths)))

    print("-" * 60)
    logging.info("开始处理图片...")

    results = []
    for i, img_path in enumerate(selected_images):
        img_name = os.path.basename(img_path)
        print(f"  [{i + 1}/{len(selected_images)}] 正在处理: {img_name}")

        result_img, total_number, class_counts = detect_and_visualize_people(img_path, model, show)

        if result_img is not None:
            log_msg = f"    -> 结果: 总数 {total_number}人 (职业: {class_counts[0]}, 普通: {class_counts[1]})"
            print(log_msg)

            result_path = os.path.join('result', f"res_{img_name}")
            cv2.imwrite(result_path, result_img)

            results.append((result_img, img_name, total_number, class_counts))

    print("-" * 60)
    logging.info("所有图片处理完成。正在生成摘要...")

    summary_img = create_text_summary_image(results)

    if summary_img is not None:
        summary_path = os.path.join('result', 'summary_report.jpg')
        cv2.imwrite(summary_path, summary_img)
        logging.info(f"结果摘要已保存到: {summary_path}")

        # 显示汇总结果
        show_image(summary_img, "检测结果汇总", wait_key=True)
    else:
        logging.warning("没有有效的检测结果，无法生成摘要。")

    print("=" * 60)
    print("程序已退出。")


if __name__ == "__main__":
    # 设置为True以显示识别过程中的图像
    main(show=True)