import fastdeploy as fd
import cv2
import glob
import os
import re
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 新增导入 rich 相关模块 ---
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

LAST_FIRE_FLOOR = []
saved_img_dir = 'result'  # 保存图片的目标文件夹

# --- 使用 rich.console 代替 print ---
console = Console()

os.makedirs(saved_img_dir, exist_ok=True)


# --- 新增字体配置 ---
class FontConfig:
    SIZE_LARGE = 96  # 大字体，用于警告信息
    SIZE_MEDIUM = 72  # 中字体，用于建筑名称和状态信息（减小为原来的一半）
    SIZE_SMALL = 96  # 小字体，用于楼层编号
    PATH = "simhei.ttf"  # 默认中文字体路径

    @staticmethod
    def get_font(size):
        """尝试加载中文字体，失败则使用默认"""
        try:
            return ImageFont.truetype(FontConfig.PATH, size)
        except Exception:
            # 尝试系统字体
            try:
                return ImageFont.truetype("simhei.ttf", size)
            except Exception:
                return ImageFont.load_default()


def add_chinese_text(img, text, position, font_size=32, color=(255, 255, 255)):
    """添加中文文本，支持自定义字体大小和颜色"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = FontConfig.get_font(font_size)
        draw.text(position, text, font=font, fill=tuple(reversed(color)))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        console.print(f"[yellow]⚠️  添加中文文本时出错: {e}，使用默认字体[/yellow]")
        return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                           font_size / 30, color, 2)


def show_image(img, title="检测结果", wait_key=True):
    """显示图像，支持调整窗口大小和等待按键"""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    # 获取屏幕分辨率并调整窗口大小
    screen_width, screen_height = 1920, 1080
    max_scale = min(screen_width / img.shape[1], screen_height / img.shape[0]) * 0.8
    window_width = int(img.shape[1] * max_scale)
    window_height = int(img.shape[0] * max_scale)
    cv2.resizeWindow(title, window_width, window_height)

    cv2.imshow(title, img)

    if wait_key:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key
    return None


class Detector:
    def __init__(self):
        # 创建推理后端
        self.runtime_option = fd.RuntimeOption()
        self.det_option = fd.RuntimeOption()
        self.cls_option = fd.RuntimeOption()
        self.rec_option = fd.RuntimeOption()

        # 使用CPU推理
        self.runtime_option.use_cpu()
        self.det_option.use_cpu()
        self.cls_option.use_cpu()
        self.rec_option.use_cpu()

        # --- 使用 rich 风格的加载信息 ---
        console.print("[yellow]⚙️  正在加载模型...", justify="left")
        try:
            self.yolo_model = fd.vision.detection.YOLOv8('building.onnx', runtime_option=self.runtime_option)
            self.fire_floor_model = fd.vision.detection.YOLOv8('fire.onnx', runtime_option=self.runtime_option)
            self.det_model = fd.vision.ocr.DBDetector('OCR/ch_PP-OCRv3_det_infer/inference.pdmodel',
                                                      'OCR/ch_PP-OCRv3_det_infer/inference.pdiparams',
                                                      runtime_option=self.det_option)
            self.cls_model = fd.vision.ocr.Classifier('OCR/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel',
                                                      'OCR/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams',
                                                      runtime_option=self.cls_option)
            self.rec_model = fd.vision.ocr.Recognizer('OCR/ch_PP-OCRv3_rec_infer/inference.pdmodel',
                                                      'OCR/ch_PP-OCRv3_rec_infer/inference.pdiparams',
                                                      'OCR/ppocr_keys_v1.txt', runtime_option=self.rec_option)
            console.print("[bold green]✅ 模型加载成功！[/bold green]\n", justify="left")
        except Exception as e:
            console.print(f"[bold red]❌ 模型加载失败，请检查模型路径和文件是否存在: {e}[/bold red]")
            raise

        # 设置参数
        self.det_model.preprocessor.max_side_len = 960
        self.det_model.postprocessor.det_db_thresh = 0.3
        self.det_model.postprocessor.det_db_box_thresh = 0.6
        self.det_model.postprocessor.det_db_unclip_ratio = 1.5
        self.det_model.postprocessor.det_db_score_mode = "slow"
        self.det_model.postprocessor.use_dilation = False
        self.cls_model.postprocessor.cls_thresh = 0.9

        self.ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=self.det_model, cls_model=self.cls_model,
                                              rec_model=self.rec_model)
        self.ppocr_v3.cls_batch_size = 1
        self.ppocr_v3.rec_batch_size = 6

    def yolo_predict(self, img):
        return self.yolo_model.predict(img)

    def ocr_predict(self, img):
        return self.ppocr_v3.predict(img)

    def fire_floor_predict(self, img):
        return self.fire_floor_model.predict(img)

    def predict_and_draw(self, img_path, show_result=True):
        """
        检测并绘制结果，可选显示图像
        """
        global LAST_FIRE_FLOOR

        img_basename = os.path.basename(img_path)
        im = cv2.imread(img_path)
        if im is None:
            report = f"{img_basename}: 无法读取图片"
            LAST_FIRE_FLOOR.append(report)
            return report, None  # 返回报告和空的保存路径

        draw_im = im.copy()
        yolo_results = self.yolo_predict(im)
        current_image_building_reports = []

        for i in range(len(yolo_results.scores)):
            if yolo_results.scores[i] > 0.8:
                x1_building, y1_building, x2_building, y2_building = [int(coord) for coord in yolo_results.boxes[i]]
                y1_building, x1_building = max(0, y1_building), max(0, x1_building)
                y2_building, x2_building = min(im.shape[0], y2_building), min(im.shape[1], x2_building)
                cut_frame = im[y1_building:y2_building, x1_building:x2_building]

                if cut_frame.shape[0] == 0 or cut_frame.shape[1] == 0:
                    current_image_building_reports.append("某区域裁剪失败")
                    continue

                # 绘制建筑物边界
                cv2.rectangle(draw_im, (x1_building, y1_building), (x2_building, y2_building), (0, 255, 0), 4)

                OCR_img = cut_frame.copy()
                detect_result_ocr = self.ocr_predict(OCR_img)
                house_name = ''
                for name_ocr in detect_result_ocr.text:
                    if '超市' in name_ocr or '商场' in name_ocr:
                        house_name = name_ocr
                        break

                if house_name == '':
                    current_image_building_reports.append("未识别到名称的建筑物")
                    continue

                # 显示建筑物名称 - 调整位置以适应调整后的字体
                draw_im = add_chinese_text(draw_im, house_name, (x1_building, y1_building - 20),
                                           FontConfig.SIZE_MEDIUM, (255, 255, 0))

                floor_img = cut_frame.copy()
                floor_results = self.fire_floor_predict(floor_img)
                floor_boxes, fire_boxes = [], []

                for j in range(len(floor_results.scores)):
                    if floor_results.scores[j] > 0.6:
                        f_label = floor_results.label_ids[j]
                        x1_rel, y1_rel, x2_rel, y2_rel = floor_results.boxes[j]
                        abs_x1, abs_y1, abs_x2, abs_y2 = int(x1_rel + x1_building), int(y1_rel + y1_building), int(
                            x2_rel + x1_building), int(y2_rel + y1_building)
                        color = (0, 0, 255) if f_label == 1 else (255, 0, 0)
                        cv2.rectangle(draw_im, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 3)

                        if f_label == 0:
                            floor_boxes.append([[abs_x1, abs_y1, abs_x2, abs_y2], (abs_y1 + abs_y2) / 2])
                        elif f_label == 1:
                            fire_center_x, fire_center_y = (abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2
                            fire_boxes.append([[abs_x1, abs_y1, abs_x2, abs_y2], fire_center_x, fire_center_y])

                floor_boxes.sort(key=lambda x: x[1], reverse=True)
                numbered_floors = []
                for idx, floor_box_info in enumerate(floor_boxes):
                    coords = floor_box_info[0]
                    current_floor_num = idx + 1
                    # 调整位置以适应调整后的字体
                    draw_im = add_chinese_text(draw_im, f"{current_floor_num}F", (coords[0], coords[1] - 20),
                                               FontConfig.SIZE_SMALL, (255, 255, 255))
                    numbered_floors.append([current_floor_num, coords])

                detected_fire_floors_list = []
                for fire_box_info in fire_boxes:
                    fire_center_x, fire_center_y = fire_box_info[1], fire_box_info[2]
                    for floor_num, floor_coords in numbered_floors:
                        fl_x1, fl_y1, fl_x2, fl_y2 = floor_coords
                        if fl_x1 <= fire_center_x <= fl_x2 and fl_y1 <= fire_center_y <= fl_y2:
                            detected_fire_floors_list.append(f"{floor_num}楼")
                            break

                if len(detected_fire_floors_list) > 0:
                    unique_fire_floors = sorted(list(set(detected_fire_floors_list)))
                    current_image_building_reports.append(f"{house_name}{'，'.join(unique_fire_floors)}着火了")
                else:
                    current_image_building_reports.append(f"{house_name}没着火")

        if current_image_building_reports:
            final_image_report_content = '，'.join(current_image_building_reports)
        else:
            final_image_report_content = "未检测到有效建筑物信息"

        # 在图片中心上方显示警告和状态信息
        img_height, img_width = draw_im.shape[:2]

        # 检查是否有着火的情况
        has_fire = any("着火了" in report for report in current_image_building_reports)

        if has_fire:
            # 有着火情况，显示醒目的警告信息
            warning_text = "⚠️ 警告：检测到火灾 ⚠️"
            draw_im = add_chinese_text(draw_im, warning_text,
                                       (img_width // 2 - 300, 60),
                                       FontConfig.SIZE_LARGE, (0, 0, 255))

            # 显示详细信息 - 调整位置以适应调整后的字体
            draw_im = add_chinese_text(draw_im, final_image_report_content,
                                       (img_width // 2 - 300, 160),  # 调整Y偏移量
                                       FontConfig.SIZE_MEDIUM, (0, 0, 255))
        else:
            # 没有着火情况，显示状态信息 - 调整位置以适应调整后的字体
            draw_im = add_chinese_text(draw_im, final_image_report_content,
                                       (img_width // 2 - 300, 120),  # 调整Y偏移量
                                       FontConfig.SIZE_MEDIUM, (0, 100, 255))

        full_report_string = f"{img_basename}: {final_image_report_content}"
        LAST_FIRE_FLOOR.append(full_report_string)

        save_path = os.path.join(saved_img_dir, f"detected_{img_basename}")
        cv2.imwrite(save_path, draw_im)

        # 显示结果图像
        if show_result:
            key = show_image(draw_im, f"分析中: {img_basename}")
            # 按ESC键退出，其他键继续
            if key == 27:
                return full_report_string, save_path, False
        return full_report_string, save_path, True


def main():
    global LAST_FIRE_FLOOR

    # 打印欢迎信息
    console.rule("[bold cyan]🔥 智能楼宇火灾检测 🔥[/bold cyan]")

    floor_detector = Detector()
    data_dir = 'images'
    all_img_paths = glob.glob(os.path.join(data_dir, 'ly*.jpg'))

    num_images_to_process = 51
    if len(all_img_paths) < num_images_to_process:
        console.print(
            f"[yellow]⚠️  警告：'{data_dir}' 中只找到 {len(all_img_paths)} 张 'ly*.jpg' 图片，将处理所有这些图片。[/yellow]")
        selected_img_paths = all_img_paths
    else:
        selected_img_paths = random.sample(all_img_paths, num_images_to_process)

    LAST_FIRE_FLOOR.clear()

    console.print(f"\n[bold]📋 任务计划：将处理以下 [cyan]{len(selected_img_paths)}[/cyan] 张图片：[/bold]")
    for img_path in selected_img_paths:
        console.print(f"  - [dim]{os.path.basename(img_path)}[/dim]")

    console.print("\n")  # 添加一个空行

    # --- 使用 rich.progress 创建进度条 ---
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]正在处理...", total=len(selected_img_paths))

        for img_path in selected_img_paths:
            report, save_path, continue_processing = floor_detector.predict_and_draw(img_path, show_result=True)

            if not continue_processing:
                console.print("[yellow]⚠️ 用户中断处理，将生成已处理图片的报告[/yellow]")
                break

            # --- 根据结果使用不同风格打印实时信息 ---
            if "着火了" in report:
                console.print(f"🔥 [bold red]火警![/bold red] {report}")
            elif "没着火" in report:
                console.print(f"✅ [green]安全[/green]   {report}")
            else:
                console.print(f"⚠️ [yellow]信息[/yellow]   {report}")

            if save_path:
                console.print(f"   [blue] -> 已保存至 [i]{save_path}[/i][/blue]")

            progress.update(task, advance=1)

    # --- 使用 rich.table 制作最终的总结报告 ---
    console.rule("[bold green]📊 所有图片最终检测结果汇总[/bold green]")

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("🖼️ 图片", style="dim", width=20)
    summary_table.add_column("🚦 状态", justify="center")
    summary_table.add_column("📝 详细信息")

    for report in LAST_FIRE_FLOOR:
        img_name, details = report.split(':', 1)
        details = details.strip()

        if "着火了" in details:
            status = "[bold red]🔥 火警[/bold red]"
        elif "没着火" in details:
            status = "[green]✅ 安全[/green]"
        else:
            status = "[yellow]⚠️ 警告[/yellow]"

        summary_table.add_row(img_name, status, details)

    console.print(summary_table)


if __name__ == '__main__':
    main()