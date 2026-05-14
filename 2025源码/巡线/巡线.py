import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SystemConfig:
    """系统参数配置类，集中管理所有可调节参数

    包含图像处理、ROI设置、方向决策、卡尔曼滤波和显示效果等参数
    """
    # 图像处理参数 - 影响图像预处理和车道线检测效果
    ADAPTIVE_THRESH_BLOCK = 23  # 自适应阈值分块大小(必须为奇数)
    ADAPTIVE_THRESH_CONST = 7  # 自适应阈值减法常量
    MIN_BLACK_PX = 8  # 有效行的最小黑色像素数
    MAX_PIXEL_GAP_DIST = 18  # 像素分组最大间隔距离
    GRAY_LEVEL_THRESH = 107  # 灰度值验证阈值(低于此值视为车道线)
    ROI_MARGIN_SIZE = 25  # ROI动态调整边距(像素)

    # 感兴趣区域(ROI)参数 - 控制图像分析区域
    ROI_START_RATIO = 0.20  # ROI起始行比例(图像顶部向下的比例)
    ROI_END_RATIO = 0.55  # ROI结束行比例(图像顶部向下的比例)
    BODY_DETECTION_RATIO = 0.55  # 车身检测区域起始比例
    BODY_BLACK_PX_RATIO = 0.12  # 车身检测黑色像素比例阈值

    # 方向决策参数 - 控制转向计算和判断逻辑
    DIRECTION_DEAD_ZONE = 30  # 方向判断死区范围(像素)
    LOOKAHEAD_POINT_RATIO = 0.5  # 前瞻点位置比例(从底部向上的比例)
    TURN_ANGLE_THRESH = 8  # 转向角度判断阈值(度)

    # 卡尔曼滤波参数 - 影响中心线平滑效果
    KF_PROCESS_NOISE = 1e-5  # 过程噪声协方差(值越小越平滑)
    KF_MEASUREMENT_NOISE = 5e-2  # 测量噪声协方差(值越小越敏感)

    # 可视化显示参数 - 控制输出画面效果
    DISPLAY_FONT_SIZE = 55  # 显示文字字体大小
    LINE_DRAW_THICKNESS = 6  # 线条绘制粗细
    LANE_LINE_COLOR = (255, 0, 0)  # 车道线颜色(蓝色)
    CENTER_LINE_COLOR = (0, 255, 0)  # 中心线颜色(绿色)
    VEHICLE_CENTER_COLOR = (0, 0, 255)  # 车辆中心颜色(红色)
    LOOKAHEAD_POINT_COLOR = (0, 0, 255)  # 前瞻点颜色(红色)
    ANGLE_TEXT_COLOR = (0, 255, 255)  # 角度文字颜色(黄色)

    # 视频处理控制参数 - 控制处理流程和性能
    SHOW_PROCESSING_FLAG = True  # 是否显示处理过程
    SAVE_PROCESS_RESULT = True  # 是否保存处理结果
    DISPLAY_DELAY_TIME = 1  # 显示延迟(毫秒)

    # 调试模式选项 - 用于开发和问题排查
    ENABLE_DEBUG_DISPLAY = False  # 启用调试信息显示
    USE_DYNAMIC_ROI_FLAG = True  # 使用动态ROI区域
    ENHANCE_VALIDATION = True  # 启用增强验证
    SHOW_ADDITIONAL_INFO_FLAG = False  # 显示附加信息


def read_image_with_zh_path(file_path):
    """安全读取包含中文路径的图像文件

    Args:
        file_path: 图像文件路径

    Returns:
        成功时返回OpenCV格式图像，失败时返回None
    """
    try:
        with open(file_path, "rb") as file_stream:
            # 以二进制方式读取并转换为OpenCV格式
            byte_data = bytearray(file_stream.read())
            numpy_array = np.asarray(byte_data, dtype=np.uint8)
            return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"图像读取失败 [{file_path}]: {str(e)}")
        return None


def draw_chinese_text(img, text_content, position,
                      font_path="C:/Windows/Fonts/simhei.ttf",
                      font_size=50, text_color=(0, 0, 255)):
    """在图像上绘制中文文本

    使用PIL库处理中文显示问题，然后转回OpenCV格式

    Args:
        img: 输入图像(OpenCV格式)
        text_content: 要显示的文本内容
        position: 文本位置(x, y)
        font_path: 字体文件路径
        font_size: 字体大小
        text_color: 文本颜色(BGR格式)

    Returns:
        添加了文本的图像(OpenCV格式)
    """
    # 转换为PIL图像以支持中文显示
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        # 尝试加载指定中文字体
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except IOError:
        # 字体加载失败时使用默认字体
        print(f"警告: 无法加载字体文件 {font_path}，将使用默认字体")
        font = ImageFont.load_default()

    # 注意: PIL使用RGB格式，而OpenCV使用BGR格式
    draw.text(position, text_content, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def image_preprocessing(input_img, config):
    """图像预处理流程

    包括高斯模糊、灰度转换、对比度增强和自适应阈值二值化

    Args:
        input_img: 输入原始图像
        config: 系统配置参数

    Returns:
        enhanced_gray: 增强后的灰度图像
        binary_img: 二值化图像
    """
    # 高斯模糊减少图像噪声
    blurred_img = cv2.GaussianBlur(input_img, (5, 5), 0)

    # 转换为灰度图以便后续处理
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    # 对比度受限的自适应直方图均衡化(CLAHE)
    # 增强图像局部对比度，有助于车道线检测
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_img)

    # 自适应阈值处理生成二值图像
    # 对不同光照条件具有更好的适应性
    binary_img = cv2.adaptiveThreshold(
        enhanced_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESH_BLOCK,
        config.ADAPTIVE_THRESH_CONST
    )

    # 形态学开运算(先腐蚀后膨胀)去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    return enhanced_gray, opened_img


def generate_roi_mask(img_height, img_width, config, body_pos=None):
    """生成感兴趣区域(ROI)掩码

    Args:
        img_height: 图像高度
        img_width: 图像宽度
        config: 系统配置参数
        body_pos: 车身位置(用于动态ROI调整)

    Returns:
        roi_mask: ROI掩码图像
        roi_start: ROI起始行
        roi_end: ROI结束行
    """
    # 基于配置比例计算ROI区域
    start_y = int(img_height * config.ROI_START_RATIO)
    end_y = int(img_height * config.ROI_END_RATIO)

    # 动态调整ROI区域(如果启用且已知车身位置)
    if config.USE_DYNAMIC_ROI_FLAG and body_pos is not None:
        # 确保ROI结束行不低于车身位置，保留一定边距
        end_y = max(int(body_pos - config.ROI_MARGIN_SIZE), start_y)

    # 创建矩形ROI掩码
    roi_mask = np.zeros((img_height, img_width), dtype="uint8")
    cv2.rectangle(roi_mask, (0, start_y), (img_width, end_y), 255, -1)

    return roi_mask, start_y, end_y


def detect_vehicle_body(binary_img, img_height, img_width, config):
    """检测车身位置用于动态ROI调整

    分析图像底部区域，寻找黑色像素密集区域作为车身

    Args:
        binary_img: 二值化图像
        img_height: 图像高度
        img_width: 图像宽度
        config: 系统配置参数

    Returns:
        车身顶部位置(y坐标)，未检测到时返回None
    """
    # 定义车身检测区域(图像下半部分)
    detection_start = int(img_height * config.BODY_DETECTION_RATIO)
    lower_region = binary_img[detection_start:, :]

    # 计算检测区域内黑色像素数量
    black_pixel_count = np.count_nonzero(lower_region == 0)
    region_area = img_width * (img_height - detection_start)

    # 如果黑色像素比例不足，逐行检测车身位置
    if black_pixel_count < region_area * config.BODY_BLACK_PX_RATIO:
        for y in range(detection_start, img_height):
            row_black = np.count_nonzero(binary_img[y, :] == 0)
            if row_black > img_width * config.BODY_BLACK_PX_RATIO:
                return y
    return None


def group_black_pixels(pixel_list, max_gap):
    """将连续的黑色像素分组

    Args:
        pixel_list: 黑色像素位置列表
        max_gap: 最大像素间隔

    Returns:
        分组后的像素列表，每组为连续的像素位置
    """
    pixel_groups = []
    if len(pixel_list) == 0:
        return pixel_groups

    # 初始化当前分组
    current_group = [pixel_list[0]]

    # 遍历所有像素，按间隔分组
    for i in range(1, len(pixel_list)):
        if pixel_list[i] - pixel_list[i - 1] <= max_gap:
            current_group.append(pixel_list[i])
        else:
            # 超过最大间隔，结束当前分组并开始新分组
            if len(current_group) > 3:  # 忽略过小的分组
                pixel_groups.append(current_group)
            current_group = [pixel_list[i]]

    # 添加最后一个分组
    if len(current_group) > 3:
        pixel_groups.append(current_group)

    return pixel_groups


def validate_lane_candidate(gray_row, pixel_groups, threshold):
    """验证像素组是否为有效车道线

    通过计算像素组的平均灰度值判断是否为车道线

    Args:
        gray_row: 灰度图像的一行
        pixel_groups: 像素分组列表
        threshold: 灰度阈值

    Returns:
        有效的车道线像素组列表
    """
    valid_lines = []
    for group in pixel_groups:
        # 计算像素组的平均灰度值
        mean_gray = np.mean(gray_row[group])
        # 灰度值低于阈值的认为是车道线(黑色或暗色)
        if mean_gray < threshold:
            valid_lines.append(group)
    return valid_lines


def detect_lane_boundaries(binary_img, gray_img, roi_start, roi_end, img_width, config):
    """检测左右车道线并计算中心线

    从下往上逐行分析，寻找左右车道线并计算中间引导线

    Args:
        binary_img: 二值化图像
        gray_img: 灰度图像
        roi_start: ROI起始行
        roi_end: ROI结束行
        img_width: 图像宽度
        config: 系统配置参数

    Returns:
        left_lane: 左车道线点列表
        right_lane: 右车道线点列表
        center_points: 中心引导线点列表
    """
    left_lane = []
    right_lane = []
    center_points = []

    # 从下往上逐行分析(更接近车辆的位置优先级更高)
    roi_y_range = range(roi_end, roi_start, -2)

    for y in roi_y_range:
        # 获取当前行的黑色像素位置
        black_pixels = np.where(binary_img[y, :] == 0)[0]

        if len(black_pixels) > config.MIN_BLACK_PX:
            # 分组连续的黑色像素
            pixel_groups = group_black_pixels(black_pixels, config.MAX_PIXEL_GAP_DIST)
            # 验证分组是否为有效车道线
            valid_lines = validate_lane_candidate(gray_img[y, :], pixel_groups, config.GRAY_LEVEL_THRESH)

            if len(valid_lines) >= 2:
                # 找到最左侧和最右侧的车道线
                sorted_lines = sorted(valid_lines, key=lambda x: np.mean(x))
                left_line = sorted_lines[0]
                right_line = sorted_lines[-1]

                # 计算车道线中心点
                left_center = int(np.mean(left_line))
                right_center = int(np.mean(right_line))

                # 确保左右车道线间距合理(避免误检)
                if right_center - left_center > img_width * 0.15:
                    left_lane.append((left_center, y))
                    right_lane.append((right_center, y))
                    center_points.append(((left_center + right_center) // 2, y))
            elif len(valid_lines) == 1:
                # 仅检测到一条车道线时的处理
                line_center = int(np.mean(valid_lines[0]))
                if line_center < img_width / 2:
                    # 假设为左车道线，右车道线使用右边界
                    left_lane.append((line_center, y))
                    right_lane.append((img_width - 1, y))
                else:
                    # 假设为右车道线，左车道线使用左边界
                    right_lane.append((line_center, y))
                    left_lane.append((0, y))
                center_points.append(((left_lane[-1][0] + right_lane[-1][0]) // 2, y))

    # 补全未检测到车道线的行，使用边界值
    all_roi_y = set(roi_y_range)
    left_detected_y = {p[1] for p in left_lane}
    right_detected_y = {p[1] for p in right_lane}

    for y in all_roi_y - left_detected_y:
        left_lane.append((0, y))  # 左边界
    for y in all_roi_y - right_detected_y:
        right_lane.append((img_width - 1, y))  # 右边界

    # 按行排序(从下到上)
    left_lane.sort(key=lambda p: p[1], reverse=True)
    right_lane.sort(key=lambda p: p[1], reverse=True)

    # 重新计算中心线，确保与车道线点对应
    center_points = [((l[0] + r[0]) // 2, l[1]) for l, r in zip(left_lane, right_lane)]

    return left_lane, right_lane, center_points


def validate_lane_lines(left_lane, right_lane, config):
    """基于间距一致性验证车道线有效性

    分析左右车道线之间的距离是否保持相对稳定，过滤异常点

    Args:
        left_lane: 左车道线点列表
        right_lane: 右车道线点列表
        config: 系统配置参数

    Returns:
        经过验证的左右车道线点列表
    """
    if not left_lane or not right_lane or len(left_lane) != len(right_lane):
        return left_lane, right_lane

    valid_left = []
    valid_right = []
    lane_distances = [r[0] - l[0] for l, r in zip(left_lane, right_lane)]

    if not lane_distances:
        return left_lane, right_lane

    # 计算平均间距和标准差
    avg_distance = np.mean(lane_distances)
    std_distance = np.std(lane_distances)

    # 保留间距在合理范围内的点(排除异常值)
    for i, (left, right) in enumerate(zip(left_lane, right_lane)):
        distance = right[0] - left[0]
        if abs(distance - avg_distance) < 2.5 * std_distance:
            valid_left.append(left)
            valid_right.append(right)

    # 如果有效点超过40%，返回有效点，否则返回原始点
    if len(valid_left) > len(left_lane) * 0.4:
        return valid_left, valid_right
    return left_lane, right_lane


def smooth_centerline(raw_points, kalman_filter, last_mid):
    """使用卡尔曼滤波平滑中心线

    减少中心线抖动，提供更稳定的导航引导

    Args:
        raw_points: 原始中心线路标点
        kalman_filter: 卡尔曼滤波器对象
        last_mid: 上一帧的中心点x坐标

    Returns:
        smoothed_points: 平滑后的中心线路标点
        current_mid: 当前帧的中心点x坐标
    """
    smoothed_points = []
    current_mid = last_mid

    for x, y in raw_points:
        # 预测下一状态
        predicted = kalman_filter.predict()

        # 更新测量值(当前检测到的中心点)
        measurement = np.array([[np.float32(x)]])

        # 结合预测和测量，得到最优估计
        corrected = kalman_filter.correct(measurement)

        # 保存平滑后的点
        smoothed_x = int(corrected[0])
        smoothed_points.append((smoothed_x, y))
        current_mid = smoothed_x

    return smoothed_points, current_mid


def calculate_steering_direction(smoothed_points, img_width, roi_end, config):
    """根据平滑后的中心线计算行驶方向和转向角度

    Args:
        smoothed_points: 平滑后的中心线路标点
        img_width: 图像宽度
        roi_end: ROI结束行(车辆位置)
        config: 系统配置参数

    Returns:
        direction: 行驶方向描述
        position_error: 位置误差(像素)
        angle_deg: 转向角度(度)
        vehicle_center_x: 车辆中心点x坐标
        vehicle_center_y: 车辆中心点y坐标
    """
    if not smoothed_points:
        return "未知", 0, 0, img_width // 2, roi_end

    # 车辆中心点(图像底部中心)
    vehicle_center_x = img_width // 2
    vehicle_center_y = roi_end

    # 计算前瞻点(基于前瞻比例选择中心线点)
    lookahead_idx = max(1, int(len(smoothed_points) * config.LOOKAHEAD_POINT_RATIO))
    lookahead_point = smoothed_points[lookahead_idx]
    lookahead_x, lookahead_y = lookahead_point

    # 计算位置误差(车辆中心与前瞻点的水平偏差)
    position_error = vehicle_center_x - lookahead_x

    # 计算转向角度(基于车辆中心和前瞻点的连线)
    dx = lookahead_x - vehicle_center_x  # x方向差值
    dy = vehicle_center_y - lookahead_y  # y方向差值(向下为正)

    # 计算角度(atan2返回弧度，转换为度)
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)

    # 根据转向角度判断行驶方向
    if abs(angle_deg) < config.TURN_ANGLE_THRESH:
        direction = "直行"
    else:
        direction = "向左转" if angle_deg < 0 else "向右转"

    return direction, position_error, angle_deg, vehicle_center_x, vehicle_center_y


def process_frame(original_img, kalman_filter, last_mid_point, config=SystemConfig, frame_idx=0, body_pos=None):
    """核心处理函数：整合所有检测步骤

    Args:
        original_img: 原始输入图像
        kalman_filter: 卡尔曼滤波器对象
        last_mid_point: 上一帧的中心点x坐标
        config: 系统配置参数
        frame_idx: 当前帧索引
        body_pos: 车身位置

    Returns:
        包含所有处理结果的元组
    """
    height, width, _ = original_img.shape

    # 步骤1: 图像预处理
    gray_img, binary_img = image_preprocessing(original_img, config)

    # 步骤2: 创建ROI掩码并应用到二值图像
    roi_mask, roi_start, roi_end = generate_roi_mask(height, width, config, body_pos)
    binary_img[roi_mask == 0] = 255  # 屏蔽ROI外区域

    # 步骤3: 动态更新车身位置(每5帧更新一次以节省计算资源)
    if config.USE_DYNAMIC_ROI_FLAG and frame_idx % 5 == 0:
        body_pos = detect_vehicle_body(binary_img, height, width, config)

    # 步骤4: 检测车道线
    left_lane, right_lane, raw_center = detect_lane_boundaries(binary_img, gray_img, roi_start, roi_end, width, config)

    # 步骤5: 车道线验证(基于间距一致性)
    if config.ENHANCE_VALIDATION and left_lane and right_lane:
        left_lane, right_lane = validate_lane_lines(left_lane, right_lane, config)
        # 确保左右车道线点数一致
        min_length = min(len(left_lane), len(right_lane))
        left_lane, right_lane = left_lane[:min_length], right_lane[:min_length]

    # 步骤6: 平滑中心线(使用卡尔曼滤波)
    if raw_center:
        smoothed_center, last_mid_point = smooth_centerline(raw_center, kalman_filter, last_mid_point)
    else:
        smoothed_center = []

    # 步骤7: 计算行驶方向和转向角度
    if smoothed_center:
        direction, error, angle, center_x, center_y = calculate_steering_direction(
            smoothed_center, width, roi_end, config
        )
    else:
        direction, error, angle, center_x, center_y = "未知", 0, 0, width // 2, roi_end

    return (direction, error, angle, left_lane, right_lane, smoothed_center,
            last_mid_point, body_pos, center_x, center_y)


def visualize_detection_results(source_frame, direction_str, pos_error, turn_angle,
                                left_lane, right_lane, center_points,
                                vehicle_center_x, vehicle_center_y):
    """可视化检测结果

    Args:
        source_frame: 原始图像
        direction_str: 行驶方向字符串
        pos_error: 位置误差(像素)
        turn_angle: 转向角度(度)
        left_lane: 左车道线点列表
        right_lane: 右车道线点列表
        center_points: 中心引导线点列表
        vehicle_center_x: 车辆中心点x坐标
        vehicle_center_y: 车辆中心点y坐标

    Returns:
        可视化后的图像
    """
    display_img = source_frame.copy()

    # 绘制车道线
    if left_lane and right_lane:
        cv2.polylines(display_img, [np.array(left_lane)], False,
                      SystemConfig.LANE_LINE_COLOR, SystemConfig.LINE_DRAW_THICKNESS)
        cv2.polylines(display_img, [np.array(right_lane)], False,
                      SystemConfig.LANE_LINE_COLOR, SystemConfig.LINE_DRAW_THICKNESS)

    # 绘制前瞻点和引导线
    if center_points:
        lookahead_idx = max(1, int(len(center_points) * SystemConfig.LOOKAHEAD_POINT_RATIO))
        lookahead = center_points[lookahead_idx]
        cv2.circle(display_img, lookahead, 12, SystemConfig.LOOKAHEAD_POINT_COLOR, -1)
        cv2.line(display_img, (vehicle_center_x, vehicle_center_y), lookahead,
                 SystemConfig.CENTER_LINE_COLOR, 2)

    # 绘制车辆中心点标记
    cv2.circle(display_img, (vehicle_center_x, vehicle_center_y), 15, SystemConfig.VEHICLE_CENTER_COLOR, -1)
    cv2.line(display_img, (vehicle_center_x - 30, vehicle_center_y), (vehicle_center_x + 30, vehicle_center_y),
             SystemConfig.VEHICLE_CENTER_COLOR, 3)
    cv2.line(display_img, (vehicle_center_x, vehicle_center_y - 30), (vehicle_center_x, vehicle_center_y + 30),
             SystemConfig.VEHICLE_CENTER_COLOR, 3)

    # 显示方向和误差信息
    if direction_str != "未知":
        display_img = draw_chinese_text(display_img, f"方向: {direction_str}", (20, 60),
                                        font_size=SystemConfig.DISPLAY_FONT_SIZE)
        display_img = draw_chinese_text(display_img, f"误差: {pos_error} 像素", (20, 120),
                                        font_size=SystemConfig.DISPLAY_FONT_SIZE)
        display_img = draw_chinese_text(display_img, f"转向角度: {turn_angle:.1f}°", (20, 180),
                                        font_size=SystemConfig.DISPLAY_FONT_SIZE,
                                        text_color=SystemConfig.ANGLE_TEXT_COLOR)

    return display_img


def process_video_file(input_path, output_path=None):
    """视频处理主函数

    读取视频文件，逐帧处理并保存结果

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径(可选)
    """
    # 打开视频文件
    video_cap = cv2.VideoCapture(input_path)
    if not video_cap.isOpened():
        print(f"错误：无法打开视频文件 - {input_path}")
        return

    # 获取视频参数
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频参数: {frame_width}x{frame_height}, {fps} FPS, 总帧数: {total_frames}")

    # 初始化视频写入器
    if output_path and SystemConfig.SAVE_PROCESS_RESULT:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    else:
        video_out = None

    # 初始化卡尔曼滤波器
    kf = cv2.KalmanFilter(2, 1, 0)
    kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)  # 状态转移矩阵
    kf.measurementMatrix = np.array([[1, 0]], np.float32)  # 测量矩阵
    kf.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * SystemConfig.KF_PROCESS_NOISE  # 过程噪声
    kf.measurementNoiseCov = np.array([[1]], np.float32) * SystemConfig.KF_MEASUREMENT_NOISE  # 测量噪声

    # 初始化处理状态
    is_initialized = False
    last_mid_point_x = frame_width // 2
    body_position = None
    current_frame = 0

    # 逐帧处理视频
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        # 初始化卡尔曼滤波器状态
        if not is_initialized:
            kf.statePost = np.array([[last_mid_point_x], [0]], np.float32)
            is_initialized = True

        # 核心处理流程
        (direction, error, angle, left_lane, right_lane, center_points,
         last_mid_point_x, body_position, center_x, center_y) = process_frame(
            frame, kf, last_mid_point_x, SystemConfig, current_frame, body_position
        )

        # 可视化处理结果
        result_frame = visualize_detection_results(
            frame, direction, error, angle, left_lane, right_lane, center_points,
            center_x, center_y
        )

        # 显示处理结果
        if SystemConfig.SHOW_PROCESSING_FLAG:
            cv2.imshow("车道线检测系统", result_frame)
            if cv2.waitKey(SystemConfig.DISPLAY_DELAY_TIME) & 0xFF == ord('q'):
                break

        # 保存处理结果
        if video_out:
            video_out.write(result_frame)

        # 更新处理进度
        current_frame += 1
        if current_frame % 50 == 0:
            print(f"处理进度: {current_frame}/{total_frames} 帧")

    # 释放资源
    video_cap.release()
    if video_out:
        video_out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！共处理 {current_frame} 帧")


if __name__ == "__main__":
    """程序入口点"""
    # 获取用户输入的视频路径
    input_video_path = input("请输入视频路径: ").strip()
    if not os.path.exists(input_video_path):
        print(f"错误：文件不存在 - {input_video_path}")
        exit(1)

    # 生成输出视频路径
    output_folder = os.path.dirname(input_video_path)
    file_base_name, file_ext = os.path.splitext(os.path.basename(input_video_path))
    output_video_path = os.path.join(output_folder, f"{file_base_name}_processed{file_ext}")

    print(f"输入文件: {input_video_path}")
    print(f"输出文件: {output_video_path}")

    # 开始处理视频
    process_video_file(input_video_path, output_video_path)