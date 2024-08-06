import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import random

def get_circle_center(p1, p2, p3):
    """
    计算三个点的外接圆圆心
    """
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1e-6:
      return None
    
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    return int(cx), int(cy)

def is_circle(contour,centers):
    """
    检查圆心分布是否接近，判断是否为圆形
    """
    # 将列表转换为 NumPy 数组
    centers = np.array(centers)
    # 计算均值中心
    mean_center = np.mean(centers, axis=0)
    # 计算每个点到均值中心的距离
    distances = np.linalg.norm(centers - mean_center, axis=1)
    # 找到距离最大的100个点的索引
    indices_to_remove = np.argpartition(distances, -100)[-100:]
    # 创建一个布尔掩码，标记出需要保留的点
    mask = np.ones(centers.shape[0], dtype=bool)
    mask[indices_to_remove] = False
    # 使用掩码创建一个新的数组，不包含最大的100个值
    filtered_centers = centers[mask]

    # 计算 去除散点的中心
    centers = np.array(filtered_centers)
    mean_center = np.mean(centers, axis=0)

    distances = (np.linalg.norm(centers - mean_center, axis=1))


    # 计算半径
    Rs = (np.linalg.norm(contour - mean_center, axis=1))
    R = np.mean(Rs)
    threshold = 0.4*R
    print('div', np.mean(distances) , 'R',R ,'threshold',threshold )
    return (np.mean(distances) < threshold ) , mean_center ,R










# 读取示例图像
frame = cv2.imread('ball.png')  # 替换为你自己的图片路径

# 将图像从 BGR 转换为 HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 定义网球的颜色范围（根据网球的实际颜色调整）
lower_green = np.array([29, 86, 6])
upper_green = np.array([64, 255, 255])

# 创建遮罩，只保留在颜色范围内的部分
mask = cv2.inRange(hsv, lower_green, upper_green)

# 使用形态学操作去除噪声
mask = cv2.erode(mask, None, iterations=4)
mask = cv2.dilate(mask, None, iterations=2)
dilated_mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=4)
cv2_imshow(dilated_mask)


# 使用Canny边缘检测
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 5)
edges = cv2.Canny(blurred, 50, 150)
cv2_imshow(edges)

# 对 edges 进行膨胀操作
edges_dilated = cv2.dilate(edges, None, iterations=1)

# 寻找轮廓
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [contour for contour in contours if len(contour) > 50]

# 为每个轮廓生成一个随机颜色
colors = [tuple(random.choices(range(256), k=3)) for _ in range(len(filtered_contours))]
# 创建一个黑底图像
black_image = np.zeros_like(frame)
# 绘制每个过滤后的轮廓在黑底图上
for contour, color in zip(filtered_contours, colors):
    cv2.drawContours(black_image, [contour], -1, color, 2)
cv2_imshow(black_image)


# 找颜色和轮廓重合部分
# 创建一个空白掩码图像
contour_mask = np.zeros_like(dilated_mask)
# 在掩码图像上绘制 filtered_contours
for contour in filtered_contours:
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
# 计算重叠部分
intersection = cv2.bitwise_and(dilated_mask, contour_mask)
# 找到重叠部分的轮廓
intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 保留重叠的轮廓黑底图上
result_image = np.zeros_like(frame)
cv2.drawContours(result_image, intersection_contours, -1, (0, 255, 0), 2)
cv2_imshow(result_image)


valid_circles = []
mean_centers = []
Radius = []
# 验证每个轮廓是否为圆形
for contour in filtered_contours:
    centers = []
    for _ in range(500):
        idx1, idx2, idx3 = np.random.choice(len(contour), 3, replace=False)
        p1, p2, p3 = contour[idx1][0], contour[idx2][0], contour[idx3][0]
        center = get_circle_center(p1, p2, p3)

        if center:
            centers.append(center)
            cv2.circle(frame, center, 1, (len(contour), len(contour), len(contour)), 2)
    check,m_center,R = is_circle(contour,centers)
    if check: 
        mean_centers.append(m_center)
        valid_circles.append(contour)
        Radius.append(R)

print('m_center', m_center)
for i in range(len(valid_circles)):
    # 将识别出的圆形轮廓标记在原图上
    contour = valid_circles[i]
    #cv2.drawContours(frame, [contour], -1, (0, 255, 0), 4)

    
# 以圆心，半径，画出框
for i in range(len(Radius)):
    Radius = np.array(Radius)
    max_R = np.max(Radius)

    if Radius[i] > 0.5 * max_R and max_R > 10:
      R_int = int(1.41*Radius[i])
      center = mean_centers[i]
      x, y = map(int, map(round, center))
      cv2.circle(frame, (x,y), 1, (len(contour), len(contour), len(contour)), 5)
      cv2.rectangle(frame, (x-R_int, y-R_int), (x + R_int, y + R_int), (255, 0, 0), 2)


# 保存结果图像
result_filename = 'detected_tennis_ball.jpg'
cv2.imwrite(result_filename, frame)

# 显示结果图像
cv2_imshow(frame)
