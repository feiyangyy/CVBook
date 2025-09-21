import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detector(im, sigma=4, thresh=0.3, nonmaxrad=5):
    """
    Harris角点检测函数
    :param im: 灰度图像 (numpy array)
    :param sigma: 高斯窗口标准差 (6*sigma决定窗口大小)
    :param thresh: 角点阈值
    :param nonmaxrad: 非极大值抑制窗口半径
    :return: rows, cols: 角点坐标
    """
    # 转 float32
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    
    # 1. 计算图像梯度 
    # 这里使用Sobel算子计算图像的梯度， 用于后续的Harris角点检测
    Ix = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. 构造矩阵 M 的分量并用高斯平滑
    ksize = int(np.ceil(6*sigma))
    if ksize % 2 == 0:
        ksize += 1  # 保证是奇数
    Ix2 = cv2.GaussianBlur(Ix*Ix, (ksize, ksize), sigma)
    Iy2 = cv2.GaussianBlur(Iy*Iy, (ksize, ksize), sigma)
    Ixy = cv2.GaussianBlur(Ix*Iy, (ksize, ksize), sigma)
    
    # 3. Harris 角点响应
    k = 0.04
    R = (Ix2 * Iy2 - Ixy**2) - k * (Ix2 + Iy2)**2
    
    # 4. 归一化
    # 这里的归一化，可能是不必要的
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    
    # 5. 阈值化 形成bool 数组
    mask = R_norm > thresh
    
    # 6. 非极大值抑制 (NMS)
    # 膨胀操作找到局部最大值
    # 这里的dilate 就是形态学中的膨胀操作，用于扩大白色区域（值大的区域，相当于放到物体），灰度图中，膨胀操作相当于在
    # 邻域内取最大值， 这里的 np.ones((2*nonmaxrad+1, 2*nonmaxrad+1) 用于定义邻域大小， 他会把邻域内每个像素替换为领域最大值
    # 这里假设有两个点靠的太近，较小的点就会被抑制掉
    R_dilated = cv2.dilate(R_norm, np.ones((2*nonmaxrad+1, 2*nonmaxrad+1), np.uint8))
    # 这里检测，那些点是邻域最大值，符合条件的点，就是角点，从而达到NMS的操作
    local_max = (R_norm == R_dilated)
    
    # 角点 = 阈值 + 局部最大
    # 这里要过滤阈值是因为要去掉那些thresh 较小的角点， 以控制次优
    # 这里是按位与，用于抑制false位置
    corners = mask & local_max
    print(corners.shape)
    # 这里找到角点的坐标，默认取非0的点
    ys, xs = np.where(corners)  # ys 对应行，xs 对应列
    rows, cols = ys, xs         # 如果你想用 rows/cols 命名
    return rows, cols
  
  
# 1. 读取图像并转灰度
img = cv2.imread("officegray.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

rows, cols = harris_corner_detector(gray)

# 在图上画角点
img_corners = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for x, y in zip(cols, rows):
    cv2.circle(img, (x,y), 3, (0,0,255), 1)

plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners with NMS")
plt.show()
