import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
"""自己实现的简单的ORB 角点检测
不知道为什么，这个效果不是很好，大部分点都落到边界上而不是角点位置了

Returns:
    _type_: _description_
"""
class SimpleORB:
  def __init__(self, n_features=1000, scale_factor=1.2, n_levels=8, fast_thresh=0.2, patch_size=31, edge_thresh=31):
    """_summary_

    Args:
        n_features (int, optional): _description_. Defaults to 1000.
        scale_factor (float, optional): 金字塔缩放因子. Defaults to 1.2.
        n_levels (int, optional): 金字塔层数. Defaults to 8.
        fast_thresh (float, optional): fast 检测阈值, 按照书上所说一般0.2. Defaults to 0.2.
        patch_size (int, optional): 领域大小. Defaults to 31.
        edge_thresh (int, optional): _description_. Defaults to 31.
    """
    self.n_features = n_features
    self.scale_factor = scale_factor
    self.n_levels = n_levels
    self.fast_thresh = fast_thresh
    self.patch_size = patch_size
    self.edge_thresh = edge_thresh
    self.pattern = self._generate_brief_pattern()
  
  def _generate_brief_pattern(self, size=256):
    """生成Brief描述子的随机模式（随机点对）

    Args:
        size (int, optional): 随机点对数量. Defaults to 256.

    Returns:
        np.array: 随机点对数组
    """
    np.random.seed(28)
    pattern = []
    window_radius = self.patch_size // 2
    for i in range(size):
      x1, y1 = np.random.randint(-window_radius, window_radius, size=2)
      x2, y2 = np.random.randint(-window_radius, window_radius, size=2)
      pattern.append((x1, y1, x2, y2))
    return np.array(pattern)
    
  def _create_scale_pyamid(self, image:np.ndarray):
    """创建图像金字塔

    Args:
        image (_type_): _description_
    """
    pyramid=[image]
    cur_image = image
    for i in range(1, self.n_levels):
      # 当前图像尺寸继续下降
      width = int(cur_image.shape[1] / self.scale_factor)
      height = int(cur_image.shape[0] / self.scale_factor)
      # cv 下采样
      resized = cv2.resize(cur_image, (width, height))
      
      pyramid.append(resized)
      cur_image = resized
    return pyramid
  
  def _fast_detect(self, image:np.ndarray, thresh = 0.2) -> list[cv2.KeyPoint]:
    """FAST 特征点检测

    Args:
        sef (_type_): _description_
        image (np.ndarray): 输入图像
        thresh (float, optional): 差分阈值. Defaults to 0.2.

    Returns:
        list: 特征点列表
    """
    h, w = image.shape
    # 上下左右 斜对角 等 共计16个点
    circle_offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]
    # note: 边界控制，btw, 需要考虑边界上的点吗？ 我觉得不需要， 到边界了很容易丢失
    border = 3
    fast_idx = [0,5,9,13]
    key_points=[]
    for y in range(border, h-border):
      for x in range(border, w - border):
        pix = image[y, x]
        count_darker = 0
        count_brighter = 0
        thresh_value_low_bound = pix * (1-thresh)
        thresh_value_high_bound = pix* (1+thresh)
        # 快速检测
        for idx in fast_idx:
          nebor = circle_offsets[idx]
          neighbor_pix = image[y+nebor[1], x+nebor[0]]
          if neighbor_pix < thresh_value_low_bound:
            count_darker += 1
          elif neighbor_pix > thresh_value_high_bound:
            count_brighter += 1
        if count_darker < 3 and count_brighter < 3:
          continue
        
        for idx  in range(0, len(circle_offsets)):
          if idx in fast_idx:
            continue
          dx, dy = circle_offsets[idx]
          # 取领域点
          neighbor_pix = image[y+dy, x+dx]
          # TODO 这里有快速算法， 先判断 1 5 9 13 这三个点的值是否都大于high_bound或者小于low_bound
          if neighbor_pix < thresh_value_low_bound:
            count_darker += 1
          elif neighbor_pix > thresh_value_high_bound:
            count_brighter += 1
        if count_brighter < 12 and count_darker < 12:
          continue
        response = 0
        # response 是: SAD
        for dx, dy in circle_offsets:
          response += abs(int(image[y + dy, x + dx]) - int(pix))
        key_points.append(cv2.KeyPoint(x, y, 1, response=response))
    return key_points
  
  def _compute_orientation(self, image:np.ndarray, keypoint:cv2.KeyPoint):
    x, y = keypoint.pt[0], keypoint.pt[1]
    x,y= int(x), int(y)
    #  patch size 是窗口大小
    raidus = self.patch_size // 2
    if  x - raidus < 0 or x + raidus >= image.shape[1] or y - raidus < 0 or y + raidus >= image.shape[0]:
      return 0
    # 图像的一阶矩
    m01, m10 = 0, 0 
    # 书上定义的主方向是质心位置， 用质心位置和中心位置定义了主方向
    for i in range(-raidus, raidus + 1):
      for j in range(-raidus, raidus + 1):
        pix_v = int(image[y + i, x+j])
        m10 += j* pix_v # y 方向一阶矩
        m01 + i * pix_v # x 方向一阶矩
    # TODO: 这里的简化是如何做的呢？
    orientation = np.arctan2(m01, m10)
    return np.degrees(orientation)

  def _compute_brief_descriptor(self, image:np.ndarray, key_point:cv2.KeyPoint, oritension):
    x,y = key_point.pt
    theta = np.radians(oritension)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    descriptor = 0
    desc_bits = []
    # 产生了点对相关随机匹配模式
    for i, (x1, y1, x2, y2) in enumerate(self.pattern):
      # do rotation in advance
      # 其实可以写成一个矩阵乘法....
      x1_rot = x1 * cos_t - y1 * sin_t
      y1_rot = x1 * sin_t + y1 * cos_t
      x2_rot = x2 * cos_t - y2 * sin_t
      y2_rot = x2 * sin_t + y2 * cos_t
      px1, py1 = int(round(x + x1_rot)), int(round(y + y1_rot))
      px2, py2 = int(round(x + x2_rot)), int(round(y + y2_rot))
      if (px1 < 0 or px1 >=  image.shape[1] or py1 < 0 or py1 >= image.shape[0]) or  (px2 < 0 or px2 >=  image.shape[1] or py2 < 0 or py2 >= image.shape[0]):
        desc_bits.append(0)
        continue
      if image[py1, px1] < image[py2, px2]:
        # 二进制描述子
        descriptor |= (1 << i)
        desc_bits.append(1)
      else:
        desc_bits.append(0)
    return descriptor, desc_bits
  
  def detect_and_compute(self, image:np.ndarray):
    if len(image.shape)  == 3:
      gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()
    pyramid = self._create_scale_pyamid(gray)
    all_key_points = []
    all_descriptors = []
    for level, level_img in enumerate(pyramid):
      print(f"Processing pyramid @level {level}, size:{level_img.shape}")
    keypoints = self._fast_detect(level_img, self.fast_thresh)
    for kp in keypoints:
      # note 这里要care
      scale = self.scale_factor ** level # 映射到原始图像坐标
      kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
      # 原始尺寸图像下检测主方向
      orientation = self._compute_orientation(gray, kp)
      # 提前对各点旋转，然后计算随机匹配点对之间的相对关系，用相对关系建立描述子
      descriptor, _ = self._compute_brief_descriptor(gray, kp, orientation)
      all_key_points.append(kp)
      all_descriptors.append(descriptor)
    if not all_key_points:
      return [], None
    responses = [kp.response for kp in all_key_points]
    # 这里就是单纯的响应值还是包含了索引?
    indcies = np.argsort(responses)[::-1] # 降序排列
    n_selected = min(self.n_features, len(indcies))
    selected_indecies = indcies[:n_selected]
    final_keypoints = [all_key_points[i] for i in selected_indecies]
    final_descriptors = [all_descriptors[i] for i in selected_indecies]
    # ??? 这里的每个descriptor 都是一个bin, 这里转成数组没看懂?
    # descripor_array = np.array(final_descriptors, dtype=np.int32).reshape(-1, 32)
    return final_keypoints, None
  

def visualize_orb(image, keypoints:list[cv2.KeyPoint], descripotrs=None):
  img_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
  for kp in keypoints:
    x,y = kp.pt
    angle = kp.angle
    size =kp.size
    cv2.circle(img_disp, (int(x), int(y)), int(size), (0, 255, 0), 1)
    angle_rad = np.radians(angle)
    end_x = int(x + size * math.cos(angle_rad))
    end_y = int(y + size * math.sin(angle_rad))
    cv2.line(img_disp, (int(x), int(y)), (end_x, end_y), (0, 255, 0), 1)
    cv2.circle(img_disp, (end_x, end_y), 2, (0, 255, 0), -1)
  
  plt.figure(figsize=(12,8))
  plt.imshow(img_disp)
  plt.axis('off')
  plt.tight_layout()
  plt.show()
  
if __name__ == "__main__":
  img1 = cv2.imread("/mnt/data/a.png")
  orb = SimpleORB(300)
  kps, descs = orb.detect_and_compute(img1)
  visualize_orb(img1, kps)