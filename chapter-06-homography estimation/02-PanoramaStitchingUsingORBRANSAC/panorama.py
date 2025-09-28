import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def read_images(img_path1, img_path2):
    """读取两张图像并转换为RGB格式"""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2

def detect_and_match_orb_features(img1, img2, nfeatures=1000):
    """检测ORB特征点并进行匹配"""
    # 为什么大部分cv操作都是在灰度图上进行?
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # 这里的特征点匹配方法？ 一般的双目深度视觉中，用的是何种特征点，何种匹配方法？
    # 关于特征点匹配方法，我还没有仔细研究过，比如张林说的汉明距离方法
    # 这里的crosscheck 是指双向验证
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def moore_penrose_pseudoinverse(matrix, tol=1e-10):
    """使用NumPy SVD实现Moore-Penrose广义逆"""
    # 这里的U V是都是正交矩阵, s是奇异值
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    print(f"U:{U} s:{s} Vt:{Vt}")
    # 构建奇异值的伪逆矩阵
    # 这里的是sigma+， 包含sigma 的转置以及取导数，这里的sigma是原矩阵的奇异值
    s_pinv = np.zeros_like(s)
    for i in range(len(s)):
        if s[i] > tol:
            s_pinv[i] = 1.0 / s[i]
    
    # 计算摩尔-彭罗斯伪逆 A^+ = V·S^+·U^T
    return Vt.T @ np.diag(s_pinv) @ U.T

def compute_homography(src_pts, dst_pts):
    """使用手动实现的Moore-Penrose广义逆求解单应性矩阵"""
    n = len(src_pts)
    if n < 4:
        raise ValueError("至少需要4个点来计算单应性矩阵")
    
    # 构建系数矩阵A和目标向量b
    A = np.zeros((2*n, 8))
    b = np.zeros(2*n)
    
    for i in range(n):
        # 取source 和目标点
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        
        # 对应x方向的方程
        # 构造系数矩阵 和 结果向量
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        b[2*i] = u
        
        # 对应y方向的方程
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
        b[2*i+1] = v
    
    # 使用手动实现的Moore-Penrose广义逆求解Ah=b
    # 摩尔彭罗斯广义逆
    A_pinv = moore_penrose_pseudoinverse(A)
    # 得到最终结果
    # 注意这里求解 并没有用到 线性齐次最小二乘和非齐次最小二乘
    h = A_pinv @ b
    
    # 重塑为3x3矩阵，设置h33=1
    # 这里注意最后一个元素固定为1， 这里选择h33是有原因的，因为在常规的变换矩阵中,h33 最不可能为0
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])
    
    # 归一化
    # 这里有必要吗?
    H = H / H[2, 2]
    return H

def ransac_homography(src_pts, dst_pts, max_iterations=1000, inlier_threshold=5.0):
    """使用RANSAC算法估计单应性矩阵
    经典RANSAC 算法框架
    1. 输入数据，这里是src_ptr dst_pts, 两者构成一个匹配点对
    2. 求解一个模型需要的最小的数据点对数量，这里是4个点对， 求解方法可以用解析法或者线性齐次最小二乘法
    3. 统计内点，相当于用已知的数据去投票（小于某个阈值）， 投票数量越多， 说明这个模型就越好
    """
    best_H = None
    best_inliers = []
    max_inlier_count = 0
    
    for _ in range(max_iterations):
        # 随机选择4个点对
        indices = sample(range(len(src_pts)), 4)
        # 随机选择匹配点对，这里的src 和 dst 每个索引应当直接对应
        sample_src = [src_pts[i] for i in indices]
        sample_dst = [dst_pts[i] for i in indices]
        
        # 计算单应性矩阵
        # 这里的是预先设定的模型
        H = compute_homography(sample_src, sample_dst)
        
        # 计算所有点的重投影误差
        inliers = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            # 齐次坐标
            pt = np.array([x, y, 1])
            # 应用单应性变换
            pt_transformed = H @ pt
            # 转换回非齐次坐标
            u, v = pt_transformed[0]/pt_transformed[2], pt_transformed[1]/pt_transformed[2]
            
            # 计算误差
            # 这里计算的是RPE
            error = np.sqrt((u - dst_pts[i][0])**2 + (v - dst_pts[i][1])**2)
            
            # 判断是否为内点
            if error < inlier_threshold:
                inliers.append(i)
        
        # 更新最佳模型
        if len(inliers) > max_inlier_count:
            max_inlier_count = len(inliers)
            best_inliers = inliers
            # 更新最佳变换矩阵
            best_H = H
    
    # 使用所有内点重新估计H
    if best_inliers:
        final_src = [src_pts[i] for i in best_inliers]
        final_dst = [dst_pts[i] for i in best_inliers]
        # 这里用一个大矩阵去更新最佳变换矩阵
        best_H = compute_homography(final_src, final_dst)
    # 最新的H，最新的内点
    return best_H, best_inliers

def warp_and_stitch_images(img1, img2, H):
    """基于单应性矩阵H对图像进行变换并拼接"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 定义源图像的四个角点
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # 计算变换后的角点位置
    # 注意这里应用的透视变换方法
    pts1_transformed = cv2.perspectiveTransform(pts1, H)
    
    # 合并两个图像的角点，计算新画布的尺寸
    # 这里其实是组合 图像的四个角点，分别是处理后的图像的四个角点，和当前图像的四个点
    # 合并成(8,1，2 ), 不过不理解这里的 min, max 是怎么去的某个轴的最小值，最大值的，这里的min,max 判断的谁?
    # 这里是似乎是指判断了x 方向，为什么不需要判断y方向呢? => 理解错误
    # 在一堆点对中，找到x 维度、y维度的最小值，最大值
    pts = np.vstack((pts1_transformed, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
    # 这里的ravel() 是指将多维数组展开成一维
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    # 计算平移量，确保所有点都在新画布内
    # 这里是找到最小坐标，然后将其移动到原点？ => yes it is
    t = [-x_min, -y_min]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # 平移矩阵
    
    # 对img1应用变换并拼接
    result = cv2.warpPerspective(img1, Ht @ H, (x_max - x_min, y_max - y_min))
    # 对图像进行拼接
    # 注意到图像已经做了位移，因此，起始位置也是从[y,x] = [t[1], t[0]] 开始的， 这里相当于直接赋值过去
    # 这里要保证result 足够大， 大的问题是在前面保证的
    # 注意，画布可能出现黑边，所以最终出来的东西要做裁切
    result[t[1]:h2 + t[1], t[0]:w2 + t[0]] = img2
    
    return result, Ht @ H

def visualize_results(img1, img2, kp1, kp2, matches, inliers, H, warped_img, stitched_img):
    """可视化所有中间结果和最终结果"""
    plt.figure(figsize=(16, 12))
    
    # 1. 显示原图
    plt.subplot(221)
    plt.title("原始图像1")
    plt.imshow(img1)
    
    plt.subplot(222)
    plt.title("原始图像2")
    plt.imshow(img2)
    
    # 2. 显示特征点检测结果
    plt.subplot(223)
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    plt.title(f"图像1的ORB特征点 ({len(kp1)}个)")
    plt.imshow(img1_kp)
    
    plt.subplot(224)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    plt.title(f"图像2的ORB特征点 ({len(kp2)}个)")
    plt.imshow(img2_kp)
    
    # 3. 显示特征点匹配结果（只显示内点）
    plt.figure(figsize=(16, 8))
    inlier_matches = [matches[i] for i in inliers]
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, 
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.title(f"特征点匹配结果（内点数量: {len(inliers)}/{len(matches)}）")
    plt.imshow(match_img)
    
    # 4. 显示单应性变换结果
    plt.figure(figsize=(10, 8))
    plt.title("图像1经过单应性变换后的结果")
    plt.imshow(warped_img)
    
    # 5. 显示最终拼接结果
    plt.figure(figsize=(12, 8))
    plt.title("全景拼接结果")
    plt.imshow(stitched_img)
    
    plt.tight_layout()
    plt.show()

def main():
    # 读取图像（请替换为你的图像路径）
    img1_path = "image1.bmp"
    img2_path = "image2.bmp"
    
    # 步骤1: 读取图像
    img1, img2 = read_images(img1_path, img2_path)
    
    # 步骤2: 检测并匹配ORB特征点
    kp1, kp2, matches = detect_and_match_orb_features(img1, img2)
    
    # 步骤3: 手动实现RANSAC估计单应性矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # 手动实现RANSAC和Moore-Penrose求解
    H, inliers = ransac_homography(src_pts, dst_pts, max_iterations=1000, inlier_threshold=5.0)
    
    # 步骤4: 图像变换与拼接
    stitched_img, full_H = warp_and_stitch_images(img1, img2, H)
    
    # 计算并显示第一张图像的变换结果
    h, w = img1.shape[:2]
    warped_img = cv2.warpPerspective(img1, full_H, (stitched_img.shape[1], stitched_img.shape[0]))
    
    # 可视化结果
    visualize_results(img1, img2, kp1, kp2, matches, inliers, H, warped_img, stitched_img)
    
    # 打印单应性矩阵
    print("完全手动实现的单应性矩阵 H:")
    print(H)

if __name__ == "__main__":
    main()