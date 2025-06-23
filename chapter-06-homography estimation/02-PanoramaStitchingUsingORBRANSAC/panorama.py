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
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def moore_penrose_pseudoinverse(matrix, tol=1e-10):
    """使用NumPy SVD实现Moore-Penrose广义逆"""
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # 构建奇异值的伪逆矩阵
    s_pinv = np.zeros_like(s)
    for i in range(len(s)):
        if s[i] > tol:
            s_pinv[i] = 1.0 / s[i]
    
    # 计算伪逆 A^+ = V·S^+·U^T
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
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        
        # 对应x方向的方程
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        b[2*i] = u
        
        # 对应y方向的方程
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
        b[2*i+1] = v
    
    # 使用手动实现的Moore-Penrose广义逆求解Ah=b
    A_pinv = moore_penrose_pseudoinverse(A)
    h = A_pinv @ b
    
    # 重塑为3x3矩阵，设置h33=1
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])
    
    # 归一化
    H = H / H[2, 2]
    return H

def ransac_homography(src_pts, dst_pts, max_iterations=1000, inlier_threshold=5.0):
    """使用RANSAC算法估计单应性矩阵"""
    best_H = None
    best_inliers = []
    max_inlier_count = 0
    
    for _ in range(max_iterations):
        # 随机选择4个点对
        indices = sample(range(len(src_pts)), 4)
        sample_src = [src_pts[i] for i in indices]
        sample_dst = [dst_pts[i] for i in indices]
        
        # 计算单应性矩阵
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
            error = np.sqrt((u - dst_pts[i][0])**2 + (v - dst_pts[i][1])**2)
            
            # 判断是否为内点
            if error < inlier_threshold:
                inliers.append(i)
        
        # 更新最佳模型
        if len(inliers) > max_inlier_count:
            max_inlier_count = len(inliers)
            best_inliers = inliers
            best_H = H
    
    # 使用所有内点重新估计H
    if best_inliers:
        final_src = [src_pts[i] for i in best_inliers]
        final_dst = [dst_pts[i] for i in best_inliers]
        best_H = compute_homography(final_src, final_dst)
    
    return best_H, best_inliers

def warp_and_stitch_images(img1, img2, H):
    """基于单应性矩阵H对图像进行变换并拼接"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 定义源图像的四个角点
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # 计算变换后的角点位置
    pts1_transformed = cv2.perspectiveTransform(pts1, H)
    
    # 合并两个图像的角点，计算新画布的尺寸
    pts = np.vstack((pts1_transformed, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    # 计算平移量，确保所有点都在新画布内
    t = [-x_min, -y_min]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # 平移矩阵
    
    # 对img1应用变换并拼接
    result = cv2.warpPerspective(img1, Ht @ H, (x_max - x_min, y_max - y_min))
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