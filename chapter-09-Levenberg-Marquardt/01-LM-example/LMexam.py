# 该程序示范了如何使用Levenberg-Marquardt算法来求解一个非线性最小二乘问题。
# 该实例求解的最小二乘问题为, f(x,y)=1/2(f1(x,y)^2+f2(x,y)^2)，其中f1(x,y)=x^2+y-11, f2(x,y)=x+y^2-7。
# 该问题的理论最优解为（3，2），最优值为0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 定义目标函数：非线性最小二乘问题
def objective_functions(params):
    """返回非线性函数向量，目标是最小化这些函数的平方和"""
    x, y = params
    f1 = x**2 + y - 11
    f2 = x + y**2 - 7
    return np.array([f1, f2])

# 计算目标函数的平方和
def sum_of_squares(params):
    f = objective_functions(params)
    return np.sum(f**2)

# 2. 计算雅可比矩阵
def jacobian(params):
    x, y = params
    df1_dx = 2 * x
    df1_dy = 1
    df2_dx = 1
    df2_dy = 2 * y
    return np.array([[df1_dx, df1_dy], 
                    [df2_dx, df2_dy]])

# 3. 实现Levenberg-Marquardt算法
def levenberg_marquardt_simple(fun, jac, x0, max_iter=100, tol=1e-8, 
                              lambda_init=1e-3, lambda_min=1e-10, lambda_max=1e10):
    """采用简化版增益比策略的Levenberg-Marquardt算法"""
    x = x0.copy()
    lambda_ = lambda_init
    fx = fun(x)
    cost = 0.5*np.sum(fx**2)
    
    costs = [cost]
    path = [x.copy()]
    
    for i in range(max_iter):
        if cost < tol:
            break
            
        J = jac(x)
        g = J.T @ fx
        H = J.T @ J
        I = np.eye(len(x))
        dx = np.linalg.solve(H + lambda_ * I, -g)
        
        # 计算预测减少量和增益比
        fx_pred = fx + J @ dx
        predicted_reduction = np.sum(fx**2) - np.sum(fx_pred**2)
        x_new = x + dx
        fx_new = fun(x_new)
        cost_new = 0.5*np.sum(fx_new**2)
        actual_reduction = cost - cost_new
        
        rho = actual_reduction / predicted_reduction if predicted_reduction > 0 else 0
        
        # 调整阻尼因子
        if rho > 0:
            x = x_new
            fx = fx_new
            cost = cost_new
            costs.append(cost)
            path.append(x.copy())
            
            if rho < 0.25:
                lambda_ = lambda_ * 2
            elif rho > 0.75:
                lambda_ = lambda_ / 3
        else:
            lambda_ = lambda_ * 2
        
        lambda_ = max(lambda_min, min(lambda_max, lambda_))
    
    return x, costs, path

# 4. 运行算法
x0 = np.array([0.0, 0.0])
x_opt, costs, path = levenberg_marquardt_simple(
    objective_functions, 
    jacobian, 
    x0, 
    max_iter=50, 
    lambda_init=1e-2
)

# 5. 输出结果
print(f"初始点: x = {x0[0]}, y = {x0[1]}")
print(f"最优解: x = {x_opt[0]:.6f}, y = {x_opt[1]:.6f}")
print(f"目标函数最小值: {sum_of_squares(x_opt):.6e}")
print(f"迭代次数: {len(costs) - 1}")

# 6. 可视化目标函数值随迭代的变化
plt.figure(figsize=(10, 6))
plt.plot(costs, 'b-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值 (平方和)')
plt.title('Levenberg-Marquardt算法迭代过程中目标函数值的变化')
plt.grid(True)
plt.yscale('log')
plt.show()

# 7. 准备100×100网格数据
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)

# 计算网格上的目标函数值
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = 0.5*sum_of_squares([X[i, j], Y[i, j]])

# 8. 等高线可视化
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, np.log10(Z + 1e-10), levels=50, cmap='viridis')
plt.colorbar(contour, label='log10(目标函数值)')
plt.contour(X, Y, np.log10(Z + 1e-10), levels=10, colors='black', linewidths=0.5)

path_np = np.array(path)
plt.plot(path_np[:, 0], path_np[:, 1], 'ro-', markersize=5, linewidth=2, label='迭代路径')
plt.scatter([x0[0]], [x0[1]], color='green', s=100, marker='o', label='初始点')
plt.scatter([x_opt[0]], [x_opt[1]], color='red', s=100, marker='*', label='最优解')

plt.xlabel('x')
plt.ylabel('y')
plt.title('目标函数等高线与迭代路径')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# 9. 三维曲面可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, np.log10(Z + 1e-10), cmap='viridis', 
                       alpha=0.7, edgecolor='none', antialiased=True)
fig.colorbar(surf, ax=ax, label='log10(目标函数值)')

path_x = path_np[:, 0]
path_y = path_np[:, 1]
path_z = np.log10([0.5*sum_of_squares([x, y]) + 1e-10 for x, y in path])

ax.plot(path_x, path_y, path_z, 'ro-', markersize=6, linewidth=2, label='迭代路径')
ax.scatter([x0[0]], [x0[1]], [path_z[0]], color='green', s=200, marker='o', label='初始点')
ax.scatter([x_opt[0]], [x_opt[1]], [path_z[-1]], color='red', s=200, marker='*', label='最优解')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('log10(目标函数值)')
ax.set_title('目标函数三维曲面与迭代路径')
ax.legend()
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()

# 10. 生成并保存三维迭代动画
try:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update_3d(frame):
        ax.clear()
        ax.plot_surface(X, Y, np.log10(Z + 1e-10), cmap='viridis', 
                       alpha=0.7, edgecolor='none', antialiased=True)
        
        # 获取当前迭代点
        current_x = path_np[frame, 0]
        current_y = path_np[frame, 1]
        
        current_path_x = path_x[:frame+1]
        current_path_y = path_y[:frame+1]
        current_path_z = path_z[:frame+1]
        ax.plot(current_path_x, current_path_y, current_path_z, 'ro-', 
                markersize=6, linewidth=2, label='迭代路径')
        
        ax.scatter([x0[0]], [x0[1]], [path_z[0]], 
                  color='green', s=200, marker='o', label='初始点')
        ax.scatter([x_opt[0]], [x_opt[1]], [path_z[-1]], 
                  color='red', s=200, marker='*', label='最优解')
        
        # 标题中包含迭代点信息
        current_cost = costs[frame] if frame < len(costs) else costs[-1]
        ax.set_title(
            f'迭代过程 (迭代次数: {frame})\n'
            f'当前点: x={current_x:.4f}, y={current_y:.4f}\n'
            f'目标函数值: {current_cost:.2e}'
        )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('log10(目标函数值)')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        return ax,
    
    ani_3d = FuncAnimation(fig, update_3d, frames=len(path), interval=800, blit=False)
    ani_3d.save('levenberg_marquardt_3d_animation.gif', writer='pillow', fps=1.5)
    print("三维曲面动画已保存为 'levenberg_marquardt_3d_animation.gif'")
    plt.close()
    
except Exception as e:
    print(f"三维动画生成失败: {e}")
    print("请确保已安装pillow库（pip install pillow）")
