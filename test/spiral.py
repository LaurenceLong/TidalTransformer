import numpy as np
import matplotlib.pyplot as plt


def polar_spiral(theta, a, b):
    """
    极坐标形式的螺线方程
    r = a * exp(b * theta)
    """
    return a * np.exp(b * theta)


def generate_spiral_points(a, b, num_points=1000, max_theta=8 * np.pi):
    thetas = np.linspace(0, max_theta, num_points)
    r = polar_spiral(thetas, a, b)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return x, y, r, thetas


def generate_dual_arm_spiral(a, b, num_points=1000, max_theta=8 * np.pi):
    x1, y1, r1, thetas1 = generate_spiral_points(a, b, num_points, max_theta)
    x2, y2, r2, thetas2 = generate_spiral_points(a, b, num_points, max_theta)
    # 第二个臂的相位差π
    x2, y2 = -x2, -y2
    return x1, y1, r1, thetas1, x2, y2, r2, thetas2


def plot_dual_arm_spiral(a, b, num_points=1000, max_theta=8 * np.pi):
    x1, y1, r1, thetas1, x2, y2, r2, thetas2 = generate_dual_arm_spiral(a, b, num_points, max_theta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 笛卡尔坐标系中的螺线
    ax1.plot(x1, y1, 'b-', label='Arm 1')
    ax1.plot(x2, y2, 'r-', label='Arm 2')
    ax1.set_title(f"Dual Arm Logarithmic Spiral (a={a}, b={b})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)

    # 极坐标系中的螺线
    ax2.plot(thetas1, r1, 'b-', label='Arm 1')
    ax2.plot(thetas2, r2, 'r-', label='Arm 2')
    ax2.set_title("Spiral in Polar Coordinates")
    ax2.set_xlabel('θ (radians)')
    ax2.set_ylabel('r')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spiral_3d(a, b, num_points=1000, max_theta=8 * np.pi):
    x1, y1, r1, thetas1, x2, y2, r2, thetas2 = generate_dual_arm_spiral(a, b, num_points, max_theta)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, thetas1, 'b-', label='Arm 1')
    ax.plot(x2, y2, thetas2, 'r-', label='Arm 2')

    ax.set_title(f"3D Visualization of Dual Arm Spiral (a={a}, b={b})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('θ (radians)')
    ax.legend()

    plt.show()


# 设置参数
a = 1  # 起始半径
b = 0.2  # 增长率

# 绘制双旋臂螺线（2D和极坐标）
plot_dual_arm_spiral(a, b)

# 3D可视化
plot_spiral_3d(a, b)

# 验证极坐标方程
theta_test = np.pi
r_test = polar_spiral(theta_test, a, b)
print(f"At θ = π, r = {r_test:.4f}")


# 计算螺线的一些特性
def spiral_properties(a, b, theta):
    r = polar_spiral(theta, a, b)
    arc_length = a / b * (np.sqrt(1 + b ** 2) * np.exp(b * theta) - 1)
    pitch_angle = np.arctan(b)
    return r, arc_length, np.degrees(pitch_angle)


theta = 2 * np.pi
r, arc_length, pitch_angle = spiral_properties(a, b, theta)
print(f"At θ = 2π:")
print(f"Radius: {r:.4f}")
print(f"Arc Length: {arc_length:.4f}")
print(f"Pitch Angle: {pitch_angle:.2f} degrees")