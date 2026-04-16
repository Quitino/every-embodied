import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def dh_matrix(theta, d, a, alpha):
    """
    根据标准DH参数计算变换矩阵 T_{i-1, i}
    单位: 角度使用弧度, 长度使用米
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def forward_kinematics_2link(joint_angles, link_lengths):
    """
    计算2连杆平面机械臂的末端位置
    """
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths
    
    # 连杆 1: frame 0 -> frame 1
    # theta=q1, d=0, a=L1, alpha=0
    T_01 = dh_matrix(theta1, 0, L1, 0)
    
    # 连杆 2: frame 1 -> frame 2
    # theta=q2, d=0, a=L2, alpha=0
    T_12 = dh_matrix(theta2, 0, L2, 0)
    
    # 总变换矩阵
    T_02 = T_01 @ T_12
    
    return T_02

def draw_frame_3d(ax, T, scale=0.2, label=None):
    """在 3D 坐标系 T 的位置画 X(红)、Y(绿)、Z(蓝) 轴箭头"""
    o = T[:3, 3]
    x = T[:3, 0]
    y = T[:3, 1]
    z = T[:3, 2]
    for vec, color in [(x, 'red'), (y, 'green'), (z, 'blue')]:
        ax.quiver(o[0], o[1], o[2],
                  vec[0], vec[1], vec[2],
                  length=scale, normalize=True,
                  color=color, linewidth=1.5, arrow_length_ratio=0.3)
    if label:
        ax.text(o[0] - 0.05, o[1] - 0.05, o[2] - 0.15,
                label, fontsize=9, color='black')

def draw_angle_arc_3d(ax, origin, theta, r=0.25, color='purple'):
    """在 XY 平面内画从0到theta的圆弧，表示关节角"""
    if abs(theta) < 1e-6:
        return
    t = np.linspace(0, theta, 30)
    xs = origin[0] + r * np.cos(t)
    ys = origin[1] + r * np.sin(t)
    zs = np.full_like(t, origin[2])
    ax.plot(xs, ys, zs, color=color, lw=1.5, ls='--')
    # 角度数值标注
    mid = theta / 2
    ax.text(origin[0] + (r + 0.08) * np.cos(mid),
            origin[1] + (r + 0.08) * np.sin(mid),
            origin[2] + 0.05,
            f"{np.degrees(theta):.0f}°", fontsize=8, color=color)

def plot_arm_3d(ax, joint_angles, link_lengths, title):
    """绘制单个姿态下的三维连杆结构与坐标系，含DH参数标注"""
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths

    T_01 = dh_matrix(theta1, 0, L1, 0)
    T_12 = dh_matrix(theta2, 0, L2, 0)
    T_02 = T_01 @ T_12

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = T_01[:3, 3]
    p2 = T_02[:3, 3]

    # 画连杆
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            'o-', color='steelblue', lw=4, ms=8, zorder=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            'o-', color='darkorange', lw=4, ms=8, zorder=2)

    # 末端效应器
    ax.scatter(*p2, s=120, color='red', marker='*', zorder=3)
    ax.text(p2[0] + 0.05, p2[1] + 0.05, p2[2] + 0.12,
            f"({p2[0]:.1f},{p2[1]:.1f},{p2[2]:.1f})", fontsize=7.5, color='red')

    # --- DH 参数标注 ---
    # θ1: joint 0 处的关节角弧线（从X轴转到连杆1方向）
    draw_angle_arc_3d(ax, p0, theta1, r=0.28, color='purple')
    if abs(theta1) < 1e-6:
        ax.text(p0[0] + 0.32, p0[1] + 0.04, p0[2] + 0.05,
                "θ1=0°", fontsize=8, color='purple')

    # θ2: joint 1 处的关节角弧线（相对连杆1方向）
    # 在 frame{1} 局部坐标下画弧，需把原点旋转到 p1
    # 直接在 frame1 的 XY 面内画弧
    R1 = T_01[:3, :3]
    t_arc = np.linspace(0, theta2, 30)
    xs2 = p1[0] + 0.28 * (R1[0, 0] * np.cos(t_arc) + R1[0, 1] * np.sin(t_arc))
    ys2 = p1[1] + 0.28 * (R1[1, 0] * np.cos(t_arc) + R1[1, 1] * np.sin(t_arc))
    zs2 = p1[2] + 0.28 * (R1[2, 0] * np.cos(t_arc) + R1[2, 1] * np.sin(t_arc))
    if abs(theta2) > 1e-6:
        ax.plot(xs2, ys2, zs2, color='darkorchid', lw=1.5, ls='--')
        mid2 = theta2 / 2
        mx = p1[0] + 0.38 * (R1[0, 0] * np.cos(mid2) + R1[0, 1] * np.sin(mid2))
        my = p1[1] + 0.38 * (R1[1, 0] * np.cos(mid2) + R1[1, 1] * np.sin(mid2))
        mz = p1[2] + 0.38 * (R1[2, 0] * np.cos(mid2) + R1[2, 1] * np.sin(mid2))
        ax.text(mx, my, mz + 0.05, f"{np.degrees(theta2):.0f}°",
                fontsize=8, color='darkorchid')
    else:
        ax.text(p1[0] + 0.32, p1[1] + 0.04, p1[2] + 0.05,
                "θ2=0°", fontsize=8, color='darkorchid')

    # a1: 连杆1长度标注（沿连杆中点偏上）
    mid1 = (p0 + p1) / 2
    ax.text(mid1[0] - 0.1, mid1[1] + 0.05, mid1[2] + 0.18,
            f"a1={L1:.0f}m", fontsize=8, color='steelblue', fontweight='bold')

    # a2: 连杆2长度标注
    mid2_pt = (p1 + p2) / 2
    ax.text(mid2_pt[0] - 0.1, mid2_pt[1] + 0.05, mid2_pt[2] + 0.18,
            f"a2={L2:.0f}m", fontsize=8, color='darkorange', fontweight='bold')

    # d=0, α=0 标注（在底部）
    ax.text(0.0, 0.0, -0.42, "d=0, α=0 (planar)", fontsize=7, color='gray')

    # 坐标系
    draw_frame_3d(ax, np.eye(4),  label="{0}")
    draw_frame_3d(ax, T_01,       label="{1}")
    draw_frame_3d(ax, T_02,       label="{2}")

    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-0.3, 2.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)


# --- 验证 & 三维可视化 ---
L_params = [1.0, 1.0]

cases = [
    ([0,              0             ], "case1: q1=0°, q2=0°\nend→(2, 0, 0)"),
    ([np.radians(90), 0             ], "case2: q1=90°, q2=0°\nend→(0, 2, 0)"),
    ([0,              np.radians(90)], "case3: q1=0°, q2=90°\nend→(1, 1, 0)"),
]

fig = plt.figure(figsize=(15, 5))
fig.suptitle("2-link Robot Arm — DH Forward Kinematics (3D)", fontsize=13, fontweight='bold')

for i, (angles, title) in enumerate(cases, 1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    plot_arm_3d(ax, angles, L_params, title)
    T = forward_kinematics_2link(angles, L_params)
    pos = T[:3, 3]
    print(f"{title.split(chr(10))[0]}  end_point: {np.round(pos, 4)}")

# 图例
red_p   = mpatches.Patch(color='red',        label='X axis')
green_p = mpatches.Patch(color='green',      label='Y axis')
blue_p  = mpatches.Patch(color='blue',       label='Z axis')
l1_p    = mpatches.Patch(color='steelblue',  label='Link 1')
l2_p    = mpatches.Patch(color='darkorange', label='Link 2')
fig.legend(handles=[red_p, green_p, blue_p, l1_p, l2_p],
           loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("dh_cases_3d.png", dpi=150, bbox_inches='tight')
print("\n图像已保存为 dh_cases_3d.png")
plt.show()