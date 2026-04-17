"""
单摆拉格朗日动力学演示
=====================
阶段1：SymPy 符号推导 E-L 方程
阶段2：数值积分（ODE）模拟运动
阶段3：四格动态演示图
  ① 单摆动画          ② 角度/角速度时间曲线
  ③ 能量守恒验证      ④ 相平面轨迹
"""

# ─────────────────────────────────────────────
# 阶段 1：符号推导
# ─────────────────────────────────────────────
import sympy as sp

t_sym = sp.symbols('t')
m_sym, l_sym, g_sym = sp.symbols('m l g', real=True, positive=True)
theta_sym  = sp.Function('theta')(t_sym)
dtheta_sym = theta_sym.diff(t_sym)
ddtheta    = dtheta_sym.diff(t_sym)

K_sym = sp.Rational(1, 2) * m_sym * (l_sym * dtheta_sym)**2
P_sym = -m_sym * g_sym * l_sym * sp.cos(theta_sym)
L_sym = K_sym - P_sym

term1 = sp.diff(sp.diff(L_sym, dtheta_sym), t_sym)
term2 = sp.diff(L_sym, theta_sym)
eq    = sp.simplify(term1 - term2)

print("=" * 50)
print("【阶段1】符号推导结果 (tau=0，自由振荡)")
print("=" * 50)
sp.pprint(eq, use_unicode=True)

# 提取 M（theta_ddot 系数）和 G（其余项）
eq_expanded = sp.expand(eq)
M_coeff = eq_expanded.coeff(ddtheta)
G_expr  = sp.simplify(eq_expanded - M_coeff * ddtheta)
print(f"\nM (惯性项系数) = {M_coeff}")
print(f"G (重力项)     = {G_expr}")
print(f"\n物理解读：({M_coeff})·θ̈  +  ({G_expr})  =  τ")

# 将方程整理为 θ̈ = f(θ, θ̇)，供数值积分使用
#   m*l²·θ̈ + m*g*l·sin(θ) = 0  →  θ̈ = -(g/l)·sin(θ)
ddtheta_sol = sp.solve(eq, ddtheta)[0]
print(f"\n状态方程：θ̈ = {ddtheta_sol}")

# ─────────────────────────────────────────────
# 阶段 2：数值积分
# ─────────────────────────────────────────────
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# 物理参数
m_val, l_val, g_val = 1.0, 1.0, 9.81

def pendulum_ode(t, state):
    """
    state = [θ, θ̇]
    从符号推导结果直接来：θ̈ = -(g/l)·sin(θ)
    """
    theta_v, dtheta_v = state
    ddtheta_v = -(g_val / l_val) * np.sin(theta_v)
    return [dtheta_v, ddtheta_v]

# 三组初始条件（小角、中角、大角）
cases = [
    {"theta0": np.radians(15),  "dtheta0": 0.0, "label": "小角 15°",  "color": "steelblue"},
    {"theta0": np.radians(60),  "dtheta0": 0.0, "label": "中角 60°",  "color": "darkorange"},
    {"theta0": np.radians(170), "dtheta0": 0.0, "label": "大角 170°", "color": "crimson"},
]

T_end, dt = 6.0, 0.02
t_eval = np.arange(0, T_end, dt)

results = []
for c in cases:
    sol = solve_ivp(pendulum_ode,
                    [0, T_end],
                    [c["theta0"], c["dtheta0"]],
                    t_eval=t_eval,
                    method="RK45",
                    rtol=1e-9, atol=1e-12)
    theta_arr  = sol.y[0]
    dtheta_arr = sol.y[1]
    K_arr = 0.5 * m_val * (l_val * dtheta_arr)**2
    P_arr = -m_val * g_val * l_val * np.cos(theta_arr)
    E_arr = K_arr + P_arr
    results.append({**c,
                    "t": sol.t,
                    "theta": theta_arr,
                    "dtheta": dtheta_arr,
                    "K": K_arr, "P": P_arr, "E": E_arr})

# ─────────────────────────────────────────────
# 阶段 3：四格静态总览图（先保存静态图）
# ─────────────────────────────────────────────
fig_static = plt.figure(figsize=(14, 10))
fig_static.suptitle("Single Pendulum — Lagrangian Dynamics",
                     fontsize=14, fontweight="bold")
gs = GridSpec(2, 2, figure=fig_static, hspace=0.38, wspace=0.32)

ax_anim  = fig_static.add_subplot(gs[0, 0])   # ① 静态末帧位置示意
ax_theta = fig_static.add_subplot(gs[0, 1])   # ② θ(t)
ax_energy= fig_static.add_subplot(gs[1, 0])   # ③ 能量
ax_phase = fig_static.add_subplot(gs[1, 1])   # ④ 相平面

# ① 摆末帧位置（三条摆杆）
ax_anim.set_title("Pendulum Positions (t=end)", fontsize=10)
ax_anim.set_xlim(-1.3, 1.3); ax_anim.set_ylim(-1.3, 0.5)
ax_anim.set_aspect("equal"); ax_anim.axhline(0, color="gray", lw=0.5)
ax_anim.plot(0, 0, "k^", ms=10, zorder=5)
for r in results:
    xb = l_val * np.sin(r["theta"][-1])
    yb = -l_val * np.cos(r["theta"][-1])
    ax_anim.plot([0, xb], [0, yb], "o-", color=r["color"],
                 lw=2.5, ms=9, label=r["label"])
ax_anim.legend(fontsize=8); ax_anim.set_xlabel("x (m)"); ax_anim.set_ylabel("y (m)")

# ② θ(t)
ax_theta.set_title("θ(t)  vs  Time", fontsize=10)
for r in results:
    ax_theta.plot(r["t"], np.degrees(r["theta"]),
                  color=r["color"], lw=1.5, label=r["label"])
ax_theta.set_xlabel("t (s)"); ax_theta.set_ylabel("θ (deg)")
ax_theta.legend(fontsize=8); ax_theta.grid(True, alpha=0.3)
ax_theta.axhline(0, color="gray", lw=0.5)

# ③ 能量守恒验证
ax_energy.set_title("Energy Conservation  K + P = E", fontsize=10)
for r in results:
    ax_energy.plot(r["t"], r["K"], "--", color=r["color"], lw=1, alpha=0.7,
                   label=f"K ({r['label']})")
    ax_energy.plot(r["t"], r["P"], ":",  color=r["color"], lw=1, alpha=0.7)
    ax_energy.plot(r["t"], r["E"], "-",  color=r["color"], lw=2,
                   label=f"E ({r['label']})")
ax_energy.set_xlabel("t (s)"); ax_energy.set_ylabel("Energy (J)")
ax_energy.legend(fontsize=7, ncol=2); ax_energy.grid(True, alpha=0.3)

# ④ 相平面 (θ, θ̇)
ax_phase.set_title("Phase Portrait  (θ, θ̇)", fontsize=10)
for r in results:
    ax_phase.plot(np.degrees(r["theta"]), np.degrees(r["dtheta"]),
                  color=r["color"], lw=1.5, label=r["label"])
    ax_phase.plot(np.degrees(r["theta"][0]), np.degrees(r["dtheta"][0]),
                  "o", color=r["color"], ms=7)
ax_phase.set_xlabel("θ (deg)"); ax_phase.set_ylabel("θ̇ (deg/s)")
ax_phase.legend(fontsize=8); ax_phase.grid(True, alpha=0.3)
ax_phase.axhline(0, color="gray", lw=0.5); ax_phase.axvline(0, color="gray", lw=0.5)

plt.savefig("pendulum_static.png", dpi=150, bbox_inches="tight")
print("\n静态总览图已保存为 pendulum_static.png")

# ─────────────────────────────────────────────
# 阶段 4：单摆动画（只演示中角60°）
# ─────────────────────────────────────────────
r_anim = results[1]   # 60° 那组

fig_anim, axes = plt.subplots(1, 3, figsize=(13, 4))
fig_anim.suptitle("Pendulum Animation — 60° Initial Angle", fontsize=12, fontweight="bold")

ax_p, ax_t, ax_e = axes
ax_p.set_title("Pendulum")
ax_t.set_title("θ(t)  &  θ̇(t)")
ax_e.set_title("Energy  K / P / E")

# 摆动画子图设置
ax_p.set_xlim(-1.3, 1.3); ax_p.set_ylim(-1.3, 0.4)
ax_p.set_aspect("equal"); ax_p.grid(True, alpha=0.3)
ax_p.axhline(0, color="gray", lw=0.5)
pivot_dot, = ax_p.plot([0], [0], "k^", ms=12, zorder=5)
rod_line,  = ax_p.plot([], [], "-", color="steelblue", lw=3)
bob_dot,   = ax_p.plot([], [], "o", color="steelblue", ms=18)
trail_line,= ax_p.plot([], [], "-", color="steelblue", alpha=0.25, lw=1)
time_text  = ax_p.text(0.02, 0.95, "", transform=ax_p.transAxes, fontsize=9)
ax_p.set_xlabel("x (m)"); ax_p.set_ylabel("y (m)")

# 角度曲线子图
ax_t.set_xlim(0, T_end); ax_t.set_ylim(-200, 200)
ax_t.set_xlabel("t (s)"); ax_t.set_ylabel("deg  /  deg·s⁻¹")
ax_t.grid(True, alpha=0.3); ax_t.axhline(0, color="gray", lw=0.5)
theta_line_full, = ax_t.plot(r_anim["t"], np.degrees(r_anim["theta"]),
                              color="steelblue", lw=1, alpha=0.3)
dtheta_line_full,= ax_t.plot(r_anim["t"], np.degrees(r_anim["dtheta"]),
                              color="darkorange", lw=1, alpha=0.3)
theta_line, = ax_t.plot([], [], color="steelblue",  lw=2, label="θ (deg)")
dtheta_line,= ax_t.plot([], [], color="darkorange", lw=2, label="θ̇ (deg/s)")
ax_t.legend(fontsize=9)
cur_t_line  = ax_t.axvline(0, color="gray", lw=1, ls="--")

# 能量子图
ax_e.set_xlim(0, T_end)
e_min = min(r_anim["E"].min(), r_anim["P"].min()) * 1.1
e_max = r_anim["E"].max() * 1.2
ax_e.set_ylim(e_min, e_max)
ax_e.set_xlabel("t (s)"); ax_e.set_ylabel("Energy (J)")
ax_e.grid(True, alpha=0.3)
ax_e.plot(r_anim["t"], r_anim["E"], "-",  color="gray",       lw=1, alpha=0.3)
ax_e.plot(r_anim["t"], r_anim["K"], "--", color="tomato",     lw=1, alpha=0.3)
ax_e.plot(r_anim["t"], r_anim["P"], ":",  color="mediumseagreen", lw=1, alpha=0.3)
K_line, = ax_e.plot([], [], color="tomato",         lw=2, label="K (kinetic)")
P_line, = ax_e.plot([], [], color="mediumseagreen", lw=2, label="P (potential)")
E_line, = ax_e.plot([], [], color="gray",           lw=2, label="E = K+P")
ax_e.legend(fontsize=9)
cur_e_line  = ax_e.axvline(0, color="gray", lw=1, ls="--")

TRAIL = 40   # 拖尾长度（帧数）

def init():
    rod_line.set_data([], [])
    bob_dot.set_data([], [])
    trail_line.set_data([], [])
    theta_line.set_data([], [])
    dtheta_line.set_data([], [])
    K_line.set_data([], [])
    P_line.set_data([], [])
    E_line.set_data([], [])
    return (rod_line, bob_dot, trail_line,
            theta_line, dtheta_line, K_line, P_line, E_line,
            time_text, cur_t_line, cur_e_line)

def update(frame):
    i = frame
    xb = l_val * np.sin(r_anim["theta"][i])
    yb = -l_val * np.cos(r_anim["theta"][i])

    rod_line.set_data([0, xb], [0, yb])
    bob_dot.set_data([xb], [yb])

    # 拖尾轨迹
    i0 = max(0, i - TRAIL)
    xs = l_val * np.sin(r_anim["theta"][i0:i+1])
    ys = -l_val * np.cos(r_anim["theta"][i0:i+1])
    trail_line.set_data(xs, ys)

    time_text.set_text(f"t = {r_anim['t'][i]:.2f} s   θ = {np.degrees(r_anim['theta'][i]):.1f}°")

    # 角度曲线
    theta_line.set_data(r_anim["t"][:i+1],  np.degrees(r_anim["theta"][:i+1]))
    dtheta_line.set_data(r_anim["t"][:i+1], np.degrees(r_anim["dtheta"][:i+1]))
    cur_t_line.set_xdata([r_anim["t"][i], r_anim["t"][i]])

    # 能量曲线
    K_line.set_data(r_anim["t"][:i+1], r_anim["K"][:i+1])
    P_line.set_data(r_anim["t"][:i+1], r_anim["P"][:i+1])
    E_line.set_data(r_anim["t"][:i+1], r_anim["E"][:i+1])
    cur_e_line.set_xdata([r_anim["t"][i], r_anim["t"][i]])

    return (rod_line, bob_dot, trail_line,
            theta_line, dtheta_line, K_line, P_line, E_line,
            time_text, cur_t_line, cur_e_line)

SKIP = 2   # 每隔一帧取一帧，压缩文件大小
N_frames = len(r_anim["t"])
frame_indices = list(range(0, N_frames, SKIP))

ani = animation.FuncAnimation(fig_anim, update, frames=frame_indices,
                               init_func=init, interval=40, blit=True)

ani.save("pendulum_animation.gif", writer="pillow", fps=25, dpi=80)
print("动画已保存为 pendulum_animation.gif")
plt.close("all")
