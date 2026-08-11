# 控制系统

本文是当前 MuJoCo 电脑仿真版的控制说明。目标是让第一次接触本项目的人能看懂：机器人测了什么量、控制器算了什么量、这些量最后怎样变成 6 个电机的力矩。

本文以论文 `ref/j.cnki.xk.2023.pdf` 的控制架构为准，并对齐当前代码实现。若本文和代码不一致，以代码为准后再修正文档；不要只改文档掩盖控制语义问题。

## 1. 总体控制链路

当前正式控制链路是：

```text
MuJoCo 状态
  -> 五连杆解析运动学，得到 L、dL、theta、dtheta
  -> 轮端模拟里程计，得到 x、dx
  -> LQR，得到纵向公共轮力矩 T 和虚拟腿俯仰力矩 Tp
  -> yaw PD，得到左右轮差动力矩
  -> 双腿同步 PD，得到左右虚拟腿差动 Tp
  -> 腿长 PID + 前馈，得到沿腿推力 F_l
  -> Roll 补偿，得到左右沿腿推力差
  -> VMC: tau_joint = J(q)^T [F_l, Tp_side]^T
  -> 6 个 MuJoCo actuator ctrl
```

对应代码：

- LQR 状态与控制：`src/robot_smoke/control/lqr.py`
- LQR 工作点与线性化设计：`src/robot_smoke/control/lqr_design.py`
- 腿长调度表：`src/robot_smoke/control/length_schedule.py`
- 五连杆与 VMC：`src/robot_smoke/model/fivebar.py`、`src/robot_smoke/control/vmc.py`
- 转向与双腿同步：`src/robot_smoke/control/turning.py`
- 腿长和 Roll：`src/robot_smoke/control/roll.py`
- 主 rollout 编排：`src/robot_smoke/experiments/virtual_rod.py`

## 2. 坐标、角度和符号

### 2.1 基本坐标

当前 MuJoCo 模型使用世界系和机体系两个概念：

- 世界系：MuJoCo 全局坐标系。
- 机体系：车身自身坐标系，随车身 pitch、roll、yaw 转动。
- 左右虚拟腿：每侧五连杆可等效成“髋部中心到轮心”的虚拟杆。

### 2.2 主要角度

| 符号 | 代码名 | 单位 | 含义 |
| --- | --- | --- | --- |
| `theta` | `theta_world` / `state.theta` | rad | 虚拟腿相对世界竖直方向的角度。`theta=0` 表示虚拟腿竖直。 |
| `dtheta` | `theta_rate` | rad/s | 虚拟腿世界角速度。 |
| `phi` | `pitch` / `state.pitch` | rad | 车身俯仰角。 |
| `dphi` | `pitch_rate` | rad/s | 车身俯仰角速度。 |
| `gamma` | `roll` | rad | 车身横滚角。 |
| `psi_dot` | `yaw_rate` | rad/s | 车身 yaw 角速度。 |

虚拟腿角在代码中按世界系定义：

```text
theta_world = atan2(r_f, -r_z)
```

其中 `r` 是从髋部指向轮侧参考点的向量，`r_f` 是沿车体前进方向的分量，`r_z` 是世界竖直方向分量。这个定义不能随意改，因为 LQR、转向同步和 VMC 都依赖它。

### 2.3 主要力矩和力

| 符号 | 代码名 | 单位 | 含义 |
| --- | --- | --- | --- |
| `T` | `wheel_torque` | N*m | 左右驱动轮公共力矩，用于纵向平衡和速度控制。 |
| `Tp` | `pitch_torque` | N*m | 虚拟腿中心轴俯仰力矩，用于调节虚拟腿和车身姿态。 |
| `tau_turn` | `turn_torque` | N*m | yaw PD 产生的左右轮差动力矩。 |
| `Tp_sync` | `sync_torque` | N*m | 双腿同步 PD 产生的左右髋部差动力矩。 |
| `F_l` | `length_force` | N | 沿虚拟腿方向的推力。正值表示倾向于增大腿长。 |
| `F_roll` | `roll_force` | N | Roll 补偿产生的左右腿推力差。 |
| `F_N` | `left/right_contact_force` | N | 轮地接触法向支持力，用于离地检测。 |

## 3. 纵向平衡 LQR

论文把轮腿机器人简化成轮式倒立摆。当前代码使用同样的六维状态和二维输入：

```text
X = [theta, dtheta, x, dx, phi, dphi]^T
U = [T, Tp]^T
```

各量含义如下：

| 状态分量 | 单位 | 当前代码来源 |
| --- | --- | --- |
| `theta` | rad | 左右虚拟腿世界角平均值。 |
| `dtheta` | rad/s | 左右虚拟腿世界角速度平均值。 |
| `x` | m | 轮端模拟里程计位置减参考位置。 |
| `dx` | m/s | 轮端模拟里程计速度减速度参考。 |
| `phi` | rad | MuJoCo freejoint 姿态换算的车身 pitch。 |
| `dphi` | rad/s | MuJoCo freejoint pitch rate。 |

代码位置：`src/robot_smoke/control/lqr.py` 的 `compute_lqr_state()` 和 `lqr_state_vector()`。

### 3.1 为什么 x/dx 用轮端里程计

论文明确说明，系统速度应为驱动轮相对惯性系的平动速度。实车上这个速度由编码器、腿部运动学和 IMU 融合得到；仿真中当前用世界坐标换算出的模拟轮端里程计近似这个量。

因此：

```text
x  = odometry.position - x_ref
dx = odometry.speed    - dx_ref
```

不要把 `base_x`、`base_x_dot` 直接塞进 LQR 状态。`base_x/base_x_dot` 可以用于诊断漂移或打滑，但不是当前主控制状态。

### 3.2 工作点和反馈律

LQR 在某个腿长工作点附近使用线性模型：

```text
dX/dt ~= A(L0) (X - X0) + B(L0) (U - U0)
```

当前运行时反馈律为：

```text
U = U0 - K(L) (X - X0)
```

其中：

- `X0`：该腿长附近的平衡状态。
- `U0`：平衡输入。
- `K(L)`：该腿长下的 LQR 反馈矩阵。
- `L`：当前左右腿平均实际腿长。

当前默认不在运行时重新搜索工作点，而是从 `config/length_schedule.yaml` 读取调度表。调度表中每个点包含：

```text
L0, F_l0, X0, U0, K
```

代码位置：`src/robot_smoke/control/length_schedule.py`。

### 3.3 LQR 代价函数

LQR 设计使用二次型代价：

```text
J = integral( (X-X_ref)^T Q (X-X_ref) + (U-U_ref)^T R (U-U_ref) ) dt
```

当前 YAML 中只配置对角阵：

```text
Q = diag(lqr_q_diag)
R = diag(lqr_r_diag)
```

当前默认顺序：

```text
lqr_q_diag = [theta, dtheta, x, dx, pitch, d_pitch]
lqr_r_diag = [T, Tp]
```

较大的 `Q[i]` 表示更重视压小该状态误差；较大的 `R[j]` 表示更不愿意使用该输入。比如当前 pitch 权重很大，是为了优先压住车身俯仰。

### 3.4 输出符号和限幅

代码中 LQR 求出的 `U=[T,Tp]` 还会经过执行符号和限幅：

```text
T_exec  = lqr_wheel_sign * clip(T,  -lqr_t_limit,  lqr_t_limit)
Tp_exec = lqr_pitch_sign * clip(Tp, -lqr_tp_limit, lqr_tp_limit)
```

当前配置：

```text
lqr_wheel_sign = +1
lqr_pitch_sign = -1
```

这两个符号是经过当前 MuJoCo actuator、VMC 和论文角度方向对齐后的结果。未重新做单通道符号测试，不要修改。

## 4. 腿长调度

论文中腿长 `L0` 不进入六维 LQR 状态，而是作为不同工况下的线性化截面。当前代码也是这样处理：

```text
L_actual = 0.5 * (L_left + L_right)
schedule = length_schedule.evaluate(L_actual)
K   = interp(K_table,   L_actual)
X0  = interp(X0_table,  L_actual)
U0  = interp(U0_table,  L_actual)
F_l0 = interp(F_l0_table, L_actual)
```

插值是分段线性插值，且会把查询腿长夹到表格范围内。代码位置：`LengthSchedule.evaluate()`。

要点：

- `leg_length` 是期望腿长命令。
- `L_actual` 是实际测得的平均腿长。
- `F_l0` 是该腿长附近每条腿的支撑前馈。
- `K/X0/U0` 随腿长变化，避免低腿长和高腿长共用同一套 LQR 参数。

## 5. 五连杆运动学和 VMC

每条腿有两个主动关节：前驱动关节和后驱动关节。五连杆解析运动学给出：

```text
y = [L, theta]^T
q = [q_front, q_rear]^T
J(q) = d[L, theta] / d[q_front, q_rear]
```

其中：

- `L`：虚拟腿长度。
- `theta`：虚拟腿角。
- `q_front/q_rear`：该侧前后驱动关节角。
- `J(q)`：任务空间到关节空间的雅可比。

VMC 使用虚功关系把任务空间力映射到关节力矩：

```text
tau_joint = J(q)^T F_task
F_task = [F_l, Tp_side]^T
```

展开为：

```text
[tau_front, tau_rear]^T
    = J(q)^T [F_l, Tp_side]^T
```

这就是论文中“利用 VMC 将虚拟倒立摆控制量转换为腿部关节力矩”的实现。

当前代码优先使用解析五连杆 Jacobian；只有解析结果不可用或接近奇异时，才回退到诊断用数值方法。主控制路径不应使用被地面接触污染的 Jacobian。

代码位置：

- 五连杆解析：`src/robot_smoke/model/fivebar.py`
- Jacobian 与 VMC 映射：`src/robot_smoke/control/vmc.py`

## 6. 腿长 PID + 前馈

论文 2.2.1 中，腿长控制采用“PID + 前馈”，让腿长变化表现为弹簧阻尼系统，同时用前馈补偿上层结构重力。

当前每条腿的沿腿推力为：

```text
e_L = L_ref - L
I_L = integral(e_L dt)

F_l,base = F_l0 + Kp_L * e_L + Ki_L * I_L - Kd_L * dL
```

各项含义：

| 项 | 代码/YAML | 含义 |
| --- | --- | --- |
| `L_ref` | `leg_length` 或 Roll 修正后的左右腿参考 | 期望腿长。 |
| `L` | `state.length` | 实际腿长。 |
| `dL` | `state.length_rate` | 实际腿长速度。 |
| `F_l0` | `length_schedule.yaml` 的 `F_l0` | 支撑前馈。 |
| `Kp_L` | `virtual_rod_length_kp` | 腿长比例刚度。 |
| `Ki_L` | `virtual_rod_length_ki` | 前馈误差修正积分。 |
| `Kd_L` | `virtual_rod_length_kd` | 腿长阻尼。 |

代码位置：`src/robot_smoke/control/vmc.py` 的 `_drive_virtual_rod_vmc_ctrl()`。

注意：

- `F_l0` 是沿腿支撑前馈，不是 LQR 输入。
- 腿长 PID 输出的是任务空间沿腿推力，之后还要经过 `J^T` 才能变成关节力矩。
- `minimum_leg_length` 和 `maximum_leg_length` 会限制腿长参考，避免给机构不可达命令。

## 7. Roll 参考与横滚补偿

论文 2.2 中有两个不同的 Roll 相关通道，不能混在一起。

### 7.1 几何腿长参考

左右腿腿长期望可由期望横滚角和地面倾角计算。当前平地模型中地面倾角估计按 0 处理，因此几何参考为：

```text
e_h = track_width / 2 * sin(gamma_ref)

L_ref,left  = clamp(L_nominal,left  + e_h)
L_ref,right = clamp(L_nominal,right - e_h)
```

其中：

- `gamma_ref`：期望 roll。
- `track_width`：左右轮距。
- `e_h`：由 roll 参考造成的左右腿长差。

### 7.2 动态横滚补偿

论文 2.2.2 中，横滚补偿不是改腿长参考，而是在腿长 PID 输出后叠加左右相反的沿腿推力：

```text
e_gamma = gamma_ref - gamma
F_roll = K_gamma * e_gamma

F_left  = F_l,base,left  + F_roll
F_right = F_l,base,right - F_roll
```

当前代码位置：`src/robot_smoke/control/roll.py` 和 `src/robot_smoke/control/whole_body.py`。

要点：

- `L_ref,left/right` 是几何参考。
- `F_roll` 是动态抗扰推力。
- 二者必须分开；不要把 `F_roll` 积分成腿长命令，也不要把几何腿长差当成横滚力补偿。

## 8. 转向控制和双腿协调

论文 2.1 中，转向控制和双腿协调是两个独立 PD。

### 8.1 yaw 角速度 PD

期望 yaw 角速度和实际 yaw 角速度之差：

```text
e_psi = psi_dot_ref - psi_dot
tau_turn = Kp_yaw * e_psi + Kd_yaw * d(e_psi)/dt
```

然后把 `tau_turn` 以相反符号叠加到左右轮：

```text
tau_left_wheel  = T / 2 - tau_turn
tau_right_wheel = T / 2 + tau_turn
```

代码位置：`src/robot_smoke/control/turning.py` 的 `yaw_turn_torque()` 和 `split_wheel_torque()`。

当前转速档位：

```text
low    = pi/2 rad/s
medium = pi rad/s
high   = 10 rad/s
```

### 8.2 双腿协调 PD

转向时左右轮差速会让左右虚拟腿出现相反方向摆动，也就是“劈叉”。论文用左右腿角差做 PD：

```text
e_sync = theta_right - theta_left
Tp_sync = Kp_sync * e_sync + Kd_sync * d(e_sync)/dt
```

然后把它以相反符号叠加到左右腿 `Tp`：

```text
Tp_left  = Tp + Tp_sync
Tp_right = Tp - Tp_sync
```

这一路只作用在左右髋部虚拟俯仰力矩差动上，不直接改变轮端 `T`。

## 9. 速度参考

速度测试使用解析梯形速度参考：

```text
dx_ref(t) = trapezoid_speed_reference(profile, t)
```

当前 LQR 状态中的速度误差为：

```text
dx_state = odometry.speed - dx_ref
```

位置 `x` 默认不作为“回到世界某个坐标”的任务。地面模式下通常把当前自身里程计位置作为参考，使位置通道中性化；运动任务主要靠 `dx_ref` 改变速度。

代码位置：

- 速度参考：`src/robot_smoke/control/trajectory.py`
- LQR 状态构造：`src/robot_smoke/control/lqr.py`

## 10. 离地检测与空中 LQR 门控

论文第 3 节指出，当驱动轮支持力过低时，轮地最大静摩擦不足以稳定系统，应认为机器人离地。论文阈值为：

```text
F_N < 20 N
```

当前 MuJoCo 仿真直接读取左右轮接触法向力：

```text
airborne_raw =
    contact_detection_armed
    and F_N,left  < threshold
    and F_N,right < threshold
```

其中：

- `threshold = flight_airborne_force_threshold`，默认 `20 N`。
- `contact_detection_armed` 用于避免 reset 初期接触还没解析时误判离地。
- `flight_airborne_confirm_seconds` 要求低支持力持续一小段时间后才确认离地。
- `flight_airborne_rearm_seconds` 防止落地后短时间内反复触发。

论文策略是在离地时把 LQR 增益矩阵中除 `K21、K22` 外全部置零。按当前代码的 0 基索引和输入定义，这等价于：

```text
T = 0
Tp 只保留对 theta 和 dtheta 的反馈
```

代码位置：`src/robot_smoke/experiments/virtual_rod.py` 的 `_airborne_lqr_torque()` 和离地状态逻辑。

当前语义：

- 离地时不靠轮端公共力矩 `T` 稳定，因为轮子没有可靠地面约束。
- 空中只用 `Tp(theta,dtheta)` 尽量保持虚拟腿姿态接近竖直。
- 重新接触后恢复完整 LQR。

## 11. 6 个电机输出的组成

最终输出给 MuJoCo 的 6 个 actuator 为：

```text
left_front_motor
left_rear_motor
left_wheel_motor
right_front_motor
right_rear_motor
right_wheel_motor
```

轮电机：

```text
left_wheel_motor  <- tau_left_wheel
right_wheel_motor <- tau_right_wheel
```

腿部四个电机：

```text
left_front/rear  <- J_left(q)^T  [F_left,  Tp_left]^T
right_front/rear <- J_right(q)^T [F_right, Tp_right]^T
```

其中：

```text
F_left  = F_l,base,left  + F_roll
F_right = F_l,base,right - F_roll
Tp_left  = Tp + Tp_sync
Tp_right = Tp - Tp_sync
```

MuJoCo actuator 使用 `tau = gear * ctrl`。代码会把目标关节力矩除以 actuator gear，再按 `ctrlrange` 限幅写入 `data.ctrl`。

## 12. 当前有效参数入口

主要参数集中在 `config/smoke.yaml`：

- `leg_length`：默认腿长。
- `minimum_leg_length / maximum_leg_length`：腿长命令限幅。
- `length_schedule / length_schedule_path`：是否启用腿长调度表。
- `virtual_rod_length_kp/kd/ki`：腿长 PID。
- `virtual_rod_length_force_ff`：调度关闭时的手动前馈。
- `lqr_q_diag / lqr_r_diag`：LQR 代价矩阵对角线。
- `lqr_t_limit / lqr_tp_limit`：LQR 输出限幅。
- `lqr_wheel_sign / lqr_pitch_sign`：执行符号。
- `yaw_turn_kp/kd`：yaw PD。
- `leg_sync_kp/kd`：双腿同步 PD。
- `roll_reference / roll_force_kp`：Roll 参考和横滚补偿。
- `flight_airborne_force_threshold`：离地检测支持力阈值。

腿长调度数据在 `config/length_schedule.yaml`：

- `L0`：腿长截面。
- `F_l0`：该截面的腿长支撑前馈。
- `X0`：该截面的 LQR 平衡状态。
- `U0`：该截面的 LQR 平衡输入。
- `K`：该截面的 2x6 LQR 增益矩阵。

## 13. 不允许随意改的物理语义

以下语义未经重新实验不得修改：

- `theta_world` 的定义和正方向。
- `F_l > 0` 表示倾向于增大虚拟腿长度。
- `tau_joint = J(q)^T [F_l, Tp_side]^T`。
- LQR 状态顺序 `[theta, dtheta, x, dx, phi, dphi]`。
- LQR 输入顺序 `[T, Tp]`。
- `x/dx` 来自轮端模拟里程计，不直接用 `base_x/base_x_dot`。
- 当前执行符号 `lqr_wheel_sign=+1`、`lqr_pitch_sign=-1`。
- 工作点反馈律 `U=U0-K(X-X0)`。

涉及平衡、摔倒、抗扰恢复和运动表现的结论，只能由 MuJoCo viewer 人工观察确认。曲线和数据只能用于定位控制通道、符号、尺度、饱和和接触问题。
