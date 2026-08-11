# MuJoCo 轮腿机器人电脑仿真

本仓库是轮腿机器人 MuJoCo 电脑仿真项目，当前正式入口为 `run_smoke.py`。主要内容包括模型加载、五连杆运动学、VMC、腿长控制、LQR 平衡、速度跟踪、转向、冲击和离地检测。

控制系统参考原论文 [ref\j.cnki.xk.2023.2533.pdf](ref\j.cnki.xk.2023.2533.pdf) 

建议先完成环境配置和基础可视化检查，再阅读 [docs/CONTROL_THEORY.md](docs/CONTROL_THEORY.md)。

## 1. 入口梳理

- `run_smoke.py`：正式运行入口，平衡、速度、转向、冲击、离地、跳跃都从这里进。
- `assets/biped_wheel_leg.xml`：MuJoCo 机器人模型。
- `config/smoke.yaml`：主参数文件。
- `src/robot_smoke/`：仿真、控制、运动学等核心 Python 代码。
- `docs/CONTROL_THEORY.md`：控制系统说明。

## 2. 第一次配置环境

下面命令默认在 Windows PowerShell 中运行。项目路径假设为：

```powershell
E:\mujoco_py_lqr
```

### 2.1 安装 Miniconda

安装 Miniconda 后重新打开 PowerShell，确认：

```powershell
conda --version
```

### 2.2 一键创建 py310 环境

在 PowerShell 中执行：

```powershell
conda create -n py310 python=3.10 -y
conda activate py310
python -m pip install --upgrade pip
pip install mujoco numpy pyyaml scipy matplotlib
```

如果安装 Python 包很慢，或者因为网络原因安装失败，可以临时使用清华源：

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mujoco numpy pyyaml scipy matplotlib
```

如果希望当前环境以后默认使用清华源，可以执行：

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 检查 Python 路径

本项目常用的 Python 路径是：

```powershell
E:\miniconda\envs\py310\python.exe
```

如果你的 Miniconda 装在别的位置，用下面命令查看当前环境 Python 路径：

```powershell
where python
```

如果 Miniconda 安装位置不同，把后续命令中的 `E:\miniconda\envs\py310\python.exe` 替换为实际路径。

## 3. MuJoCo / OSMesa 常见问题

### 3.1 有 viewer 窗口时不要使用 OSMesa

如果之前跑过无显示/headless 仿真，环境变量 `MUJOCO_GL` 可能被设置成 `osmesa`。这会导致本地 viewer 不弹窗、窗口卡住，或者出现 OpenGL 相关错误。

在每次打开 viewer 前，先执行：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
```

### 3.2 如果你看到 OSMesa / OpenGL 报错

检查当前环境：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
conda activate py310
python -c "import mujoco; print(mujoco.__version__)"
```

如果 `import mujoco` 失败，重新安装 MuJoCo：

```powershell
pip install --upgrade mujoco
```

常见现象和处理方式：

| 现象                       | 优先处理                                                           |
| ------------------------ | -------------------------------------------------------------- |
| viewer 不弹窗               | 先执行 `Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue`。 |
| 报 `OSMesa` 或 OpenGL 相关错误 | 不要在本地 viewer 模式使用 `MUJOCO_GL=osmesa`。                          |
| `import mujoco` 失败       | 在 `py310` 环境里重新执行 `pip install --upgrade mujoco`。              |
| 命令里找不到 Python            | 用 `where python` 找到自己的 `python.exe`，替换命令里的路径。                  |

## 4. 第一次运行检查

进入项目目录：

```powershell
cd E:\mujoco_py_lqr
```

先确认脚本能显示帮助：

```powershell
python run_smoke.py --help
```

再跑一个不打开仿真画面的测试

```powershell
python run_smoke.py --virtual-rod-steps 10
```

正常结果示例：

```text
result: PASS finite model/load/step smoke
```

## 5. 第一次打开画面

平衡可视化：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --visualize --visualize-seconds 10
```

观察重点：

- 小车是否能站住。
- 腿长是否大致保持在默认高度附近。

## 6. 常用实验命令

默认腿长来自 `config/smoke.yaml`，当前传承版默认值为 `0.24 m`。腿长调度表默认启用。

### 6.1 平衡测试

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --visualize --visualize-seconds 10
```

### 6.2 直线速度测试

低速：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --speed-profile low --visualize --visualize-seconds 10
```

中速：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --speed-profile medium --visualize --visualize-seconds 10
```

高速：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --speed-profile high --visualize --visualize-seconds 10
```

### 6.3 平衡冲击测试

小冲击：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --impact small --visualize --visualize-seconds 10
```

中冲击：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --lqr-true-equilibrium --impact medium --visualize --visualize-seconds 10
```

### 6.4 原地旋转测试

`--turn-speed` 可取 `low`、`medium`、`high`：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --turn-test --turn-speed low --visualize --visualize-seconds 6
```

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --turn-test --turn-speed medium --visualize --visualize-seconds 6
```

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --turn-test --turn-speed high --visualize --visualize-seconds 6
```

### 6.5 变腿长高速旋转测试

这个测试会高速原地旋转，同时让腿长参考在允许范围内做正弦变化：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --turn-length-sine-test --visualize --visualize-seconds 10
```

### 6.6 Roll 坡道测试

用于观察论文 2.2 的双腿长度控制和横滚补偿：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --roll-test --visualize --visualize-seconds 10
```

### 6.7 飞坡 / 离地检测测试

飞坡：

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --flight-test --flight-test-speed high --visualize --visualize-seconds 10
```

### 6.8 原地跳跃测试

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --jump-test --visualize --visualize-seconds 10
```

### 6.9 手动驾驶场景

这个场景会一直运行到关闭 MuJoCo viewer。按键为：

- 按住 `↑` / `↓`：前进 / 后退。
- 按住 `←` / `→`：左旋 / 右旋。

```powershell
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
python run_smoke.py --manual-drive
```

## 7. 绘图诊断

图片默认输出到 `output\HHMMSS.png`。

转向 PD 图：

```powershell
python run_smoke.py --turn-test --turn-speed high --visualize-seconds 6 --turn-pd-plot
```

腿长 / Roll 图：

```powershell
python run_smoke.py --lqr-true-equilibrium --visualize-seconds 10 --roll-length-plot
```

LQR 调试图：

```powershell
python run_smoke.py --lqr-true-equilibrium --speed-profile high --visualize-seconds 10 --lqr-debug-plot
```

## 8. 后续阅读顺序

推荐阅读顺序：

1. 先跑 `README` 里的平衡测试。
2. 再看 [docs/CONTROL_THEORY.md](docs/CONTROL_THEORY.md) 控制算法文档和原论文 [ref\j.cnki.xk.2023.2533.pdf](ref\j.cnki.xk.2023.2533.pdf) 的相关章节，理解 `theta/pitch/T/Tp`。
3. 再跑速度测试和转向测试。
4. 最后再看腿长、Roll 和离地检测。


## 9. 参数分组

`config/smoke.yaml` 里的参数可以按用途理解，不用一上来全看完：

- 运行与工作点：`leg_length`、`minimum_leg_length`、`maximum_leg_length`、`initial_leg_length`、`length_schedule`
- 平衡工作点诊断：`equilibrium_*`
- 腿长 PID 与前馈：`virtual_rod_length_*`
- VMC 与整体平衡：`virtual_rod_theta_*`、`lqr_*`
- 转向与双腿同步：`yaw_turn_*`、`leg_sync_*`
- 横滚与腿高测试：`roll_reference`、`roll_force_kp`、`leg_height_*`
- 飞坡与离地检测：`flight_*`

## 10. 结尾

本项目仅用于开源控制算法的简单验证，一些任务效果并未优化的很好，欢迎指出；
如果对本项目有问题，欢迎交流询问，QQ：2470519590。
