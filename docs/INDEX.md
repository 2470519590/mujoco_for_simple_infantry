# INDEX.md

本文件是当前传承版项目的目录索引。正式维护范围是电脑端 MuJoCo 仿真。

## 当前目录

- `assets/`：MuJoCo 模型。
- `config/`：仿真参数和腿长调度表。
- `src/robot_smoke/`：本地 smoke、控制、模型语义、实验和绘图代码。
- `tools/`：离线诊断工具，不作为正式仿真入口。
- `run_smoke.py`：根目录正式仿真入口。
- `docs/`：目录索引和控制理论文档。

## 关键文档

- `README.md`：社团传承版入口和常用命令。
- `docs/CONTROL_THEORY.md`：当前控制框架、公式和有效物理语义。

## 当前正式入口

- `run_smoke.py`

## 代码分包

- `src/robot_smoke/core/`：常量、类型、配置和 MuJoCo 通用工具。
- `src/robot_smoke/model/`：模型语义、actuator、五连杆、运动学和接触采样。
- `src/robot_smoke/control/`：IK、VMC、LQR、转向、腿长调度和控制辅助模块。
- `src/robot_smoke/experiments/`：本地 smoke、equilibrium、诊断和 trace。
- `src/robot_smoke/io/`：CLI 和绘图输出。
- `src/robot_smoke/runner.py`：入口编排。
- `tools/analyze_length_workpoints.py`：离线腿长工作点诊断工具。

## 默认不要读取或提交

- `__pycache__/`
- `.pytest_cache/`
- 日志和缓存
- checkpoint
- 大体积训练输出
- 临时实验垃圾文件
