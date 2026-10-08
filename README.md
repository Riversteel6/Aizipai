# Aizipai — 字牌 AI

基于本地视觉识别、规则引擎与 AI 策略的字牌自动对战项目。支持识牌、合法动作判断、自动打牌和对局复盘；实时决策无需连接云端大模型。

## 功能

- **视觉识别**：通过 ADB 或 Android 本地运行环境识别手牌、桌面牌和操作按钮。
- **规则引擎**：支持字牌、红黑牌和王牌（癞子）规则，校验吃、碰、偎、跑、提、胡等动作。
- **AI 策略**：基于当前可见牌局、合法动作、胜率、胡息与风险进行决策。
- **自动执行**：检查执行前后的状态，防止误点和重复动作。
- **训练与测试**：支持本地模拟、策略评估、决策日志与回放。

## 开发与测试

需要 Python 3.11 或更高版本。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Android 客户端源码位于 `chenzhou_zipai_ai/android_app/`。本仓库仅提供源码，不提供 APK 安装包。

## 项目结构

- `chenzhou_zipai_ai/ai/`：AI 策略、搜索与对手模型
- `chenzhou_zipai_ai/engine/`：牌局规则与合法动作
- `chenzhou_zipai_ai/vision/`：视觉识别
- `chenzhou_zipai_ai/capture/`、`control/`：画面采集与操作
- `chenzhou_zipai_ai/android_app/`：Android 客户端
- `chenzhou_zipai_ai/models/`：运行必需的模型文件
- `tests/`、`chenzhou_zipai_ai/tests/`：自动化测试

本仓库只保留源码、必要配置与运行资源，不包含私人设备数据、运行日志或开发过程报告。视觉识别与设备控制的效果可能随设备、系统和游戏版本变化。
