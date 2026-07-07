# OmniCut

**360° Equirectangular Splitter + Video Frame Extractor**

A portable GUI tool with two functions:

| Tab | Function |
|-----|----------|
| **视频抽帧** (Video Extract) | Extract frames from video via FFmpeg |
| **图片裁切** (Image Split) | Split equirectangular 360° images into multi-angle perspective views |

### Image Split Features

- **Multi-axis control** — Yaw (horizontal), Pitch (vertical), independently configurable
- **Custom FOV, resolution, quality**
- **Single image or batch folder** processing
- **No AliceVision dependency** — pure NumPy/OpenCV implementation

### Quick Start

```bash
pip install -r requirements.txt
python main.py
```

For video extraction, click **Download FFmpeg** on the Video tab first.

### Build exe

```bash
pip install pyinstaller -r requirements.txt
pyinstaller --onefile --name OmniCut --noconsole main.py
```

Or push a tag (`v*`) to trigger GitHub Actions auto-build.

---

## OmniCut

**360° 等距柱状图裁切 + 视频抽帧工具**

两个功能独立 Tab：

| Tab | 功能 |
|-----|------|
| **视频抽帧** | 通过 FFmpeg 从视频提取帧序列 |
| **图片裁切** | 将 360° 全景图裁切成多张透视视角图 |

### 图片裁切特点

- **多轴控制** — Yaw（水平旋转）、Pitch（俯仰），各自独立配置
- **自定义 FOV、分辨率、画质**
- **支持单张或批量文件夹处理**
- **不依赖 AliceVision** — 纯 NumPy / OpenCV 实现

### 快速开始

```bash
pip install -r requirements.txt
python main.py
```

视频抽帧需先在「视频抽帧」Tab 中点击 **Download FFmpeg**。

### 打包 exe

```bash
pip install pyinstaller -r requirements.txt
pyinstaller --onefile --name OmniCut --noconsole main.py
```

或推送 tag（`v*`）触发 GitHub Actions 自动编译。

---

### License

This project bundles logic that replaces `aliceVision_split360Images.exe` (MPL-2.0).
FFmpeg is downloaded separately under LGPL/GPL.
The core projection algorithm is original work.
