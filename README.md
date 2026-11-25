# AI Manga Studio

一个完全本地运行的漫画壁纸工作室。输入中文或英文提示词，生成构图、检查人物和物体的结构、精绘画面，并在 8K 原像素下逐块复审。通过审查的任务会输出一对 4K 和 8K PNG。这里的 4K 和 8K 指长边 3840 和 7680 像素，短边由画幅决定。

## 安装和打开

下载仓库 ZIP 并解压，或用 Git 克隆。**第一次和以后都打开同一个文件 `Studio.command`**。首次运行会从上游下载 Python、ComfyUI、Ollama、绘图和审图模型到项目的 `runtime/`。中断后重新打开可继续安装。安装成功后可以断网生成，无需账户、API 密钥或订阅。

macOS 使用 Apple Silicon，本项目在配有 64 GB 统一内存的 M4 Max 上开发。Linux 安装脚本支持 x86-64、Bash、curl 和兼容的 GPU 驱动，神经放大需要 Vulkan。Windows 可在具备相同 GPU 能力的 Linux 环境中使用。建议为模型、缓存和成图预留 120 GB 以上。新克隆中可以执行以下命令。

```bash
bash Studio.command --help
bash Studio.command --install
bash Studio.command
```

菜单默认选「生成壁纸」。还可以打开图库、设置成品位置、只出底图或进入节点编辑器。菜单启动的生成任务在后台运行，日志在任务目录的 `run.log`。终端直接传参时会在前台显示进度。加上 `--background` 可转入后台并打印任务目录。

## 命令行

```bash
./Studio.command -p "雨后的小镇，一位穿黄色雨衣的女孩牵着一只柴犬"
./Studio.command -p "狐狸沿着海岸散步" --style "日系透明水彩" --aspect 16:9 \
  --seed 42 --candidates 2 --max-rounds 3 -o "$HOME/Pictures/Manga"
./Studio.command --set-output-dir "$HOME/Pictures/Manga"
./Studio.command --reset-output-dir
./Studio.command --job "images/new-jobs/JOB_ID"
```

| 参数 | 省略时的行为 |
| --- | --- |
| `-p, --prompt` | 从无参数菜单输入场景 |
| `--style TEXT` | 使用提示词中的画风，未指定时采用工作室插画风格 |
| `--aspect auto/2:3/9:16/16:9/1:1` | 从描述推断，未注明时竖幅 2:3 |
| `--seed N` | 随机选择起始种子 |
| `--candidates 1…8` | 完整流程每轮 3 张初始候选，快捷底图模式 1 张 |
| `--max-rounds N` | 0，持续尝试至通过。达到显式上限时保留进度 |
| `-o, --output-dir DIR` | 使用菜单保存的成品目录，未设置时使用任务目录 |
| `--native-4k` | 常规底图尺寸，例如竖幅 1248 × 1872 |
| `--full-8k-diffusion` | 4K 全幅精绘、2 倍神经超分与 8K 焦点重绘 |
| `--draft` | 完整出片。加此参数可只做底图和选片 |
| `--from-image PNG` | 从提示词生成新构图 |

使用 `--help` 查看日常参数，`--advanced-help` 查看局部返修选项。节点编辑器使用本机地址 127.0.0.1:8193。

## 成品在哪里

- 图库入口为项目根目录的 `gallery.html` 或 `images/gallery.html`。
- 默认成品为 `images/new-jobs/<任务>/4K.png` 和 `8K.png`。校样、审图记录和中间文件都在同一个任务目录。
- 指定成品目录后，工作室会在该位置创建按任务和版本区分的子文件夹，复制两张通过终审的 PNG 和两份审图凭据。跨磁盘复制也会校验文件指纹。底图模式只导出选中的 `selected-proof.png`。
- 命令行 `-o` 优先于保存的默认目录。任务续跑会记住自己的导出位置。默认目录仅保存在本机的 `runtime/settings.json`。

## 工作流与审图

FLUX.2 klein 和 Anima 负责生成及精绘。初审结合本地视觉模型、数量核对、接触和遮挡检查，终审使用 Qwen3.8 27B 看全图及重叠的原像素图块。发现问题时尝试局部重绘或重新构图，然后重新检查。模型可能误判；复杂提示词可能经历多轮仍无定稿。种子可复用，跨硬件和依赖版本的像素结果可能不同。

节点编辑器附带三份可修改的工作流，分别用于文字起稿、参考图起稿和 4K 精绘。安装包内的画法参考和灰度材质样片是运行所需的图像资源。模型由安装器从原发布站点获取，各模型沿用上游许可。下载地址、固定版本与 SHA-256 可在 `scripts/models.json` 和安装脚本中查看。

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) · [FLUX.2 klein](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) · [Anima](https://huggingface.co/circlestone-labs/Anima) · [Ollama 视觉模型](https://ollama.com/library/qwen3.8:27b) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
