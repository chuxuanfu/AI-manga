# AI Manga Studio

一个完全本地运行的漫画壁纸工作室。输入中文或英文提示词，生成构图、检查人物和物体的结构、精绘画面，并在交付分辨率下逐块复审。默认生成一张 4K PNG。加 `--8k` 会额外制作、审查并导出 8K PNG。这里的 4K 和 8K 指长边 3840 和 7680 像素，短边由画幅决定。

## 安装和打开

下载仓库 ZIP 并解压，或用 Git 克隆。**第一次和以后都打开同一个文件 `Studio.command`**。首次运行会从上游下载 Python、ComfyUI、Ollama、绘图和审图模型到项目的 `runtime/`。中断后重新打开可继续安装。安装成功后可以断网生成，无需账户、API 密钥或订阅。

macOS 使用 Apple Silicon，本项目在配有 64 GB 统一内存的 M4 Max 上开发。Linux 安装脚本支持 x86-64、Bash、curl 和兼容的 GPU 驱动，神经放大需要 Vulkan。Windows 可在具备相同 GPU 能力的 Linux 环境中使用。基础安装建议预留 120 GB。64 GiB 及以上的 Apple Silicon 或显存达到 48 GiB 的 CUDA 设备，安装器另外下载约 59 GB 的 Qwen-Image-2512 起稿与 Qwen-Image-Edit-2511 返修组件，另含约 7.3 GB 的 SeedVR2 图像重建模型，建议共预留 195 GB 以上。新克隆中可以执行以下命令。

```bash
bash Studio.command --help
bash Studio.command --install
bash Studio.command
```

菜单默认选「生成壁纸」。画面描述也可以用「日系商业插画，」开头来选用商业插画上色流程。还可以打开图库、设置成品位置、只出底图或进入节点编辑器。菜单启动的生成任务在后台运行，日志在任务目录的 `run.log`。终端直接传参时会在前台显示进度。加上 `--background` 可转入后台并打印任务目录。

## 命令行

```bash
./Studio.command -p "雨后的小镇，一位穿黄色雨衣的女孩牵着一只柴犬"
./Studio.command -p "狐狸沿着海岸散步" --style "日系透明水彩" --aspect 16:9 \
  --seed 42 --candidates 2 --max-rounds 3 -o "$HOME/Pictures/Manga"
./Studio.command -p "海港车站，一位少女和一只三花猫" --style anime-key-visual --aspect 16:9 --8k
./Studio.command --set-output-dir "$HOME/Pictures/Manga"
./Studio.command --reset-output-dir
./Studio.command --job "images/new-jobs/JOB_ID"
```

| 参数 | 省略时的行为 |
| --- | --- |
| `-p, --prompt` | 从无参数菜单输入场景 |
| `--style TEXT` | 使用提示词中的画风，未指定时采用工作室插画风格。内置的 anime-key-visual 使用有粗细变化的线稿、分明的明暗面和受控高光 |
| `--aspect auto/2:3/3:2/9:16/16:9/1:1` | 从描述推断，未注明时竖幅 2:3 |
| `--review-print PNG --reviewed-image DRAFT` | 对已有 4K 修订稿核验同源构图、实际像素和全部终审项。默认从底图精绘 |
| `--base-model MODEL` | 自动混合底图候选。支持的设备可指定 `qwen-image-2512` 四步起稿或 `qwen-image-2512-full` 五十步起稿 |
| `--seed N` | 随机选择起始种子 |
| `--candidates 1…8` | 完整流程每轮 3 张初始候选，快捷底图模式 1 张 |
| `--max-rounds N` | 0，持续尝试至通过。达到显式上限时保留进度 |
| `-o, --output-dir DIR` | 使用菜单保存的成品目录，未设置时使用任务目录 |
| `--8k` | 默认只精绘和复审 4K。指定后同时交付 4K 和 8K |
| `--native-4k` | 常规底图尺寸，例如竖幅 1248 × 1872 |
| `--full-8k-diffusion` | 默认不启用。指定后开启 8K 并使用全幅分块扩散 |
| `--draft` | 完整出片。加此参数可只做底图和选片 |
| `--from-image PNG` | 从提示词生成新构图 |

使用 `--help` 查看日常参数，`--advanced-help` 查看局部返修选项。节点编辑器使用本机地址 127.0.0.1:8193。

Qwen 的完整起稿模式使用基础模型、50 步和 4.0 引导强度，按模型原生画幅从空白画布采样。每张底图的计算量较大。自动模式在前两轮退稿后需要重新构图时，才会为一个独立文字候选启用完整采样。两种模式都经过相同的构图、精绘和原尺寸审查。相同区域跨轮退稿、清除错误轮廓后仍未改好时，工具会尝试一次 40 步的完整编辑模型，再检查修改前后的实际像素。[Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen-Image-2512)

## 成品在哪里

- 图库入口为项目根目录的 `gallery.html` 或 `images/gallery.html`。
- 默认成品为 `images/new-jobs/<任务>/4K.png`。启用 8K 时同目录还会有 `8K.png`。旧任务续跑沿用创建时的规格。对已定稿的 4K 任务运行 `./Studio.command --job images/new-jobs/JOB_ID --8k`，会以已发布的 4K 为底图追加 8K。校样、审图记录和中间文件都在同一个任务目录。
- 指定成品目录后，工作室会在该位置创建按任务和版本区分的子文件夹，复制通过终审的 PNG 和两份审图凭据。跨磁盘复制也会校验文件指纹。底图模式只导出选中的 `selected-proof.png`。
- 命令行 `-o` 优先于保存的默认目录。任务续跑会记住自己的导出位置。默认目录仅保存在本机的 `runtime/settings.json`。

## 工作流与审图

FLUX.2 klein 和 Anima 负责生成及精绘。具备 Qwen-Image-2512 权重时，纯文字任务的第二个候选可由它从空白画布独立构图。传入 `--base-model qwen-image-2512` 可直接使用该模型起稿，也可搭配 `--draft` 单独查看底图。初审结合本地视觉模型、数量核对、接触和遮挡检查，提示词明确要求全图无文字时，工具还会用本地文字检测器定位疑似字样，由视觉模型辨认，再用 LaMa 修补相应笔画。仅限制字幕时保留场景文字。常规底图先经过本地超分，再在 4K 分块画布上重绘。启用 8K 时，重绘区覆盖模型定位的焦点以及全图中有可辨结构的区域，平坦背景可以略过重绘。蓝晒、水彩、摄影、纺织等有连续色调的画材会在两级放大时使用通用 RealESRGAN x2plus，以保留纸面颗粒和颜料渐变。动漫线稿使用动漫超分模型。两级精绘还从底图取回超分时损失的局部色调变化，以保留水彩、版画等画材的痕迹。终审使用 Qwen3.8 27B 对照底图和 4K 成品。启用 8K 后还会对照两级成品的色调与同位置的原像素裁切，并检查全图和所有重叠的原像素图块。每个原像素图块先记录判断和依据，疑似缺陷再定位和复核。结构返工会在合成后的原像素上先复查受损部位及其连接处。未修好的局部会按反馈换种子重画。常规情况下依次比较最多三个候选；同一处反复出现错误时，还可尝试清除错误轮廓及完整模型重绘。局部通过后再复查整幅成图。再次返工时会用本机 LaMa 清除错误轮廓；安装了 Qwen-Image-Edit-2511 的设备会在前两个候选未达标时尝试独立的指令编辑模型，同一区域跨轮反复失败时优先使用它。安装器选用 INT8 编辑权重，已有 BF16 安装也可使用。纯文字任务若整轮修补都没有改善，就换种子重新构图。指定了源图的任务保留原构图继续返修。4K 任务跨轮续修时，会核对提示词、底图和退稿指纹，以保留其他已完成的修改。焦点模糊先按位置局部重画。若问题遍及画面或局部重画后仍模糊，工具可用已选构图作参考，重新采样高分辨率底图。它会核对主体、姿势、构图和画法，再重新精绘并终审。初审或终审退回的任务可换种子重试。模型可能误判；复杂提示词可能经历多轮仍无定稿。种子可复用，跨硬件和依赖版本的像素结果可能不同。

默认 4K 在装有 SeedVR2 的设备上优先从已选的 1024 至 3839 像素长边底图重建整幅画，再恢复原画颜色，检查主体、接触关系与原尺寸细节。未安装该模型的设备使用分块精绘。可以用 `--print-method tiled` 指定分块扩散，或用 `--print-method restoration` 指定本机 SeedVR2。原生 4K 底图直接进入原尺寸处理。

每轮目录里的 `*-task.json` 分别记录等待设备的 `gpu_queue_seconds` 和实际绘图的 `renderer_seconds`。多个任务共享同一块 GPU 时，等待时间会计入总耗时。审图时，本地视觉模型接收无损 PNG。复核细节清晰度时，工具可将同一位置的两个原像素裁切并排展示，交换左右再比较。

在分块精绘时，本地视觉模型先看底图的实际局部，逐块描述主体与笔触。审图发现焦点仍模糊时，支持指令编辑的设备会参考原画和局部重画，再交换原稿与候选的顺序比较两次。局部改善后，整幅成品仍须通过全部原尺寸图块的复查。提示词明确禁止文字时，印刷字检测和离线 OCR 会定位字迹，修补后重新检查画面。已有安装更新此版本后，运行一次 `./Studio.command --install` 可同步节点和文字识别权重。

节点编辑器附带文字起稿、参考图起稿、商业插画上色和 4K 精绘工作流。使用命令行的 anime-key-visual 风格时，会先构图，再由本地模型把人物、物体、颜色和动作转成标签，并用 Animagine XL 4.0 Opt 根据构图上色。原生 4K 模式使用构图与笔触样片的双参考上色。旧安装缺少新版权重时也可走双参考路线，执行 `./Studio.command --install` 可补齐约 6.94 GB 的新权重。在节点编辑器打开 anime-key-visual-color.json，可替换「本次构图」并修改画面描述。安装包内的画法参考和灰度材质样片是运行所需的图像资源。模型由安装器从原发布站点获取，各模型沿用上游许可。下载地址、固定版本与 SHA-256 可在 `scripts/models.json` 和安装脚本中查看。

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) · [FLUX.2 klein](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) · [Anima](https://huggingface.co/circlestone-labs/Anima) · [Animagine XL 4.0 Opt](https://huggingface.co/cagliostrolab/animagine-xl-4.0) · [Ollama 视觉模型](https://ollama.com/library/qwen3.8:27b) · [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · [SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2)

局部结构返工可读取随项目安装的实物参考。离线索引位于 `assets/repair-guides/index.json`，每项记录匹配词、具体连接关系、图片作者和许可。只有本次修正描述同时命中各组词时，编辑模型才会收到相应参考图。

当两个手腕挨得较近、视觉模型无法确认指尖与袖口的归属时，工具会在 CPU 上用 OpenCV 定位手掌，对照原图和镜像，再把每只手的独立原图裁切交给两个本地视觉模型复查。检测到手掌只提供取景位置。现有安装执行一次 `./Studio.command --install` 可补齐约 3.9 MB 权重。[OpenCV Zoo 手掌检测模型](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/palm_detection_mediapipe)

木版画与分层剪纸的自动构图可以参考本地灰度技法样片，观察刻线、留白和纸层的遮挡。样片取自大都会艺术博物馆开放馆藏。作品、图源和使用许可记录在 `assets/medium-studies/credits.json`。

在已安装指令编辑模型的设备上，若整幅图的主体和结构通过检查、终审确认画法仍低于参考，默认流程会按具体评语在该构图上重新绘制。它先核对身份、动作、画幅和色彩，再对新版重新精绘并逐块复审。
