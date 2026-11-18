# Reproduce the README gallery / 复现展示图

Use the published DSFQ checkpoint and the environment in the main README. Each inference process sees one GPU. The two commands below may run independently on GPU0 and GPU1 when both have sufficient free memory. Output directories must be new.

使用已发布的 DSFQ 权重和 README 中的环境；每个推理进程只看到一张卡。GPU0／GPU1 显存足够时，可分别运行以下两条推理命令。输出目录必须尚不存在。

```bash
CUDA_VISIBLE_DEVICES=0 python -m qmap infer --method dsfq \
  --deployment models/dsfq/model.pt --images data/xyz/images \
  --start 64 --frames 732 --output runs/xyz_dsfq

CUDA_VISIBLE_DEVICES=1 python -m qmap infer --method dsfq \
  --deployment models/dsfq/model.pt --images data/indoor/images \
  --start 1120 --frames 496 --output runs/showroom_dsfq
```

`data/xyz` uses the chronological TUM preparation in the README. `data/indoor/images` refers to the existing 5,000-frame Indoor Travel sequence, sampled at 10 FPS from the 50 FPS source video (source frames 0, 5, 10, …). The original video/image sequence is not redistributed here. Use your own chronological PNG/JPEG sequence with suitable frame ranges if unavailable; it will produce a different scene.

`data/xyz` 使用 README 的 TUM 数据准备流程；`data/indoor/images` 指现有的 Indoor Travel 5000 帧序列，由 50 FPS 原视频按 10 FPS 采样，源帧为 0、5、10……。本仓库不分发原视频或图像集；没有该数据时，可以替换为自己的时序 PNG／JPEG 和对应帧范围，展示场景会随之改变。

## Render / 渲染

```bash
CUDA_VISIBLE_DEVICES='' python -m qmap gallery --run runs/xyz_dsfq \
  --output runs/xyz_gallery --title 'TUM XYZ / workspace' \
  --frame 200 --map-start 136 --map-end 264 --map-frames 64 \
  --pixel-stride 2 --confidence-percentile 20 \
  --yaw -25 --elevation 5 --animation-frames 40

CUDA_VISIBLE_DEVICES='' python -m qmap gallery --run runs/showroom_dsfq \
  --output runs/showroom_gallery --title 'Indoor Travel / showroom' \
  --frame 130 --map-start 96 --map-end 160 --map-frames 64 \
  --pixel-stride 2 --confidence-percentile 20 \
  --yaw -25 --elevation -20 --animation-frames 40

CUDA_VISIBLE_DEVICES='' python -m qmap depth-gallery \
  --runs BF16=runs/xyz_bf16 'Calibrated W4A4=runs/xyz_w4a4' DSFQ=runs/xyz_dsfq \
  --frames 100 365 630 --output runs/depth_gallery
```

The last command uses the three completed XYZ runs from the main README. `--frame`, `--frames`, `--map-start` and `--map-end` refer to zero-based indices **within the run**, with an exclusive map end. Thus the XYZ map uses source frames 200–327 and the showroom map uses 1216–1279. Source frame numbers in figure labels are also zero-based.

最后一条命令使用主 README 中已经完成的三方法 XYZ 推理。所有帧参数都是**当前推理序列内部**从 0 开始的索引，`--map-end` 不包含结束帧。因此 XYZ 点云对应源帧 200–327，展厅对应源帧 1216–1279。图中的源帧编号也从 0 开始。

## Outputs and interpretation / 输出与说明

| File | Content / 内容 |
|---|---|
| `gallery.png` | 2160×1120 RGB/depth/map panel / 展示面板 |
| `point_map.png`, `point_map.ply` | Render and colored predicted 3D points / 点云图与彩色三维点 |
| `orbit.gif` | Optional 40-frame virtual-camera orbit / 可选的 40 帧虚拟视角动图 |
| `gallery.json` | Source frames, sampling, confidence filtering and camera settings / 来源与渲染参数 |
| `depth_comparison.png`, `.pdf`, `.json` | Same-frame comparison with shared row colors / 同帧深度对比及记录 |

- Model input remains 294×518; a larger output canvas does not increase prediction resolution. / 模型输入仍为 294×518，大画布不会提高预测分辨率。
- Point maps use predicted depth, intrinsics and poses, with 64 sampled views, pixel stride 2 and the bottom 20% of confidence removed per frame. The renderer uses orthographic point splats and frames all retained points; it adds no GT, mesh completion or inpainting. / 点云由预测深度、内参和位姿生成，采用 64 个视图、像素步长 2、逐帧去除最低 20% 置信度；使用正交点渲染，静态构图包含全部保留点，不补入真值、网格或生成内容。
- The GIF shows the **same fixed point map** from moving virtual viewpoints, at a presentation frame rate. / 动图是**固定点云**的虚拟视角变化，播放帧率不代表推理速度。
- RGB/depth panels are not confidence masked. Showcase depth uses its own 2nd/98th-percentile colors. The comparison divides each method/frame by its own median and uses the same BF16 2nd/98th-percentile limits within each row. Values outside the color limits saturate. / RGB／深度面板不做置信度遮罩；展示深度采用自身第 2／98 百分位色阶。三方法对比先分别除以各帧中位数，再使用同一行 BF16 的第 2／98 百分位色阶，超出范围的颜色饱和。
- These curated local views are qualitative. They do not replace full-sequence accuracy results, show absolute depth scale, or estimate generalization. No calibration or weight update is performed. / 这些选取的局部图是定性展示，不替代全序列精度结果，也不用于展示绝对深度尺度或估计泛化性能；没有重新校准或修改权重。

The published source/selection records are in [gallery.json](../results/gallery.json). / 已发布图片的来源与筛选记录见 [gallery.json](../results/gallery.json)。
