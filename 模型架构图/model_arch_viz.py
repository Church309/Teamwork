"""
Fashion CNN 模型架构可视化
生成模型结构图，保存到 可视化图表/model_architecture.png
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # [B,32,14,14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # [B,64,7,7]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B,128,7,7]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # [B,128,3,3]
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 3 * 3, 256),                  # [B,1152] → [B,256]
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10)                             # [B,256] → [B,10]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ─── 1. 用 torchinfo 打印详细摘要 ───
try:
    from torchinfo import summary
    model = FashionCNN()
    print("=" * 70)
    print("模型详细摘要 (torchinfo)")
    print("=" * 70)
    summary(model, input_size=(1, 1, 28, 28),
            col_names=['input_size', 'output_size', 'num_params', 'kernel_size'],
            depth=5, verbose=0)
except ImportError:
    print("torchinfo 未安装，跳过详细摘要。")
    print("如需安装: pip install torchinfo")

print("\n")


# ─── 2. matplotlib 架构图 ───
def draw_model_architecture():
    """绘制 CNN 架构图"""
    # 定义各层的输入输出形状
    layers = [
        ('Input\n图像', '1×28×28', '#E8F5E9'),
        ('Conv2d\n3×3, pad=1', '32×28×28', '#BBDEFB'),
        ('ReLU', '32×28×28', '#FFF9C4'),
        ('MaxPool\n2×2', '32×14×14', '#FFE0B2'),
        ('Conv2d\n3×3, pad=1', '64×14×14', '#BBDEFB'),
        ('ReLU', '64×14×14', '#FFF9C4'),
        ('MaxPool\n2×2', '64×7×7', '#FFE0B2'),
        ('Conv2d\n3×3, pad=1', '128×7×7', '#BBDEFB'),
        ('ReLU', '128×7×7', '#FFF9C4'),
        ('MaxPool\n2×2', '128×3×3', '#FFE0B2'),
        ('Flatten\nview', '1152', '#E1BEE7'),
        ('Dropout(0.25)\nLinear → 256', '256', '#C8E6C9'),
        ('ReLU\nDropout(0.25)', '256', '#FFF9C4'),
        ('Linear → 10\nOutput', '10', '#F8BBD0'),
    ]

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)
    ax.axis('off')
    ax.set_title('Fashion CNN 模型架构图', fontsize=15, fontweight='bold', pad=15)

    box_width, box_height = 4.5, 0.55
    arrow_dy = 0.15
    start_y = len(layers) - 0.5

    legend_patches = []
    legend_labels_map = {
        '#E8F5E9': '输入层',
        '#BBDEFB': '卷积层 (Conv2d)',
        '#FFF9C4': '激活层 (ReLU)',
        '#FFE0B2': '池化层 (MaxPool)',
        '#E1BEE7': '形状变换',
        '#C8E6C9': '全连接 + Dropout',
        '#F8BBD0': '输出层',
    }
    seen = set()
    for color, label in legend_labels_map.items():
        if color not in seen:
            seen.add(color)
            legend_patches.append(mpatches.Patch(color=color, label=label))

    for i, (name, shape, color) in enumerate(layers):
        y_pos = start_y - i
        box = FancyBboxPatch(
            (10 / 2 - box_width / 2, y_pos - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(5, y_pos + 0.08, name, ha='center', va='bottom',
                fontsize=8, fontweight='bold')
        ax.text(5, y_pos - 0.08, shape, ha='center', va='top',
                fontsize=7.5, color='#555555')

        # 向下箭头
        if i < len(layers) - 1:
            ax.annotate('', xy=(5, y_pos - box_height / 2 - arrow_dy),
                        xytext=(5, y_pos - box_height / 2),
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

    fig.legend(handles=legend_patches, loc='lower center',
               ncol=4, fontsize=7, framealpha=0.9)
    plt.subplots_adjust(bottom=0.06, top=0.96)
    plt.savefig('可视化图表/model_architecture.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("已保存: 可视化图表/model_architecture.png")


draw_model_architecture()


# ─── 3. 参数统计饼图 ───
def draw_param_distribution():
    """绘制各层参数量分布"""
    model = FashionCNN()

    layer_params = []
    layer_names = []

    # 卷积层参数
    for i, module in enumerate(model.conv_layers):
        if isinstance(module, nn.Conv2d):
            n_params = sum(p.numel() for p in module.parameters())
            in_ch = module.in_channels
            out_ch = module.out_channels
            k = module.kernel_size[0]
            layer_params.append(n_params)
            layer_names.append(f'Conv2d\n{in_ch}→{out_ch}\n{k}×{k}')

    # 全连接层参数
    for i, module in enumerate(model.fc_layers):
        if isinstance(module, nn.Linear):
            n_params = sum(p.numel() for p in module.parameters())
            layer_names.append(f'Linear\n{module.in_features}\n→{module.out_features}')
            layer_params.append(n_params)

    colors = plt.cm.Set3(range(len(layer_params)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    wedges, texts, autotexts = ax1.pie(
        layer_params, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=1.15
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title('各层参数量占比', fontsize=13, fontweight='bold')

    # 图例
    legend_labels = [f'{n}: {p:,}' for n, p in zip(layer_names, layer_params)]
    ax1.legend(wedges, legend_labels, loc='center left',
               fontsize=8, bbox_to_anchor=(-0.3, 0.5))

    # 柱状图
    bars = ax2.barh(range(len(layer_params)), layer_params, color=colors, edgecolor='#333')
    ax2.set_yticks(range(len(layer_params)))
    ax2.set_yticklabels([n.replace('\n', ' ') for n in layer_names], fontsize=8)
    ax2.set_xlabel('参数量', fontsize=11)
    ax2.set_title('各层参数量 (params)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for bar, v in zip(bars, layer_params):
        ax2.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
                 f'{v:,}', ha='left', va='center', fontsize=8)

    total_params = sum(layer_params)
    fig.suptitle(f'模型总参数量: {total_params:,}  (≈{total_params/1024:.1f}K)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('可视化图表/model_param_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 可视化图表/model_param_distribution.png")


draw_param_distribution()

print("\n✅ 全部完成！模型架构图已生成到「可视化图表」文件夹。")
