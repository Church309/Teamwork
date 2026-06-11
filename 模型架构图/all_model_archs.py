"""
4 种模型架构可视化 (Fashion MNIST)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- CNN (PyTorch)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = '可视化图表'
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
N_CLASSES = 10


# ═══════════════════════════════════════════════
# 1. Logistic Regression 架构图
# ═══════════════════════════════════════════════
def draw_logistic_regression():
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Logistic Regression — 多分类逻辑回归', fontsize=14, fontweight='bold', pad=10)

    # 输入层 (784 像素)
    for i in range(28):
        for j in range(14):
            if i < 14:
                val = 0.3 + 0.5 * np.random.random()
                color = (val, val, val)
                rect = plt.Rectangle((0.5 + j * 0.2, 5.3 - i * 0.18), 0.18, 0.16,
                                     facecolor=color, edgecolor='#999', linewidth=0.3)
                ax.add_patch(rect)
    ax.text(3.3, 7.7, '输入图像 28×28×1', ha='center', fontsize=9, color='#555')
    ax.text(3.3, 7.4, '(展平为 784 维向量)', ha='center', fontsize=8, color='#999')
    box_in = FancyBboxPatch((0.2, 3.0), 6.2, 0.7, boxstyle="round,pad=0.1",
                            facecolor='#E8F5E9', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_in)
    ax.text(3.3, 3.45, '输入层: 784 个像素特征 (x₁, x₂, ..., x₇₈₄)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # 权重连接线 (简化，用弧形示意)
    for i in range(8):
        y_start = 2.7 - i * 0.3
        line_color = plt.cm.RdBu(i / 8)
        ax.plot([6.5, 7.5], [y_start, 4.15], color=line_color, linewidth=0.5, alpha=0.4)

    # Softmax 层
    box_softmax = FancyBboxPatch((7.5, 2.6), 2.5, 1.6, boxstyle="round,pad=0.15",
                                 facecolor='#FFF9C4', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_softmax)
    ax.text(8.75, 4.1, 'Softmax', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 3.75, 'P(y=k) = \nfrac{e^{w_k^T x}}{\sum_{j=1}^{10} e^{w_j^T x}}',
            ha='center', va='top', fontsize=7.5, color='#555')

    # 权重矩阵
    ax.text(8.75, 2.8, '权重矩阵: 784 × 10', ha='center', va='bottom', fontsize=8, color='#666')
    ax.text(8.75, 2.6, '偏置: 10', ha='center', va='bottom', fontsize=8, color='#666')

    # 输出层
    box_out = FancyBboxPatch((10.5, 2.0), 1.3, 3.2, boxstyle="round,pad=0.1",
                             facecolor='#F8BBD0', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_out)
    ax.text(11.15, 3.6, '输出', ha='center', fontsize=9, fontweight='bold', color='#333')
    for i in range(10):
        y = 3.2 - i * 0.28
        prob = 0.02 + 0.1 * (10 - i) * np.random.random()
        rect_bg = plt.Rectangle((10.7, y - 0.1), 0.9, 0.18, facecolor='#f0f0f0', edgecolor='none', linewidth=0.3)
        ax.add_patch(rect_bg)
        rect_bar = plt.Rectangle((10.7, y - 0.1), 0.9 * prob * 5, 0.18,
                                 facecolor='#E91E63', alpha=0.7, edgecolor='none', linewidth=0.3)
        ax.add_patch(rect_bar)

    # 箭头
    ax.annotate('', xy=(7.2, 3.5), xytext=(6.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.annotate('', xy=(10.3, 3.5), xytext=(10.1, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # 信息面板
    info_text = (
        "模型特点:\n"
        "• 线性决策边界\n"
        "• 无隐层，直接 784→10\n"
        "• 参数量: 784×10 + 10 = 7,850\n"
        "• 训练快，适合 baseline"
    )
    ax.text(0.5, 0.4, info_text, ha='left', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/07_lr_diagram.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'已保存: {SAVE_DIR}/07_lr_diagram.png')


# ═══════════════════════════════════════════════
# 2. Random Forest 架构图
# ═══════════════════════════════════════════════
def draw_random_forest():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Random Forest — 随机森林 (150 棵决策树)', fontsize=14, fontweight='bold', pad=10)

    # 输入
    box_in = FancyBboxPatch((0.3, 6.5), 3.0, 0.7, boxstyle="round,pad=0.1",
                            facecolor='#E8F5E9', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_in)
    ax.text(1.8, 6.85, '训练集 (bootstrap 采样)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.8, 6.5, 'N=20000 样本, 784 维特征', ha='center', va='bottom', fontsize=8, color='#666')

    # 多个子集 + 决策树
    tree_positions = [
        (0.3, 5.1), (1.1, 4.9), (1.9, 5.2), (2.7, 4.8),
        (0.5, 3.8), (1.3, 4.1), (2.1, 3.6), (2.9, 4.0),
        (0.7, 2.8), (1.5, 2.5), (2.3, 2.9), (2.8, 2.6),
        (0.6, 1.3), (1.4, 1.6), (2.2, 1.1), (2.9, 1.4),
    ]

    for xi, yi in tree_positions:
        # 树冠 (三角形)
        tri = plt.Polygon([[xi + 0.35, yi + 0.35], [xi, yi], [xi + 0.7, yi]],
                          facecolor='#2E7D32', alpha=0.5 + 0.3 * np.random.random(),
                          edgecolor='#1B5E20', linewidth=0.5)
        ax.add_patch(tri)
        # 树干
        ax.plot([xi + 0.35, xi + 0.35], [yi - 0.1, yi], color='#5D4037', linewidth=1.5)
        # 子集标签
        ax.text(xi + 0.35, yi - 0.25, '子集', ha='center', fontsize=5.5, color='#999')

    ax.text(1.8, 5.7, 'bootstrap 采样 × 150 棵决策树',
            ha='center', fontsize=8.5, color='#555', style='italic')

    # 投票环节
    box_vote = FancyBboxPatch((4.0, 2.5), 2.8, 1.8, boxstyle="round,pad=0.12",
                              facecolor='#E1BEE7', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_vote)
    ax.text(5.4, 3.8, '投票机制', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.4, 3.45, '每棵树对测试样本\n独立预测类别', ha='center', fontsize=8, color='#555')
    ax.text(5.4, 2.95, '众数投票\n(多数决定最终类别)', ha='center', fontsize=8.5, fontweight='bold', color='#6A1B9A')
    ax.text(5.4, 2.7, '150 棵树 = 150 票', ha='center', fontsize=8, color='#888')

    # 从树到投票的箭头
    for xi, yi in tree_positions:
        if xi < 3:
            ax.annotate('', xy=(4.0, 2.5 + 1.8 * np.random.random()),
                        xytext=(xi + 0.35, yi - 0.3),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=0.4, alpha=0.3))

    # 输出
    box_out = FancyBboxPatch((7.5, 2.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                             facecolor='#F8BBD0', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_out)
    ax.text(8.75, 3.8, '最终预测', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 3.45, '得票最多的类别', ha='center', fontsize=8.5, color='#333')
    ax.text(8.75, 3.1, '例如: "Trouser"\n获得 42/150 票', ha='center', fontsize=8, color='#E91E63')
    ax.text(8.75, 2.7, '输出: 10 分类之一', ha='center', fontsize=8, color='#666')

    ax.annotate('', xy=(7.3, 3.5), xytext=(6.85, 3.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # 信息面板
    info_text = (
        "模型特点:\n"
        "• 集成学习: Bagging + 决策树\n"
        "• 每棵树随机选特征子集\n"
        "• 超参数: n=150, max_depth=20\n"
        "• 抗过拟合，无需特征缩放"
    )
    ax.text(7.8, 1.0, info_text, ha='left', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/08_rf_diagram.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'已保存: {SAVE_DIR}/08_rf_diagram.png')


# ═══════════════════════════════════════════════
# 3. K-Nearest Neighbors 架构图
# ═══════════════════════════════════════════════
def draw_knn():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('K-Nearest Neighbors — K 最近邻 (K=5)', fontsize=14, fontweight='bold', pad=10)

    # 在二维空间中绘制样本点 (2 个 PCA 维度示意)
    np.random.seed(42)
    points_by_class = {}
    for c in range(N_CLASSES):
        center_x = 2 + 4 * np.random.random()
        center_y = 2 + 4 * np.random.random()
        pts = []
        for _ in range(12):
            pts.append([center_x + 0.3 * np.random.randn(),
                        center_y + 0.3 * np.random.randn()])
        points_by_class[c] = np.array(pts)

    colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
    for c, pts in points_by_class.items():
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors[c]], s=25, alpha=0.5, edgecolors='none')

    # 测试点 (红色大五角星)
    query = np.array([3.8, 4.2])
    ax.scatter(*query, c='red', s=200, marker='*', edgecolors='darkred', linewidths=1, zorder=5)
    ax.text(query[0] + 0.1, query[1] + 0.2, '测试样本', fontsize=9, fontweight='bold', color='red')

    # 画 K=5 的圆 (最近邻范围)
    circle = plt.Circle(query, 0.75, facecolor='none', edgecolor='red', linewidth=1.5, linestyle='--')
    ax.add_patch(circle)
    ax.text(query[0] + 0.6, query[1] + 0.6, 'K=5', fontsize=10, fontweight='bold', color='red')

    # 找到最近的 5 个点并连线
    all_points = []
    all_classes = []
    for c, pts in points_by_class.items():
        for pt in pts:
            all_points.append(pt)
            all_classes.append(c)
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points - query, axis=1)
    nearest_idx = np.argsort(dists)[:5]

    for idx in nearest_idx:
        pt = all_points[idx]
        c = all_classes[idx]
        ax.plot([query[0], pt[0]], [query[1], pt[1]], color=colors[c], linewidth=1.2, alpha=0.6)
        ax.scatter(*pt, c=[colors[c]], s=80, edgecolors='#333', linewidths=1.2, zorder=4)

    # 图例 (只展示 5 个)
    legend_elements = []
    shown = set()
    for idx in nearest_idx:
        c = all_classes[idx]
        if c not in shown:
            shown.add(c)
            legend_elements.append(mpatches.Patch(color=colors[c], label=f'{CLASS_NAMES[c]}'))
    legend_elements.insert(0, plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                         markersize=10, label='测试样本'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9, title='最近邻类别', title_fontsize=8)

    # 投票框
    box_vote = FancyBboxPatch((0.2, 0.2), 3.5, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#E1BEE7', edgecolor='#333', linewidth=1.2)
    ax.add_patch(box_vote)
    ax.text(1.95, 1.0, 'K=5 最近邻投票', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.95, 0.75, '多数邻居的类别 = 预测结果', ha='center', va='center', fontsize=8.5, color='#6A1B9A')

    # 信息面板
    info_text = (
        "模型特点:\n"
        "• 非参数化: 无训练过程\n"
        "• 直接存储所有训练样本\n"
        "• 预测时计算欧氏距离\n"
        "• 需要特征缩放 (StandardScaler)\n"
        "• K=5, 距离度量: L2"
    )
    ax.text(4.5, 1.2, info_text, ha='left', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/09_knn_diagram.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'已保存: {SAVE_DIR}/09_knn_diagram.png')


# ═══════════════════════════════════════════════
# 4. CNN 架构图 (加强版)
# ═══════════════════════════════════════════════
def draw_cnn():
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

    legend_map = {
        '#E8F5E9': '输入层', '#BBDEFB': '卷积层 (Conv2d)',
        '#FFF9C4': '激活层 (ReLU)', '#FFE0B2': '池化层 (MaxPool)',
        '#E1BEE7': '形状变换', '#C8E6C9': '全连接 + Dropout',
        '#F8BBD0': '输出层',
    }
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in legend_map.items()]

    for i, (name, shape, color) in enumerate(layers):
        y_pos = start_y - i
        box = FancyBboxPatch(
            (5 - box_width / 2, y_pos - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(5, y_pos + 0.08, name, ha='center', va='bottom',
                fontsize=8, fontweight='bold')
        ax.text(5, y_pos - 0.08, shape, ha='center', va='top',
                fontsize=7.5, color='#555555')

        if i < len(layers) - 1:
            ax.annotate('', xy=(5, y_pos - box_height / 2 - arrow_dy),
                        xytext=(5, y_pos - box_height / 2),
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=7, framealpha=0.9)
    plt.subplots_adjust(bottom=0.06, top=0.96)
    plt.savefig(f'{SAVE_DIR}/10_cnn_diagram.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'已保存: {SAVE_DIR}/10_cnn_diagram.png')


# ═══════════════════════════════════════════════
# 5. 四模型综合对比图
# ═══════════════════════════════════════════════
def draw_all_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('4 种模型架构对比总览', fontsize=16, fontweight='bold', y=0.98)

    info = [
        {
            'name': 'Logistic Regression',
            'color': '#4CAF50',
            'desc': '线性模型 + Softmax',
            'arch': '784 输入 → 10 输出\n(无隐层)',
            'params': '7,850',
            'pros': ['简单快速', '可解释性强', '适合 baseline'],
            'cons': ['线性边界', '无法处理图像空间结构'],
        },
        {
            'name': 'Random Forest',
            'color': '#FF9800',
            'desc': '150 棵决策树集成',
            'arch': 'Bootstrap 采样\n→ 每棵树独立预测\n→ 多数投票',
            'params': '~1,200 万 (估算)',
            'pros': ['抗过拟合', '捕捉非线性', '无需归一化'],
            'cons': ['模型体积大', '预测较慢', '高维稀疏表现一般'],
        },
        {
            'name': 'K-Nearest Neighbors',
            'color': '#9C27B0',
            'desc': '基于距离的非参数模型',
            'arch': '存储全部训练集\n→ 计算 L2 距离\n→ K=5 投票',
            'params': '无 (懒惰学习)',
            'pros': ['无需训练', '直觉直观', '适合小数据集'],
            'cons': ['预测 O(N)', '受维度灾难影响', '需要特征缩放'],
        },
        {
            'name': 'CNN (PyTorch)',
            'color': '#2196F3',
            'desc': '卷积神经网络',
            'arch': 'Conv×3 + Pool×3\n→ Flatten → FC\n→ 10 分类',
            'params': '~300K',
            'pros': ['捕捉空间特征', '参数共享', '准确率最高'],
            'cons': ['需要 GPU', '训练时间较长', '超参数调优复杂'],
        },
    ]

    for ax, m in zip(axes.flatten(), info):
        ax.axis('off')

        # 标题
        ax.add_patch(FancyBboxPatch((0.05, 0.85), 0.9, 0.12,
                                    boxstyle="round,pad=0.02",
                                    facecolor=m['color'], alpha=0.2, edgecolor=m['color'], linewidth=2))
        ax.text(0.5, 0.91, m['name'], ha='center', va='center',
                fontsize=14, fontweight='bold', color=m['color'])
        ax.text(0.5, 0.83, m['desc'], ha='center', va='center',
                fontsize=9, color='#666')

        # 架构
        ax.add_patch(FancyBboxPatch((0.08, 0.55), 0.84, 0.22,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#f9f9f9', edgecolor='#ddd'))
        ax.text(0.5, 0.72, '网络结构', ha='center', fontsize=9, fontweight='bold', color='#444')
        ax.text(0.5, 0.61, m['arch'], ha='center', va='center',
                fontsize=8.5, color='#333')

        # 参数量
        ax.text(0.5, 0.52, f'参数量: {m["params"]}', ha='center', fontsize=9,
                fontweight='bold', color=m['color'])

        # Pros
        ax.text(0.12, 0.47, '优势:', ha='left', fontsize=8.5, fontweight='bold', color='#2E7D32')
        for j, pro in enumerate(m['pros']):
            ax.text(0.15, 0.47 - (j + 1) * 0.06, f'✔ {pro}', ha='left', fontsize=7.5, color='#333')

        # Cons
        ax.text(0.55, 0.47, '劣势:', ha='left', fontsize=8.5, fontweight='bold', color='#C62828')
        for j, con in enumerate(m['cons']):
            ax.text(0.58, 0.47 - (j + 1) * 0.06, f'✘ {con}', ha='left', fontsize=7.5, color='#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{SAVE_DIR}/11_all_models_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'已保存: {SAVE_DIR}/11_all_models_comparison.png')


if __name__ == '__main__':
    draw_logistic_regression()
    draw_random_forest()
    draw_knn()
    draw_cnn()
    draw_all_comparison()
    print('\n全部完成！4 个模型架构图已生成到「可视化图表」文件夹。')
