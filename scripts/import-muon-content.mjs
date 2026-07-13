import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, posix } from 'node:path';

const sourceRoot = process.argv[2];
if (!sourceRoot) {
  throw new Error('Usage: node scripts/import-muon-content.mjs /path/to/Muon');
}

const entries = [
  ['MUON_LEARNING_GUIDE.md', 'muon-system-guide', 'Muon 优化器系统学习指南', '从更新几何、Newton–Schulz 数值近似到分布式实现，建立一套可验证的 Muon 心智模型。', 'Muon 专题', 0, 30, '2026-07-06'],
  ['必备知识地图/优化基础/SGD.md', 'sgd', 'SGD', '从随机梯度估计、学习率动力学到 Muon 的一阶优化起点。', '优化基础', 1, 8, '2026-07-01'],
  ['必备知识地图/优化基础/momentum.md', 'momentum', 'Momentum', '把动量理解成方向历史的低通滤波，并看清不同实现约定的尺度差异。', '优化基础', 2, 7, '2026-07-01'],
  ['必备知识地图/优化基础/Nesterov.md', 'nesterov', 'Nesterov 动量', '提前看向更新后的参数位置：公式、实现约定与 Muon 中被正交化的对象。', '优化基础', 3, 7, '2026-07-01'],
  ['必备知识地图/优化基础/AdamW.md', 'adamw', 'AdamW', '理解逐元素二阶矩、参数尺度和 Muon 混合优化器中的职责分工。', '优化基础', 4, 8, '2026-07-01'],
  ['必备知识地图/优化基础/解耦 weight decay.md', 'decoupled-weight-decay', '解耦 Weight Decay', '为什么权重衰减不等于 L2 正则，以及它为何是现代 Muon 配方的一部分。', '优化基础', 5, 8, '2026-07-01'],
  ['必备知识地图/线性代数/SVD.md', 'svd', '奇异值分解（SVD）', '把矩阵看成线性变换，理解 Muon 为什么保留奇异向量并压平奇异值。', '线性代数', 10, 7, '2026-07-01'],
  ['必备知识地图/线性代数/谱范数.md', 'spectral-norm', '谱范数', '从最坏方向的长度放大，到谱范数几何下的最陡下降。', '线性代数', 11, 7, '2026-07-01'],
  ['必备知识地图/线性代数/Frobenius 范数.md', 'frobenius-norm', 'Frobenius 范数', '矩阵版欧氏长度、RMS 缩放，以及 Muon 更新尺度的形状依赖。', '线性代数', 12, 8, '2026-07-01'],
  ['必备知识地图/线性代数/核范数.md', 'nuclear-norm', '核范数', '理解谱范数的对偶与 Muon 极分解方向之间的理论桥梁。', '线性代数', 13, 8, '2026-07-01'],
  ['必备知识地图/线性代数/极分解.md', 'polar-decomposition', '极分解', '从 SVD 推出 polar factor，并区分最近正交矩阵、QR 与 Muon 近似。', '线性代数', 14, 8, '2026-07-01'],
  ['必备知识地图/线性代数/半正交矩阵.md', 'semi-orthogonal-matrix', '半正交矩阵', '非方阵中的行半正交与列半正交，以及 update 和 weight 的关键区别。', '线性代数', 15, 8, '2026-07-01'],
  ['必备知识地图/数值计算/Newton–Schulz 迭代.md', 'newton-schulz', 'Newton–Schulz 迭代', 'Muon 的计算核心：用少量低精度矩阵乘近似压平动量矩阵的奇异谱。', '数值计算', 20, 13, '2026-07-01'],
  ['必备知识地图/数值计算/条件数.md', 'condition-number', '条件数', '输入谱的难度如何影响极分解近似、数值误差与训练稳定性。', '数值计算', 21, 6, '2026-07-01'],
  ['必备知识地图/数值计算/低精度矩阵乘.md', 'low-precision-matmul', '低精度矩阵乘', '输入精度、累加精度与舍入误差怎样进入优化器更新路径。', '数值计算', 22, 6, '2026-07-01'],
  ['必备知识地图/深度学习工程/mixed precision.md', 'mixed-precision', 'Mixed Precision', '理解 autocast、loss scaling、master weights，以及 BF16 Muon 的数值边界。', '深度学习工程', 30, 6, '2026-07-01'],
  ['必备知识地图/深度学习工程/optimizer state.md', 'optimizer-state', 'Optimizer State', '逐项计算优化器状态内存，理解 Muon 相对 AdamW 的真实收益。', '深度学习工程', 31, 7, '2026-07-01'],
  ['必备知识地图/深度学习工程/ZeRO-FSDP.md', 'zero-fsdp', 'ZeRO 与 FSDP', '参数、梯度与优化器状态如何分片，以及 Muon 的整矩阵语义为何棘手。', '深度学习工程', 32, 7, '2026-07-01'],
  ['必备知识地图/深度学习工程/张量并行.md', 'tensor-parallelism', '张量并行', 'Column/row parallel、fused QKV 和局部正交化中的高频正确性陷阱。', '深度学习工程', 33, 8, '2026-07-01'],
  ['必备知识地图/深度学习工程/Megatron-LM Muon 实现解析.md', 'megatron-muon-implementation', 'Megatron-LM Muon 实现解析', '沿代码路径拆解参数路由、TensorParallelMuon、LayerWise 布局与 DDP 同步。', '深度学习工程', 34, 30, '2026-07-06'],
  ['必备知识地图/LLM 实验方法/scaling law.md', 'scaling-law', 'Scaling Law', '读懂 compute-optimal 比较，不把特定拟合区间的优势误写成普遍规律。', 'LLM 实验方法', 40, 7, '2026-07-01'],
  ['必备知识地图/LLM 实验方法/critical batch size.md', 'critical-batch-size', 'Critical Batch Size', '理解 batch 增大何时不再带来等比例优化收益，以及它对 Muon 实验的影响。', 'LLM 实验方法', 41, 7, '2026-07-01'],
  ['必备知识地图/LLM 实验方法/muP.md', 'mup', 'μP', '把超参数从小模型迁移到大模型时，哪些尺度规则不能靠直觉复制。', 'LLM 实验方法', 42, 6, '2026-07-01'],
  ['必备知识地图/LLM 实验方法/token-FLOP-wall-clock 公平比较.md', 'token-flop-wall-clock', 'Token、FLOP 与 Wall-clock', '建立公平比较框架：更少 token、更少 FLOP 和更短墙钟时间不是同一件事。', 'LLM 实验方法', 43, 7, '2026-07-01'],
];

const routeBySource = new Map(entries.map(([source, slug]) => [source, `/notes/${slug}/`]));
const megatronBase = 'https://github.com/NVIDIA/Megatron-LM/blob/0823c731ed7d793aef047b6a64f2dbbf32bf6e2c/';

function yamlString(value) {
  return JSON.stringify(value);
}

function githubSourceUrl(path) {
  return `https://github.com/J-shang/Muon/blob/main/${path.split('/').map(encodeURIComponent).join('/')}`;
}

function rewriteLinks(body, currentSource) {
  let output = body;
  output = output.replace(/\]\(([^)#]+\.md)(#[^)]*)?\)/g, (match, target, hash = '') => {
    if (/^https?:/.test(target)) return match;
    let decoded;
    try { decoded = decodeURIComponent(target); } catch { decoded = target; }
    const resolved = posix.normalize(posix.join(posix.dirname(currentSource), decoded));
    if (resolved === '必备知识地图/README.md') return '](/notes/)';
    const route = routeBySource.get(resolved);
    return route ? `](${route}${hash})` : match;
  });
  if (currentSource.includes('Megatron-LM Muon')) {
    output = output.replaceAll(/\]\(\.\.\/\.\.\/Megatron-LM\/([^\)#]+)(#[^\)]*)?\)/g, (_, path, hash = '') => {
      return `](${megatronBase}${path}${hash})`;
    });
  }
  return output;
}

await mkdir(resolve('src/data/notes'), { recursive: true });

for (const [source, slug, title, description, category, order, readtime, date] of entries) {
  let body = await readFile(resolve(sourceRoot, source), 'utf8');
  body = body.replace(/^# .+\r?\n+/, '');
  body = rewriteLinks(body, source).trimStart();
  body = body.replace(/ {2}\r?\n/g, '<br>\n');
  const frontmatter = [
    '---',
    `title: ${yamlString(title)}`,
    `description: ${yamlString(description)}`,
    `category: ${yamlString(category)}`,
    `date: ${date}`,
    `updated: ${date}`,
    ...(category === 'Muon 专题' ? ['cutoff: 2026-07-01', 'featured: true'] : []),
    `order: ${order}`,
    `readtime: ${readtime}`,
    `source: ${yamlString(githubSourceUrl(source))}`,
    '---',
    '',
  ].join('\n');
  await writeFile(resolve('src/data/notes', `${slug}.md`), `${frontmatter}${body}`);
}

console.log(`Imported ${entries.length} Muon documents.`);
