# 配置变体说明

本文档提供不同场景下的配置调优建议。

## 大规模数据配置 (>100万问题)

```yaml
# 推荐修改项
recall:
  topk: 100                # 降低TopK减少内存使用
  faiss:
    index_type: "ivf_flat_ip"  # 使用IVF索引
    nlist: 8192             # 增加聚类中心数
    nprobe: 32              # 增加探测簇数保持精度

embeddings:
  batch_size: 32            # 降低批次大小
  
rerank:
  batch_size: 32            # 降低CE批次大小

# 可选：降低精度换取速度
consistency:
  cos_a: 0.85              # 稍微降低阈值
  cos_b: 0.85
  cos_c: 0.85
```

## 低内存配置 (<8GB显存)

```yaml
embeddings:
  batch_size: 16            # 小批次
  device: "cpu"             # 强制CPU

recall:
  topk: 50                  # 减少候选
  faiss:
    index_type: "flat_ip"   # CPU友好

rerank:
  batch_size: 8             # 最小批次
  device: "cpu"
```

## 高精度配置 (质量优先)

```yaml
recall:
  topk: 500                 # 更大召回
  char_ngram:
    threshold: 0.55         # 提高n-gram阈值

consistency:
  cos_a: 0.90              # 更严格阈值
  cos_b: 0.88
  cos_c: 0.88
  std_max: 0.02            # 更严格一致性
  vote_2_of_3: false       # 要求3/3通过

rerank:
  thresholds:
    high: 0.88             # 更高置信阈值
    mid_low: 0.82

cluster:
  center_constraints:
    coverage: 0.90         # 更严格中心约束
    mean: 0.90
    median: 0.88
    p10: 0.85
```

## 性能优化配置 (速度优先)

```yaml
recall:
  topk: 100
  faiss:
    index_type: "hnsw_ip"   # HNSW平衡速度精度
    hnsw_m: 16              # 减少连接数
    ef_search: 100          # 减少搜索广度

consistency:
  cos_a: 0.85              # 放宽阈值
  cos_b: 0.82
  cos_c: 0.82
  std_max: 0.06

cluster:
  second_merge:
    enable: false           # 关闭二次聚合加速
```

## 调优建议

### 精度 vs 召回权衡
- `consistency.cos_*`: 0.85-0.90，越高越严格
- `rerank.thresholds.high`: 0.80-0.90
- `recall.topk`: 100-500，越大召回越全

### 内存 vs 速度权衡  
- `embeddings.batch_size`: 8-256
- `rerank.batch_size`: 8-128
- `recall.faiss.index_type`: flat_ip(精确) > hnsw_ip(平衡) > ivf_flat_ip(大规模)

### 聚类质量控制
- `cluster.center_constraints.*`: 调节簇质量
- `cluster.second_merge.ce_min`: 0.75-0.85，控制二次聚合
- `cluster.min_cluster_size`: 最小簇大小

### 答案冲突控制
- `govern.number_tolerance_pct`: 数字容差
- `govern.date_tolerance_days`: 日期容差
- 根据业务需求调整BLACK_PATTERNS和WHITELIST_PATTERNS
