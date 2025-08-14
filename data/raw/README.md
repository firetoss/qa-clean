# 数据目录说明

请将您的中文问答数据放置到此目录下。支持多种文件格式：

- `input.parquet` (推荐，性能最佳)
- `input.xlsx` 或 `input.xls` (Excel格式)
- `input.csv` (CSV格式，自动检测编码和分隔符)

## 数据格式要求

数据文件必须包含以下列：

- `question`: 问题文本 (string) **必需**
- `answer`: 答案文本 (string) **必需**
- `id`: 唯一标识符 (string/int) **可选** - 如果不提供，程序会自动创建行索引作为id

## 示例数据结构

```python
import pandas as pd

# 示例数据（最小格式，仅包含必需列）
data = {
    'question': [
        '如何开发票？',
        '可以报销吗？',
        '发票怎么开？',
        '售后流程是什么？',
        '退款政策如何？'
    ],
    'answer': [
        '请登录官网开具增值税电子发票',
        '支持按规定报销，请保留原始小票',
        '请联系客服或在官网自助开票',
        '售后请联系官方客服并提供订单号',
        '支持7天无理由退款，详见退款政策'
    ]
}

# 可选：如果需要自定义id
# data['id'] = [1, 2, 3, 4, 5]

df = pd.DataFrame(data)

# 保存为不同格式
df.to_parquet('input.parquet', index=False)  # Parquet格式 (推荐)
df.to_excel('input.xlsx', index=False)       # Excel格式
df.to_csv('input.csv', index=False)          # CSV格式
```

## 注意事项

1. **支持的文件格式**：Parquet (推荐)、Excel (.xlsx/.xls)、CSV
2. **必需列**：`question`, `answer`（必须严格匹配列名）
3. **可选列**：`id`（如果不存在，程序自动创建行索引作为id）
4. **文本语言**：问题和答案文本建议为中文
5. **数据量建议**：1000条以上以获得更好的聚类效果
6. **CSV文件**：程序会自动检测编码(UTF-8/GBK/GB2312)和分隔符
7. **性能建议**：大数据量推荐使用Parquet格式，读取速度最快
