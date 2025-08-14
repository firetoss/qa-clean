# 数据目录说明

请将您的中文问答数据放置到此目录下，命名为 `input.parquet`。

## 数据格式要求

数据文件必须包含以下三列：

- `id`: 唯一标识符 (string/int)
- `question`: 问题文本 (string)
- `answer`: 答案文本 (string)

## 示例数据结构

```python
import pandas as pd

# 示例数据
data = {
    'id': [1, 2, 3, 4, 5],
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

df = pd.DataFrame(data)
df.to_parquet('input.parquet', index=False)
```

## 注意事项

1. 文件格式必须为 Parquet
2. 列名必须严格匹配：`id`, `question`, `answer`
3. 问题和答案文本建议为中文
4. 数据量建议在1000条以上以获得更好的聚类效果
