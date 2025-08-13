# PostgreSQL + pgvector 设置指南（可选）

本指南将帮助你设置 PostgreSQL 数据库和 pgvector 扩展，作为 FAISS GPU 的替代方案。

## 🆕 PostgreSQL 17 支持

**好消息：** 本项目的代码完全兼容 PostgreSQL 17！无需任何代码修改。

### 推荐版本组合
- **PostgreSQL**: 17.x (最新稳定版)
- **pgvector**: 0.7+ (与 PG 17 完全兼容)
- **psycopg2-binary**: 2.9.9+ (确保最佳兼容性)

### 使用场景
- 当需要数据持久化时
- 当没有 GPU 资源时
- 当需要 SQL 查询支持时

## 1. 安装 PostgreSQL

### macOS (使用 Homebrew)
```bash
# 安装 PostgreSQL 17 (推荐)
brew install postgresql@17
brew services start postgresql@17

# 或者安装 PostgreSQL 15 (稳定版)
# brew install postgresql@15
# brew services start postgresql@15
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql-17 postgresql-contrib-17
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## 2. 安装 pgvector 扩展

### macOS
```bash
brew install pgvector
```

### Ubuntu/Debian
```bash
# 添加 pgvector 仓库
sudo sh -c 'echo "deb https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list'
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install postgresql-17-pgvector
```

## 3. 创建数据库和用户

```bash
# 连接到 PostgreSQL
sudo -u postgres psql

# 创建数据库
CREATE DATABASE qa_clean;

# 创建用户
CREATE USER qa_user WITH PASSWORD 'your_password';

# 授权
GRANT ALL PRIVILEGES ON DATABASE qa_clean TO qa_user;

# 退出
\q
```

## 4. 启用 pgvector 扩展

```bash
# 连接到 qa_clean 数据库
psql -h localhost -U qa_user -d qa_clean

# 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;

# 验证安装
SELECT * FROM pg_extension WHERE extname = 'vector';

# 退出
\q
```

## 5. 环境变量配置

创建 `.env` 文件：

```bash
# PostgreSQL 连接配置
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=qa_clean
export POSTGRES_USER=qa_user
export POSTGRES_PASSWORD=your_password

# 向量表配置
export VECTOR_TABLE=qa_vectors
```

或者在运行前设置：

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=qa_clean
export POSTGRES_USER=qa_user
export POSTGRES_PASSWORD=your_password
export VECTOR_TABLE=qa_vectors

# 然后运行
uv run qa-clean --input "data.xlsx"
```

## 6. 验证设置

运行以下命令测试连接：

```bash
uv run python -c "
from qa_clean.vector_store import PGVectorStore
from qa_clean.config import POSTGRES_CONFIG

try:
    store = PGVectorStore(POSTGRES_CONFIG)
    print('✅ PostgreSQL + pgvector 连接成功！')
    store.close()
except Exception as e:
    print(f'❌ 连接失败: {e}')
"
```

## 7. 性能优化建议

### 索引优化
```sql
-- 为向量列创建 HNSW 索引（PostgreSQL 15+）
CREATE INDEX ON qa_vectors 
USING hnsw (embedding_a vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX ON qa_vectors 
USING hnsw (embedding_b vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

### 连接池配置
```sql
-- 调整连接参数
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
SELECT pg_reload_conf();
```

## 8. 故障排除

### 常见问题

1. **扩展未找到**
   ```sql
   -- 检查扩展是否安装
   SELECT * FROM pg_available_extensions WHERE name = 'vector';
   ```

2. **权限不足**
   ```sql
   -- 确保用户有创建扩展的权限
   GRANT CREATE ON DATABASE qa_clean TO qa_user;
   ```

3. **连接被拒绝**
   - 检查 PostgreSQL 服务是否运行
   - 检查防火墙设置
   - 检查 `pg_hba.conf` 配置

4. **内存不足**
   - 调整 `shared_buffers` 和 `effective_cache_size`
   - 减少并发连接数

## 9. 监控和维护

### 查看向量表状态
```sql
-- 查看表大小
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename = 'qa_vectors';

-- 查看索引使用情况
SELECT 
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename = 'qa_vectors';
```

### 定期维护
```sql
-- 分析表统计信息
ANALYZE qa_vectors;

-- 清理死元组
VACUUM qa_vectors;
```
