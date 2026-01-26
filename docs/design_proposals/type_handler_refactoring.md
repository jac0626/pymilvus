---
title: TypeHandler 重构提案（精简版）
author: Deepmind-Agent
date: 2026-01-26
status: DRAFT
---

# TypeHandler 重构提案

## TL;DR

**问题**：类型处理逻辑分散在 5+ 个文件的 if-else 链中，难以维护和测试。

**方案**：用 16 个 TypeHandler 类替代 if-else，每个类型一个 Handler。

**原则**：最少抽象，最大实用，零拷贝优先。

---

## 1. 核心设计

### 1.1 TypeHandler 接口（极简版）

```python
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

class TypeHandler(ABC):
    """类型处理器 - 每个 DataType 对应一个实现"""
    
    @abstractmethod
    def pack(self, value: Any, field_data, field_info: dict) -> None:
        """将 Python 值写入 FieldData（包含验证和转换）"""
        pass
    
    @abstractmethod
    def extract(self, field_data, index: int) -> Any:
        """从 FieldData 读取 Python 值"""
        pass
    
    # 可选：批量优化
    def pack_batch(self, values: List[Any], field_data, field_info: dict) -> None:
        """批量写入（默认循环调用 pack，子类可覆盖优化）"""
        for v in values:
            self.pack(v, field_data, field_info)
    
    def extract_batch(self, field_data, start: int, end: int) -> List[Any]:
        """批量读取（默认循环调用 extract，子类可覆盖优化）"""
        return [self.extract(field_data, i) for i in range(start, end)]
```

**关键决策**：
- ✅ 直接操作 `field_data` (FieldData proto)，不引入 Sink/Source 抽象
- ✅ 保留 `field_info` dict 兼容现有代码
- ✅ 批量方法可选覆盖，默认实现兜底

### 1.2 Handler 注册表

```python
from pymilvus.client.types import DataType

# 简单的全局字典，不需要单例类
_HANDLERS: dict[DataType, TypeHandler] = {}

def register_handler(dtype: DataType, handler: TypeHandler) -> None:
    _HANDLERS[dtype] = handler

def get_handler(dtype: DataType) -> TypeHandler:
    handler = _HANDLERS.get(dtype)
    if handler is None:
        raise ValueError(f"No handler for {dtype}")
    return handler
```

---

## 2. Handler 实现示例

### 2.1 FloatVectorHandler（展示零拷贝优化）

```python
class FloatVectorHandler(TypeHandler):
    def pack(self, value, field_data, field_info):
        if value is None:
            return
        
        # 零拷贝：如果已经是 float32 np.array，直接使用
        if isinstance(value, np.ndarray):
            if value.dtype == np.float32:
                data = value
            else:
                data = value.astype(np.float32, copy=False)
        else:
            data = np.asarray(value, dtype=np.float32)
        
        field_data.vectors.dim = len(data)
        field_data.vectors.float_vector.data.extend(data.tolist())
    
    def pack_batch(self, values, field_data, field_info):
        """批量优化：一次性分配内存"""
        if not values:
            field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            return
        
        dim = len(values[0])
        field_data.vectors.dim = dim
        
        # 批量 extend
        all_data = np.concatenate([
            v if isinstance(v, np.ndarray) and v.dtype == np.float32 
            else np.asarray(v, dtype=np.float32)
            for v in values if v is not None
        ])
        field_data.vectors.float_vector.data.extend(all_data.tolist())
    
    def extract(self, field_data, index):
        dim = field_data.vectors.dim
        start = index * dim
        return list(field_data.vectors.float_vector.data[start:start + dim])
```

### 2.2 Int64Handler（最简实现）

```python
class Int64Handler(TypeHandler):
    def pack(self, value, field_data, field_info):
        if value is None:
            return
        field_data.scalars.long_data.data.append(value)
    
    def pack_batch(self, values, field_data, field_info):
        field_data.scalars.long_data.data.extend(v for v in values if v is not None)
    
    def extract(self, field_data, index):
        return field_data.scalars.long_data.data[index]
```

---

## 3. 集成方式

### 3.1 替换 pack_field_value_to_field_data

```python
# entity_helper.py 修改
def pack_field_value_to_field_data(
    field_value: Any,
    field_data: schema_types.FieldData,
    field_info: Any,
    vector_bytes_cache: Dict[int, List[bytes]],
):
    field_type = field_data.type
    
    # nullable 处理（保留在调用层）
    if field_info.get("nullable", False):
        field_data.valid_data.append(field_value is not None)
        if field_value is None:
            return
    
    # 委托给 Handler
    handler = get_handler(field_type)
    handler.pack(field_value, field_data, field_info)
```

### 3.2 替换 get_field_data + extract 逻辑

```python
# search_result.py 修改
def extract_field_value(field_data, index: int) -> Any:
    """统一的字段值提取"""
    # nullable 检查
    if field_data.valid_data and not field_data.valid_data[index]:
        return None
    
    handler = get_handler(field_data.type)
    return handler.extract(field_data, index)
```

---

## 4. 职责边界

| 场景 | 处理位置 | 理由 |
|------|----------|------|
| **类型验证/转换** | Handler | 核心职责 |
| **nullable / valid_data** | 调用层 | 存储逻辑，非类型逻辑 |
| **default_value** | prepare.py | 业务逻辑 |
| **is_dynamic 展开** | search_result | 呈现逻辑 |
| **highlight** | search_result | 搜索元数据 |

**Handler 只负责**：`值 ↔ FieldData` 的转换。其他逻辑保持原位。

---

## 5. 文件结构

```
pymilvus/client/handlers/
├── __init__.py          # 导出 get_handler, register_handler
├── base.py              # TypeHandler ABC
├── registry.py          # _HANDLERS dict
├── scalars.py           # Bool, Int8/16/32/64, Float, Double
├── strings.py           # VARCHAR, GEOMETRY, TIMESTAMPTZ
├── vectors.py           # FLOAT_VECTOR, BINARY, F16, BF16, INT8, SPARSE
└── complex.py           # JSON, ARRAY
```

**共 16 个 Handler 类覆盖 22 个 DataType**（INT8/16/32 共用一个，STRING/VARCHAR 共用一个）。

---

## 6. 类型覆盖矩阵

| 类别 | DataType | Handler | 优先级 |
|------|----------|---------|--------|
| **标量** | BOOL, INT8/16/32, INT64, FLOAT, DOUBLE | 6 个 Handler | P0 |
| **字符串** | VARCHAR, GEOMETRY, TIMESTAMPTZ | 3 个 Handler | P0 |
| **向量** | FLOAT_VECTOR, BINARY_VECTOR | 2 个 Handler | P0 |
| **特殊向量** | FLOAT16, BFLOAT16, INT8_VECTOR | 3 个 Handler | P1 |
| **稀疏向量** | SPARSE_FLOAT_VECTOR | 1 个 Handler | P1 |
| **复杂类型** | JSON, ARRAY | 2 个 Handler | P1 |

---

## 7. 迁移计划

| 周次 | 任务 |
|------|------|
| Week 1 | 创建 handlers/ 目录，实现 6 个 scalar handlers |
| Week 2 | 实现 6 个 vector handlers |
| Week 3 | 实现 JSON, ARRAY handlers + 集成 entity_helper |
| Week 4 | 集成 search_result + 删除旧 if-else 代码 |

**渐进式迁移**：每个 Handler 独立可测试，不需要一次性全部完成。

---

## 8. 与复杂版设计的对比

| 维度 | 复杂版 | 精简版 |
|------|--------|--------|
| 接口数量 | TypeHandler + Sink + Source + Registry + FieldMeta | TypeHandler + get_handler |
| 方法数量 | validate + normalize + pack + extract | pack + extract |
| 抽象层数 | 3 层（Logic/Interface/Storage） | 1 层（Handler） |
| nullable 处理 | Handler 内部 | 调用层 |
| 代码复杂度 | 中 | 低 |
| 文档长度 | ~600 行 | ~200 行 |

---

## 9. 开放问题

1. **Arrow 支持**：如果未来需要 Arrow，是否要引入 Sink/Source？
   - **建议**：到时再说。现在 Handler 直接操作 FieldData，未来可以加参数区分。

2. **STRUCT 类型**：当前实现复杂，是否单独处理？
   - **建议**：是，STRUCT 逻辑保持在 prepare.py，不纳入 Handler。

---

## 修订历史

| 版本 | 日期 | 作者 | 变更 |
|------|------|------|------|
| 0.1 | 2026-01-23 | Deepmind-Agent | 初稿 |
| 0.2 | 2026-01-23 | Deepmind-Agent | 添加代码调研，补充完整设计 |
| 0.3 | 2026-01-23 | Deepmind-Agent | 优化结构，添加 Mermaid 图 |
| 1.0 | 2026-01-26 | Deepmind-Agent | **精简版**：移除过度设计，保留核心 |
