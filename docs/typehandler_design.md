# TypeHandler 设计方案

## 📐 接口设计

```python
class DataTypeHandler(ABC):
    """基类 - 仅核心方法"""
    supported_types: Tuple[DataType, ...]
    
    # 写
    def pack_single(self, value, field_data, field_info) -> None
    def pack_batch(self, values, field_data, field_info, valid_data=None) -> None
    
    # 读
    def extract_payload(self, field_data) -> Any
    def create_accessor(self, payload, start, valid_data=None) -> Callable
    def get_slice(self, payload, start, end) -> List


class VectorHandler(DataTypeHandler):
    """向量子类 - 添加维度相关方法"""
    def get_dim(self, field_data) -> int
    def get_bytes_per_element(self, dim) -> int
    # create_accessor/get_slice 需要额外 dim 参数


class BytesVectorHandler(VectorHandler):
    """字节向量子类 - 添加缓存 flush"""
    def flush(self, field_data) -> None
```

---

## 🔍 用途对照

| 方法 | 用于 |
|------|------|
| `pack_single` | row insert |
| `pack_batch` | columnar insert |
| `extract_payload` | search/query 读取 |
| `create_accessor` | 单值随机访问 |
| `get_slice` | 批量访问 |
