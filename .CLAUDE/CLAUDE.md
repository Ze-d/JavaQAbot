# 项目开发约束与规范

## 开发约束

### 1. 编码规范
- **禁止使用表情包**：在所有代码、注释、文档中严禁使用表情包符号
- **编码风格**：遵循项目现有代码风格，保持一致性

### 2. 注释要求
- **详细注释**：所有代码必须书写详细的英文注释
- **行内注释**：复杂逻辑前必须添加行内注释说明
- **方法头注释**：每个方法必须有完整的方法头注释，包含：
  - 方法功能描述
  - 参数说明
  - 返回值说明
  - 可能抛出的异常
- **类注释**：每个类必须有详细的类注释说明

### 3. 日志要求
- **完整日志**：所有方法必须添加日志输出，日志输出使用英文输出
- **日志级别**：
  - DEBUG：详细调试信息（方法参数、变量值、流程步骤）
  - INFO：重要操作记录（初始化成功、方法调用完成）
  - WARNING：潜在问题（降级处理、参数调整）
  - ERROR：错误异常（异常捕获、失败记录）
- **日志格式**：统一使用 `[模块名] - 级别 - 文件:行号 - 消息` 的格式

### 4. 测试要求
- **全面测试**：对所有的类和方法添加测试
- **测试类型**：
  - 单元测试：测试单个方法的正确性
  - 集成测试：测试多个组件的协作
  - 边界测试：测试边界条件和异常情况
- **测试框架**：
  - Python：使用 pytest
  - Java：使用 JUnit
  - 其他语言：根据项目技术栈选择合适框架

## 文档注释规范

### 方法注释模板
```python
def method_name(param1: type, param2: type) -> return_type:
    """
    方法功能描述

    Args:
        param1: 参数1的详细说明
        param2: 参数2的详细说明

    Returns:
        返回值的详细说明

    Raises:
        ExceptionType: 异常情况的详细说明

    Examples:
        >>> method_name(arg1, arg2)
        'expected_result'
    """
```

### 类注释模板
```python
class ClassName:
    """
    类的详细功能描述

    类的用途、主要功能、设计思路等

    Attributes:
        attr1: 属性1的详细说明
        attr2: 属性2的详细说明
    """
```

## 日志输出规范

### 日志级别使用场景
```python
# DEBUG级别
logger.debug(f"method_name() - 输入参数: param1={param1}, param2={param2}")
logger.debug(f"method_name() - 处理步骤: {step}")

# INFO级别
logger.info("method_name() - 开始执行...")
logger.info("method_name() - 执行完成")
logger.info(f"method_name() - 结果: {result}")

# WARNING级别
logger.warning(f"method_name() - 检测到异常值: {value}")
logger.warning(f"method_name() - 使用降级方案: {fallback}")

# ERROR级别
logger.error(f"method_name() - 执行失败: {error}")
logger.error(f"method_name() - 异常类型: {type(e).__name__}")
```

## 测试代码规范

### 单元测试模板
```python
def test_method_name():
    """
    测试方法的基本功能
    """
    # 测试用例1：正常输入
    input_data = "test_input"
    expected_output = "expected_result"
    result = method_name(input_data)
    assert result == expected_output

    # 测试用例2：边界条件
    input_data = "edge_case"
    result = method_name(input_data)
    assert result is not None

    # 测试用例3：异常情况
    input_data = None
    with pytest.raises(ExceptionType):
        method_name(input_data)
```

## 代码审查检查清单

- [ ] 没有使用表情包
- [ ] 所有方法都有详细的文档注释
- [ ] 所有复杂逻辑都有行内注释
- [ ] 所有方法都有日志输出
- [ ] 所有类和方法都有测试
- [ ] 测试覆盖率 >= 80%
- [ ] 日志格式统一
- [ ] 异常处理完整
- [ ] 代码风格一致

## 注意事项

1. 注释和文档必须使用中文
2. 日志消息必须清晰明了，便于调试
3. 测试代码同样需要详细注释
4. 确保所有新增代码都遵循这些规范
5. 在代码审查时重点检查这些约束的执行情况
