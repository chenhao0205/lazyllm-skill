#!/usr/bin/env python3
"""
使用LazyLLM Flow创建九九乘法表打印程序
演示了Pipeline、Parallel、IFS等多种Flow组件的使用
"""

import lazyllm
from lazyllm import pipeline, parallel, ifs

# 方法1: 基础Pipeline - 生成格式化的九九乘法表
def generate_table(input_data):
    """生成九九乘法表数据"""
    table = []
    for i in range(1, 10):
        row = []
        for j in range(1, i + 1):
            row.append(f"{j}×{i}={i*j}")
        table.append("  ".join(row))
    return table

def format_with_row_numbers(table_rows):
    """为表格添加行号"""
    header = "九九乘法表\n" + "=" * 30 + "\n"
    content = "\n".join([f"第{i+1:2d}行: {row}" for i, row in enumerate(table_rows)])
    return header + content + "\n"

# 方法2: Parallel - 分别生成头部、内容和尾部
def generate_header(input_data):
    """生成表格头部"""
    return "九九乘法表\n" + "=" * 30 + "\n"

def generate_body(input_data):
    """生成表格内容"""
    rows = []
    for i in range(1, 10):
        row_items = [f"{j}×{i}={i*j}" for j in range(1, i + 1)]
        rows.append("  ".join(row_items))
    return "\n".join(rows)

def generate_footer(input_data):
    """生成表格尾部"""
    return "\n" + "=" * 30

def combine_parts(*args):
    """组合Parallel生成的各个部分"""
    parallel_result = args[0] if args else ()
    if len(parallel_result) == 3:
        header, body, footer = parallel_result
        return header + body + footer
    return str(parallel_result)

# 方法3: 简洁版Pipeline - 直接生成最终结果
def generate_complete_table(input_data):
    """一次性生成完整的九九乘法表"""
    result = "九九乘法表\n" + "=" * 30 + "\n"
    for i in range(1, 10):
        row_items = [f"{j}×{i}={i*j:2d}" for j in range(1, i + 1)]
        result += "  ".join(row_items) + "\n"
    return result + "=" * 30 + "\n"

# 方法4: 使用IFS演示条件分支
def generate_small_table(input_data):
    """生成5x5小乘法表"""
    result = "5×5乘法表\n" + "=" * 20 + "\n"
    for i in range(1, 6):
        row_items = [f"{j}×{i}={i*j:2d}" for j in range(1, i + 1)]
        result += "  ".join(row_items) + "\n"
    return result + "=" * 20 + "\n"

def generate_large_table(input_data):
    """生成12×12大乘法表"""
    result = "12×12乘法表\n" + "=" * 40 + "\n"
    for i in range(1, 13):
        row_items = [f"{j}×{i}={i*j:3d}" for j in range(1, i + 1)]
        result += "  ".join(row_items) + "\n"
    return result + "=" * 40 + "\n"

# 主程序
def main():
    print("LazyLLM Flow 九九乘法表示例\n")
    
    # 创建并执行方法1: 基础Pipeline
    print("方法1: 基础Pipeline - 带行号格式")
    print("=" * 50)
    pipeline1 = pipeline(generate_table, format_with_row_numbers)
    pipeline1.start()
    result1 = pipeline1(None)
    print(result1)
    
    # 创建并执行方法2: Parallel
    print("方法2: Parallel - 并行生成各部分")
    print("=" * 50)
    pipeline2 = pipeline(
        parallel(generate_header, generate_body, generate_footer),
        combine_parts
    )
    pipeline2.start()
    result2 = pipeline2(None)
    print(result2)
    
    # 创建并执行方法3: 简洁版Pipeline
    print("方法3: 简洁版Pipeline - 一步到位")
    print("=" * 50)
    pipeline3 = pipeline(generate_complete_table)
    pipeline3.start()
    result3 = pipeline3(None)
    print(result3)
    
    # 创建并执行方法4: IFS条件分支
    print("方法4: IFS条件分支 - 动态选择表类型")
    print("=" * 50)
    for table_type in ["small", "normal", "large"]:
        print(f"\n{table_type.upper()} 版本:")
        print("-" * 30)
        if_table = ifs(
            lambda x: table_type == "small",
            generate_small_table,
            ifs(
                lambda x: table_type == "large",
                generate_large_table,
                generate_complete_table
            )
        )
        if_table.start()
        result4 = if_table(None)
        print(result4)

if __name__ == "__main__":
    main()