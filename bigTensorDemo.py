import torch
import time

# 检查MPS是否可用
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print(f"MPS可用: {mps_available}")
print(f"MPS已构建: {mps_built}")
print("-" * 50)

# 设置设备
if mps_available:
    device = torch.device("mps")
    print(f"使用设备: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"使用设备: CPU (MPS不可用)")

print("-" * 50)

# 定义不同的张量大小进行测试
sizes = [
    (1000, 1000),
    (2000, 2000),
    (5000, 5000),
    (10000, 10000),
]

def benchmark_matrix_multiplication(size, device, num_runs=5):
    """执行矩阵乘法并测量耗时"""
    m, n = size
    k = m  # 确保矩阵可以相乘
    
    # 创建随机大张量
    A = torch.randn(m, k, device=device)
    B = torch.randn(k, n, device=device)
    
    # 预热（MPS需要预热）
    if device.type == 'mps':
        _ = torch.mm(A, B)
        torch.mps.synchronize()
    
    # 测量多次运行的平均时间
    times = []
    for _ in range(num_runs):
        if device.type == 'mps':
            torch.mps.synchronize()  # 确保之前的操作完成
            start = time.time()
            C = torch.mm(A, B)
            torch.mps.synchronize()  # 等待GPU计算完成
            end = time.time()
        else:
            start = time.time()
            C = torch.mm(A, B)
            end = time.time()
        
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算FLOPS（浮点运算次数）
    # 矩阵乘法 A(m×k) × B(k×n) = C(m×n)
    # 每个元素需要k次乘法和k-1次加法，约等于k次运算
    # 总共有m×n个元素，所以总运算次数约为 m × n × k
    flops = m * n * k * 2  # 乘法和加法各算一次
    gflops = flops / (avg_time * 1e9)  # 转换为GFLOPS
    
    return {
        'size': size,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'gflops': gflops
    }

# 执行基准测试
print("开始大张量乘法性能测试...")
print("=" * 50)

results = []
for size in sizes:
    print(f"\n测试矩阵大小: {size[0]} × {size[1]}")
    result = benchmark_matrix_multiplication(size, device)
    results.append(result)
    
    print(f"  平均耗时: {result['avg_time']:.4f} 秒")
    print(f"  最小耗时: {result['min_time']:.4f} 秒")
    print(f"  最大耗时: {result['max_time']:.4f} 秒")
    print(f"  性能: {result['gflops']:.2f} GFLOPS")

# 总结
print("\n" + "=" * 50)
print("测试总结:")
print("-" * 50)
print(f"{'矩阵大小':<15} {'平均耗时(秒)':<15} {'性能(GFLOPS)':<15}")
print("-" * 50)
for result in results:
    size_str = f"{result['size'][0]}×{result['size'][1]}"
    print(f"{size_str:<15} {result['avg_time']:<15.4f} {result['gflops']:<15.2f}")

# 可选：对比CPU性能（如果MPS可用）
if mps_available:
    print("\n" + "=" * 50)
    print("对比测试: MPS vs CPU")
    print("=" * 50)
    
    # 选择一个中等大小的矩阵进行对比
    test_size = (2000, 2000)
    print(f"\n测试矩阵大小: {test_size[0]} × {test_size[1]}")
    
    # MPS测试
    print("\nMPS设备:")
    mps_result = benchmark_matrix_multiplication(test_size, torch.device("mps"))
    print(f"  平均耗时: {mps_result['avg_time']:.4f} 秒")
    print(f"  性能: {mps_result['gflops']:.2f} GFLOPS")
    
    # CPU测试
    print("\nCPU设备:")
    cpu_result = benchmark_matrix_multiplication(test_size, torch.device("cpu"))
    print(f"  平均耗时: {cpu_result['avg_time']:.4f} 秒")
    print(f"  性能: {cpu_result['gflops']:.2f} GFLOPS")
    
    # 加速比
    speedup = cpu_result['avg_time'] / mps_result['avg_time']
    print(f"\n加速比: {speedup:.2f}x (MPS比CPU快 {speedup:.2f} 倍)")

print("\n测试完成！")

