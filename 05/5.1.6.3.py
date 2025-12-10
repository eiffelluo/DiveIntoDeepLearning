import torch.nn as nn
import torch

def create_network(num_instances, input_size, hidden_size, output_size):
    # 创建一个线性层
    linear_layer = nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, input_size)
    )
    
    # 创建多个实例并连接
    instances = [linear_layer for _ in range(num_instances)]
    network = nn.Sequential(*instances)
    
    # 添加输出层
    output_layer = nn.Linear(input_size, output_size)
    network.add_module("output", output_layer)
    
    return network
# 示例用法
net = create_network(num_instances=2, input_size=3, hidden_size=3, output_size=2)
print(net(torch.rand(1, 3)))
print(net.state_dict())
