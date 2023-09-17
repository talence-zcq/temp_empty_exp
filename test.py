import torch

# 创建一个示例张量
x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

# 对张量进行升序排序
sorted_values, sorted_indices = torch.sort(x)
print("升序排序后的值：", sorted_values)
print("升序排序后的索引：", sorted_indices)

# 对张量进行降序排序
sorted_values_desc, sorted_indices_desc = torch.sort(x, descending=True)
print("降序排序后的值：", sorted_values_desc)
print("降序排序后的索引：", sorted_indices_desc)

# 在指定维度上排序
matrix = torch.tensor([[3, 1, 4],
                      [1, 5, 9],
                      [2, 6, 5]])
sorted_rows, _ = torch.sort(matrix, dim=1)
print("按行升序排序：", sorted_rows)
