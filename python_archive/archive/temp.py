import os
# 告诉 XLA 显式使用大写的 METAL
os.environ["JAX_PLATFORMS"] = "METAL"

import jax
import jax.numpy as jnp

# 1. 尝试直接获取设备
try:
    # 强制指定后端名称为大写
    devices = jax.devices("METAL")
    device = devices[0]
    print(f"✅ 成功连接设备: {device.device_kind}")
except Exception as e:
    print(f"❌ 依然无法识别 METAL: {e}")
    # 打印出所有可用的后端供参考
    print(f"当前可用后端: {jax.devices()}")
    device = jax.devices()[0]

# 2. 极简计算
def smoke_test():
    # 注意：在 Metal 上有时直接创建 ones 会触发分配器错误
    # 我们先在 CPU 创建再转过去，或者直接用 jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = x * 2.0
    print(f"计算结果: {y}")
    print("🚀 烟测完成！")

if __name__ == "__main__":
    smoke_test()