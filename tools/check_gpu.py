# 检查 Kaggle 的 GPU 状态：有没有、是哪块卡、用了多少、还有多少空间。
# 贴进 Kaggle 一格跑就行。

import subprocess


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as e:
        return '（跑不了：%s）' % e


print('=' * 60)
print('1. PyTorch 认不认 GPU')
print('=' * 60)
import torch
print('   torch 版本      :', torch.__version__)
print('   CUDA 可用       :', torch.cuda.is_available())
print('   CUDA 版本       :', torch.version.cuda)
print('   显卡数量        :', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print('   卡 %d            : %s' % (i, torch.cuda.get_device_name(i)))
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print('   显存            : %.1f GB' % mem)
        print('   算力            : sm_%d%d' % cap)
else:
    print('   ⚠️ GPU 不可用。右边栏 Settings → Accelerator 选 GPU T4 x2，保存后重启 session。')

print()
print('=' * 60)
print('2. nvidia-smi：卡的实时状态')
print('=' * 60)
print(run('nvidia-smi'))

print()
print('=' * 60)
print('3. 现在显存占用')
print('=' * 60)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print('   已用 %.2f GB / 总共 %.2f GB，剩 %.2f GB'
          % ((total - free) / 1024**3, total / 1024**3, free / 1024**3))
else:
    print('   没有 GPU，跳过')

print()
print('=' * 60)
print('4. 跑个小测试（在 GPU 上算个矩阵乘法）')
print('=' * 60)
if torch.cuda.is_available():
    import time
    a = torch.randn(4000, 4000, device='cuda')
    b = torch.randn(4000, 4000, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print('   4000x4000 矩阵乘一次：%.1f 毫秒（%.1f TFLOPS）'
          % (dt * 1000, 2 * 4000**3 / dt / 1e12))
    del a, b, c
    torch.cuda.empty_cache()
    print('   ✅ GPU 能正常干活')
else:
    print('   没有 GPU，跳过')
