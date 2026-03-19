import numpy as np
import matplotlib.pyplot as plt

path = "/root/scaffoldfull/700_20.npy"
save_path = "/root/700_20_mask.png"

data = np.load(path)

# 只取第一个通道
mask = (data[:,:,0] == 1)

plt.figure(figsize=(5,5))
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close()

print("saved:", save_path)