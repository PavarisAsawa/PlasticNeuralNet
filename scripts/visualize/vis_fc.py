import numpy as np
import matplotlib.pyplot as plt

data = np.load("foot_contact.npy")          # shape (T,)  – one scalar per step
print("samples:", data.shape[0], "min/max:", data.min(), data.max())

plt.figure(figsize=(8, 4))
plt.plot(data, lw=1)
plt.title("Left‑front foot – contact force magnitude")
plt.xlabel("simulation step")
plt.ylabel("‖F‖  [N]")
plt.grid(alpha=0.3)
plt.tight_layout()

window = 50
smoothed = np.convolve(data, np.ones(window)/window, mode="valid")
plt.plot(smoothed, color="orange", label=f"moving‑avg ({window})")
plt.legend()

plt.show()