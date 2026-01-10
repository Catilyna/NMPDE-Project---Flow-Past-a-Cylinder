import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# We use sep='\s+' to handle spaces or tabs as separators
# header=None tells pandas there is no first row with names
filename = "../results/drag_lift_history.txt"
data = pd.read_csv(filename, sep='\s+', header=None, names=["Time", "Drag", "Lift"])

# 2. Create the plot
plt.figure(figsize=(12, 5)) # Width, Height in inches

# --- Drag Plot ---
plt.subplot(1, 2, 1) # 1 row, 2 columns, plot number 1
plt.plot(data["Time"], data["Drag"], color="blue", linewidth=1.5, label="Drag ($C_D$)")
plt.title("Drag Coefficient ($C_D$)")
plt.xlabel("Time (s)")
plt.ylabel("Coefficient Value")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- Lift Plot ---
plt.subplot(1, 2, 2) # 1 row, 2 columns, plot number 2
plt.plot(data["Time"], data["Lift"], color="red", linewidth=1.5, label="Lift ($C_L$)")
plt.title("Lift Coefficient ($C_L$)")
plt.xlabel("Time (s)")
plt.ylabel("Coefficient Value")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 3. Show/Save
plt.tight_layout()
plt.savefig("../results/coefficients_plot.png", dpi=300) # Save high-quality image
plt.show()