"""
Plot of MBH masses vs distance from 2020ARA&A..58..257G
Shows upper limits (blue arrows), confirmed detections (green circles), 
and G1 observation (orange square).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Use the physrev style if available
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(script_dir)
    plt.style.use(os.path.join(pipeline_dir, 'physrev.mplstyle'))
except:
    pass

# Data from 2020ARA&A..58..257G
# Data for upper limits (blue arrows)
upper_limits = {
    'LMC': (0.05, 1e6),
    'Fornax': (0.14, 1e5),
    'UMi': (0.06, 1e5),
}

# Data for green circles
green_points = {
    'M32': (0.785, 2.5e6),
    'N5102': (3.4, 9e5),
    'N5206': (3.8, 5.4e5),
    'N4395': (4.3, 3.6e5),
    'N205': (0.824, 2e4),
}

# Data for orange square (G1)
orange_points = {
    'G1': (0.785, 1.7e4),
}

fig, ax = plt.subplots()

# Plot green circles
for name, (d, m) in green_points.items():
    ax.plot(d, m, marker='o', color='green', markersize=8)
    ax.text(d * 1.05, m, name, color='green', fontsize=10, va='center')

# Plot orange square
for name, (d, m) in orange_points.items():
    ax.plot(d, m, marker='s', color='orange', markersize=6)
    ax.text(d * 1.05, m, name, color='orange', fontsize=10, va='center')

# Plot blue arrows for upper limits
for name, (d, m) in upper_limits.items():
    # Arrow pointing down
    arrow_length = m * 0.3  # Adjust length for visibility in log scale
    ax.arrow(d, m + arrow_length, 0, -arrow_length, head_width=d*0.1, head_length=arrow_length*0.3, fc='blue', ec='blue', width=d*0.01)
    ax.text(d * 1.05, m + arrow_length, name, color='blue', fontsize=10, va='bottom')

# Set scales and limits
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-3, 2e1)
ax.set_ylim(1e2, 1e7)

# Labels
ax.set_xlabel('D (Mpc)')
ax.set_ylabel(r'$M_{BH} (M_{\odot})$')

plt.tight_layout()
# plt.savefig('mbh_distance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('mbh_distance.png', dpi=300, bbox_inches='tight')
# plt.show()

print("Figure saved to figures/mbh_distance.pdf and figures/mbh_distance.png")
