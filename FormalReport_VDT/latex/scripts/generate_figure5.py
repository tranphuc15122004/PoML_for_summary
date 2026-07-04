"""
Generate Figure 5: Baseline-relative improvement heatmap.
Two panels: left = Base+SFT+GRPOv5, right = Instruct+SFT+GRPOv5.
Rows = baselines, columns = 4 datasets × 2 metrics (ΔR2, LenRed).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Raw data from Tables 1 & 2 ──────────────────────────────────────────────

datasets = ['VietNews', 'WikiLingua', 'ViMs', 'VLSP']

# System ROUGE-2
base_system_r2  = [28.97, 26.35, 18.16, 17.46]
instruct_system_r2 = [29.62, 25.59, 21.27, 24.39]

# System LenDist (lower is better)
base_system_ld  = [1.16, 8.35, 89.23, 100.50]
instruct_system_ld = [1.52, 10.25, 84.45, 82.71]

# Baseline ROUGE-2: rows in order
baseline_names = [
    'Qwen3-4B-Base (raw)',
    'Qwen3-4B-Instruct (raw)',
    'VietAI',
    'GPT-4o',
    'GPT-3.5-Turbo',
    'Qwen3-14B',
    'Llama3.3-70B-Instr.',
    'Phi4-14B',
    'Sailor-20B-chat',
    'VinBigdata-7B',
]

baseline_r2 = [
    [12.13,  3.18, 13.46,  9.49],   # Base raw
    [16.48,  7.44, 17.83, 15.51],   # Instruct raw
    [34.24, 33.12,   None,  None],   # VietAI
    [21.61, 20.65, 44.26, 43.37],   # GPT-4o
    [28.13, 21.09, 19.28, 35.79],   # GPT-3.5-Turbo
    [18.51, 20.24, 44.38, 44.27],   # Qwen3-14B
    [22.17, 19.40, 37.54, 39.02],   # Llama3.3-70B
    [13.17, 14.28, 41.85, 42.31],   # Phi4-14B
    [17.70, 18.74, 39.47, 40.96],   # Sailor-20B
    [20.59, 21.02, 37.98, 40.23],   # VinBigdata-7B
]

baseline_ld = [
    [33.03, 61.59, 92.01, 120.33],  # Base raw
    [4.01,  23.42, 59.80,  70.17],  # Instruct raw
    [None,   None,  None,   None],   # VietAI
    [8,     11,    187,    58],      # GPT-4o
    [18,    17,    47,     30],      # GPT-3.5-Turbo
    [50,    45,    131,    34],      # Qwen3-14B
    [14,    13,    186,    73],      # Llama3.3-70B
    [205,   214,   166,    134],     # Phi4-14B
    [38,    43,    268,    41],      # Sailor-20B
    [115,   89,    98,     82],      # VinBigdata-7B
]

# ── Compute ΔR2 and LenRed ──────────────────────────────────────────────────

def compute_deltas(system_r2, system_ld, baseline_r2_rows, baseline_ld_rows):
    """Return two 2D lists: dr2[n_baseline][n_dataset], lenred[n_baseline][n_dataset]."""
    n_rows = len(baseline_r2_rows)
    n_cols = len(datasets)
    dr2 = [[None]*n_cols for _ in range(n_rows)]
    lenred = [[None]*n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            # ΔR2 = (system - baseline) / baseline * 100
            b_r2 = baseline_r2_rows[i][j]
            if b_r2 is not None and b_r2 != 0:
                dr2[i][j] = (system_r2[j] - b_r2) / b_r2 * 100
            # LenRed = (baseline - system) / baseline * 100  (positive = improvement)
            b_ld = baseline_ld_rows[i][j]
            if b_ld is not None and b_ld != 0:
                lenred[i][j] = (b_ld - system_ld[j]) / b_ld * 100
    return dr2, lenred

dr2_base, lenred_base = compute_deltas(base_system_r2, base_system_ld, baseline_r2, baseline_ld)
dr2_instruct, lenred_instruct = compute_deltas(instruct_system_r2, instruct_system_ld, baseline_r2, baseline_ld)

# ── Build combined data arrays ──────────────────────────────────────────────
# Columns: for each dataset -> (ΔR2, LenRed)
# So indices: 0=VN_ΔR2, 1=VN_LenRed, 2=WL_ΔR2, 3=WL_LenRed, 4=ViMs_ΔR2, ...

def interleave(dr2, lenred):
    """Interleave ΔR2 and LenRed columns: [dr2[0], lenred[0], dr2[1], lenred[1], ...]"""
    n_rows = len(dr2)
    n_cols = len(dr2[0])
    out = [[None]*(2*n_cols) for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            out[i][2*j]   = dr2[i][j]
            out[i][2*j+1] = lenred[i][j]
    return out

data_base = interleave(dr2_base, lenred_base)
data_instruct = interleave(dr2_instruct, lenred_instruct)

col_labels = []
for ds in datasets:
    col_labels.append(f'{ds}\nΔR2')
    col_labels.append(f'{ds}\nLenRed')

# ── Clipping for color scaling ──────────────────────────────────────────────
# Most values lie within ±200%; extreme outliers (raw Qwen3 baselines on
# WikiLingua ROUGE-2, ~+729%) are saturated at the clip boundary.
# The exact value is still shown in the cell text.
CLIP = 200  # clip |value| > 200% for color mapping; text shows real value

def clip_data(data):
    clipped = []
    for row in data:
        clipped.append([max(-CLIP, min(CLIP, v)) if v is not None else None for v in row])
    return clipped

plot_data_base = clip_data(data_base)
plot_data_instruct = clip_data(data_instruct)

# Print data range for diagnostics
def find_extremes(data):
    vals = []
    for row in data:
        for v in row:
            if v is not None:
                vals.append(v)
    return min(vals), max(vals)
min_b, max_b = find_extremes(data_base)
min_i, max_i = find_extremes(data_instruct)
print(f'Base range: [{min_b:.1f}%, {max_b:.1f}%], '
      f'Instruct range: [{min_i:.1f}%, {max_i:.1f}%], CLIP={CLIP}')

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1, 1]})

titles = [
    'Qwen3-4B-Base + SFT + GRPO v5',
    'Qwen3-4B-Instruct + SFT + GRPO v5',
]

for ax, data, title in zip(axes, [plot_data_base, plot_data_instruct], titles):
    n_rows, n_cols = len(data), len(data[0])
    
    # Create numeric array with NaN for missing
    arr = np.full((n_rows, n_cols), np.nan)
    for i in range(n_rows):
        for j in range(n_cols):
            if data[i][j] is not None:
                arr[i, j] = data[i][j]
    
    # Softer diverging colormap: teal (positive) -> white -> coral (negative)
    norm = mcolors.TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)
    # BrBG is a perceptually柔和 (soft) brown-blue-green diverging map
    cmap = plt.cm.BrBG  # brown=negative, blue-green=positive
    
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect='auto')
    
    # Grid lines
    # Vertical separators between datasets (after every 2 columns)
    for sep in range(2, n_cols, 2):
        ax.axvline(x=sep-0.5, color='white', linewidth=2)
    # Horizontal separators after the two raw Qwen3 rows
    ax.axhline(y=1.5, color='white', linewidth=2, linestyle='-')
    
    # Text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i][j]
            if val is None:
                # Gray cell
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           fill=True, color='0.85', ec='none'))
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='0.5')
            else:
                color = 'white' if abs(val) > 60 else 'black'
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)
    
    # Tick labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, ha='center')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(baseline_names, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('% change', fontsize=9)

# Overall title
fig.suptitle('Baseline-Relative Improvement of Final Project Systems',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/scratch/jp09/dd9648/PoML_for_summary/FormalReport_VDT/latex/figures/section52_tradeoff.pdf',
            bbox_inches='tight', dpi=200)
print('Figure saved successfully.')
