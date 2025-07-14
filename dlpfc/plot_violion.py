import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1.  ADD YOUR DATA HERE  (list of dicts per method)
# ---------------------------------------------------------------------------
records_unaligned = [  # Example for method 1
    {'Class-wise Dice Coefficient': 0.7706694788800533, 'Spatial Cross-Correlation': 0.8509856257687658,
     'Mean Centroid Shift': 18.74953803406123, 'Mutual Information': 1.3820463368508342, 'SSIM': 0.5248053157086725,
     'NCC': 0.795317801534296},
    {'Class-wise Dice Coefficient': 0.11463122456393122, 'Spatial Cross-Correlation': 0.14324634475338732,
     'Mean Centroid Shift': 61.60758265725576, 'Mutual Information': 0.8778307688456355, 'SSIM': 0.447884987338847,
     'NCC': 0.5520502375455478},
    {'Class-wise Dice Coefficient': 0.7767972169432995, 'Spatial Cross-Correlation': 0.8428419807050608,
     'Mean Centroid Shift': 9.970960611694322, 'Mutual Information': 1.287896825060816, 'SSIM': 0.4671519874211662,
     'NCC': 0.7550135772345167},
    {'Class-wise Dice Coefficient': 0.6421180695044131, 'Spatial Cross-Correlation': 0.744388183790651,
     'Mean Centroid Shift': 19.002098530295193, 'Mutual Information': 1.0253090512434362, 'SSIM': 0.49842117731288305,
     'NCC': 0.6805493669923863},
    {'Class-wise Dice Coefficient': 0.39622557647360745, 'Spatial Cross-Correlation': 0.46111385562852397,
     'Mean Centroid Shift': 36.51947251203146, 'Mutual Information': 1.017749394668657, 'SSIM': 0.46316997011472666,
     'NCC': 0.690357394474187},
    {'Class-wise Dice Coefficient': 0.7316147259630693, 'Spatial Cross-Correlation': 0.7966924944967634,
     'Mean Centroid Shift': 19.85088763943637, 'Mutual Information': 1.1599039029217497, 'SSIM': 0.42808179631461235,
     'NCC': 0.7570687648769131},
    {'Class-wise Dice Coefficient': 0.7847436979217163, 'Spatial Cross-Correlation': 0.8848197049890351, 'Mean Centroid Shift': 7.066182667905086, 'Mutual Information': 1.434284422993768, 'SSIM': 0.5937008014849005, 'NCC': 0.8194896499294927},
{'Class-wise Dice Coefficient': 0.5963123899756224, 'Spatial Cross-Correlation': 0.7157637564733832, 'Mean Centroid Shift': 18.3962622722477, 'Mutual Information': 1.2545409423185827, 'SSIM': 0.5279818800740347, 'NCC': 0.6814886026118167},
{'Class-wise Dice Coefficient': 0.6470452025802882, 'Spatial Cross-Correlation': 0.7363280589424893, 'Mean Centroid Shift': 20.019643890088382, 'Mutual Information': 1.3235882944455852, 'SSIM': 0.6004645471181943, 'NCC': 0.727987723631598}
]

records_method2 = [  # ← paste method-2 runs
    {'Class-wise Dice Coefficient': 0.7824620246941222, 'Spatial Cross-Correlation': 0.8807919382249251,
     'Mean Centroid Shift': 13.931910564863152, 'Mutual Information': 1.188362047467269, 'SSIM': 0.65641594,
     'NCC': 0.4269155},
    {'Class-wise Dice Coefficient': 0.20808948311143935, 'Spatial Cross-Correlation': 0.27810118825731855,
     'Mean Centroid Shift': 49.99093630370655, 'Mutual Information': 0.9631767131596618, 'SSIM': 0.45199883,
     'NCC': -0.16340454},
    {'Class-wise Dice Coefficient': 0.754082015663007, 'Spatial Cross-Correlation': 0.8265979355653961,
     'Mean Centroid Shift': 11.342567348364973, 'Mutual Information': 1.1586558430712035, 'SSIM': 0.60669285,
     'NCC': 0.7047028},
    {'Class-wise Dice Coefficient': 0.8208254286685619, 'Spatial Cross-Correlation': 0.8952868339011065,
     'Mean Centroid Shift': 7.2160061082719995, 'Mutual Information': 0.9453672113073317, 'SSIM': 0.74381113,
     'NCC': 0.76006925},
    {'Class-wise Dice Coefficient': 0.30962499435561497, 'Spatial Cross-Correlation': 0.3346076182231942,
     'Mean Centroid Shift': 35.90279783373902, 'Mutual Information': 0.816220715062844, 'SSIM': 0.5640429,
     'NCC': -0.07646794},
    {'Class-wise Dice Coefficient': 0.8300972377106082, 'Spatial Cross-Correlation': 0.8885980457847603,
     'Mean Centroid Shift': 8.762618324249996, 'Mutual Information': 1.0139748291672426, 'SSIM': 0.68627065,
     'NCC': 0.37944272},
    {'Class-wise Dice Coefficient': 0.7839487847238058, 'Spatial Cross-Correlation': 0.8799881335734634,
     'Mean Centroid Shift': 9.320081250603058, 'Mutual Information': 1.1196982628631424, 'SSIM': 0.71171093,
     'NCC': 0.60242075},
    {'Class-wise Dice Coefficient': 0.5312555819244257, 'Spatial Cross-Correlation': 0.5564028422879466,
     'Mean Centroid Shift': 24.95881248769815, 'Mutual Information': 0.9880174472873193, 'SSIM': 0.66407824,
     'NCC': 0.23007031},
    {'Class-wise Dice Coefficient': 0.6375776677297234, 'Spatial Cross-Correlation': 0.7234530678831332,
     'Mean Centroid Shift': 11.93194609609825, 'Mutual Information': 1.0096016171781983, 'SSIM': 0.66655993,
     'NCC': 0.21412154}
]

records_method3 = [  # ← paste method-3 runs
    {'Class-wise Dice Coefficient': 0.697097111416658, 'Spatial Cross-Correlation': 0.7933371425065637,
     'Mean Centroid Shift': 15.820941723896404, 'Mutual Information': 1.3949588394837455, 'SSIM': 0.5399693235934527,
     'NCC': 0.11126194776811377},
    {'Class-wise Dice Coefficient': 0.107468983294814, 'Spatial Cross-Correlation': 0.13361384079117225,
     'Mean Centroid Shift': 62.34542663284823, 'Mutual Information': 0.9302239554678611, 'SSIM': 0.5221562747798366,
     'NCC': 0.530169710812968},
    {'Class-wise Dice Coefficient': 0.7991034292391566, 'Spatial Cross-Correlation': 0.8729218878515125,
     'Mean Centroid Shift': 7.614881838057851, 'Mutual Information': 1.3873508045316925, 'SSIM': 0.5276221866474097,
     'NCC': 0.3564169010940287},
    {'Class-wise Dice Coefficient': 0.8332897860967121, 'Spatial Cross-Correlation': 0.9056254275368755,
     'Mean Centroid Shift': 8.515710983293044, 'Mutual Information': 1.1807563304901585, 'SSIM': 0.5199400110147847,
     'NCC': 0.3014979554594893},
    {'Class-wise Dice Coefficient': 0.39962336889172184, 'Spatial Cross-Correlation': 0.4659818202938088,
     'Mean Centroid Shift': 35.898453829427915, 'Mutual Information': 1.1563677175822473, 'SSIM': 0.5578553056151104,
     'NCC': 0.7138243000418218},
    {'Class-wise Dice Coefficient': 0.7890720336092862, 'Spatial Cross-Correlation': 0.8458426986426222,
     'Mean Centroid Shift': 16.665881255384527, 'Mutual Information': 1.3116357907830452, 'SSIM': 0.5225428004429241,
     'NCC': 0.4169805285079389},
    {'Class-wise Dice Coefficient': 0.7921356135318666, 'Spatial Cross-Correlation': 0.8894440276355416,
     'Mean Centroid Shift': 9.04116332779755, 'Mutual Information': 1.4868470925126633, 'SSIM': 0.5950849440565368,
     'NCC': 0.5494444582355821},
    {'Class-wise Dice Coefficient': 0.6807587058730987, 'Spatial Cross-Correlation': 0.7933128898829462,
     'Mean Centroid Shift': 15.307510442650072, 'Mutual Information': 1.4240206975608682, 'SSIM': 0.5698511903214923,
     'NCC': 0.6773519290711113},
    {'Class-wise Dice Coefficient': 0.6814013575794249, 'Spatial Cross-Correlation': 0.7751249916732605,
     'Mean Centroid Shift': 16.44096969809178, 'Mutual Information': 1.4581642897300284, 'SSIM': 0.5913246512185228,
     'NCC': 0.6797940123365019}
]

records_method4 = [  # ← paste method-4 runs
    {'Class-wise Dice Coefficient': 0.778548598938323, 'Spatial Cross-Correlation': 0.847257519660881,
     'Mean Centroid Shift': 15.37583379833727, 'Mutual Information': 1.8964008531765444, 'SSIM': 0.6875019,
     'NCC': 0.9499311},
    {'Class-wise Dice Coefficient': 0.43897513545087974, 'Spatial Cross-Correlation': 0.5015286911078461,
     'Mean Centroid Shift': 40.27381821336137, 'Mutual Information': 1.4830741154959526, 'SSIM': 0.65045774,
     'NCC': 0.8376941},
    {'Class-wise Dice Coefficient': 0.8614903192437655, 'Spatial Cross-Correlation': 0.917791115006162,
     'Mean Centroid Shift': 6.921284045839256, 'Mutual Information': 1.8938317051935716, 'SSIM': 0.67124414,
     'NCC': 0.9435308},
    {'Class-wise Dice Coefficient': 0.8329201810017324, 'Spatial Cross-Correlation': 0.8927792163616715,
     'Mean Centroid Shift': 8.06536475234555, 'Mutual Information': 1.4921405127491116, 'SSIM': 0.64439636,
     'NCC': 0.88875735},
    {'Class-wise Dice Coefficient': 0.5083283400799038, 'Spatial Cross-Correlation': 0.5519129369689248,
     'Mean Centroid Shift': 31.34751218719503, 'Mutual Information': 1.4753123555182128, 'SSIM': 0.6245076,
     'NCC': 0.872972},
    {'Class-wise Dice Coefficient': 0.8265907600179606, 'Spatial Cross-Correlation': 0.8704485543050262,
     'Mean Centroid Shift': 12.777310889847453, 'Mutual Information': 1.625196180692683, 'SSIM': 0.61052364,
     'NCC': 0.90897065},
    {'Class-wise Dice Coefficient': 0.8037419676527413, 'Spatial Cross-Correlation': 0.882981066479898,
     'Mean Centroid Shift': 8.001422411716812, 'Mutual Information': 1.7820796436154573, 'SSIM': 0.74472046,
     'NCC': 0.9327356},
    {'Class-wise Dice Coefficient': 0.6653928478884487, 'Spatial Cross-Correlation': 0.7416033908646291,
     'Mean Centroid Shift': 16.225295620788245, 'Mutual Information': 1.6934859057155904, 'SSIM': 0.6938059,
     'NCC': 0.87111276},
    {'Class-wise Dice Coefficient': 0.7519668953841185, 'Spatial Cross-Correlation': 0.8008013239373389,
     'Mean Centroid Shift': 16.47633025371601, 'Mutual Information': 1.7920108342426708, 'SSIM': 0.6875827,
     'NCC': 0.91063595}
]

# Put all methods into a list (only include the ones you have)
all_methods = [
    ("Unaligned",  records_unaligned),
    ("Method-2",   records_method2),
    ("Method-3",   records_method3),
    ("Method-4",   records_method4),
]

# ---------------------------------------------------------------------------
# 2.  PLOT SETTINGS
# ---------------------------------------------------------------------------
method_palette = {        # customise colours as you like
    "Unaligned": "#4C72B0",
    "Method-2" : "#55A868",
    "Method-3" : "#C44E52",
    "Method-4" : "#8172B3",
}
groups = [
    ["Class-wise Dice Coefficient", "Spatial Cross-Correlation"],  # subplot-1
    ["Mean Centroid Shift"],                                       # subplot-2
    ["Mutual Information", "SSIM", "NCC"]                          # subplot-3
]
titles = ["Dice & SCC", "Mean Centroid Shift", "MI, SSIM & NCC"]

# horizontal offsets so methods don't overlap (max 4 methods)
offsets = np.linspace(-0.25, 0.25, len(all_methods))

def draw_violin(ax, data, pos, color):
    """One violin at x-pos with alpha+points."""
    vp = ax.violinplot(data, positions=[pos], vert=True,
                       showmeans=False, showmedians=False, widths=0.35)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.5)
    for part in ("cbars", "cmins", "cmaxes"):
        if part in vp: vp[part].set_alpha(0)
    jitter = (np.random.rand(len(data)) - 0.5) * 0.15
    ax.scatter(np.full_like(data, pos) + jitter, data,
               facecolor=color, edgecolor="black", s=60, zorder=3)

# ---------------------------------------------------------------------------
# 3.  BUILD THE FIGURE
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, metric_group, title in zip(axes, groups, titles):
    for m_idx, (m_name, recs) in enumerate(all_methods):
        # Skip empty method slots
        if not recs:
            continue
        metric_vals = {k: np.array([r[k] for r in recs]) for k in recs[0]}
        for g_idx, metric in enumerate(metric_group):
            draw_violin(
                ax,
                metric_vals[metric],
                g_idx + offsets[m_idx],      # base position + method offset
                method_palette[m_name]
            )
    ax.set_xticks(range(len(metric_group)))
    ax.set_xticklabels(metric_group, rotation=20, ha="right")
    ax.set_xlim(-0.7, len(metric_group)-0.3)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

# ----------------------  OPTIONAL LEGEND  ----------------------------------
handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                      markersize=10, markerfacecolor=method_palette[n],
                      markeredgecolor='black') for n, _ in all_methods if _]
labels  = [n for n, _ in all_methods if _]
fig.legend(handles, labels, loc="upper center", ncol=len(labels),
           bbox_to_anchor=(0.5, 1.05), frameon=False)

plt.tight_layout()
plt.show()
