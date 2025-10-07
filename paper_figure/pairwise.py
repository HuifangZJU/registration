import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppose you already collected the data into a nested dictionary:
# metrics[method] = list of dicts for each pair
metrics = {
    "Unaligned": [{'Class-wise Dice Coefficient': 0.7768701441967623, 'Mean Centroid Shift': 4.976276719154381, 'Spatial Cross-Correlation': 0.8917915757845025},
{'Class-wise Dice Coefficient': 0.11381712833303122, 'Mean Centroid Shift': 17.009418566749925, 'Spatial Cross-Correlation': 0.17260286833376343},
{'Class-wise Dice Coefficient': 0.7716638336154529, 'Mean Centroid Shift': 2.929752233430087, 'Spatial Cross-Correlation': 0.8919177212989803},
{'Class-wise Dice Coefficient': 0.6537677193788716, 'Mean Centroid Shift': 4.936118497208948, 'Spatial Cross-Correlation': 0.803405981553255},
{'Class-wise Dice Coefficient': 0.4283162146232627, 'Mean Centroid Shift': 9.27682541771767, 'Spatial Cross-Correlation': 0.5113613571816132},
{'Class-wise Dice Coefficient': 0.7318337194104668, 'Mean Centroid Shift': 4.528260956071636, 'Spatial Cross-Correlation': 0.8447927863789882},
{'Class-wise Dice Coefficient': 0.7878531890570014, 'Mean Centroid Shift': 1.7715230369593875, 'Spatial Cross-Correlation': 0.9343656859791082},
{'Class-wise Dice Coefficient': 0.6101366111033418, 'Mean Centroid Shift': 5.459304507548953, 'Spatial Cross-Correlation': 0.7909307095263987},
{'Class-wise Dice Coefficient': 0.6665280823227714, 'Mean Centroid Shift': 5.586649501530565, 'Spatial Cross-Correlation': 0.8059942839801577}],
    "SimpleITK": [{'Class-wise Dice Coefficient': 0.7833975254420932, 'Mean Centroid Shift': 4.418763648308256, 'Spatial Cross-Correlation': 0.9183695973916125},
{'Class-wise Dice Coefficient': 0.25110088340415593, 'Mean Centroid Shift': 11.914192092888815, 'Spatial Cross-Correlation': 0.34573729143248855},
{'Class-wise Dice Coefficient': 0.8563863168425763, 'Mean Centroid Shift': 2.0246699346403885, 'Spatial Cross-Correlation': 0.9521977660401664},
{'Class-wise Dice Coefficient': 0.8062835813663058, 'Mean Centroid Shift': 2.827141280874249, 'Spatial Cross-Correlation': 0.9225488701819545},
{'Class-wise Dice Coefficient': 0.32002360867733515, 'Mean Centroid Shift': 10.430159878601744, 'Spatial Cross-Correlation': 0.3998089616638382},
{'Class-wise Dice Coefficient': 0.8296217413177859, 'Mean Centroid Shift': 2.6396156301541445, 'Spatial Cross-Correlation': 0.9206647976199125},
{'Class-wise Dice Coefficient': 0.7841290805491005, 'Mean Centroid Shift': 2.9619820290802514, 'Spatial Cross-Correlation': 0.9167323281220511},
{'Class-wise Dice Coefficient': 0.5312753818775872, 'Mean Centroid Shift': 9.091159156833992, 'Spatial Cross-Correlation': 0.6273503570426324},
{'Class-wise Dice Coefficient': 0.6583069467923091, 'Mean Centroid Shift': 4.8180530122517595, 'Spatial Cross-Correlation': 0.8304774212294392}],
    "PASTE": [{'Class-wise Dice Coefficient': 0.7040958310137437, 'Mean Centroid Shift': 4.189968705015194, 'Spatial Cross-Correlation': 0.8556616480838272},
{'Class-wise Dice Coefficient': 0.10818847600247591, 'Mean Centroid Shift': 17.217823427139464, 'Spatial Cross-Correlation': 0.16307149568678472},
{'Class-wise Dice Coefficient': 0.7960658681741369, 'Mean Centroid Shift': 2.289769561068192, 'Spatial Cross-Correlation': 0.9178825404931378},
{'Class-wise Dice Coefficient': 0.8361504343835273, 'Mean Centroid Shift': 2.5285479805214663, 'Spatial Cross-Correlation': 0.9439317639232699},
{'Class-wise Dice Coefficient': 0.4321909117485333, 'Mean Centroid Shift': 9.12809492797445, 'Spatial Cross-Correlation': 0.5139023829090827},
{'Class-wise Dice Coefficient': 0.7894719351368391, 'Mean Centroid Shift': 3.8514742921784686, 'Spatial Cross-Correlation': 0.881397512581264},
{'Class-wise Dice Coefficient': 0.7999391481633978, 'Mean Centroid Shift': 2.1131740915165556, 'Spatial Cross-Correlation': 0.9394717886796827},
{'Class-wise Dice Coefficient': 0.6902331870585445, 'Mean Centroid Shift': 4.7729695833601005, 'Spatial Cross-Correlation': 0.8599983379125758},
{'Class-wise Dice Coefficient': 0.7018269880242143, 'Mean Centroid Shift': 4.722761411230415, 'Spatial Cross-Correlation': 0.837620656131101}],
    "GPSA": [{'Class-wise Dice Coefficient': 0.7494602836312754, 'Mean Centroid Shift': 3.3679885997370023, 'Spatial Cross-Correlation': 0.8004786584569772},
{'Class-wise Dice Coefficient': 0.05582964713916009, 'Mean Centroid Shift': 26.36439588323585, 'Spatial Cross-Correlation': 0.08406038304536334},
{'Class-wise Dice Coefficient': 0.7784622362973723, 'Mean Centroid Shift': 2.216172686367079, 'Spatial Cross-Correlation': 0.8122318422089396},
{'Class-wise Dice Coefficient': 0.7773401679205533, 'Mean Centroid Shift': 2.694080869316649, 'Spatial Cross-Correlation': 0.7590354465960457},
{'Class-wise Dice Coefficient': 0.47625762684152145, 'Mean Centroid Shift': 10.773872213339716, 'Spatial Cross-Correlation': 0.4326733029481131},
{'Class-wise Dice Coefficient': 0.5421414763702532, 'Mean Centroid Shift': 7.761855679800493, 'Spatial Cross-Correlation': 0.5265672506794438},
{'Class-wise Dice Coefficient': 0.7350801510537365, 'Mean Centroid Shift': 2.8887605884213414, 'Spatial Cross-Correlation': 0.7925270640811112},
{'Class-wise Dice Coefficient': 0.6824664900265718, 'Mean Centroid Shift': 3.9493465791876536, 'Spatial Cross-Correlation': 0.7383734192098135},
{'Class-wise Dice Coefficient': 0.44136501362663044, 'Mean Centroid Shift': 5.115700825587744, 'Spatial Cross-Correlation': 0.5032254910537841}],
    "SANTO":[{'Class-wise Dice Coefficient': 0.7816913102183762, 'Mean Centroid Shift': 4.345158011539273, 'Spatial Cross-Correlation': 0.9101451069268085},
{'Class-wise Dice Coefficient': 0.23650635215612809, 'Mean Centroid Shift': 14.163979684116173, 'Spatial Cross-Correlation': 0.36165657544332674},
{'Class-wise Dice Coefficient': 0.608325658891023, 'Mean Centroid Shift': 3.984241376804729, 'Spatial Cross-Correlation': 0.7686263288943002},
{'Class-wise Dice Coefficient': 0.7010844049751929, 'Mean Centroid Shift': 5.895678185864474, 'Spatial Cross-Correlation': 0.840170129722533},
{'Class-wise Dice Coefficient': 0.3983088613029944, 'Mean Centroid Shift': 11.93103809358755, 'Spatial Cross-Correlation': 0.5742424748664229},
{'Class-wise Dice Coefficient': 0.7211740260957658, 'Mean Centroid Shift': 5.076087120186747, 'Spatial Cross-Correlation': 0.8334661064873494},
{'Class-wise Dice Coefficient': 0.6688005206900023, 'Mean Centroid Shift': 2.865206288646258, 'Spatial Cross-Correlation': 0.8594946537923862},
{'Class-wise Dice Coefficient': 0.36717360572833113, 'Mean Centroid Shift': 17.202382624689964, 'Spatial Cross-Correlation': 0.504426254030043},
{'Class-wise Dice Coefficient': 0.42328996928637724, 'Mean Centroid Shift': 11.257182552133328, 'Spatial Cross-Correlation': 0.5797351561303347}],
    "Voxelmorph": [{'Class-wise Dice Coefficient': 0.7795561795051488, 'Mean Centroid Shift': 5.1253359492007595, 'Spatial Cross-Correlation': 0.8937467284098428},
{'Class-wise Dice Coefficient': 0.14499755929988722, 'Mean Centroid Shift': 16.00478398589941, 'Spatial Cross-Correlation': 0.21835477757970378},
{'Class-wise Dice Coefficient': 0.7750418653534643, 'Mean Centroid Shift': 2.8498043380984224, 'Spatial Cross-Correlation': 0.8993167477513262},
{'Class-wise Dice Coefficient': 0.6794128909565785, 'Mean Centroid Shift': 4.301885217175544, 'Spatial Cross-Correlation': 0.8192170916594819},
{'Class-wise Dice Coefficient': 0.434514513106455, 'Mean Centroid Shift': 8.53713207430587, 'Spatial Cross-Correlation': 0.5289327977567757},
{'Class-wise Dice Coefficient': 0.7485656869080397, 'Mean Centroid Shift': 4.080346668215727, 'Spatial Cross-Correlation': 0.8525145478212405},
{'Class-wise Dice Coefficient': 0.8334551288461233, 'Mean Centroid Shift': 1.7113595231208656, 'Spatial Cross-Correlation': 0.9445793139506543},
{'Class-wise Dice Coefficient': 0.6174440117188598, 'Mean Centroid Shift': 7.776861991560051, 'Spatial Cross-Correlation': 0.7696003689748874},
{'Class-wise Dice Coefficient': 0.7066427583167966, 'Mean Centroid Shift': 5.16325231286924, 'Spatial Cross-Correlation': 0.8286765462716259}],
    "Nicetrans": [{'Class-wise Dice Coefficient': 0.7849618363820051, 'Mean Centroid Shift': 4.395024783982191, 'Spatial Cross-Correlation': 0.9151299290908885},
{'Class-wise Dice Coefficient': 0.43148565294410474, 'Mean Centroid Shift': 9.863630864621882, 'Spatial Cross-Correlation': 0.5855547357279207},
{'Class-wise Dice Coefficient': 0.6373740301973182, 'Mean Centroid Shift': 4.018304982062853, 'Spatial Cross-Correlation': 0.761783045293073},
{'Class-wise Dice Coefficient': 0.8382103982522489, 'Mean Centroid Shift': 2.6150792908092373, 'Spatial Cross-Correlation': 0.9369228192955502},
{'Class-wise Dice Coefficient': 0.3486122976238098, 'Mean Centroid Shift': 10.497665353018046, 'Spatial Cross-Correlation': 0.4138190657482054},
{'Class-wise Dice Coefficient': 0.8264631005381812, 'Mean Centroid Shift': 2.552232502424565, 'Spatial Cross-Correlation': 0.9101677957924273},
{'Class-wise Dice Coefficient': 0.722577967578163, 'Mean Centroid Shift': 3.594866832893564, 'Spatial Cross-Correlation': 0.8602561970757167},
{'Class-wise Dice Coefficient': 0.5644462450048818, 'Mean Centroid Shift': 6.688049828433136, 'Spatial Cross-Correlation': 0.7214956687497475},
{'Class-wise Dice Coefficient': 0.7583411976612376, 'Mean Centroid Shift': 4.255785578950518, 'Spatial Cross-Correlation': 0.8699993008669378},
{'Class-wise Dice Coefficient': 0.6569414140202167, 'Spatial Cross-Correlation': 0.7750142841822741, 'Mean Centroid Shift': 5.386737779688443}],
    "Ours": [{'Class-wise Dice Coefficient': 0.7649853362834426, 'Mean Centroid Shift': 4.0879035301490845, 'Spatial Cross-Correlation': 0.8853936492587853},
{'Class-wise Dice Coefficient': 0.5359967816169615, 'Mean Centroid Shift': 9.405747135002027, 'Spatial Cross-Correlation': 0.6707080268713957},
{'Class-wise Dice Coefficient': 0.8134206779364644, 'Mean Centroid Shift': 2.1230904347281125, 'Spatial Cross-Correlation': 0.910560097230335},
{'Class-wise Dice Coefficient': 0.8202323499960318, 'Mean Centroid Shift': 2.925141250636122, 'Spatial Cross-Correlation': 0.9183722165348304},
{'Class-wise Dice Coefficient': 0.7373385538259113, 'Mean Centroid Shift': 2.491250730768191, 'Spatial Cross-Correlation': 0.845693875594532},
{'Class-wise Dice Coefficient': 0.8284548685484318, 'Mean Centroid Shift': 2.0276576692470236, 'Spatial Cross-Correlation': 0.8811760521574838},
{'Class-wise Dice Coefficient': 0.8240488470205747, 'Mean Centroid Shift': 2.392767163297619, 'Spatial Cross-Correlation': 0.9292669885203699},
{'Class-wise Dice Coefficient': 0.7051723351111479, 'Mean Centroid Shift': 5.098329820960093, 'Spatial Cross-Correlation': 0.822714573705074},
{'Class-wise Dice Coefficient': 0.7722937180849898, 'Mean Centroid Shift': 4.7932225102606525, 'Spatial Cross-Correlation': 0.8493225365753668},
{'Class-wise Dice Coefficient': 0.7557714964915506, 'Spatial Cross-Correlation': 0.8570231129386859, 'Mean Centroid Shift': 3.9272344716721026}]
}


plt.rcParams.update({
    'font.size': 20,        # base font size
    'axes.titlesize': 22,   # title size
    'axes.labelsize': 20,   # x/y label size
    'xtick.labelsize': 18,  # x tick labels
    'ytick.labelsize': 18,  # y tick labels
    'legend.fontsize': 18,  # legend text
    'legend.title_fontsize': 20
})

base_colors = ['#dcd2da','#376795', '#528fad', '#aadce0', '#ffe6b7', '#ffd06f', '#ef8a47']
ours_color = '#e76254'
# Flatten into DataFrame
rows = []
for method, dict_list in metrics.items():
    for i, d in enumerate(dict_list[:9]):
        rows.append({
            "Method": method,
            "Pair": f"Pair{i+1}",
            "Dice": d["Class-wise Dice Coefficient"],
            "CentroidShift": d["Mean Centroid Shift"],
            "CrossCorr": d["Spatial Cross-Correlation"]
        })
df = pd.DataFrame(rows)

# Assign each method a unique color
methods = list(df["Method"].unique())
colors = {}

# Fill purples in order, reserve green for Ours
purple_iter = iter(base_colors)
for m in methods:
    if m == "Ours":
        colors[m] = ours_color
    else:
        colors[m] = next(purple_iter)

# --- Example: Dice barplot ---
fig, ax = plt.subplots(figsize=(21,7))
pairs = sorted(df["Pair"].unique())
x = np.arange(len(pairs))
width = 0.8 / len(methods)

for j, m in enumerate(methods):
    vals = df[df["Method"]==m]["CrossCorr"].values
    ax.bar(x + j*width, vals, width, label=m, color=colors[m])

ax.set_xticks(x + width*len(methods)/2)
ax.set_xticklabels(pairs, rotation=45)
# ax.set_ylabel("Class-wise Dice Coefficient")
# ax.set_title("Pairwise Dice Comparison Across Methods")
# ax.set_ylabel("Mean Centroid Shift")
# ax.set_title("Pairwise Mean Centroid Shift Across Methods")
ax.set_ylabel("Spatial Cross-Correlation")
ax.set_title("Pairwise Spatial Cross-Correlation Across Methods")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
