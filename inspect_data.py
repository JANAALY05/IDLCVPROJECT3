"""Quick look at the raw data: size, mode and values of one sample of each file type,
and all PH2 image sizes.  Run on the cluster:  python3 inspect_data.py"""
import os
from PIL import Image
import numpy as np

root = "/dtu/datasets1/02516/"
paths = [
    "PH2_Dataset_images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp",
    "PH2_Dataset_images/IMD002/IMD002_lesion/IMD002_lesion.bmp",
    "DRIVE/training/images/21_training.tif",
    "DRIVE/training/1st_manual/21_manual1.gif",
    "DRIVE/training/mask/21_training_mask.gif",
]

for p in paths:
    im = Image.open(root + p)
    a = np.array(im)
    print(p.split("/")[-1], im.size, im.mode, a.dtype, np.unique(a)[:10])

sizes = set()
ph2 = root + "PH2_Dataset_images/"
for case in os.listdir(ph2):
    sizes.add(Image.open(f"{ph2}{case}/{case}_Dermoscopic_Image/{case}.bmp").size)
print("PH2 sizes:", sizes)
