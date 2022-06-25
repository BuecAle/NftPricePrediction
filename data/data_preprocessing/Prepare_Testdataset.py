import re
from pathlib import Path
import numpy as np
from os import rename
import imageio
import parameter


# Set directory
directory = parameter.Prepare_Testdataset.dir

# Insert pattern for asset id in filename (e.g. _0.02_1234568976_.jpg)
pattern = r'_(\d+)_'
p = re.compile(pattern)

# Sort and rename images best on brightness
def sort_images(path, thrshld):
    for img in Path(path).iterdir():
        # print(img.name)
        if not img.is_file():
            continue
        image = imageio.imread(img, as_gray=True)
        r = p.search(img.name)
        asset_id = r.group(1)
        fname = '_' + asset_id + '_.jpg'
        is_light = np.mean(image) > thrshld
        if is_light:
            rename(img, path + '/_100' + fname)
        else:
            rename(img, path + '/_0' + fname)


sort_images(directory, 110)

