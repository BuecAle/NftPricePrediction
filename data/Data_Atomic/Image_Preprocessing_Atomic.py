import pathlib
import imghdr
import os
from PIL import Image

dir = r'C:\Users\BuecAle\Documents\GitHub\NFTPricePrediction\data\Data_Atomic\images'
data_dir = pathlib.Path(dir)
image_extensions = [".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

counter_bad = 0
counter_good = 0
for filepath in pathlib.Path(data_dir).rglob("*.*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        print(counter_good)
        if img_type == "jpeg":
            counter_good += 1
        if img_type != "jpeg":
            try:
                input_image = Image.open(filepath).convert('RGB')
                os.remove(filepath)
                input_image.save(str(filepath)[:-4] + ".jpg", "jpeg")
                counter_good += 1
            except:
                counter_bad += 1
                os.remove(filepath)
            test = True
        elif img_type is None:
            counter_bad += 1
            print(f"{filepath} is not an image")
            os.remove(filepath)
        elif img_type not in img_type_accepted_by_tf:
            counter_bad += 1
            os.remove(filepath)
            print(f"{filepath.name} is a {img_type}, not accepted by TensorFlow")
print(str(counter_good), str(counter_bad))
