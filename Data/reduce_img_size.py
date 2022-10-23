from PIL import Image
import os
import glob
"""
Loading images during training consumes a lot of time if it's size is big
"""

def reduce_size(path,target_size=(256,256)):
    foo = Image.open(path) 
    foo = foo.resize(target_size,Image.ANTIALIAS)    
    foo.save(path, optimize=True, quality=95)

def reduce_all_images(data_version):
    data_path ="Data"
    for folder in os.listdir(os.path.join(data_path, data_version))[:-1]: # -1 because we want to exclude test folder
        path = os.path.join(data_path, data_version, folder)
        origin_path = os.path.join(path, "origin")
        origin_paths = []
        origin_types = (origin_path+"/*.jpeg",origin_path+"/*.jpg")

        for type in origin_types:
            for pat in glob.glob(type):
                origin_paths.append(pat)
        origin_path = origin_paths[0] #It's one image so no problem
        reduce_size(origin_path)
        image_types = (path+"/*.jpeg",path+"/*.jpg")
        images_paths = []
        for type in image_types:
            for pat in glob.glob(type):
                images_paths.append(pat)
        
        for image_path in images_paths:
            reduce_size(image_path)


#reduce_all_images("3-10-22")