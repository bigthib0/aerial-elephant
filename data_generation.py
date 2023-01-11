# tiré du code de ....

# import tensorflow.keras
import os
import numpy as np
import cv2
import pandas as pd
import shutil
from scipy.ndimage.filters import gaussian_filter


FOLDERPATH = "./AED"
DEBUG = False
SIGMA_ELEPHANT = 10

def read_coordinates(img_id, path):
    df_coordinates = pd.read_csv(path)
    
    img_df = df_coordinates[df_coordinates['tid'] == os.path.splitext(img_id)[0]]
    if DEBUG:
        print("Function: read_coordinates() {}".format(len(img_df)))
    x_coord = img_df['row'].tolist()
    y_coord = img_df['col'].tolist()
    return x_coord, y_coord

def gen_data(image_list, output_folder, input_folder, path_csv):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder)
    os.makedirs(output_folder + '/images')
    os.makedirs(output_folder + '/masks')

    iter_count = 0
    err_count = 0

    elephs = 0
    backgr = 0

    for image in image_list:
        if DEBUG:
            print("Reading {}".format(image))

        # Creation d'un mask ou les pixels des elephants sont mis a 1
        elephant_image = cv2.imread(os.path.join(input_folder, image))
        
        elephant_mask = np.zeros((elephant_image.shape[0], elephant_image.shape[1]), dtype=np.float32)
        if DEBUG:
            print("Image Shape {}".format(elephant_image.shape))

        # Recupère les coordonnées du/des elephant(s) present sur l'image
        x_coord, y_coord = read_coordinates(image, path_csv)
        for x, y in zip(x_coord, y_coord):
            try:
                elephant_mask[y, x] = 255
            except IndexError:
                print("IndexError: x={}, y={}".format(x, y))
                err_count += 1
                continue
        if DEBUG:
            print("Before Density Count", elephant_mask.sum())

        # Smooth the mask to get a density map of elephants (proposé par copilot)
        elephant_mask[:, :] = gaussian_filter(elephant_mask[:, :], sigma=SIGMA_ELEPHANT)


        # print(elephant_mask.min())
        # print(elephant_mask.max())
        # testos = (elephant_mask * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join('./', os.path.splitext(image)[0] + '.png'), testos)

        # do the same but with cv2
        # elephant_mask = cv2.GaussianBlur(elephant_mask, (5, 5), 0)
        # elephant_mask = np.clip(elephant_mask, 0, 255)

        if DEBUG:
            print("After Density Count", elephant_mask.sum())

        # slidingwindow algorithm
        elephant, background = sliding_window_crop(elephant_image, elephant_mask, os.path.splitext(image)[0], output_folder)
        iter_count += 1
        elephs += elephant
        backgr += background
        print(".....Image {}/{} cropping done".format(iter_count, len(image_list)))

    print("#####################################################")
    print("Dataset Summary")
    print("#####################################################")
    print("Total no.of Images:              {}".format(len(image_list)))
    print("No.of patches with elephant:     {}".format(elephs))
    print("No.of patches without elephant:  {}".format(backgr))
    print("No.of errors:                    {}".format(err_count))
    print("#####################################################")

def sliding_window_crop(elephant_image, elephant_mask, image_id, output_folder):
    width = height = 256
    stride = 0
    background_count = 3
    elephant_crop=0
    background_crop=0 

    factor_x = elephant_image.shape[0] / 256
    factor_y = elephant_image.shape[1] / 256
    fact_new_x = int(height - (height * (factor_x % 1))) if factor_x != 0 else 0
    fact_new_y = int(width - (width * (factor_y % 1))) if factor_y != 0 else 0
    image_resize = cv2.copyMakeBorder(elephant_image, 0, fact_new_x, 0, fact_new_y, cv2.BORDER_CONSTANT)
    mask_resize = cv2.copyMakeBorder(elephant_mask, 0, fact_new_x, 0, fact_new_y, cv2.BORDER_CONSTANT)
    if DEBUG:
        print("Before resize: Image {} and Mask {} ---> After resize: Image {} and Mask {}".format(elephant_image.shape,
                                                                                                   elephant_mask.shape,
                                                                                                   image_resize.shape,
                                                                                                   mask_resize.shape))
    cpt = 0
    for i in range(0, int(image_resize.shape[0] / 256)):
        for j in range(0, int(image_resize.shape[1] / 256)):
            
            # print("Image {} Patch {}x{} - cpt : {}".format(image_id, i, j, cpt))

            if width * i - stride > 0:
                xstart = width * i - stride
            else:
                xstart = 0

            if width * j - stride > 0:
                ystart = width * j - stride
            else:
                ystart = 0

            xstop = width + width * i
            ystop = height + height * j

            image_crop = image_resize[xstart:xstop, ystart:ystop, :]
            mask_crop = mask_resize[xstart:xstop, ystart:ystop]

            if mask_crop.sum() == 0:
                if background_crop < background_count:
                    np.save(f'{output_folder}/images/{image_id}_{cpt}', image_crop)
                    np.save(f'{output_folder}/masks/{image_id}_{cpt}', mask_crop)
                    background_crop += 1    
            else:
                np.save(f'{output_folder}/images/{image_id}_{cpt}', image_crop)
                np.save(f'{output_folder}/masks/{image_id}_{cpt}', mask_crop)

                elephant_crop += 1
            
            cpt += 1
            
    return elephant_crop, background_crop

train_image_list = os.listdir(FOLDERPATH + '/training_images')
test_image_list = os.listdir(FOLDERPATH + '/test_images')

print(f"Number of train source images : {len(train_image_list)}")
gen_data(train_image_list, "train", FOLDERPATH+'/training_images', FOLDERPATH + '/training_elephants.csv')

print(f"Number of test source images : {len(test_image_list)}")
gen_data(test_image_list, "test", FOLDERPATH+'/test_images', FOLDERPATH + '/test_elephants.csv')