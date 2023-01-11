import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def rmse_func_tf(y_pred, y_true):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.sqrt(mse)

def load_data(data_path):
    # get the filenames of all the images and masks
    list_fn_img = os.listdir(f"{data_path}/images")
    list_fn_mask = os.listdir(f"{data_path}/masks")

    imgs = []
    masks = []

    skip = len(list_fn_img)
    # load all the images and masks
    for fn_img, fn_mask in zip(list_fn_img[:skip], list_fn_mask[:skip]):
        img = np.load(f"{data_path}/images/{fn_img}")
        mask = np.load(f"{data_path}/masks/{fn_mask}")

        imgs.append(img)
        masks.append(mask)
    
    iimgs = []
    mmasks = []

    for im,ma in zip(imgs,masks):
        if ma.sum() > 120:
            # plt.imshow(ma)
            # plt.show()
            iimgs.append(im)
            mmasks.append(ma)

    # imgs = np.array(imgs)
    # masks = np.array(masks)

    iimgs = np.array(iimgs)
    mmasks = np.array(mmasks)

    return iimgs, mmasks
    # return imgs, masks

imgs, masks = load_data('./test')

imgs = imgs[:7]
masks = masks[:7]


# exit()
print(imgs.shape)

for folder in os.listdir('check_backup'):
    for files in os.listdir(f'check_backup/{folder}'):
        if files.startswith('model'):

            model = keras.models.load_model(f'check_backup/{folder}/{files}', custom_objects={"rmse_func_tf": rmse_func_tf})
            # model.summary()
            
            cpt = 0
            fig, ax = plt.subplots(len(imgs), 3, figsize=(15, 15))
            for i in range(len(imgs)):
                
                pred = model.predict(np.expand_dims(imgs[i], axis=0))
                pred = np.squeeze(pred)

                ax[cpt, 0].imshow(imgs[i])
                # ax[cpt, 0].set_title('Image')
                ax[cpt, 0].set_axis_off()
                ax[cpt, 1].imshow(masks[i])
                # ax[cpt, 1].set_title('Mask')
                ax[cpt, 1].set_axis_off()
                ax[cpt, 2].imshow(pred)
                # ax[cpt, 2].set_title('Prediction')
                ax[cpt, 2].set_axis_off()
                
                cpt += 1
            plt.tight_layout()
            plt.show()
    