import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, Input, BatchNormalization, UpSampling2D
import os
import numpy as np
import uuid
import time
import matplotlib.pyplot as plt

FOLDERPATH = os.path.expanduser("~/AED")
CHECKPOINTPATH = os.path.expanduser("~/checkpoint")

class UNetElephant():
    def __init__(self) -> None:
        pass

    def rmse_func_tf(y_pred, y_true):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.sqrt(mse)

    def down_block_x(nb_conv, x, filters, stride=1, kernel=(3,3), padding='same', activation_id=1):
        c = x        
        for i in range(nb_conv):
            c = Conv2D(filters, kernel, strides=stride, padding=padding)(c)
            
            if activation_id == 1:
                c = keras.layers.ReLU()(c)
            elif activation_id == 2:
                c = keras.layers.PReLU()(c)
            elif activation_id == 3:
                c = keras.layers.LeakyReLU()(c)

            if (i+1) % 2 == 0:
                c = BatchNormalization()(c)
        p = MaxPooling2D((2,2), (2,2))(c)
        p = BatchNormalization()(p)
        return c, p

    def up_block_x(nb_conv, x, s, filters, stride=1, kernel=(3,3), padding='same', activation_id=1):
        u = UpSampling2D((2,2))(x)
        cc = concatenate(inputs=[u, s])
        c = cc
        for i in range(nb_conv):
            c = Conv2D(filters, kernel, strides=stride, padding=padding)(c)
            
            if activation_id == 1:
                c = keras.layers.ReLU()(c)
            elif activation_id == 2:
                c = keras.layers.PReLU()(c)
            elif activation_id == 3:
                c = keras.layers.LeakyReLU()(c)

            if (i > 2 and i % 2 == 0):
                c = BatchNormalization()(c)
        return c

    def bottom_transition_x(nb_conv, x, filters, stride=1, kernel=(3,3), padding='same', activation_id=1):
        c = x
        for i in range(nb_conv):
            c = Conv2D(filters, kernel, strides=stride, padding=padding)(c)
            
            if activation_id == 1:
                c = keras.layers.ReLU()(c)
            elif activation_id == 2:
                c = keras.layers.PReLU()(c)
            elif activation_id == 3:
                c = keras.layers.LeakyReLU()(c)
            
            if (i > 2 and i % 2 == 0):
                c = BatchNormalization()(c)
        return c

    def create_unet(complexity, img_size = 256, nb_filters_start = 16, lr=0.001, activation_id=1):
        inp = Input((img_size, img_size, 3)) # size of the image

        p0 = inp

        c1, p1 = UNetElephant.down_block_x(complexity, p0, nb_filters_start, activation_id=activation_id)
        c2, p2 = UNetElephant.down_block_x(complexity, p1, nb_filters_start * 2, activation_id=activation_id)
        c3, p3 = UNetElephant.down_block_x(complexity, p2, nb_filters_start * 4, activation_id=activation_id)
        c4, p4 = UNetElephant.down_block_x(complexity, p3, nb_filters_start * 8, activation_id=activation_id)
        c5, p5 = UNetElephant.down_block_x(complexity, p4, nb_filters_start * 16, activation_id=activation_id)

        bt = UNetElephant.bottom_transition_x(complexity, p5, nb_filters_start * 32, activation_id=activation_id)

        u1 = UNetElephant.up_block_x(complexity, bt, c5, nb_filters_start * 16, activation_id=activation_id)
        u2 = UNetElephant.up_block_x(complexity, u1, c4, nb_filters_start * 8, activation_id=activation_id)
        u3 = UNetElephant.up_block_x(complexity, u2, c3, nb_filters_start * 4, activation_id=activation_id)
        u4 = UNetElephant.up_block_x(complexity, u3, c2, nb_filters_start * 2, activation_id=activation_id)
        u5 = UNetElephant.up_block_x(complexity, u4, c1, nb_filters_start, activation_id=activation_id)

        out = Conv2D(1, (1,1), padding="same", activation="sigmoid")(u5)

        model = keras.models.Model(inp, out)
        optimizer=keras.optimizers.Adam(learning_rate=lr,beta_1=0.9,beta_2=0.999) # taken from the github code
        metrics = ["mean_squared_error","mean_squared_logarithmic_error","mean_absolute_error","squared_hinge"]
        model.compile(optimizer=optimizer, loss=UNetElephant.rmse_func_tf, metrics=metrics)

        return model

    def load_data(data_path, size_of_dataset=1, with_splitting=True, elephant_only=False):

        data_path = data_path if data_path.find("~") == -1 else os.path.expanduser(data_path)

        # get the filenames of all the images and masks
        list_fn_img = os.listdir(f"{data_path}/images")
        list_fn_mask = os.listdir(f"{data_path}/masks")

        imgs = []
        masks = []

        skip = int(size_of_dataset*len(list_fn_img)) if not elephant_only else len(list_fn_img)

        # load all the images and masks
        for fn_img, fn_mask in zip(list_fn_img[:skip], list_fn_mask[:skip]):
            img = np.load(f"{data_path}/images/{fn_img}")
            mask = np.load(f"{data_path}/masks/{fn_mask}")

            if elephant_only and np.sum(mask) > 100:
                imgs.append(img)
                masks.append(mask)
            else:
                imgs.append(img)
                masks.append(mask)
        
        imgs = np.array(imgs)
        masks = np.array(masks)
        if not with_splitting:
            return imgs, masks

        # split the data into training (85%), validation(15%)
        nb_samples = int(len(imgs) * 0.85)

        x_train = imgs[:nb_samples]
        y_train = masks[:nb_samples]
        x_val = imgs[nb_samples:]
        y_val = masks[nb_samples:]

        return x_train, y_train, x_val, y_val

    def _train(self):
        # load the data
        # x_train, y_train, x_val, y_val = UNetElephant.load_data('~/train', size_of_dataset=self.mini_dataset_testing, elephant_only=self.elephant_only)
        x_train, y_train, x_val, y_val = UNetElephant.load_data('~/test', size_of_dataset=self.mini_dataset_testing, elephant_only=self.elephant_only)
        
        print(f"Training on {len(x_train)} samples, validating on {len(x_val)} samples")

        # train the model
        if self.epochs > 0:
            self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.cb_checkpoint, self.cb_early_stopping])

        if self.log:
            # save the training history
            plt.plot(self.model.history.history['loss'], label='train')
            plt.plot(self.model.history.history['val_loss'], label='val')
            plt.title(f'model {self.name} loss')
            plt.legend()
            plt.xlabel('epoch');plt.ylabel('loss')
            plt.savefig(f"{self.checkpoint_folder}/history.png")

    def _build_model(self):
        self.model = UNetElephant.create_unet(self.model_complexity, img_size=self.image_size, nb_filters_start=self.nb_filters_start, lr=self.lr, activation_id=self.activation_id)
        # self.model.summary()

    def _load_model(self):
        # check if model exist otherwise open the checkpoint
        file2load = "model.h5" if os.path.exists(f"{self.checkpoint_folder}/{self.name_continue_training}/model.h5") else "checkpoint.h5"
        self.model = keras.models.load_model(os.path.join(self.checkpoint_folder, file2load), custom_objects={"rmse_func_tf": UNetElephant.rmse_func_tf})

    def _save_model(self):
        name = "model.h5" if self.name_continue_training is None else "model-continue.h5"
        self.model.save(f"{self.checkpoint_folder}/{name}")

    def _validate_args(self, args):
        self.epochs = args.epochs
        assert self.epochs >= 0, "Epochs must be a positive integer"
        self.batch_size = args.batch
        assert self.batch_size > 0, "Batch size must be a positive integer"
        self.image_size = args.image_size
        assert self.image_size > 0, "Image size must be a positive integer"
        self.model_complexity = args.model_complexity
        assert self.model_complexity > 0, "Model complexity must be a positive integer"
        self.mini_dataset_testing = args.mini_dataset_testing
        assert self.mini_dataset_testing > 0.0, f"Mini dataset testing must be a positive float in ]0,1] not {args.mini_dataset_testing}"
        self.lr = args.lr
        assert self.lr > 0.0, "Learning rate must be a positive float"
        self.nb_filters_start = args.nb_filters_start
        assert self.nb_filters_start > 0, "Number of filters must be a positive integer"
        self.activation_id = args.activation_id
        assert self.activation_id in [1, 2, 3], "Activation id must be 0, 1 or 2"

        self.elephant_only = args.elephant_only

        self.name_continue_training = args.name_continue_training
        assert self.name_continue_training is None, "Checkpoint name does not exist"
        self.log = args.log

        # Get or create the training name
        self.name = uuid.uuid4().hex if self.name_continue_training is None else self.name_continue_training
        print("Checkpoint name: ", self.name)
        
        self.checkpoint_folder = os.path.join(CHECKPOINTPATH, self.name)

        if self.name_continue_training is None:
            os.mkdir(self.checkpoint_folder)

        # create a file for the checkpoint save
        nn = "checkpoint.h5" if self.name_continue_training is None else "checkpoint-continue.h5"
        checkpoint_path = os.path.join(self.checkpoint_folder, nn)

        # create training callbacks
        self.cb_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_freq='epoch')
        self.cb_early_stopping = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    def _evaluate(self):
        x_test, y_test = UNetElephant.load_data('~/test', with_splitting=False)
        self.model.evaluate(x_test, y_test, verbose=1)

    def main(self, args):
        self._validate_args(args)

        # test if not None to decide if continue or not
        self.name_continue_training and self._load_model() or self._build_model()

        self._train()
        print("\nEvaluating...\n")
        self._evaluate()
        self._save_model()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Unet Elephant")
    parser.add_argument("--epochs", type=int, default=5, dest="epochs", help="Number of epochs to train the model")
    parser.add_argument("--batch", type=int, default=32, dest="batch", help="Number of image in a batch")
    parser.add_argument("--image_size", type=int, default=256, dest="image_size", help="Size of the image")
    parser.add_argument("--model_complexity", type=int, default=2, dest="model_complexity", help="Number of convolution layers in each down and up block, power of 2 preferred")
    parser.add_argument("--name_continue_training", type=str, default=None, dest="name_continue_training", help="Give the name of the training inside the checkpoint folder")
    parser.add_argument("--log", type=bool, default=True, dest="log", help="Log the training history")
    parser.add_argument("--mini_dataset_testing", type=float, default=0.2, dest="mini_dataset_testing", help="Use a mini dataset for testing purposes, float in ]0,1]")
    parser.add_argument("--lr", type=float, default=0.001, dest="lr", help="Learning rate")
    parser.add_argument("--nb_filters_start", type=int, default=16, dest="nb_filters_start", help="Starting number of filters learn in the convolutional layers, double each block. Power of 2 preferred")
    parser.add_argument("--activation_id", type=int, default=1, dest="activation_id", help="Activation function to use in the convolutional layers, dict[1: 'relu', 2:'leaky', 3:'prelu']")
    parser.add_argument("--elephant_only", type=bool, default=False, dest="elephant_only", help="Use only elephant images for training, doesnt work with mini_dataset_testing")

    args = parser.parse_args()

    st = time.time()
    eleph = UNetElephant()
    eleph.main(args)
    print(f"Time taken: {time.time() - st}")