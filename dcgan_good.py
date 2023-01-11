
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.initializers import RandomNormal
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path

class GanAlgorithm():

  def __init__(self) -> None:
    self.filters = [512, 256, 128, 64, 64, 32]
    self.generator_optimizer = Adam(1.5e-4, 0.5)
    self.discriminator_optimizer = Adam(1.5e-4, 0.5)
    # for image display
    self.preview_row = 4
    self.preview_cols = 7
    self.preview_margin = 16

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
    self.buffer_size = 60000


  def _hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

  def _load_image(self):
    path = self.inputdir

    training_binary_path = os.path.join(path, f'training_data_{self.generate_square}_{self.generate_square}.npy')
    print(f"Looking for file: {training_binary_path}")
    if not os.path.isfile(training_binary_path):
      start = time.time()
      print("Loading training images...")

      training_data = []

      errorCnt = 0
      first = True

      # images_path = os.path.join(self.inputdir + "/images")
      # masks_path = os.path.join(self.inputdir + "/masks")

      images_path = os.path.join(self.inputdir)

      for filename in os.listdir(images_path):
        curr = []
        # test pour prendre que des fichiers de type images
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
          im_path = os.path.join(images_path, filename)
          # mask_path = os.path.join(masks_path, filename)

          image = Image.open(im_path).resize((self.generate_square, self.generate_square), Image.LANCZOS)
          # mask = Image.open(path).resize((self.generate_square, self.generate_square), Image.LANCZOS)

          curr.append(np.array(image))

          # test pour eviter les erreurs sur certaines images
          try:
            converted = np.reshape(curr, (-1, self.generate_square, self.generate_square, self.channels))
          except ValueError:
            print(f"Error with one image {filename}")
            errorCnt+=1
            continue

          if first:
            training_data = converted
            first = False
          else:
            training_data = np.concatenate((training_data, converted))
      
      # print(f"training_data shape {training_data.shape}\n")
      print(f"Error with {errorCnt} images during loading")

      print("Images normalizing ...")
      training_data = training_data.astype(np.float32)
      training_data = training_data / 127.5 - 1.

      print("Saving training image binary in file ...")
      np.save(training_binary_path, training_data)
      elapsed = time.time()-start
      print (f'Image preprocess time: {GanAlgorithm._hms_string(elapsed)}')
    else:
      print("Loading previous training pickle...")
      training_data = np.load(training_binary_path)

    # shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(self.buffer_size).batch(self.batch_size)

    print(f"Number of training images: {training_data.shape[0]}")
    
    self.train_dataset = train_dataset
    
  def _upsampling_block(model, filters, kernel_size, strides=(1, 1), padding='same', activation=None, momentum=0.8, alpha=0.2):
    model.add(UpSampling2D())
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))
    return model
  
  def _downsampling_block(model, filters, kernel_size, strides=(2, 2), padding='same', activation=None, momentum=0.8, alpha=0.2, dropout=0.25):
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))
    return model
    
  def _build_generator(self):
    model = Sequential()

    model.add(Dense(self.base_square*self.base_square*self.base_length, input_dim=self.latent_dim, activation="relu"))
    model.add(Reshape((self.base_square, self.base_square, self.base_length)))

    for f in self.filters:
      model = GanAlgorithm._upsampling_block(model, f, self.kernel_size, momentum=self.momentum, alpha=self.alpha)

    # Final CNN layer
    model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
    model.add(Activation("tanh"))

    return model

  def _build_discriminator(self):
    model = Sequential()

    self.filters.reverse()

    # Entry layer
    model.add(Conv2D(self.filters[0], kernel_size=self.kernel_size, strides=self.strides, padding="same", input_shape=self.image_shape))
    model.add(LeakyReLU(alpha=self.alpha))

    for f in self.filters[1:]:
      model = GanAlgorithm._downsampling_block(model, f, self.kernel_size, momentum=self.momentum, alpha=self.alpha, dropout=self.dropout, strides=self.strides)

    # Output layer (1 for real, 0 for fake)
    model.add(Dropout(self.dropout))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

  def _save_images(self, cnt, noise):
    image_array = np.full((
        self.preview_margin + (self.preview_row * (self.generate_square+self.preview_margin)),
        self.preview_margin + (self.preview_cols * (self.generate_square+self.preview_margin)), 3),
        255, dtype=np.uint8)

    generated_images = self.generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(self.preview_row):
        for col in range(self.preview_cols):
          r = row * (self.generate_square + 16) + self.preview_margin
          c = col * (self.generate_square + 16) + self.preview_margin
          image_array[r : r + self.generate_square, c : c + self.generate_square] = generated_images[image_count] * 255
          image_count += 1

    output_path = os.path.join(self.outputdir)
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    a = output_path.removeprefix("./output_")
    filename = os.path.join(output_path, f"{a}-{self.generate_square}-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

  def _discriminator_loss(self, real_output, fake_output):
      real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss

  def _generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  @tf.function
  def _train_step(self, images):
    # Setup the seed for generation
    seed = tf.random.normal([self.batch_size, self.latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Generate a image (noise)
      generated_images = self.generator(seed, training=True)

      # Get the decision from the discriminator with real and fake image
      real_output = self.discriminator(images, training=True)
      fake_output = self.discriminator(generated_images, training=True)

      # Calculate the loss of generator and discriminator
      gen_loss = self._generator_loss(fake_output)
      disc_loss = self._discriminator_loss(real_output, fake_output)

      if self.log:
        # Save for TensorBoard
        self.tb_gen_loss(gen_loss)
        self.tb_disc_loss(disc_loss)
        self.tb_gen_acc(tf.ones_like(fake_output), fake_output)
        self.tb_disc_acc(tf.zeros_like(real_output), real_output)

      # Calculate the gradients of generator and discriminator
      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

      # Apply the gradients to the optimizer
      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
    return gen_loss, disc_loss

  def _check_saved_model(self):
    if os.path.exists(self.outputdir + "/generator.h5") and os.path.exists(self.outputdir + "/discriminator.h5"):
      os.remove(self.outputdir + "/generator.h5")
      os.remove(self.outputdir + "/discriminator.h5")


    self._save_model()
  
  def train(self):
    fixed_seed = np.random.normal(0, 1, (self.preview_row * self.preview_cols, self.latent_dim))
    start = time.time()
    g_global_loss = []
    d_global_loss = []
    
    for epoch in range(self.epochs):
      epoch_start = time.time()
      gen_loss_list = []
      disc_loss_list = []

      # Iterate over the batches of the dataset
      for image_batch in self.train_dataset:
        t = self._train_step(image_batch)
        gen_loss_list.append(t[0])
        disc_loss_list.append(t[1])

      # Compute the average loss
      g_loss = sum(gen_loss_list) / len(gen_loss_list)
      d_loss = sum(disc_loss_list) / len(disc_loss_list)
      g_global_loss.append(g_loss)
      d_global_loss.append(d_loss)

      if self.log:
        # Save logs of the training process
        with self.gen_summary_writer.as_default():
          tf.summary.scalar('g_loss', self.tb_gen_loss.result(), step=epoch)
          tf.summary.scalar('g_acc', self.tb_gen_acc.result(), step=epoch)
        with self.disc_summary_writer.as_default():
          tf.summary.scalar('d_loss', self.tb_disc_loss.result(), step=epoch)
          tf.summary.scalar('d_acc', self.tb_disc_acc.result(), step=epoch)

      epoch_elapsed = time.time() - epoch_start
      print (f'Epoch {epoch+1}, gen loss={g_loss}, disc loss={d_loss}, elapsed_epoch {epoch_elapsed}')      
      
      if self.log:
        # Reset for the next step
        self.tb_gen_loss.reset_states()
        self.tb_disc_loss.reset_states()

      if (epoch % self.each == 0 or epoch == self.epochs-1) and epoch > 1000:
        self._save_images(epoch + self.base_epochs, fixed_seed)
        self._check_saved_model()

      # sauvegarde en plus au cas ou le 6000 est nul      
      if epoch == 4000:
        count = 4000 + self.base_epochs
        self._save_model(name=str(count))

    elapsed = time.time()-start
    print (f'Training time: {GanAlgorithm._hms_string(elapsed)}')

  def _save_model(self, name=""):
    self.generator.save(os.path.join(self.outputdir, f"{name}_generator.h5"))
    self.discriminator.save(os.path.join(self.outputdir, f"{name}_discriminator.h5"))

  def _load_model(self):
    print(f"Load model : {self.continue_training_path}")
    if self.continue_training_path is not None:
      self.generator = tf.keras.models.load_model(os.path.join(self.continue_training_path, self.generator_file))
      self.discriminator = tf.keras.models.load_model(os.path.join(self.continue_training_path, self.discriminator_file))
    else:
      self.generator = self._build_generator()
      self.discriminator = self._build_discriminator()

    print(f"Generator: {self.generator.summary()}")
    print(f"Discriminator: {self.discriminator.summary()}")

  def _setup_tensorboard(self):
    # Tensorboard variables
    self.tb_gen_loss = Mean('gen_loss', dtype=tf.float32)
    self.tb_disc_loss = Mean('disc_loss', dtype=tf.float32)
    self.tb_gen_acc = SparseCategoricalAccuracy('gen_acc', dtype=tf.float32)
    self.tb_disc_acc = SparseCategoricalAccuracy('disc_acc', dtype=tf.float32)
    self.gen_log_dir = f'{self.outputdir}/log/generator'
    self.disc_log_dir = f'{self.outputdir}/log/discriminator'
    self.gen_summary_writer = tf.summary.create_file_writer(self.gen_log_dir)
    self.disc_summary_writer = tf.summary.create_file_writer(self.disc_log_dir)

  def _validate_args(self, args):
    self.inputdir = Path(args.inputdir)
    assert self.inputdir.exists(), "Input directory does not exist"
    
    # images = Path(self.inputdir + "/images")
    # masks = Path(self.inputdir + "/masks")
    # assert images.exists(), "Image directory doesn't exit, Input folder must contains 'images' and 'masks' folders"
    # assert masks.exists(), "Mask directory doesn't exist, Input folder must contains 'images' and 'masks' folders"
    # assert len(images.glob("*.png")) == len(masks.glob("*.png")), "Amount of images and masks doesn't match"

    self.outputdir = args.outputdir
    self.epochs = args.epochs
    assert self.epochs > 0, "Epochs must be greater than 0"
    self.kernel_size = args.kernel_size
    self.strides = args.strides
    self.base_square = args.base_square
    self.base_length = args.base_length
    self.latent_dim = args.latent_dim # seed_size
    self.momentum = args.momentum
    self.dropout = args.dropout
    self.alpha = args.alpha
    if args.filters is not None:
      self.filters = [int(f) for f in args.filters]

    self.channels = args.channels
    self.batch_size = args.batch_size

    self.each = args.each
    self.log = args.log

    if args.continue_training_path is not None:
      assert os.path.exists(args.continue_training_path), "Continue training file does not exist"

      files = os.listdir(args.continue_training_path)
      generator_file = [f for f in files if f.endswith('generator.h5')]
      discriminator_file = [f for f in files if f.endswith('discriminator.h5')]


      assert len(generator_file) == 1, "Generator file not found"
      assert len(discriminator_file) == 1, "Discriminator file not found"

      self.generator_file = generator_file[0]
      self.discriminator_file = discriminator_file[0]

      self.base_epochs = int(self.generator_file.split("_")[0])
    else :
      self.base_epochs = 0
      self.generator_file = None
      self.discriminator_file = None
    # None if not specified
    self.continue_training_path = args.continue_training_path # path to the folder containing the generator and discriminator


  def main(self, args):
    self.__init__()
    self._validate_args(args)

    if self.log:
      self._setup_tensorboard()

    # create new variables from the arguments
    self.generate_square = self.base_square * (2**len(self.filters)) # calcul de la taille de l'image en sortie par rapport à la taille de base et le nombre de filtres
    self.image_shape = (self.generate_square, self.generate_square, self.channels)
    self.noise = tf.random.normal([1, self.latent_dim])

    print(f"Will generate {self.generate_square}px square images. - {self.image_shape}")
    print(f"Model parameters: {vars(self)}")

    self._load_image()
    
    # Create both models or load them from the continue_training folder
    self._load_model()

    self.train()

    # print("Saving model...")
    self._save_model(name=str(self.epochs))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("GAN")
    parser.add_argument("--data_path", type=str, required=True, dest="inputdir", help="Path to the data containing the dataset, must content two sub-folder 'images' and 'masks'")
    parser.add_argument("--output_dir", type=str, required=True, dest="outputdir", help="Path to the output directory")
    parser.add_argument("--epochs", type=int, default=2000, dest="epochs", help="Number of epochs to train the model")
    parser.add_argument("--kernel_size", type=int, default=5, dest="kernel_size", help="Kernel size (must be odd)")
    parser.add_argument("--strides", type=int, default=2, dest="strides", help="Stride size")
    parser.add_argument("--base_square", type=int, default=4, dest="base_square", help="The base dimension of the generator image, default : 4x4 px image")
    parser.add_argument("--base_length", type=int, default=1024, dest="base_length", help="The base length of the generator image")
    parser.add_argument("--latent_dim", type=int, default=100, dest="latent_dim", help="Latent dimension (seed_size)")
    parser.add_argument("--momentum", type=float, default=0.8, dest="momentum", help="Momentum for the batch normalization")
    parser.add_argument("--alpha", type=float, default=0.2, dest="alpha", help="Alpha for the leaky relu")
    parser.add_argument("--dropout", type=float, default=0.25, dest="dropout", help="Dropout rate")
    parser.add_argument("--filters", type=str, dest="filters", nargs="+", help="List of filters size to be applied, each filter will double the size of the image")
    parser.add_argument("--channels", type=int, default=3, dest="channels", help="Number of channels")
    parser.add_argument("--batch_size", type=int, default=32, dest="batch_size", help="Batch size")
    parser.add_argument("--each", type=int, default=25, dest="each", help="Save images each n epochs")
    parser.add_argument("--log", action="store_true", dest="log", help="Activate the logging of training with TensorBoard")
    parser.add_argument("--continue_training_path", type=str, dest="continue_training_path", help="Continue training from a previous models, path folder containing '<nb epochs>_generator.h5' and '<nb epochs>_discriminator.h5' (e.g. 4000_generator.h5)")


    args = parser.parse_args()

    gan = GanAlgorithm()
    gan.main(args)
