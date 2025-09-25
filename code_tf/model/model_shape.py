import tensorflow as tf

import keras
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, Conv3D, Reshape, Dropout, MaxPool2D,UpSampling2D, ZeroPadding2D, Activation, Permute
from keras import metrics
import keras.backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable(package="metrics_losses")
class maeloss(tf.keras.losses.Loss):
    def __init__(self, depth_padding=0.0, name="tumor_mae_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.depth_padding = depth_padding

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, self.depth_padding)
        err  = tf.abs(y_true - y_pred)
        masked_err = tf.boolean_mask(err, mask)
        return tf.cond(
            tf.size(masked_err) > 0,
            lambda: tf.reduce_mean(masked_err),
            lambda: tf.constant(0.0)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "depth_padding": self.depth_padding,
        })
        return config



@register_keras_serializable(package="metrics_losses")
class diceloss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-7, name="dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        return 1.0 - dice

    def get_config(self):
        config = super().get_config()
        config.update({
            "smooth": self.smooth,
        })
        return config

@register_keras_serializable(package="metrics_losses")
class focalBCEDice(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-7, alpha=0.25, gamma=2.0,
                 dice_weight=0.5, name="focal_bce_dice_loss", **kwargs):
        """
        Focal BCE + Dice Loss
        alpha: class weight (higher -> more weight on positive class)
        gamma: focusing parameter (higher -> focus more on hard pixels)
        dice_weight: relative weight of Dice loss vs focal BCE
        """
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # --- BCE (manual, pixel-wise) ---
        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))

        # --- Focal modulation ---
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_bce = self.alpha * tf.pow((1 - p_t), self.gamma) * bce
        focal_bce = tf.reduce_mean(focal_bce)

        # --- Dice ---
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth
        )
        dice_loss = 1.0 - dice

        return (1 - self.dice_weight) * focal_bce + self.dice_weight * dice_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "smooth": self.smooth,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "dice_weight": self.dice_weight,
        })
        return config

@register_keras_serializable(package="metrics_losses")
class TumorMAE(tf.keras.metrics.Metric):
    def __init__(self, depth_padding=0.0, name="tumor_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.depth_padding = depth_padding
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.depth_padding)
        err  = tf.abs(y_true - y_pred)
        masked_err = tf.boolean_mask(err, mask)
        value = tf.cond(
            tf.size(masked_err) > 0,
            lambda: tf.reduce_mean(masked_err),
            lambda: tf.constant(0.0)
        )
        self.total.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count


class ModelInit():  

    def __init__(self, params):
            super().__init__()
            self.params = params
    
    def build_attention_gate(self, g, s, num_filters):
        Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
        Ws = Conv2D(num_filters, 1, padding="same")(s)

        out = Activation("relu")(Wg + Ws)
        out = Conv2D(1, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
        out = UpSampling2D()(out)
        return out * g
    
    def build_model(self):

        """The deep learning architecture gets defined here"""
        # Input Optical Properties
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence
        inFL_beg = Input(shape=(self.params['xX'], self.params['yY'], self.params['nF'], 1))


        #3D CNN for all layers

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        
        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)

        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape
        inFL = Conv3D(filters=self.params['nFilters3D']//2,
              kernel_size=self.params['kernelConv3D'],
              strides=self.params['strideConv3D'],
              padding='same',
              activation=self.params['activation'],
              data_format="channels_last")(inFL_beg)


        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)

        ## Concatenate Branch ##
        # inFL = Permute((2, 3, 1, 4))(inFL)
        inFL = Reshape((inFL.shape[1], inFL.shape[2], inFL.shape[3] * inFL.shape[4]))(inFL)
        concat = concatenate([inOP,inFL],axis=-1)

        Max_Pool_1 = MaxPool2D()(concat)

        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_1)
        
        Max_Pool_2 = MaxPool2D()(Conv_1)

        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_2)

        Max_Pool_3 = MaxPool2D()(Conv_2)

        Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_3)

        #decoder 

        #adjust size of Conv_2
        long_path_1 = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]
        attention_1 = self.build_attention_gate(long_path_1, Conv_3, 512)

        Up_conv_1 = UpSampling2D()(Conv_3)

        
        Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Up_conv_1)

        #attention block 
        concat_1 = concatenate([Up_conv_1,attention_1],axis=-1)

        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_1)

        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_4)
        
        long_path_2 = Conv_1
        Conv_4_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_4)

        attention_2 = self.build_attention_gate(long_path_2, Conv_4_zero_pad, 256)

        Up_conv_2 = UpSampling2D()(Conv_4)

        Up_conv_2 = Conv2D(filters=256, kernel_size = (2,2), strides=(1,1), padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Up_conv_2)

        Up_conv_2 = ZeroPadding2D()(Up_conv_2)

        concat_2 = concatenate([Up_conv_2,attention_2],axis=-1)

        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_2)
        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_5)
        
        long_path_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(concat)
        Conv_5_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_5)

        attention_3 = self.build_attention_gate(long_path_3, Conv_5_zero_pad, 128)

        Up_conv_3 = UpSampling2D()(Conv_5)
        Up_conv_3 = Conv2D(filters=128, kernel_size = (2,2), strides=(1,1), padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Up_conv_3)
                        
        Up_conv_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(Up_conv_3)

        attention_3 = attention_3[:,0:attention_3.shape[1] - 1, 0:attention_3.shape[2] - 1, :]
        concat_3 = concatenate([Up_conv_3,attention_3],axis=-1)  

        Conv_6 = Conv2D(filters=128, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_3)
        
        # --------------- Shape Head (binary mask) -----------------
        shape = Conv2D(64, self.params['kernelConv2D'], self.params['strideConv2D'],
                padding='same', activation=self.params['activation'])(Conv_6)
        shape = Conv2D(32, self.params['kernelConv2D'], self.params['strideConv2D'],
                padding='same', activation=self.params['activation'])(shape)
        shape = Conv2D(1, (1, 1), (1, 1), padding='same',
               activation='sigmoid',  # binary output
               name='shape_logits')(shape)
        outShape = keras.layers.Reshape((self.params['yY'], self.params['xX']),
                                name='outShape')(shape)

        # Depth head (DF)
        df = Conv2D(64, self.params['kernelConv2D'], self.params['strideConv2D'],
                padding='same', activation=self.params['activation'],
                data_format="channels_last")(Conv_6)
        df = Conv2D(32, self.params['kernelConv2D'], self.params['strideConv2D'],
                padding='same', activation=self.params['activation'],
                data_format="channels_last")(df)
        df = Conv2D(1, (1, 1), (1, 1), padding='same',
                activation=None,
                data_format="channels_last", name='df_logits')(df)  # <-- different name

        outDF = keras.layers.Reshape((self.params['yY'], self.params['xX']),
                                name='outDF')(df)


        self.model = Model(inputs=[inOP_beg, inFL_beg], outputs=[outShape, outDF])

        self.model.compile(
            loss={
                'outShape': focalBCEDice(alpha=0.25, gamma=2.0, dice_weight=0.5),  # <- hybrid loss
                'outDF': maeloss()
            },
            loss_weights={'outShape': 5.0, 'outDF': 1.0},
            optimizer=getattr(tf.keras.optimizers, self.params['optimizer'])(learning_rate=self.params['learningRate']),
            metrics={
                'outShape': [
                    metrics.BinaryAccuracy(name='acc_shape'),
                    metrics.Precision(name='prec_shape'),
                    metrics.Recall(name='rec_shape')
                ],
                'outDF': [TumorMAE()]
            }
        )
        
        # self.model.summary()

        return None
    
    def load_model(self, model_path):
        self.model = load_model(model_path)

        return None
    
    def save_model(self, model_path):
        self.model.save(model_path)

        return None