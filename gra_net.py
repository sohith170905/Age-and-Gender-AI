"""Compact educational implementation of the GRA_Net ideas described in the paper.
It is a faithful implementation scaffold, not a claim of exact reproduction of the authors' private training code.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="GRA_Net")
class _GatedBlend(layers.Layer):
    """Blend trunk and mask features with a trainable gate in [0, 1]."""

    def build(self, input_shape):
        self.gamma_logit = self.add_weight(
            name="gamma_logit",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        trunk, mask = inputs
        gamma = tf.sigmoid(self.gamma_logit)
        return (gamma + (1.0 - gamma) * mask) * trunk


def conv_bn_act(x, filters, k=3, s=1, name=None):
    x = layers.Conv2D(filters, k, strides=s, padding='same', use_bias=False, name=None if not name else name+'_conv')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_bn')(x)
    return layers.Activation('relu', name=None if not name else name+'_act')(x)


def gated_residual(x, filters, name):
    shortcut = x
    y = layers.BatchNormalization(name=name+'_bn1')(x)
    y = layers.Activation('relu', name=name+'_relu1')(y)
    y = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=name+'_conv1')(y)
    y = layers.BatchNormalization(name=name+'_bn2')(y)
    y = layers.Activation('relu', name=name+'_relu2')(y)
    y = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=name+'_conv2')(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False, name=name+'_proj')(shortcut)
    return layers.Add(name=name+'_add')([shortcut, y])


def soft_mask(x, name):
    """Bottom-up/top-down soft mask branch with sigmoid mask and skip connection."""
    skip = x
    d1 = layers.MaxPooling2D(2, name=name+'_pool1')(x)
    d1 = conv_bn_act(d1, x.shape[-1], name=name+'_down1')
    d2 = layers.MaxPooling2D(2, name=name+'_pool2')(d1)
    d2 = conv_bn_act(d2, x.shape[-1], name=name+'_down2')
    u1 = layers.UpSampling2D(2, interpolation='bilinear', name=name+'_up1')(d2)
    u1 = layers.Concatenate(name=name+'_skip1')([u1, d1])
    u1 = conv_bn_act(u1, x.shape[-1], name=name+'_upconv1')
    u2 = layers.UpSampling2D(2, interpolation='bilinear', name=name+'_up2')(u1)
    u2 = layers.Concatenate(name=name+'_skip2')([u2, skip])
    u2 = layers.Conv2D(x.shape[-1], 1, padding='same', activation='relu', name=name+'_conv1x1a')(u2)
    return layers.Conv2D(x.shape[-1], 1, padding='same', activation='sigmoid', name=name+'_mask')(u2)


def gated_attention(x, filters, name, gamma_init=0.5):
    trunk = gated_residual(x, filters, name=name+'_trunk')
    mask = soft_mask(trunk, name=name+'_mask')
    # K in [0,1], gated residual attention: (gamma + (1-gamma)K)F.
    # Store the gate as a logit so optimization cannot make it negative or
    # larger than one, which destabilizes deeper paper-style models.
    return _GatedBlend(name=name+'_apply')([trunk, mask])


def build_gra_net(input_shape=(32,32,3), num_age_classes=10, gender_classes=2):
    inp = layers.Input(shape=input_shape, name='face_image')
    x = conv_bn_act(inp, 64, 3, name='stem')
    x = gated_attention(x, 64, 'attention1')
    x = layers.MaxPooling2D(2)(x)
    x = gated_attention(x, 128, 'attention2')
    x = layers.MaxPooling2D(2)(x)
    x = gated_attention(x, 256, 'attention3')
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(0.3)(x)
    gender = layers.Dense(gender_classes, activation='softmax', name='gender')(x)
    age_class = layers.Dense(num_age_classes, activation='softmax', name='age_class')(x)
    age_reg = layers.Dense(1, activation='linear', name='age_regression')(x)
    return Model(inp, [gender, age_class, age_reg], name='GRA_Net_Compact')


def compile_model(model, lr=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
        loss={'gender':'sparse_categorical_crossentropy','age_class':'sparse_categorical_crossentropy','age_regression':'mae'},
        metrics={'gender':'accuracy','age_class':'accuracy','age_regression':'mae'}
    )
    return model


def _residual_stack(x, filters, blocks, name):
    """Stack pre-activation residual units used by the paper variant."""
    for index in range(blocks):
        x = gated_residual(x, filters, f"{name}_unit{index + 1}")
    return x


def build_paper_gra_net(
    input_shape=(32, 32, 3),
    num_age_classes=10,
    gender_classes=2,
    stage_widths=(96, 192, 384),
    blocks_per_side=2,
):
    """Build a deeper GRA_Net-inspired model for paper-protocol training.

    The published paper describes three stacked gated residual-attention
    stages and decomposes age into a decade class plus a 0-9 remainder. The
    existing compact model is kept unchanged; this separate builder makes the
    new experiment reversible and lets the app distinguish its age head by
    name.
    """
    inp = layers.Input(shape=input_shape, name="face_image")
    width1, width2, width3 = stage_widths
    x = conv_bn_act(inp, width1, 3, name="paper_stem")

    x = _residual_stack(x, width1, blocks_per_side, "paper_stage1_pre")
    x = gated_attention(x, width1, "paper_attention1")
    x = _residual_stack(x, width1, blocks_per_side, "paper_stage1_post")
    x = layers.MaxPooling2D(2, name="paper_pool1")(x)

    x = _residual_stack(x, width2, blocks_per_side, "paper_stage2_pre")
    x = gated_attention(x, width2, "paper_attention2")
    x = _residual_stack(x, width2, blocks_per_side, "paper_stage2_post")
    x = layers.MaxPooling2D(2, name="paper_pool2")(x)

    x = _residual_stack(x, width3, blocks_per_side, "paper_stage3_pre")
    x = gated_attention(x, width3, "paper_attention3")
    x = _residual_stack(x, width3, blocks_per_side, "paper_stage3_post")
    x = layers.GlobalAveragePooling2D(name="paper_global_pool")(x)
    x = layers.Dropout(0.3, name="paper_dropout")(x)

    gender = layers.Dense(gender_classes, activation="softmax", name="gender")(x)
    age_class = layers.Dense(num_age_classes, activation="softmax", name="age_class")(x)
    # The remainder represents the units digit of age, so keep it in the
    # valid 0-9 interval.  Bounding this head also prevents a large transient
    # regression value from dominating the multi-task loss during early
    # training.
    age_remainder = layers.Dense(1, activation="sigmoid", name="age_remainder")(x)
    return Model(inp, [gender, age_class, age_remainder], name="GRA_Net_PaperProtocol")


def build_paper_gra_net_fast(input_shape=(32, 32, 3), num_age_classes=10, gender_classes=2):
    """Feasible paper-protocol configuration for CPU training.

    It preserves the paper's three attention stages and age decomposition while
    using one residual unit per side and narrower stages than the large
    reference configuration.
    """
    return build_paper_gra_net(
        input_shape=input_shape,
        num_age_classes=num_age_classes,
        gender_classes=gender_classes,
        stage_widths=(64, 128, 256),
        blocks_per_side=1,
    )


def compile_paper_model(model, lr=0.001):
    """Compile the paper-style multi-task outputs with Nadam."""
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
        loss={
            "gender": "sparse_categorical_crossentropy",
            "age_class": "sparse_categorical_crossentropy",
            "age_remainder": "mae",
        },
        metrics={
            "gender": "accuracy",
            "age_class": "accuracy",
            "age_remainder": "mae",
        },
    )
    return model
