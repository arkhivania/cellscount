from tensorflow.keras.layers import Conv2D,MaxPooling2D, BatchNormalization, Dropout, Activation, Conv2DTranspose, LeakyReLU, AveragePooling2D
from tensorflow.keras.layers import concatenate
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.merge import concatenate, add
import tensorflow.keras.regularizers as tfg


def conv2d_bn(x, filters, ksize, d_rate, strides,padding='same', activation='relu', groups=1, name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate = d_rate, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def CFPModule(inp, filters, d_size):
    '''
    CFP module for medicine
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    x_inp = conv2d_bn(inp, filters//4, ksize=1, d_rate=1, strides=1)
    
    x_1_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
    x_1_2 = conv2d_bn(x_1_1, filters//16, ksize=3, d_rate=1, strides=1,groups=filters//16)
    x_1_3 = conv2d_bn(x_1_2, filters//8, ksize=3, d_rate=1, strides=1,groups=filters//8)
    
    x_2_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
    x_2_2 = conv2d_bn(x_2_1, filters//16, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//16)
    x_2_3 = conv2d_bn(x_2_2, filters//8, ksize=3, d_rate=d_size//4+1, strides=1, groups=filters//8)

    x_3_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
    x_3_2 = conv2d_bn(x_3_1, filters//16, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//16)
    x_3_3 = conv2d_bn(x_3_2, filters//8, ksize=3, d_rate=d_size//2+1, strides=1, groups=filters//8)
    
    x_4_1 = conv2d_bn(x_inp, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
    x_4_2 = conv2d_bn(x_4_1, filters//16, ksize=3, d_rate=d_size+1, strides=1, groups=filters//16)
    x_4_3 = conv2d_bn(x_4_2, filters//8, ksize=3, d_rate=d_size+1, strides=1, groups=filters//8)
    
    o_1 = concatenate([x_1_1,x_1_2,x_1_3], axis=3)
    o_2 = concatenate([x_2_1,x_2_2,x_2_3], axis=3)
    o_3 = concatenate([x_1_1,x_3_2,x_3_3], axis=3)
    o_4 = concatenate([x_1_1,x_4_2,x_4_3], axis=3)
    
    o_1 = BatchNormalization(axis=3)(o_1)
    o_2 = BatchNormalization(axis=3)(o_2)
    o_3 = BatchNormalization(axis=3)(o_3)
    o_4 = BatchNormalization(axis=3)(o_4)
    
    ad1 = o_1
    ad2 = add([ad1,o_2])
    ad3 = add([ad2,o_3])
    ad4 = add([ad3,o_4])
    output = concatenate([ad1,ad2,ad3,ad4],axis=3)
    output = BatchNormalization(axis=3)(output)
    output = conv2d_bn(output, filters, ksize=1, d_rate=1, strides=1,padding='valid')
    output = add([output, inp])

    return output

def CFPNetM(inputs, slicesParameters):
    nfilters = 16
    nfilters *= slicesParameters['filtersFactor']
    conv1 = conv2d_bn(inputs, nfilters * slicesParameters['firstElementFiltersFactor'], 3, 1, 2)
    conv2 = conv2d_bn(conv1, nfilters, 3, 1, 1)
    conv3 = conv2d_bn(conv2, nfilters, 3, 1, 1)
    
    injection_1 = AveragePooling2D()(inputs)
    injection_1 = BatchNormalization(axis=3)(injection_1)
    injection_1 = Activation('relu')(injection_1)
    opt_cat_1 = concatenate([conv3,injection_1], axis=3)
    
    #CFP block 1
    opt_cat_1_0 = conv2d_bn(opt_cat_1, nfilters * 2, 3, 1, 2)
    cfp_1 = CFPModule(opt_cat_1_0, nfilters * 2, 2)
    cfp_2 = CFPModule(cfp_1, nfilters * 2, 2)
    
    injection_2 = AveragePooling2D()(injection_1)
    injection_2 = BatchNormalization(axis=3)(injection_2)
    injection_2 = Activation('relu')(injection_2)
    opt_cat_2 = concatenate([cfp_2,opt_cat_1_0,injection_2], axis=3)
    
    #CFP block 2
    opt_cat_2_0 = conv2d_bn(opt_cat_2, nfilters * 4, 3, 1, 2)
    cfp_3 = CFPModule(opt_cat_2_0, nfilters * 4, 4)
    cfp_4 = CFPModule(cfp_3, nfilters * 4, 4)
    cfp_5 = CFPModule(cfp_4, nfilters * 4, 8)
    cfp_6 = CFPModule(cfp_5, nfilters * 4, 8)
    cfp_7 = CFPModule(cfp_6, nfilters * 4, 16)
    cfp_8 = CFPModule(cfp_7, nfilters * 4, 16)
    
    injection_3 = AveragePooling2D()(injection_2)
    injection_3 = BatchNormalization(axis=3)(injection_3)
    injection_3 = Activation('relu')(injection_3)
    opt_cat_3 = concatenate([cfp_8,opt_cat_2_0,injection_3], axis=3)
    
    
    conv4 = Conv2DTranspose(nfilters * 4,(2,2),strides=(2,2),padding='same',activation='relu')(opt_cat_3)
    up_1 = concatenate([conv4,opt_cat_2])    
    conv5 = Conv2DTranspose(nfilters * 2,(2,2),strides=(2,2),padding='same',activation='relu')(up_1)
    up_2 = concatenate([conv5, opt_cat_1],axis=3)        
    conv6 = Conv2DTranspose(nfilters,(2,2),strides=(2,2),padding='same',activation='relu')(up_2)    
    conv7 = conv2d_bn(conv6, slicesParameters['nclasses'], 1, 1, 1, activation='sigmoid', padding='valid')
    
    model = tf.keras.Model(inputs=inputs, outputs=conv7)
    
    return model