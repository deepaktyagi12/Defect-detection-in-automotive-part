from keras import backend as K
from keras.applications  import MobileNetV2
from keras.layers import Input, GlobalAveragePooling2D, Activation,Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Dense
import numpy as np
from mobile_model import convert_to_tf, convert_to_tflite
import keras
np.random.seed(0)

###************Mobilenet_v2 model************************************
def train_L2(x, y, classes, input_shape, epochs, alpha, batchsize, metric, adam, trained_models, Keras_model,  Tf_model, tflite_model_name):
    train_label = y
    y = np.eye(2)[train_label]

    mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,  weights='imagenet')
    mobile.layers.pop()
    model = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)
    # c = Dense(classes, activation='softmax')(model.output)      
    if metric == True:
        c = keras.layers.Lambda(lambda xx: 5*(xx)/K.sqrt(K.sum(xx**2)))(model.output) #metric learning
        c = Dense(classes, activation='softmax')(c)
    else:
        c = Dense(classes, activation='softmax')(model.output)

    model = Model(inputs=model.input,outputs=c) 

    if adam == True:
        opt = Adam(lr=0.00005, amsgrad=True)
    else:
        opt = SGD(lr=5e-4, decay=0.00005)

    model.compile(loss='categorical_crossentropy', optimizer=opt,  metrics=['accuracy'])
    model.summary()
    hist = model.fit(x, y, batch_size=batchsize, epochs=epochs, verbose=1, validation_split=0.1)
    print(hist)
    keras_model_file=trained_models+"/"+Keras_model
    keras_model_weight_file=trained_models+"/MobileNet_model_wieghts.h5"
    model.save_weights(keras_model_weight_file)
    model.save(keras_model_file) 
    # convert_to_tf(trained_models, Keras_model,  Tf_model)
    # convert_to_tflite(trained_models,  Tf_model, tflite_model_name)
    return model
