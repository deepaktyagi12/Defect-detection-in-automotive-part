from keras import backend as K
from keras.models import load_model
import tensorflow as tf
'''
This code convert the keras .h5 model to TF and TFLite model.
'''
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

def convert_to_tf(trained_models, Keras_model,  Tf_model):
    ###**************This function convert keras .h5 model to TF .pb model
    K.set_learning_phase(0)
    keras_model_file=trained_models+"/"+Keras_model
    model = load_model(keras_model_file)
    sess = K.get_session()
    graph_def = sess.graph.as_graph_def()
    print(model.outputs, model.output_names)
    print(model.inputs, model.input)
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    print("write .pb model")
    tf.io.write_graph(frozen_graph, trained_models, Tf_model, as_text=False)

def convert_to_tflite(trained_models,  Tf_model, tflite_model_name):
    ## Convert the TF .pb model to TFLite .tflite.
    graph_def_file = trained_models+"/"+Tf_model
    input_var=["input_1"]
    output_var= ["dense_1/Softmax"]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_var, output_var)
    tflite_model = converter.convert()
    tflite_model_file=trained_models+"/"+tflite_model_name
    open(tflite_model_file, "wb").write(tflite_model)

def main():
    trained_models = 'Trained_model'
    Keras_model='MobileNet_model_keras.h5'
    Tf_model = 'model.pb'
    tflite_model_name='converted_model.tflite'
    convert_to_tf(trained_models, Keras_model,  Tf_model)
    convert_to_tflite(trained_models,  Tf_model, tflite_model_name)
if __name__ == "__main__":
    main()