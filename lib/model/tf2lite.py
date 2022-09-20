TF_PATH = "./hw_rec_model_pad_0809.pb" # where the forzen graph is stored
TFLITE_PATH = "./hw_rec_model_pad_0809.tflite"
import tensorflow as tf

# protopuf needs your virtual environment to be explictly exported in the path
# os.environ["PATH"] = "/opt/miniconda3/envs/convert/bin:/opt/miniconda3/bin:/usr/local/sbin:...."

# make a converter object from the saved tensorflow file
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(TF_PATH,  # TensorFlow freezegraph .pb model file
                                                      input_arrays=['input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['output'] # name of output arrays defined in torch.onnx.export function before.
                                                      )
# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]
# converter.allow_custom_ops=True
converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS ] #, tf.compat.v1.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)