from onnx_tf.backend import prepare
import onnx

TF_PATH = "./hw_rec_model_pad_0809.pb" # where the representation of tensorflow model will be stored
ONNX_PATH = "./hw_rec_model_pad_0809.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model

# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)  # creating TensorflowRep object

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)