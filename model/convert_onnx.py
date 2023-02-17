import torch
from model import build_model
import onnx
import numpy as np
import onnxruntime
import time
from class_names import class_names

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )  

def pt_to_onnx(checkpoint = "output/checkpoint/resnext_best.pt", onnx_model_path = "output/checkpoint_onnx/carmodel_classification_resnext_best_0102.onnx",size=(224, 244)):
# Load pretrained model weights
    batch_size = 1  # just a random number
    
    torch_model = build_model(num_classes=len(class_names))
    # Initialize model with the pretrained weights
    def map_location(storage, loc):
        return storage

    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(
        torch.load(
            checkpoint, map_location=torch.device("cpu")
        )["model_state_dict"]
    )
    # set the model to inference mode
    torch_model.eval()
    x = torch.randn(batch_size, 3, size[0], size[1])
    start_time = time.time()
    torch_out = torch_model(x)
    print("Pytorch model inference", time.time() - start_time)
    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        # where to save the model (can be a file or file-like object)
        onnx_model_path,
    )
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    start_time = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX model inference", time.time() - start_time)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    pt_to_onnx()