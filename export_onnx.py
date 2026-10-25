import torch
import torchvision
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
import onnx

# Pretrained semantic segmentation model.
weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
model = lraspp_mobilenet_v3_large(weights=weights)
model.eval()

# Export with a fixed input resolution for simplicity.
dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "person_lraspp.onnx",
    input_names=["images"],
    output_names=["logits"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False,
    external_data=False,
)
onnx_model = onnx.load("person_lraspp.onnx")
onnx_model.ir_version = 8
onnx.save(onnx_model, "person_lraspp.onnx")


print("Exported person_lraspp.onnx")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
