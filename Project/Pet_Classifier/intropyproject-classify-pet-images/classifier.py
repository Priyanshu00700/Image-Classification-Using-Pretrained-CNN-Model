from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch import __version__

# Import weights enums
from torchvision.models import (
    resnet18, ResNet18_Weights,
    alexnet, AlexNet_Weights,
    vgg16, VGG16_Weights
)

# Load models with proper weights
weights_dict = {
    "resnet": ResNet18_Weights.DEFAULT,
    "alexnet": AlexNet_Weights.DEFAULT,
    "vgg": VGG16_Weights.DEFAULT
}

models_dict = {
    "resnet": resnet18(weights=weights_dict["resnet"]),
    "alexnet": alexnet(weights=weights_dict["alexnet"]),
    "vgg": vgg16(weights=weights_dict["vgg"])
}

# Get labels (categories) for each model
labels_dict = {
    model_name: weights.meta["categories"]
    for model_name, weights in weights_dict.items()
}

def classifier(img_path, model_name):
    # load the image
    img_pil = Image.open(img_path)

    # define transforms (from weights)
    preprocess = weights_dict[model_name].transforms()

    # preprocess the image
    img_tensor = preprocess(img_pil).unsqueeze(0)  # add batch dim

    # handle PyTorch version
    pytorch_ver = __version__.split('.')
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)

    # get model
    model = models_dict[model_name].eval()

    # forward pass
    output = model(img_tensor)

    # get predicted class index
    pred_idx = output.data.numpy().argmax()

    # return human-readable label
    return labels_dict[model_name][pred_idx]