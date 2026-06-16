import torch.nn as nn
import torchvision


def DenseNet121_pretrain(num_classes):
    api_model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
    api_model.classifier = nn.Linear(api_model.classifier.in_features, num_classes)
    return api_model


def ResNet50_pretrain(num_classes):
    api_model = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
    api_model.fc = nn.Linear(api_model.fc.in_features, num_classes)
    return api_model


def ConvNeXt_pretrain(num_classes):
    api_model = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1")
    num_features = api_model.classifier[2].in_features
    api_model.classifier[2] = nn.Linear(num_features, num_classes)
    return api_model