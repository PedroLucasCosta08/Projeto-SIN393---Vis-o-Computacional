from torchvision import models
import torch.nn as nn 

import timm

def create_model(arch, ft, num_classes, bce, fe=False):
    """
    fe : perform feature extraction
    """

    input_size = 224

    """ torchvision """
    if arch == 'alexnet':
        print(f'\nModel: AlexNet')
        if ft:
            model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        else:
            model = models.alexnet(weights=None)

        if fe:
            #Congelar o treinamento para todas as camadas
            for param in model.features.parameters():
                param.requires_grad = False

            #Remover as camadas de classificação
            features = list(model.classifier.children())[:-7]

            #Substituir o classificador do modelo
            model.classifier = nn.Sequential(*features)

        else: # trainamento
            num_ftrs = model.classifier[6].in_features # alexnet: 4096

            if num_classes == 2 and bce:
                # Classificação binária:
                # Apenas 1 neurônio na camada de saída e função de perda (loss) BCELoss.
                model.classifier[6] = nn.Linear(num_ftrs, 1)
            else:
                # Classicação não binária: 
                # Número de neurônios na camada de sáida igual ao número de classes. 
                # Função de perda CrossEntropyLoss.
                model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    elif arch == 'vgg11':
        print('\nModel: VGG11_BN')
        if ft:
            model = models.vgg11_bn(weights='VGG11_BN_Weights.IMAGENET1K_V1')
        else:
            model = models.vgg11_bn(weights=None)

        if fe:
            #Congelar o treinamento para todas as camadas
            for param in model.features.parameters():
                param.requires_grad = False

            #Remover as camadas de classificação
            features = list(model.classifier.children())[:-7]

            #Substituir o classificador do modelo
            model.classifier = nn.Sequential(*features)

        else: # treinamento
            num_ftrs = model.classifier[6].in_features # vgg11_bn: 4096

            if num_classes == 2 and bce:
                model.classifier[6] = nn.Linear(num_ftrs, 1)
            else:
                model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    # elif arch == 'vgg16':
    #     print(f'\nModel: VGG16_BN')
    #     if ft:
    #         model = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
    #     else:
    #         model = models.vgg16_bn(weights=None)

    #     num_ftrs = model.classifier[6].in_features # vgg16_bn: 4096

    #     if num_classes == 2 and bce:
    #         model.classifier[6] = nn.Linear(num_ftrs, 1)
    #     else:
    #         model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    #     # Grad-cam
    #     target_layers = model.features[-1]

    
    elif arch == 'vgg16':
        print(f'\nModel: VGG16_BN')
        if ft:
            model = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        else:
            model = models.vgg16_bn(weights=None)

        #Congelar o treinamento para todas as camadas
        for param in model.features.parameters():
            param.requires_grad = False

        #Remover as camadas de classificação
        features = list(model.classifier.children())[:-7]

        #Substituir o classificador do modelo
        model.classifier = nn.Sequential(*features)

        target_layers = None

    elif arch == 'resnet18':
        print('\nModel: ResNet18')
        if ft:
            model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet18(weights=None)

        num_ftrs = model.fc.in_features # resnet18: 512

        if num_classes == 2 and bce:
            model.fc = nn.Linear(num_ftrs, 1)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]]

    elif arch == 'resnet50':
        print('\nModel: ResNet50')
        if ft:
            model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        else:
            model = models.resnet50(weights=None)


        if fe: 
            #remover a camada totalmente conectada
            model.fc = nn.Identity()
        else:
            num_ftrs = model.fc.in_features # resnet18: 2048

            if num_classes == 2 and bce:
                model.fc = nn.Linear(num_ftrs, 1)
            else:
                model.fc = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = [model.layer4[-1]]

    elif arch == 'squeezenet':
        print('\nModel: SqueezeNet1_0')
        if ft:
            model = models.squeezenet1_0(weights='SqueezeNet1_0_Weights.IMAGENET1K_V1')
        else:
            model = models.squeezenet1_0(weights=None)

        if num_classes == 2 and bce:
            model.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))
        else:
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        # Grad-cam
        target_layers = model.features[-1] # Defined by joaofmari!

    elif arch == 'densenet':
        print('\nModel: DenseNet121')
        if ft:
            model = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        else:
            model = models.densenet121(weights=None)

        num_ftrs = model.classifier.in_features # Densenet121: 1024

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(num_ftrs, 1)
        else:
            model.classifier = nn.Linear(num_ftrs, num_classes)

        # Grad-cam
        target_layers = model.features[-1]

    elif arch == 'inception':
        print('\nModel: Inception_V3')
        if ft:
            model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        else:
            model = models.inception_v3(weights=None)

        num_ftrs1 = model.AuxLogits.fc.in_features # Inception_v3: 768
        num_ftrs2 = model.fc.in_features # Inception_v3: 2048

        if num_classes == 2 and bce:
            model.AuxLogits.fc = nn.Linear(num_ftrs1, 1)
            model.fc = nn.Linear(num_ftrs2, 1)
        else:
            model.AuxLogits.fc = nn.Linear(num_ftrs1, num_classes)
            model.fc = nn.Linear(num_ftrs2, num_classes)

        # *** ATENTION *** A entrada da rede Inception possui tamanho 299x299
        input_size = 299
        
        # Grad-cam (TODO)
        target_layers = None

    # **** BETA 2 ****
    elif arch == 'vit':
        print('\nModel: ViT_B_16')

        if ft:
            model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        else:
            model = models.vit_b_16(weights=None)

        num_ftrs = model.heads.head.in_features # vit_b_16: 768

        if fe: 
            model.heads = nn.Identity()
        else:
            if num_classes == 2 and bce:
                model.heads.head = nn.Linear(num_ftrs, 1, bias=True)
            else:
                model.heads.head = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam
        target_layers = None

    elif arch == 'efficientnet_b4':
        print('\nModel: EfficientNet')

        if ft:
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_b4(weights=None)
        
        if fe:
            #Congelar o treinamento para todas as camadas
            for param in model.features.parameters():
                param.requires_grad = False

            #Remover as camadas de classificação
            features = list(model.classifier.children())[:-2]

            #Substituir o classificador do modelo
            model.classifier = nn.Sequential(*features)
        
        else:

            num_ftrs = model.classifier[1].in_features # efficientnet_b4: 1792

            if num_classes == 2 and bce:
                model.classifier[1] = nn.Linear(num_ftrs, 1, bias=True)
            else:
                model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[8][2]
        # Grad-cam. #2
        target_layers = model.avgpool

    elif arch == 'mobilenet_v3_large':
        print('\nModel: MobileNet_V3_Large')

        if ft:
            model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1')
        else:
            model = models.mobilenet_v3_large(weights=None)

        num_ftrs = model.classifier[3].in_features # mobilenet_v3_large: 1280

        if num_classes == 2 and bce:
            model.classifier[3] = nn.Linear(num_ftrs, 1, bias=True)
        else:
            model.classifier[3] = nn.Linear(num_ftrs, num_classes, bias=True)

        # Grad-cam. #1
        target_layers = model.features[16][2]
        # Grad-cam. #2
        target_layers = model.avgpool

    # **** BETA ****
    elif arch == 'efficientnet_v2_m':
        print('\nModel: EfficientNet')
        model = models.efficientnet_v2_m(pretrained=ft)
        if ft:
            model = models.efficientnet_b4(weights='EfficientNet_V2_M_Weights.IMAGENET1K_V1')
        else:
            model = models.efficientnet_b4(weights=None)

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(1024, 1)
        else:
            model.classifier = nn.Linear(1024, num_classes)

        # Grad-cam
        target_layers = model.blocks[-1] # Defined by joaofmari


    elif arch == 'convnext':
        print('\nModel: ConvNeXt')
        model = models.convnext_small(pretrained=ft)
        ### model = models.convnext_small(weights='DenseNet121_Weights.IMAGENET1K_V1')

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(1024, 1)
        else:
            model.classifier = nn.Linear(1024, num_classes)

        input_size = 224

        # Grad-cam
        target_layers = None

    elif arch == 'swin':
        print('\nModel: ConvNeXt')
        model = models.convnext_small(pretrained=ft)
        ### model = models.convnext_small(weights='DenseNet121_Weights.IMAGENET1K_V1')

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(1024, 1)
        else:
            model.classifier = nn.Linear(1024, num_classes)

        input_size = 224

        # Grad-cam
        target_layers = None

    elif arch == 'resnext':
        print('\nModel: ResNeXt')
        model = models.efficientnet_v2_m(pretrained=ft)
        ### model = models.convnext_small(weights='DenseNet121_Weights.IMAGENET1K_V1')

        if num_classes == 2 and bce:
            model.classifier = nn.Linear(1024, 1)
        else:
            model.classifier = nn.Linear(1024, num_classes)

        input_size = 224

        # Grad-cam
        target_layers = None

    # **** TIMM ****
    elif arch == 'timm_vit':
        # https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
        print('\nModel: Timm ViT')
        model = timm.create_model("vit_base_patch16_224", pretrained=ft, num_classes=num_classes)

        # Grad-cam
        target_layers = model.blocks[-1].norm1

    elif arch == 'timm_efficientnet':
        #  'efficientnet_b0'
        ### model = timm.create_model("efficientnet_b0", pretrained=ft, num_classes=num_classes)
        model = timm.create_model("efficientnet_b2", pretrained=ft, num_classes=num_classes)

        # Grad-cam
        target_layers = model.blocks[-1] # Defined by joaofmari

    elif arch == 'timm_mobilenet':
        #  'mobilenetv3_large_100'
        model = timm.create_model("mobilenetv3_large_100", pretrained=ft, num_classes=num_classes)

        # Grad-cam
        target_layers = model.blocks[-1] # Defined by joaofmari

    elif arch == 'timm_convnext':
        #  'convnext_base' 'convnext_small'
        model = timm.create_model("convnext_small", pretrained=ft, num_classes=num_classes)

        # Grad-cam
        target_layers = None

    elif arch == 'timm_coatnet':
        #  'coatnet_0_rw_224'   
        model = timm.create_model("coatnet_0_rw_224", pretrained=ft, num_classes=num_classes)

        # Grad-cam
        target_layers = None


    return model, input_size, target_layers