import torch


from .AANet import AANet
from .AANet_weight import AANet_weight
model_dict = {
    "AANet",
    "AANet_weight",
}

def get_model(args):
    model_name = args.model
    num_classes = args.num_classes
    
    assert model_name in model_dict, '{} is not in model dictionary'.format(model_name)
    
    try:
        if args.model == 'AANet':
            model = AANet(args,num_classes=num_classes)
        elif args.model == 'AANet_weight':
            model = AANet_weight(args,num_classes=num_classes)
    except:
        raise 'Error occured in model building...\n model name is {}'.format(model_name)

    return model


    
