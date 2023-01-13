import torch


from .AANet import AANet
model_dict = {
    "AANet",
}

def get_model(args):
    model_name = args.model
    num_classes = args.num_classes
    
    assert model_name in model_dict, '{} is not in model dictionary'.format(model_name)
    
    try:
        model = AANet(args,num_classes=num_classes)
    except:
        raise 'Error occured in model building...\n model name is {}'.format(model_name)

    return model


    
