import torch
from model import MobileFormer

def mobile_former_508(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf508',
        'token': 6,  # tokens and embed_dim
        'embed': 192,
        'stem': 24,
        'bneck': {'e': 48, 'o': 24, 's': 1},
        'body': [
            {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1920,  # hid_layer
        'fc2': num_class   # num_classes
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model


def mobile_former_294(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf294',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 16,
        # stage1
        'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1920,  # hid_layer
        'fc2': num_class   # num_classes
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model

def mobile_former_151(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf151',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 12,
        # stage1
        'bneck': {'e': 24, 'o': 12, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 16, 'exp': 48, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 16, 'exp': 96, 'out': 32, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 32, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 32, 'exp': 192, 'out': 64, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 64, 'exp': 256, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 64, 'exp': 384, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 88, 'exp': 528, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 88, 'exp': 528, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1280,  # hid_layer
        'fc2': num_class   # num_classes`
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model
