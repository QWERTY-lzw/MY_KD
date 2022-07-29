from .layers import QConv2d, QAdd, QConcat, Focus, SPP, Upsample

def build_torch_layers(layers):
    torch_layers = []
    for info in layers:
        if info['type'] == 'Conv':
            torch_layer = QConv2d
        elif info['type'] == 'Add':
            torch_layer = QAdd
        elif info['type'] == 'Concat':
            torch_layer = QConcat
        elif info['type'] == 'Resize':
            torch_layer = Upsample
        elif info['type'] == 'Focus':
            torch_layer = Focus
        elif info['type'] == 'SPP':
            torch_layer = SPP
        else:
            raise NotImplementedError
        torch_layers.append(torch_layer(info))
    return torch_layers
