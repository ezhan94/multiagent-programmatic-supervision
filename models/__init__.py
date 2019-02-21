from .rnn_gauss import RNN_GAUSS

from .vrnn_single import VRNN_SINGLE
from .vrnn_indep import VRNN_INDEP
from .vrnn_mixed import VRNN_MIXED

from .macro_vrnn import MACRO_VRNN
from .macro_shared_vrnn import MACRO_SHARED_VRNN

from .vrae_mi import VRAE_MI


def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == 'rnn_gauss':
        return RNN_GAUSS(params, parser)
    elif model_name == 'vrnn_single':
        return VRNN_SINGLE(params, parser)
    elif model_name == 'vrnn_indep':
        return VRNN_INDEP(params, parser)
    elif model_name == 'vrnn_mixed':
        return VRNN_MIXED(params, parser)
    elif model_name == 'macro_vrnn':
        return MACRO_VRNN(params, parser)
    elif model_name == 'macro_shared_vrnn':
        return MACRO_SHARED_VRNN(params, parser)
    elif model_name == 'vrae_mi':
        return VRAE_MI(params, parser)
    else:
        raise NotImplementedError
