"""
    A simple factory module that returns instances of possible modules 

"""

from .models import ETE, VAE,  Separate_Affordances, STDIM, ACTION_PREDICTION, FIMBC, \
    Affordances_Separate, ETE_inverse_model, ForwardInverse, FIMBC2


def CoILModel(architecture_name, architecture_configuration, encoder_architecture_configuration = None):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition
    if architecture_name in ['separate-affordances']:

        return Separate_Affordances(architecture_configuration, encoder_architecture_configuration)



def EncoderModel(architecture_name, architecture_configuration):
    if architecture_name == 'ETE':
        return ETE(architecture_configuration)

    elif architecture_name == 'one-step-affordances':
        return Affordances_Separate(architecture_configuration)

    elif architecture_name == 'stdim':
        return STDIM(architecture_configuration)

    elif architecture_name == 'action_prediction':
        return ACTION_PREDICTION(architecture_configuration)

    elif architecture_name == 'ETE_inverse_model':
        return ETE_inverse_model(architecture_configuration)

    elif architecture_name == 'forward':
        return ForwardInverse(architecture_configuration)

    elif architecture_name == 'FIMBC':
        return FIMBC2(architecture_configuration)

    #elif architecture_name == 'FIMBC2':
    #    return FIMBC2(architecture_configuration)

    else:

        raise ValueError(" Not found architecture name")