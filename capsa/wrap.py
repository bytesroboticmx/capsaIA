import tensorflow as tf
from tensorflow import keras
from .aleatoric import MVEWrapper
from .bias import HistogramWrapper
from .epistemic import VAEWrapper, DropoutWrapper, EnsembleWrapper
from .controller_wrapper import ControllerWrapper


def wrap(model, bias=True, aleatoric=True, epistemic=True):
    """Abstract away the Wrapper and most parameters to simplify the wrapping process for the user."""
    metric_wrappers = []
    hist_with_vae = None
    add_vae = False
    if bias == True:
        add_vae = True
        hist = HistogramWrapper(model, is_standalone=False)
        metric_wrappers.append(hist)
        hist_with_vae = hist
    elif bias == False:
        pass
    elif type(bias) == list:
        add_vae = False
        for i in bias:
            out, vae = _check_bias_compatibility(i)
            add_vae = add_vae or vae
            if type(i) == type:
                metric_wrappers.append(i(model, is_standalone=False))
            else:
                metric_wrappers.append(i)
            if type(i) == HistogramWrapper and type(i.metric_wrapper) != VAEWrapper:
                hist_with_vae = i
    else:
        out, add_vae = _check_bias_compatibility(bias)
        if type(out) == type:
            out = out(model, is_standalone=False)
        metric_wrappers.append(out)

    if aleatoric == True:
        metric_wrappers.append(MVEWrapper(model, is_standalone=False))
    elif aleatoric == False:
        pass
    elif type(aleatoric) == list:
        out = [_check_aleatoric_compatibility(i) for i in aleatoric]
        metric_wrappers.extend(
            [i(model, is_standalone=False) for i in out if type(i) == type]
        )
    else:
        out = _check_aleatoric_compatibility(aleatoric)
        if type(out) == type:
            out = out(model, is_standalone=False)
        metric_wrappers.append(out)

    if epistemic == True:
        metric_wrappers.append(VAEWrapper(model, is_standalone=False))
    elif epistemic == False:
        pass
    elif type(epistemic) == list:
        out = [_check_epistemic_compatibility(i) for i in epistemic]
        metric_wrappers.extend(
            [i(model, is_standalone=False) for i in out if type(i) == type]
        )
        metric_wrappers.extend([i for i in out if type(i) != type])
    else:
        out = _check_epistemic_compatibility(epistemic)
        if type(out) == type:
            out = out(model, is_standalone=False)
        metric_wrappers.append(out)

    vae_exists = False
    for i in metric_wrappers:
        if type(i) == VAEWrapper:
            vae_exists = True
            if hist_with_vae is not None:
                hist_with_vae.metric_wrapper = i

    if add_vae and not (vae_exists):
        vae = VAEWrapper(model, is_standalone=False)
        hist_with_vae.metric_wrapper = vae
        metric_wrappers.append(vae)

    return ControllerWrapper(model, metrics=metric_wrappers)


def _check_bias_compatibility(bias):
    bias_named_wrappers = {
        "HistogramWrapper": HistogramWrapper,
    }
    add_vae = False
    if type(bias) == str and bias in bias_named_wrappers.keys():
        add_vae = bias == "HistogramWrapper"
        return bias_named_wrappers[bias], add_vae
    elif type(bias) == type and bias in bias_named_wrappers.values():
        add_vae = bias == HistogramWrapper
        return bias, add_vae
    elif type(bias) in bias_named_wrappers.values():
        if type(bias) == HistogramWrapper:
            if type(bias.metric_wrapper) == type:
                return bias, add_vae
            elif type(bias.metric_wrapper) == VAEWrapper:
                return bias, bias.metric_wrapper
    else:
        raise ValueError(
            f"Must pass in either a string (one of {bias_named_wrappers.keys()}) or wrapper types (one of {bias_named_wrappers.values()}) or an instance of a wrapper type. Received {bias}"
        )


def _check_aleatoric_compatibility(aleatoric):
    aleatoric_named_wrappers = {"MVEWrapper": MVEWrapper}
    if type(aleatoric) == str and aleatoric in aleatoric_named_wrappers.keys():
        return aleatoric_named_wrappers[aleatoric]
    elif type(aleatoric) == type and aleatoric in aleatoric_named_wrappers.values():
        return aleatoric
    elif type(aleatoric) in aleatoric_named_wrappers.values():
        return aleatoric
    else:
        raise ValueError(
            f"Must pass in either a string (one of {aleatoric_named_wrappers.keys()}) or wrapper types (one of {aleatoric_named_wrappers.values()}) or an instance of a wrapper type. Received {aleatoric}"
        )


def _check_epistemic_compatibility(epistemic):
    epistemic_named_wrappers = {
        "DropoutWrapper": DropoutWrapper,
        "EnsembleWrapper": EnsembleWrapper,
        "VAEWrapper": VAEWrapper,
    }
    if type(epistemic) == str and epistemic in epistemic_named_wrappers.keys():
        return epistemic_named_wrappers[epistemic]
    elif type(epistemic) == type and epistemic in epistemic_named_wrappers.values():
        return epistemic
    elif type(epistemic) in epistemic_named_wrappers.values():
        return epistemic
    else:
        raise ValueError(
            f"Must pass in either a string (one of {epistemic_named_wrappers.keys()}) or wrapper types (one of {epistemic_named_wrappers.values()}) or an instance of a wrapper type. Received {epistemic}"
        )
