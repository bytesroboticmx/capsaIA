from .wrapper import Wrapper


def wrap(model, includeHistogram, includeAleatoric, mode):
    # currently only support histogram biases in simplest version of this skeleton

    # TODO: allow selecting what awarenesses to add (histogram, ensemble, mve,
    # etc) and then dynamically construct the tf.keras.Model wrapper with these
    # components.
    return Wrapper(model, includeHistogram, includeAleatoric, mode)
