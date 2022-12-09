<br />
<p align="center">
    <a href="https://github.com/themis-ai/capsa#gh-light-mode-only" class="only-light">
      <img src="https://raw.githubusercontent.com/themis-ai/capsa/main/docs/source/assets/header-light.svg" width="50%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
    <a href="https://github.com/themis-ai/capsa#gh-dark-mode-only" class="only-dark">
      <img src="https://raw.githubusercontent.com/themis-ai/capsa/main/docs/source/assets/header-dark.svg" width="50%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>

<h2><p align="center">A Library for Risk-Aware and Trustworthy Machine Learning</p></h2>

<h4><p align='center'>
<a href="https://www.themisai.io">[🌐 Website]</a>
- <a href="https://themisai.io/capsa/getting_started/basic_usage.html">[🚀 Getting Started]</a>
- <a href="https://themisai.io/capsa/">[📄 Docs]</a>
- <a href="https://themisai.io/company.html">[🧠 We're Hiring!]</a>
</p></h4>

<p align="center">
    <a href="https://pypi.org/project/capsa/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/capsa">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/capsa">
    </a>
    <!--
    <a href="https://pypi.org/project/capsa/">
        <img alt="PyPi Downloads" src="https://pepy.tech/badge/capsa">
    </a>
    -->
    <!--
    <a href="https://themisai.io/capsa">
        <img alt="Documentation" src="https://readthedocs.org/projects/capsa/badge/?version=stable">
    </a>
    -->
    <a href="https://github.com/themis-ai/capsa/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/themis-ai/capsa?color=green&logo=slack">
    </a>
</p>
<br />

# 👋 Welcome

We know deploying machine learning models can be tough. Today's models are notoriously bad at understanding their own risks -- they are biased on underrepresented data, brittle on challenging out-of-distribution scenarios, and can fail without warning when insufficiently trained.

Ensuring awareness of not one, but all of these risks, requires a tedious process involving changes to your model, its architecture, loss function, optimization procedure, and more.

Luckily, capsa has got you covered! Capsa automatically wraps your model (i.e., like a <i>capsule</i>!) and makes all of the internal changes so it can be end-to-end risk-aware. Capsa abstracts away all of those changes so you don't have to change any of your existing training or deployment pipelines in order to build state-of-the-art trustworthy machine learning solutions.

# 🚀 Quickstart

## 💾 Installation
capsa is available to be downloaded with Pip:

```bash
pip install capsa
```

## ⭐ Wrap your model!
Eager to make your models risk-aware? Let's go through a quick example of wrapping your model (e.g., using an `MVEWrapper`) to estimate risk from noise in your labels (i.e., aleatoric uncertainty).

```python
import capsa
import tensorflow as tf

# Build your model
model = tf.keras.Sequential(...)

# Wrap the model with capsa to make it risk-aware.
#   Capsa takes care of all the architecture, loss,
#   and deployment changes so you don't have to!
model = MVEWrapper(model)

# Compile and train the wrapped model the
#   same as you would have done with the
#   original model. No changes!
model.compile(...)
model.fit(train_x, train_y, epochs=5)

# The model now outputs `RiskTensor` objects, which
#   behave just like a normal `Tensor`, except they also
#   contain multiple different quantitative risk measures.
pred_y = model(test_x)

# Returns the aleatoric uncertainty of this prediction
risk = pred_y.aleatoric
```

## 🧠 Tutorials
Hungry for more?

Checkout our <a href="https://themisai.io/capsa/tutorials">tutorials</a> on some more advanced functions with capsa including other forms of risk, composing wrappers together, high-dimensional datasets, and more! All tutorials can be opened directly in Google Collab so you can play around without needing access to GPUs.


# 💪 Contribution

Capsa is being actively maintained and advanced. It has been built with research, extensibility, and community development as a priority. We greatly appreciate contributions to the capsa repository and codebase, including issues, enhancements, and pull requests.

For more details please see <a href="https://themisai.io/capsa/contribute/">here</a>.
