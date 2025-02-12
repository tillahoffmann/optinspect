# optinspect [![optinspect](https://github.com/tillahoffmann/optinspect/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/optinspect/actions/workflows/build.yml)

`optinspect` provides `optax` gradient transformation to inspect optimization algorithms while leaveing updates unchanged. `optinspect` transformations belong to two categories:

- `*_update` transformations to inspect updates directly.
- `*_wrapped` transformations to inspect the state of a wrapped transformation after it is applied.

```python
>>> import jax
>>> from jax import numpy as jnp
>>> import optax
>>> import optinspect

>>> value_and_grad = jax.value_and_grad(jnp.square)
>>> params = 4.0

>>> optim = optax.chain(
...     optinspect.trace_update("raw"),
...     optax.clip_by_global_norm(1.0),
...     optinspect.trace_update("clipped"),
...     optinspect.trace_wrapped(
...         optax.scale_by_adam(),
...         "adam_nu",
...         key=lambda _, state, *args, **kwargs: state.nu,
...     ),
...     optax.scale_by_learning_rate(0.01),
... )
>>> state = optim.init(params)
>>> value, grad = value_and_grad(params)
>>> updates, state = optim.update(grad, state, value=value)
>>> optinspect.get_traced_values(state)
{'raw': Array(8., ...), 'clipped': Array(1., ...), 'adam_nu': Array(0.001, ...)}

```
