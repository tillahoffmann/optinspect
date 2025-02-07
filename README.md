# optinspect [![optinspect](https://github.com/tillahoffmann/optinspect/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/optinspect/actions/workflows/build.yml)

`optinspect` provides `optax` gradient transformation to inspect optimization algorithms while leaveing updates unchanged. `optinspect` transformations belong to two categories:

- `*_on_update` transformations to inspect updates directly.
- `*_before_after_update` transformations to inspect the state of a wrapped transformation before and after it is applied.
