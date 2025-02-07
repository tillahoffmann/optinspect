# optinspect

`optinspect` provides `optax` gradient transformation that leave updates unchanged but are useful for inspecting and debugging optimization algorithms. `optinspect` transformations belong to two classes: `*_on_update` transformations to inspect updates directly and `*_before_after_update` wrappers that facilitate inspection of the state of another transformation before and after it is applied. All `optinspect` transformations can be skipped if code is `jit`-compiled to remove performance overheads.

## Printing Gradient and State Information

- `print_on_update`: Print updates, parameters, or extra arguments.
- `print_before_after_update`: Print state information before and/or after updates.

## Accumulating Gradient and State Information

- `accumulate_on_update`: Accumulate updates, parameters, or extra arguments.

## Trace Gradient Information

- `trace_on_update`: Add a traced value to the state.

## Applying Arbitrary Functions

- `on_update`: Call an arbitrary function.
- `before_after_update`: Call functions before and after applying a transformation.
