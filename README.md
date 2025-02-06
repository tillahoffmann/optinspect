# optinspect

Inspect and debug optax gradient transformations.

## Printing Gradient and State Information

- `print_on_update`: Print updates, parameters, or extra arguments without changing updates.
- `print_before_after_update`: Print state information before and/or after updates.

## Applying Arbitrary Functions

- `on_update`: Call a function and leave the updates unchanged.
- `before_after_update`: Call functions before and after applying a transformation.
