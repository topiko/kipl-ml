# Clarifications

## Replacement counting for padding buffers

- `replace=True` on padding buffers is counted per direction.
- Count `UP` and `DOWN` normal packets separately.
- Do not subtract packets from the opposite direction when applying replacement.
- Replacement counting is based on scheduling state, not packet timestamps.

## Delay semantics

- Delay state is tracked explicitly in the cursor.
- `replace=False` must not overwrite an already active delay.
- `replace=True` overwrites the active delay state.
- `bypass=True` only allows bypass-enabled traffic to skip the delay.

## Open questions

- Should replacement counting remain direction-specific for all future buffer types?
- Should delay replacement follow the same direction-specific rule as padding replacement?
