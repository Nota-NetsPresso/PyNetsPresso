# Policy

::: netspresso.enums.Policy
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Policies

| Name    | Value   | Description                           |
|---------|---------|---------------------------------------|
| SUM     | sum     | Sum policy for handling connected filters |
| AVERAGE | average | Average policy for handling connected filters |

## Example

```python
from netspresso.enums import Policy

# Select a policy
policy = Policy.AVERAGE

# Available policies
print(f"Sum: {Policy.SUM}")
print(f"Average: {Policy.AVERAGE}")
```

