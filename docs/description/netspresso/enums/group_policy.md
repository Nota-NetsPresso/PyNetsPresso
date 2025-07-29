# GroupPolicy

::: netspresso.enums.GroupPolicy
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Group Policies

| Name    | Value   | Description                                   |
|---------|---------|-----------------------------------------------|
| SUM     | sum     | Sum group policy for group convolutions       |
| AVERAGE | average | Average group policy for group convolutions   |
| COUNT   | count   | Count group policy for group convolutions     |
| NONE    | none    | No group policy applied                       |

## Example

```python
from netspresso.enums import GroupPolicy

# Select a group policy
group_policy = GroupPolicy.AVERAGE

# Available policies
print(f"Sum: {GroupPolicy.SUM}")
print(f"Average: {GroupPolicy.AVERAGE}")
print(f"Count: {GroupPolicy.COUNT}")
print(f"None: {GroupPolicy.NONE}")
```

