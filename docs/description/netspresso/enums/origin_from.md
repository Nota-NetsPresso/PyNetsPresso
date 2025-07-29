# OriginFrom

::: netspresso.enums.OriginFrom
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Origin Sources

| Name   | Value  | Description                    | Source     |
|--------|--------|--------------------------------|------------|
| CUSTOM | custom | Custom user-uploaded model    | Custom     |
| NPMS   | npms   | NetsPresso Model Store model   | NPMS       |

## Example

```python
from netspresso.enums import OriginFrom

# Select an origin source
origin = OriginFrom.CUSTOM

# Available origins
print(f"Custom: {OriginFrom.CUSTOM}")
print(f"NPMS: {OriginFrom.NPMS}")
```

