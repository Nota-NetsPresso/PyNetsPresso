# HardwareType

::: netspresso.enums.HardwareType
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Hardware Types

| Name   | Value  | Description                          | Category    |
|--------|--------|--------------------------------------|-------------|
| HELIUM | helium | ARM Helium vector processing unit    | ARM         |

## Example

```python
from netspresso.enums import HardwareType

# Select a hardware type
hardware = HardwareType.HELIUM

# Available hardware types
print(f"Helium: {HardwareType.HELIUM}")
```
