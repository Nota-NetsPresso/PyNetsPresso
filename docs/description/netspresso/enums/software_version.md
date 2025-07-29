# SoftwareVersion

::: netspresso.enums.SoftwareVersion
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Software Versions

| Name          | Value        | Description                   | JetPack Version |
|---------------|--------------|-------------------------------|-----------------|
| JETPACK_4_4_1 | 4.4.1-b50    | JetPack 4.4.1                | 4.4.1          |
| JETPACK_4_6   | 4.6-b199     | JetPack 4.6                  | 4.6            |
| JETPACK_5_0_1 | 5.0.1-b118   | JetPack 5.0.1                | 5.0.1          |
| JETPACK_5_0_2 | 5.0.2-b231   | JetPack 5.0.2                | 5.0.2          |
| JETPACK_6_1   | 6.1+b123     | JetPack 6.1                  | 6.1            |

## Example

```python
from netspresso.enums import SoftwareVersion

# Select a software version
version = SoftwareVersion.JETPACK_5_0_2

# Available versions
print(f"JetPack 4.4.1: {SoftwareVersion.JETPACK_4_4_1}")
print(f"JetPack 4.6: {SoftwareVersion.JETPACK_4_6}")
print(f"JetPack 5.0.1: {SoftwareVersion.JETPACK_5_0_1}")
print(f"JetPack 5.0.2: {SoftwareVersion.JETPACK_5_0_2}")
print(f"JetPack 6.1: {SoftwareVersion.JETPACK_6_1}")
```
