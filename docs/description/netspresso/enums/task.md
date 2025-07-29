# Task

::: netspresso.enums.Task
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Tasks

| Name                    | Value         | Description                   | Domain          |
|-------------------------|---------------|-------------------------------|-----------------|
| IMAGE_CLASSIFICATION    | classification | Image classification task     | Computer Vision |
| OBJECT_DETECTION        | detection     | Object detection task         | Computer Vision |
| SEMANTIC_SEGMENTATION   | segmentation  | Semantic segmentation task    | Computer Vision |

## Example

```python
from netspresso.enums import Task

# Select a task
task = Task.IMAGE_CLASSIFICATION

# Available tasks
print(f"Image Classification: {Task.IMAGE_CLASSIFICATION}")
print(f"Object Detection: {Task.OBJECT_DETECTION}")
print(f"Semantic Segmentation: {Task.SEMANTIC_SEGMENTATION}")
```
