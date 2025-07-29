# TaskStatus

::: netspresso.enums.TaskStatus
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Task Status

| Name        | Value       | Description                             | State    |
|-------------|-------------|----------------------------------------|----------|
| IN_QUEUE    | IN_QUEUE    | Task is waiting in queue to be processed | Waiting  |
| IN_PROGRESS | IN_PROGRESS | Task is currently being executed       | Active   |
| FINISHED    | FINISHED    | Task has been completed successfully    | Finished |
| ERROR       | ERROR       | Task has failed with an error          | Error    |
| TIMEOUT     | TIMEOUT     | Task has timed out                      | Error    |
| USER_CANCEL | USER_CANCEL | Task has been cancelled by user         | Stopped  |

## Example

```python
from netspresso.enums import TaskStatus

# Check task status
status = TaskStatus.IN_PROGRESS

# Available statuses
print(f"In Queue: {TaskStatus.IN_QUEUE}")
print(f"In Progress: {TaskStatus.IN_PROGRESS}")
print(f"Finished: {TaskStatus.FINISHED}")
print(f"Error: {TaskStatus.ERROR}")
print(f"Timeout: {TaskStatus.TIMEOUT}")
print(f"User Cancel: {TaskStatus.USER_CANCEL}")
```

