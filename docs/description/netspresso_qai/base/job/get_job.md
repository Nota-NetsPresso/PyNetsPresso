# Get Job

::: netspresso.np_qai.base.NPQAIBase.get_job
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Example

```python
from netspresso import NPQAI

QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
job = np_qai.get_job("YOUR_JOB_ID")
``` 