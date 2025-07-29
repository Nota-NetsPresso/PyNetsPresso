# Upload Dataset

::: netspresso.np_qai.base.NPQAIBase.upload_dataset
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
dataset = np_qai.upload_dataset("YOUR_DATASET_PATH")
``` 