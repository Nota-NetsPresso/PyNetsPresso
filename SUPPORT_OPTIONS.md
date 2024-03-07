## Supported options for Converter & Benchmarker

### Frameworks that support conversion for model's framework

| Target / Source Framework | ONNX | TENSORFLOW_KERAS | TENSORFLOW |
|:--------------------------|:----:|:----------------:|:----------:|
| TENSORRT                  |  ✔️  |                  |            |
| DRPAI                     |  ✔️  |                  |            |
| OPENVINO                  |  ✔️  |                  |            |
| TENSORFLOW_LITE           |  ✔️  |        ✔️        |     ✔️     |

### Devices that support benchmarks for model's framework

| Device / Framework           | ONNX | TENSORRT | TENSORFLOW_LITE | DRPAI | OPENVINO |
|:-----------------------------|:----:|:--------:|:---------------:|:-----:|:--------:|
| RASPBERRY_PI_5               |  ✔️  |          |       ✔️        |       |          |
| RASPBERRY_PI_4B              |  ✔️  |          |       ✔️        |       |          |
| RASPBERRY_PI_3B_PLUS         |  ✔️  |          |       ✔️        |       |          |
| RASPBERRY_PI_ZERO_W          |  ✔️  |          |       ✔️        |       |          |
| RASPBERRY_PI_ZERO_2W         |  ✔️  |          |       ✔️        |       |          |
| ARM_ETHOS_U_SERIES           |      |          |  ✔️(only INT8)  |       |          |
| ALIF_ENSEMBLE_E7_DEVKIT_GEN2 |      |          |  ✔️(only INT8)  |       |          |
| RENESAS_RA8D1                |      |          |  ✔️(only INT8)  |       |          |
| NXP_iMX93                    |      |          |  ✔️(only INT8)  |       |          |
| RENESAS_RZ_V2L               |  ✔️  |          |                 |  ✔️   |          |
| RENESAS_RZ_V2M               |  ✔️  |          |                 |  ✔️   |          |
| JETSON_NANO                  |  ✔️  |    ✔️    |                 |       |          |
| JETSON_TX2                   |  ✔️  |    ✔️    |                 |       |          |
| JETSON_XAVIER                |  ✔️  |    ✔️    |                 |       |          |
| JETSON_NX                    |  ✔️  |    ✔️    |                 |       |          |
| JETSON_AGX_ORIN              |  ✔️  |    ✔️    |                 |       |          |
| JETSON_ORIN_NANO             |  ✔️  |    ✔️    |                 |       |          |
| AWS_T4                       |  ✔️  |    ✔️    |                 |       |          |
| INTEL_XEON_W_2233            |      |          |                 |       |    ✔️    |

### Software versions that support conversions and benchmarks for specific devices 

Software Versions requires only Jetson Device. If you are using a different device, you do not need to enter it.

| Software Version / Device | JETSON_NANO | JETSON_TX2 | JETSON_XAVIER | JETSON_NX | JETSON_AGX_ORIN | JETSON_ORIN_NANO |
|:--------------------------|:-----------:|:----------:|:-------------:|:---------:|:---------------:|:----------------:|
| JETPACK_4_4_1             |     ✔️      |            |               |           |                 |                  |
| JETPACK_4_6               |     ✔️      |     ✔️     |      ✔️       |    ✔️     |                 |                  |
| JETPACK_5_0_1             |             |            |               |           |       ✔️        |                  |
| JETPACK_5_0_2             |             |            |               |    ✔️     |                 |                  |
| JETPACK_6_0               |             |            |               |           |                 |        ✔️        |

The code below is an example of using software version.

```python
conversion_result = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=Framework.TENSORRT,
    target_device_name=DeviceName.JETSON_AGX_ORIN,
    target_software_version=SoftwareVersion.JETPACK_5_0_1,
)
benchmark_result = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=DeviceName.JETSON_AGX_ORIN,
    target_software_version=SoftwareVersion.JETPACK_5_0_1,
)
```

### Hardware type that support benchmarks for specific devices

Benchmark and compare models with and without Arm Helium.

`RENESAS_RA8D1` and `ALIF_ENSEMBLE_E7_DEVKIT_GEN2` are available for use.

The benchmark results with Helium can be up to twice as fast as without Helium.

The code below is an example of using hardware type.

```python
benchmark_result = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=DeviceName.RENESAS_RA8D1,
    target_data_type=DataType.INT8,
    target_hardware_type=HardwareType.HELIUM
)
```