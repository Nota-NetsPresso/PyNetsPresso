# DeviceName

::: netspresso.enums.DeviceName
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Device Names

| Name                      | Value                              | Description                        | Category      |
|---------------------------|------------------------------------|------------------------------------|---------------|
| RASPBERRY_PI_5            | RaspberryPi5                       | Raspberry Pi 5                     | Raspberry Pi  |
| RASPBERRY_PI_4B           | RaspberryPi4B                      | Raspberry Pi 4 Model B            | Raspberry Pi  |
| RASPBERRY_PI_3B_PLUS      | RaspberryPi3BPlus                  | Raspberry Pi 3 Model B+           | Raspberry Pi  |
| RASPBERRY_PI_3B           | RaspberryPi3B                      | Raspberry Pi 3 Model B            | Raspberry Pi  |
| RASPBERRY_PI_2B           | RaspberryPi2B                      | Raspberry Pi 2 Model B            | Raspberry Pi  |
| RASPBERRY_PI_ZERO_W       | RaspberryPi-ZeroW                  | Raspberry Pi Zero W                | Raspberry Pi  |
| RASPBERRY_PI_ZERO_2W      | RaspberryPi-Zero2W                 | Raspberry Pi Zero 2 W              | Raspberry Pi  |
| RENESAS_RZ_V2L            | rzv2l_avnet                        | Renesas RZ/V2L                     | Renesas       |
| RENESAS_RZ_V2M            | rzv2m                              | Renesas RZ/V2M                     | Renesas       |
| RENESAS_RA8D1             | Renesas-RA8D1                      | Renesas RA8D1                      | Renesas       |
| JETSON_NANO               | Jetson-Nano                        | NVIDIA Jetson Nano                 | Jetson        |
| JETSON_TX2                | Jetson-Tx2                         | NVIDIA Jetson TX2                  | Jetson        |
| JETSON_XAVIER             | Jetson-Xavier                      | NVIDIA Jetson Xavier               | Jetson        |
| JETSON_NX                 | Jetson-Nx                          | NVIDIA Jetson NX                   | Jetson        |
| JETSON_AGX_ORIN           | Jetson-AGX-Orin                    | NVIDIA Jetson AGX Orin             | Jetson        |
| JETSON_ORIN_NANO          | Jetson-Orin-Nano                   | NVIDIA Jetson Orin Nano            | Jetson        |
| AWS_T4                    | AWS-T4                             | AWS T4 Instance                    | Cloud         |
| INTEL_XEON_W_2233         | Intel-Xeon                         | Intel Xeon W-2233                  | Intel         |
| ALIF_ENSEMBLE_E7_DEVKIT_GEN2 | Ensemble-E7-DevKit-Gen2         | Alif Ensemble E7 DevKit Gen2       | Alif          |
| ARM_ETHOS_U_SERIES        | Arm Virtual Hardware Ethos-U Series | ARM Ethos-U Series                | ARM           |
| NXP_iMX93                 | nxp_imx93_ethos_u65                | NXP iMX93 with Ethos-U65           | NXP           |
| ARDUINO_NICLA_VISION      | arduino_nicla_vision               | Arduino Nicla Vision               | Arduino       |

## Example

```python
from netspresso.enums import DeviceName

# Select a device
device = DeviceName.JETSON_NANO

# Available devices
print(f"Raspberry Pi 5: {DeviceName.RASPBERRY_PI_5}")
print(f"Jetson Nano: {DeviceName.JETSON_NANO}")
print(f"AWS T4: {DeviceName.AWS_T4}")
print(f"Intel Xeon: {DeviceName.INTEL_XEON_W_2233}")
```

