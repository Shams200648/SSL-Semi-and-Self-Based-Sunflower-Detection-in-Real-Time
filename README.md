## Installation (Windows)

    > pip install --upgrade setuptools
    > pip install hidapi
    > pip install xarm
    > pip instal pyserial

## Methods and examples

### Connecting to the xArm

This example is the bare minimum to connect to the xArm via USB and read the battery voltage.

```py
import xarm

# arm is the first xArm detected which is connected to USB
arm = xarm.Controller('USB')
print('Battery voltage in volts:', arm.getBatteryVoltage())
```
Output:
> Battery voltage: 7.677