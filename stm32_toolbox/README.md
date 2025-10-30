# STM32 Toolbox

## Preparation and Build Instructions

### 1. Prepare Generated Files

- Run validate of `stm32ai`, then generate output.
- Replace the contents of `stm32ai_ws/../generated` with the contents of `stm32ai_output/*`.

### 2. Manual Build Steps

To build the project manually, run the following commands in your shell:

```sh
# Compile generated network files
gcc  -c generated/network_data_params.c -Iinclude/ -Igenerated/  -o network_data_params.o

gcc  -c generated/network_data.c -Iinclude/ -Igenerated/  -o network_data.o

gcc  -c generated/network.c -Iinclude/ -Igenerated/  -o network.o

# Compile main C++ source file
g++  -c Source.cpp -Iinclude/ -Igenerated/  -o Source.o

# Link everything into the 'out' executable
g++ Source.o network.o network_data.o network_data_params.o -L./lib/static -lruntime -lst_cmsis_nn -lcmsis-nn -lx86_cmsis -lm  -o out
```
