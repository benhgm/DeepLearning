# Experimenting with Accelerated PyTorch Training on Mac
On 18 May 2022, PyTorch announced the release of GPU-acceleration on Mac with the upcoming v1.12 release. This is huge. As a heavy user of PyTorch myself, the idea of training deep neural networks on Apple's famous M1 chip excited me for three reasons.

## 1. More efficient power consumption during training
Training deep learning networks consumes a lot of power. Those familiar with this would hav had experienced their GPUs turning up the heat and the sound when they push their batch sizes up.
M1 is famous for being incredibly quiet and power-efficient when in operation. If this trait is maintained during deep learning training, image what it could do for systems engineering teams around the world.

## 2. Saying goodbye to builky workstations and computers
We've seen how chunky those RTXs are. If GPU-accelearation with the M1 takes off, think about what this means for data server designs, onboard computers for self-driving cars, UAVs, etc.

## 3. Reduce memory leaks, improve latency during inference
Conventional CPU-GPU architectures may experience latency issues and (sometimes) memory leaks during data transfer between the CPU and GPU, especially in times of high inference rates. In robotics, where real-time data is extremely important, this is a big problem. M1 uses a unified memory architecture, which means the CPU and GPU cores share the same memory pool. This also means that both the CPU and GPU directly read from this memory pool, instead of having data transfer operations between the two, therby giving us the possibility of reduced memory leaks and faster data transfer & processing.

In this repository, I make a modest attempt to experiment with GPU-acceleration on my Mac using PyTorch. I am currently using a base model 14-inch Macbook Pro with M1 Pro chip. To start, I will be experimenting with the following, with more experiments to add as I go along.
- Image Processing
- Model Inference
- Finetune a Pre-trained Model

In addition, I will also update on my findings on the capabilities of GPU-acceleration. Given it is still new, I expect many functions to still be unsupported.
