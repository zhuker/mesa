/*
 * cudapipe clear kernel — fills a buffer with a constant value.
 */
#include "cp_rast_types.h"

extern "C" __global__ void
cp_clear_kernel(struct cp_clear_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= args.width || y >= args.height)
      return;

   uint8_t *base = (uint8_t *)(uintptr_t)args.target;
   uint8_t *pixel = base + y * args.stride + x * args.pixel_size;

   if (args.pixel_size == 4) {
      *(uint32_t *)pixel = args.clear_value[0];
   } else if (args.pixel_size == 8) {
      ((uint32_t *)pixel)[0] = args.clear_value[0];
      ((uint32_t *)pixel)[1] = args.clear_value[1];
   } else if (args.pixel_size == 16) {
      ((uint32_t *)pixel)[0] = args.clear_value[0];
      ((uint32_t *)pixel)[1] = args.clear_value[1];
      ((uint32_t *)pixel)[2] = args.clear_value[2];
      ((uint32_t *)pixel)[3] = args.clear_value[3];
   } else if (args.pixel_size == 1) {
      *pixel = (uint8_t)args.clear_value[0];
   } else if (args.pixel_size == 2) {
      *(uint16_t *)pixel = (uint16_t)args.clear_value[0];
   }
}

extern "C" __global__ void
cp_clear_depth_kernel(struct cp_clear_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= args.width || y >= args.height)
      return;

   uint8_t *base = (uint8_t *)(uintptr_t)args.target;
   uint8_t *pixel = base + y * args.stride + x * args.pixel_size;

   if (args.pixel_size == 4) {
      /* D32F or D24X8 */
      *(uint32_t *)pixel = args.clear_value[0];
   } else if (args.pixel_size == 2) {
      /* D16 */
      *(uint16_t *)pixel = (uint16_t)args.clear_value[0];
   }
}

