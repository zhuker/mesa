/* Image-copy corner cases: compressed-compatible elements and 2D/3D slices. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(expr)                                                        \
   do {                                                                       \
      VkResult _result = (expr);                                              \
      if (_result != VK_SUCCESS) {                                            \
         fprintf(stderr, "%s -> %d\n", #expr, _result);                    \
         return 1;                                                            \
      }                                                                       \
   } while (0)

static uint32_t
memory_type(VkPhysicalDevice pdev, uint32_t bits,
            VkMemoryPropertyFlags wanted)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(pdev, &props);
   for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & wanted) == wanted)
         return i;
   }
   return UINT32_MAX;
}

static int
make_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size,
            VkBufferUsageFlags usage, VkBuffer *buffer, VkDeviceMemory *memory)
{
   VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_CHECK(vkCreateBuffer(dev, &info, NULL, buffer));

   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *buffer, &req);
   VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(
         pdev, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, memory));
   VK_CHECK(vkBindBufferMemory(dev, *buffer, *memory, 0));
   return 0;
}

static int
make_image(VkDevice dev, VkPhysicalDevice pdev, VkImageType type,
           VkFormat format, uint32_t width, uint32_t height, uint32_t depth,
           uint32_t layers, VkImage *image, VkDeviceMemory *memory)
{
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = type,
      .format = format,
      .extent = { width, height, depth },
      .mipLevels = 1,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VK_CHECK(vkCreateImage(dev, &info, NULL, image));

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, *image, &req);
   VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(pdev, req.memoryTypeBits, 0),
   };
   VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, memory));
   VK_CHECK(vkBindImageMemory(dev, *image, *memory, 0));
   return 0;
}

int
main(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   VK_CHECK(vkCreateInstance(&instance_info, NULL, &instance));

   uint32_t count = 1;
   VkPhysicalDevice pdev;
   VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));

   VkFormatProperties props;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                       &props);
   if (props.optimalTilingFeatures &
       (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      fprintf(stderr, "BC1 unexpectedly advertises unsupported blits\n");
      return 5;
   }
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R32_SINT, &props);
   if (props.optimalTilingFeatures &
       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) {
      fprintf(stderr, "R32_SINT unexpectedly advertises linear filtering\n");
      return 6;
   }
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R16_SINT, &props);
   if (props.optimalTilingFeatures &
       (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      fprintf(stderr, "R16_SINT unexpectedly advertises unsupported blits\n");
      return 7;
   }
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R16G16_UNORM, &props);
   if (props.optimalTilingFeatures &
       (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      fprintf(stderr, "R16G16_UNORM unexpectedly advertises unsupported blits\n");
      return 8;
   }
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R8G8B8A8_SRGB,
                                       &props);
   if (props.optimalTilingFeatures &
       (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      fprintf(stderr, "RGBA8 SRGB unexpectedly advertises unsupported blits\n");
      return 9;
   }
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R8G8B8A8_UNORM,
                                       &props);
   if ((props.optimalTilingFeatures &
        (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) !=
       (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      fprintf(stderr, "RGBA8 UNORM lost supported blit features\n");
      return 10;
   }

   uint32_t queue_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_count, NULL);
   VkQueueFamilyProperties queues[8];
   if (queue_count > 8)
      queue_count = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_count, queues);
   uint32_t family = 0;
   while (family < queue_count &&
          !(queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      family++;

   float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice dev;
   VK_CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   VkBuffer upload, readback;
   VkDeviceMemory upload_memory, readback_memory;
   if (make_buffer(dev, pdev, 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   &upload, &upload_memory) ||
       make_buffer(dev, pdev, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   &readback, &readback_memory))
      return 2;

   const uint8_t expected[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
   uint8_t *mapped;
   VK_CHECK(vkMapMemory(dev, upload_memory, 0, 8, 0, (void **)&mapped));
   memcpy(mapped, expected, sizeof(expected));
   vkUnmapMemory(dev, upload_memory);
   VK_CHECK(vkMapMemory(dev, readback_memory, 0, 8, 0, (void **)&mapped));
   memset(mapped, 0, 8);
   vkUnmapMemory(dev, readback_memory);

   VkImage bc, raw;
   VkDeviceMemory bc_memory, raw_memory;
   if (make_image(dev, pdev, VK_IMAGE_TYPE_2D,
                  VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, 1, 1,
                  &bc, &bc_memory) ||
       make_image(dev, pdev, VK_IMAGE_TYPE_2D,
                  VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, 1, 1,
                  &raw, &raw_memory))
      return 3;

   VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   VK_CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));
   VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   VK_CHECK(vkAllocateCommandBuffers(dev, &command_info, &command));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   VK_CHECK(vkBeginCommandBuffer(command, &begin));

   VkImageMemoryBarrier initialize[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = bc,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = raw,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, initialize);

   VkBufferImageCopy to_bc = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { 4, 4, 1 },
   };
   vkCmdCopyBufferToImage(command, upload, bc, VK_IMAGE_LAYOUT_GENERAL,
                          1, &to_bc);
   VkImageMemoryBarrier bc_ready = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = bc,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &bc_ready);

   /* The extent is in source texels: one 4x4 BC1 block becomes one
    * copy-compatible 8-byte texel. */
   VkImageCopy image_copy = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .extent = { 4, 4, 1 },
   };
   vkCmdCopyImage(command, bc, VK_IMAGE_LAYOUT_GENERAL,
                  raw, VK_IMAGE_LAYOUT_GENERAL, 1, &image_copy);
   VkImageMemoryBarrier raw_ready = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = raw,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &raw_ready);

   VkImageMemoryBarrier bc_writable = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = bc,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &bc_writable);

   /* In the reverse direction the one source texel becomes one BC block. */
   VkImageCopy reverse_copy = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .extent = { 1, 1, 1 },
   };
   vkCmdCopyImage(command, raw, VK_IMAGE_LAYOUT_GENERAL,
                  bc, VK_IMAGE_LAYOUT_GENERAL, 1, &reverse_copy);
   VkImageMemoryBarrier bc_readable = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = bc,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &bc_readable);

   VkBufferImageCopy from_bc = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { 4, 4, 1 },
   };
   vkCmdCopyImageToBuffer(command, bc, VK_IMAGE_LAYOUT_GENERAL,
                          readback, 1, &from_bc);

   VK_CHECK(vkEndCommandBuffer(command));

   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_CHECK(vkQueueWaitIdle(queue));

   VK_CHECK(vkMapMemory(dev, readback_memory, 0, 8, 0, (void **)&mapped));
   bool pass = memcmp(mapped, expected, sizeof(expected)) == 0;
   fprintf(stderr, "BC1 <-> RGBA16 raw copy: %s\n", pass ? "PASS" : "FAIL");
   vkUnmapMemory(dev, readback_memory);
   if (!pass)
      return 4;

   /* Exercise both valid cross-dimensional copy directions. Three 2D array
    * layers are one sequence of three slices, not layerCount * extent.depth. */
   const uint32_t slice_values[3] = {
      0x11223344u, 0x55667788u, 0x99aabbccu,
   };
   VK_CHECK(vkMapMemory(dev, upload_memory, 0, sizeof(slice_values), 0,
                     (void **)&mapped));
   memcpy(mapped, slice_values, sizeof(slice_values));
   vkUnmapMemory(dev, upload_memory);
   VK_CHECK(vkMapMemory(dev, readback_memory, 0, sizeof(slice_values), 0,
                     (void **)&mapped));
   memset(mapped, 0, sizeof(slice_values));
   vkUnmapMemory(dev, readback_memory);

   VkImage array_image, volume_image;
   VkDeviceMemory array_memory, volume_memory;
   if (make_image(dev, pdev, VK_IMAGE_TYPE_2D,
                  VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 3,
                  &array_image, &array_memory) ||
       make_image(dev, pdev, VK_IMAGE_TYPE_3D,
                  VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 3, 1,
                  &volume_image, &volume_memory))
      return 11;

   VkCommandBuffer command2;
   VK_CHECK(vkAllocateCommandBuffers(dev, &command_info, &command2));
   VK_CHECK(vkBeginCommandBuffer(command2, &begin));
   VkImageMemoryBarrier init_slices[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = array_image,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 3 },
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = volume_image,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
   };
   vkCmdPipelineBarrier(command2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, init_slices);

   VkBufferImageCopy upload_array = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 3 },
      .imageExtent = { 1, 1, 1 },
   };
   vkCmdCopyBufferToImage(command2, upload, array_image,
                          VK_IMAGE_LAYOUT_GENERAL, 1, &upload_array);
   VkImageMemoryBarrier array_readable = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = array_image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 3 },
   };
   vkCmdPipelineBarrier(command2, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &array_readable);

   VkImageCopy array_to_volume = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 3 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .extent = { 1, 1, 3 },
   };
   vkCmdCopyImage(command2, array_image, VK_IMAGE_LAYOUT_GENERAL,
                  volume_image, VK_IMAGE_LAYOUT_GENERAL, 1, &array_to_volume);

   VkImageMemoryBarrier reverse_ready[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = array_image,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 3 },
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = volume_image,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
   };
   vkCmdPipelineBarrier(command2, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, reverse_ready);

   VkImageCopy volume_to_array = {
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 3 },
      .extent = { 1, 1, 3 },
   };
   vkCmdCopyImage(command2, volume_image, VK_IMAGE_LAYOUT_GENERAL,
                  array_image, VK_IMAGE_LAYOUT_GENERAL, 1, &volume_to_array);
   VkImageMemoryBarrier array_final = array_readable;
   vkCmdPipelineBarrier(command2, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &array_final);
   VkBufferImageCopy download_array = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 3 },
      .imageExtent = { 1, 1, 1 },
   };
   vkCmdCopyImageToBuffer(command2, array_image, VK_IMAGE_LAYOUT_GENERAL,
                          readback, 1, &download_array);
   VK_CHECK(vkEndCommandBuffer(command2));

   submit.pCommandBuffers = &command2;
   VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_CHECK(vkQueueWaitIdle(queue));
   VK_CHECK(vkMapMemory(dev, readback_memory, 0, sizeof(slice_values), 0,
                     (void **)&mapped));
   pass = memcmp(mapped, slice_values, sizeof(slice_values)) == 0;
   fprintf(stderr, "2D-array <-> 3D slice copy: %s\n",
           pass ? "PASS" : "FAIL");
   return pass ? 0 : 12;
}
