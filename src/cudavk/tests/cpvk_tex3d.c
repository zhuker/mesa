/*
 * Differential test for linear filtering between z slices of a 3D texture.
 *
 * The optimal-tiled texture is 1x1x2: opaque red followed by opaque blue.
 * A compute shader samples its centre through a linear sampler and writes the
 * result to a storage buffer. The result must be half red/half blue; a sampler
 * which only filters x/y returns the blue slice and fails.
 *
 * The embedded shader is generated from:
 *
 *   #version 450
 *   layout(local_size_x=1) in;
 *   layout(set=0,binding=0) uniform sampler3D tex;
 *   layout(std430,set=0,binding=1) buffer Result { vec4 color[3]; } result;
 *   void main() {
 *      result.color[0] = texelFetch(tex, ivec3(0,0,0), 0);
 *      result.color[1] = texelFetch(tex, ivec3(0,0,1), 0);
 *      result.color[2] = texture(tex, vec3(0.5));
 *   }
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

static const uint32_t compute_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002cu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0005000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00060010u, 0x00000004u, 0x00000011u,
   0x00000001u, 0x00000001u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00040005u, 0x0000000bu, 0x75736552u, 0x0000746cu, 0x00050006u, 0x0000000bu,
   0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00040005u, 0x0000000du, 0x75736572u, 0x0000746cu, 0x00030005u,
   0x00000013u, 0x00786574u, 0x00040047u, 0x0000000au, 0x00000006u, 0x00000010u, 0x00030047u, 0x0000000bu,
   0x00000003u, 0x00050048u, 0x0000000bu, 0x00000000u, 0x00000023u, 0x00000000u, 0x00040047u, 0x0000000du,
   0x00000021u, 0x00000001u, 0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000013u,
   0x00000021u, 0x00000000u, 0x00040047u, 0x00000013u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000002bu,
   0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x0004001cu, 0x0000000au,
   0x00000007u, 0x00000009u, 0x0003001eu, 0x0000000bu, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000002u,
   0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000002u, 0x00040015u, 0x0000000eu, 0x00000020u,
   0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u, 0x00090019u, 0x00000010u, 0x00000006u,
   0x00000002u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x00000011u,
   0x00000010u, 0x00040020u, 0x00000012u, 0x00000000u, 0x00000011u, 0x0004003bu, 0x00000012u, 0x00000013u,
   0x00000000u, 0x00040017u, 0x00000015u, 0x0000000eu, 0x00000003u, 0x0006002cu, 0x00000015u, 0x00000016u,
   0x0000000fu, 0x0000000fu, 0x0000000fu, 0x00040020u, 0x00000019u, 0x00000002u, 0x00000007u, 0x0004002bu,
   0x0000000eu, 0x0000001bu, 0x00000001u, 0x0006002cu, 0x00000015u, 0x0000001du, 0x0000000fu, 0x0000000fu,
   0x0000001bu, 0x0004002bu, 0x0000000eu, 0x00000021u, 0x00000002u, 0x00040017u, 0x00000023u, 0x00000006u,
   0x00000003u, 0x0004002bu, 0x00000006u, 0x00000024u, 0x3f000000u, 0x0006002cu, 0x00000023u, 0x00000025u,
   0x00000024u, 0x00000024u, 0x00000024u, 0x0004002bu, 0x00000006u, 0x00000026u, 0x00000000u, 0x00040017u,
   0x00000029u, 0x00000008u, 0x00000003u, 0x0004002bu, 0x00000008u, 0x0000002au, 0x00000001u, 0x0006002cu,
   0x00000029u, 0x0000002bu, 0x0000002au, 0x0000002au, 0x0000002au, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x00000011u, 0x00000014u, 0x00000013u,
   0x00040064u, 0x00000010u, 0x00000017u, 0x00000014u, 0x0007005fu, 0x00000007u, 0x00000018u, 0x00000017u,
   0x00000016u, 0x00000002u, 0x0000000fu, 0x00060041u, 0x00000019u, 0x0000001au, 0x0000000du, 0x0000000fu,
   0x0000000fu, 0x0003003eu, 0x0000001au, 0x00000018u, 0x0004003du, 0x00000011u, 0x0000001cu, 0x00000013u,
   0x00040064u, 0x00000010u, 0x0000001eu, 0x0000001cu, 0x0007005fu, 0x00000007u, 0x0000001fu, 0x0000001eu,
   0x0000001du, 0x00000002u, 0x0000000fu, 0x00060041u, 0x00000019u, 0x00000020u, 0x0000000du, 0x0000000fu,
   0x0000001bu, 0x0003003eu, 0x00000020u, 0x0000001fu, 0x0004003du, 0x00000011u, 0x00000022u, 0x00000013u,
   0x00070058u, 0x00000007u, 0x00000027u, 0x00000022u, 0x00000025u, 0x00000002u, 0x00000026u, 0x00060041u,
   0x00000019u, 0x00000028u, 0x0000000du, 0x0000000fu, 0x00000021u, 0x0003003eu, 0x00000028u, 0x00000027u,
   0x000100fdu, 0x00010038u,
};

static uint32_t
pick_memory(VkPhysicalDevice pdev, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(pdev, &props);
   for (uint32_t i = 0; i < props.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static int
make_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size,
            VkBufferUsageFlags usage, VkBuffer *buffer,
            VkDeviceMemory *memory)
{
   VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(dev, &info, NULL, buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *buffer, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX)
      type = pick_memory(pdev, req.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (type == UINT32_MAX) {
      fprintf(stderr, "no host-visible buffer memory\n");
      return 1;
   }
   VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = type,
   };
   CHECK(vkAllocateMemory(dev, &alloc, NULL, memory));
   CHECK(vkBindBufferMemory(dev, *buffer, *memory, 0));
   return 0;
}

int
main(void)
{
   VkApplicationInfo application = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));

   uint32_t count = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));
   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; i++)
      if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
         family = i;
         break;
      }
   free(families);
   if (family == UINT32_MAX) {
      fprintf(stderr, "no compute queue\n");
      return 1;
   }

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
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   VkFormatProperties props;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R8G8B8A8_UNORM,
                                       &props);
   VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((props.optimalTilingFeatures & required) != required) {
      fprintf(stderr, "RGBA8 optimal 3D linear sampling is not advertised\n");
      return 1;
   }
   VkImageFormatProperties image_props;
   CHECK(vkGetPhysicalDeviceImageFormatProperties(
      pdev, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_3D,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0,
      &image_props));

   VkBuffer upload, result;
   VkDeviceMemory upload_memory, result_memory;
   if (make_buffer(dev, pdev, 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   &upload, &upload_memory) ||
       make_buffer(dev, pdev, 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   &result, &result_memory))
      return 1;
   unsigned char *mapped;
   CHECK(vkMapMemory(dev, upload_memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   const unsigned char voxels[8] = {
      255, 0, 0, 255,
      0, 0, 255, 255,
   };
   memcpy(mapped, voxels, sizeof(voxels));
   VkMappedMemoryRange upload_flush = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = upload_memory,
      .size = VK_WHOLE_SIZE,
   };
   CHECK(vkFlushMappedMemoryRanges(dev, 1, &upload_flush));
   vkUnmapMemory(dev, upload_memory);

   VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_3D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { 1, 1, 2 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &image_info, NULL, &image));
   VkMemoryRequirements image_req;
   vkGetImageMemoryRequirements(dev, image, &image_req);
   uint32_t image_type = pick_memory(pdev, image_req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (image_type == UINT32_MAX)
      image_type = pick_memory(pdev, image_req.memoryTypeBits, 0);
   if (image_type == UINT32_MAX)
      return 1;
   VkMemoryAllocateInfo image_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_req.size,
      .memoryTypeIndex = image_type,
   };
   VkDeviceMemory image_memory;
   CHECK(vkAllocateMemory(dev, &image_alloc, NULL, &image_memory));
   CHECK(vkBindImageMemory(dev, image, image_memory, 0));

   VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_3D,
      .format = image_info.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &view_info, NULL, &view));
   VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sampler_info, NULL, &sampler));

   VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };
   VkDescriptorSetLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &layout_info, NULL, &set_layout));
   VkDescriptorPoolSize pool_sizes[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(dev, &pool_info, NULL, &pool));
   VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &set_info, &set));
   VkDescriptorImageInfo descriptor_image = {
      .sampler = sampler,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkDescriptorBufferInfo descriptor_buffer = {
      .buffer = result,
      .range = 48,
   };
   VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descriptor_image,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &descriptor_buffer,
      },
   };
   vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

   VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   CHECK(vkCreatePipelineLayout(dev, &pipeline_layout_info, NULL,
                                &pipeline_layout));
   VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(compute_spv),
      .pCode = compute_spv,
   };
   VkShaderModule shader;
   CHECK(vkCreateShaderModule(dev, &shader_info, NULL, &shader));
   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = shader,
         .pName = "main",
      },
      .layout = pipeline_layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeline_info,
                                  NULL, &pipeline));

   VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool command_pool;
   CHECK(vkCreateCommandPool(dev, &command_pool_info, NULL, &command_pool));
   VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   CHECK(vkAllocateCommandBuffers(dev, &command_info, &command));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(command, &begin));
   VkBufferMemoryBarrier upload_ready = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = upload,
      .size = VK_WHOLE_SIZE,
   };
   VkImageMemoryBarrier image_writable = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL,
                        1, &upload_ready, 1, &image_writable);
   VkBufferImageCopy copy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { 1, 1, 2 },
   };
   vkCmdCopyBufferToImage(command, upload, image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
   VkImageMemoryBarrier image_readable = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                        0, NULL, 1, &image_readable);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &set, 0, NULL);
   vkCmdDispatch(command, 1, 1, 1);
   VkBufferMemoryBarrier result_ready = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = result,
      .size = VK_WHOLE_SIZE,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL,
                        1, &result_ready, 0, NULL);
   CHECK(vkEndCommandBuffer(command));
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   CHECK(vkMapMemory(dev, result_memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   VkMappedMemoryRange result_invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = result_memory,
      .size = VK_WHOLE_SIZE,
   };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &result_invalidate));
   const float *color = (const float *)mapped;
   int pass = fabsf(color[8] - 0.5f) < 0.01f && fabsf(color[9]) < 0.01f &&
              fabsf(color[10] - 0.5f) < 0.01f &&
              fabsf(color[11] - 1.0f) < 0.01f;
   fprintf(stderr, "fetch0=(%.3f,%.3f,%.3f,%.3f) "
                   "fetch1=(%.3f,%.3f,%.3f,%.3f) "
                   "linear=(%.3f,%.3f,%.3f,%.3f): %s\n",
           color[0], color[1], color[2], color[3],
           color[4], color[5], color[6], color[7],
           color[8], color[9], color[10], color[11], pass ? "PASS" : "FAIL");
   return pass ? 0 : 2;
}
