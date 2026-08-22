/*
 * Focused descriptor-semantics regression for the native Vulkan ICD.
 *
 * The shader uses the deliberately sparse bindings 3, 19 and 47, while the
 * layout presents them unsorted.  Its live set is populated only with
 * VkCopyDescriptorSet.  The copied combined-image descriptor has a linear
 * immutable sampler in the source layout and a nearest immutable sampler in
 * the destination layout, so preserving the destination sampler changes the
 * observable result (1250 rather than about 1178).
 *
 * Finally a poison set is bound at the graphics bind point after the live set
 * is bound for compute.  The dispatch must retain compute's descriptor state;
 * a driver which keeps one shared table writes the poison buffer instead.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* Generated from:
 *   layout(local_size_x=1) in;
 *   layout(binding=3,std430) buffer O { uint value; } outbuf;
 *   layout(binding=19,std430) readonly buffer I { uint value; } inbuf;
 *   layout(binding=47) uniform sampler2D tex;
 *   outbuf.value = inbuf.value +
 *      uint(textureLod(tex, vec2(.60,.50), 0).r * 255.0 + .5);
 */
static const uint32_t comp_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000002d, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
   0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2,
   0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00040005, 0x00000008,
   0x706d6173, 0x0064656c, 0x00030005, 0x0000000d, 0x00786574, 0x00040005,
   0x0000001c, 0x7074754f, 0x00007475, 0x00050006, 0x0000001c, 0x00000000,
   0x756c6176, 0x00000065, 0x00050005, 0x0000001e, 0x7074756f, 0x625f7475,
   0x00006675, 0x00040005, 0x00000021, 0x75706e49, 0x00000074, 0x00050006,
   0x00000021, 0x00000000, 0x756c6176, 0x00000065, 0x00050005, 0x00000023,
   0x75706e69, 0x75625f74, 0x00000066, 0x00040047, 0x0000000d, 0x00000021,
   0x0000002f, 0x00040047, 0x0000000d, 0x00000022, 0x00000000, 0x00030047,
   0x0000001c, 0x00000003, 0x00050048, 0x0000001c, 0x00000000, 0x00000023,
   0x00000000, 0x00040047, 0x0000001e, 0x00000021, 0x00000003, 0x00040047,
   0x0000001e, 0x00000022, 0x00000000, 0x00030047, 0x00000021, 0x00000003,
   0x00040048, 0x00000021, 0x00000000, 0x00000018, 0x00050048, 0x00000021,
   0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000023, 0x00000018,
   0x00040047, 0x00000023, 0x00000021, 0x00000013, 0x00040047, 0x00000023,
   0x00000022, 0x00000000, 0x00040047, 0x0000002c, 0x0000000b, 0x00000019,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
   0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007,
   0x00000006, 0x00030016, 0x00000009, 0x00000020, 0x00090019, 0x0000000a,
   0x00000009, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001,
   0x00000000, 0x0003001b, 0x0000000b, 0x0000000a, 0x00040020, 0x0000000c,
   0x00000000, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000000,
   0x00040017, 0x0000000f, 0x00000009, 0x00000002, 0x0004002b, 0x00000009,
   0x00000010, 0x3f19999a, 0x0004002b, 0x00000009, 0x00000011, 0x3f000000,
   0x0005002c, 0x0000000f, 0x00000012, 0x00000010, 0x00000011, 0x0004002b,
   0x00000009, 0x00000013, 0x00000000, 0x00040017, 0x00000014, 0x00000009,
   0x00000004, 0x0004002b, 0x00000006, 0x00000016, 0x00000000, 0x0004002b,
   0x00000009, 0x00000018, 0x437f0000, 0x0003001e, 0x0000001c, 0x00000006,
   0x00040020, 0x0000001d, 0x00000002, 0x0000001c, 0x0004003b, 0x0000001d,
   0x0000001e, 0x00000002, 0x00040015, 0x0000001f, 0x00000020, 0x00000001,
   0x0004002b, 0x0000001f, 0x00000020, 0x00000000, 0x0003001e, 0x00000021,
   0x00000006, 0x00040020, 0x00000022, 0x00000002, 0x00000021, 0x0004003b,
   0x00000022, 0x00000023, 0x00000002, 0x00040020, 0x00000024, 0x00000002,
   0x00000006, 0x00040017, 0x0000002a, 0x00000006, 0x00000003, 0x0004002b,
   0x00000006, 0x0000002b, 0x00000001, 0x0006002c, 0x0000002a, 0x0000002c,
   0x0000002b, 0x0000002b, 0x0000002b, 0x00050036, 0x00000002, 0x00000004,
   0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
   0x00000008, 0x00000007, 0x0004003d, 0x0000000b, 0x0000000e, 0x0000000d,
   0x00070058, 0x00000014, 0x00000015, 0x0000000e, 0x00000012, 0x00000002,
   0x00000013, 0x00050051, 0x00000009, 0x00000017, 0x00000015, 0x00000000,
   0x00050085, 0x00000009, 0x00000019, 0x00000017, 0x00000018, 0x00050081,
   0x00000009, 0x0000001a, 0x00000019, 0x00000011, 0x0004006d, 0x00000006,
   0x0000001b, 0x0000001a, 0x0003003e, 0x00000008, 0x0000001b, 0x00050041,
   0x00000024, 0x00000025, 0x00000023, 0x00000020, 0x0004003d, 0x00000006,
   0x00000026, 0x00000025, 0x0004003d, 0x00000006, 0x00000027, 0x00000008,
   0x00050080, 0x00000006, 0x00000028, 0x00000026, 0x00000027, 0x00050041,
   0x00000024, 0x00000029, 0x0000001e, 0x00000020, 0x0003003e, 0x00000029,
   0x00000028, 0x000100fd, 0x00010038,
};

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint32_t *map;
   VkDeviceSize allocation_size;
};

static uint32_t
pick_memory(VkPhysicalDevice physical, uint32_t bits,
            VkMemoryPropertyFlags wanted)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(physical, &props);
   for (uint32_t i = 0; i < props.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & wanted) == wanted)
         return i;
   return UINT32_MAX;
}

static int
make_buffer(VkPhysicalDevice physical, VkDevice device, uint32_t initial,
            struct buffer *out)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(uint32_t),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(device, &bci, NULL, &out->buffer));

   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, out->buffer, &req);
   uint32_t type = pick_memory(physical, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX)
      type = pick_memory(physical, req.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (type == UINT32_MAX) {
      fprintf(stderr, "no host-visible buffer memory type\n");
      return 1;
   }

   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size, .memoryTypeIndex = type,
   };
   out->allocation_size = req.size;
   CHECK(vkAllocateMemory(device, &mai, NULL, &out->memory));
   CHECK(vkBindBufferMemory(device, out->buffer, out->memory, 0));
   CHECK(vkMapMemory(device, out->memory, 0, VK_WHOLE_SIZE, 0,
                     (void **)&out->map));
   *out->map = initial;
   VkMappedMemoryRange range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = out->memory, .offset = 0, .size = VK_WHOLE_SIZE,
   };
   CHECK(vkFlushMappedMemoryRanges(device, 1, &range));
   return 0;
}

int
main(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   if (!count) {
      fprintf(stderr, "no Vulkan physical device\n");
      return 1;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);

   struct buffer good_in, good_out, poison_in, poison_out;
   if (make_buffer(physical, device, 1000, &good_in) ||
       make_buffer(physical, device, 0xdeadbeef, &good_out) ||
       make_buffer(physical, device, 9000, &poison_in) ||
       make_buffer(physical, device, 0xcafebabe, &poison_out))
      return 1;

   /* A two-texel image: at u=.60 nearest returns 250, while linear returns
    * about 178.  This makes the immutable sampler selected by a copy visible. */
   VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { 2, 1, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
   };
   VkImage image;
   CHECK(vkCreateImage(device, &imci, NULL, &image));
   VkMemoryRequirements ireq;
   vkGetImageMemoryRequirements(device, image, &ireq);
   uint32_t image_type = pick_memory(physical, ireq.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (image_type == UINT32_MAX)
      image_type = pick_memory(physical, ireq.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (image_type == UINT32_MAX) {
      fprintf(stderr, "no host-visible image memory type\n");
      return 1;
   }
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = ireq.size, .memoryTypeIndex = image_type,
   };
   VkDeviceMemory image_memory;
   CHECK(vkAllocateMemory(device, &imai, NULL, &image_memory));
   CHECK(vkBindImageMemory(device, image, image_memory, 0));

   VkImageSubresource subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
   VkSubresourceLayout sublayout;
   vkGetImageSubresourceLayout(device, image, &subresource, &sublayout);
   unsigned char *pixels;
   CHECK(vkMapMemory(device, image_memory, 0, VK_WHOLE_SIZE, 0,
                     (void **)&pixels));
   pixels += sublayout.offset;
   const unsigned char texels[8] = { 10, 0, 0, 255, 250, 0, 0, 255 };
   memcpy(pixels, texels, sizeof(texels));
   VkMappedMemoryRange image_range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = image_memory, .offset = 0, .size = VK_WHOLE_SIZE,
   };
   CHECK(vkFlushMappedMemoryRanges(device, 1, &image_range));
   vkUnmapMemory(device, image_memory);

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = imci.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView image_view;
   CHECK(vkCreateImageView(device, &ivci, NULL, &image_view));

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod = 0.0f, .maxLod = 0.0f,
   };
   VkSampler nearest, linear;
   CHECK(vkCreateSampler(device, &sci, NULL, &nearest));
   sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
   CHECK(vkCreateSampler(device, &sci, NULL, &linear));

   /* Unsorted on purpose. Binding 47's immutable sampler differs between the
    * copy source and destination layouts. */
   VkDescriptorSetLayoutBinding dst_bindings[3] = {
      { .binding = 47, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL,
         .pImmutableSamplers = &nearest },
      { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
      { .binding = 19, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL },
   };
   VkDescriptorSetLayoutBinding src_bindings[3];
   memcpy(src_bindings, dst_bindings, sizeof(src_bindings));
   src_bindings[0].pImmutableSamplers = &linear;
   VkDescriptorSetLayoutCreateInfo slci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
   };
   VkDescriptorSetLayout src_layout, dst_layout;
   slci.pBindings = src_bindings;
   CHECK(vkCreateDescriptorSetLayout(device, &slci, NULL, &src_layout));
   slci.pBindings = dst_bindings;
   CHECK(vkCreateDescriptorSetLayout(device, &slci, NULL, &dst_layout));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dst_layout,
   };
   VkPipelineLayout pipeline_layout;
   CHECK(vkCreatePipelineLayout(device, &plci, NULL, &pipeline_layout));

   VkDescriptorPoolSize pool_sizes[2] = {
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 },
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 3, .poolSizeCount = 2, .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   CHECK(vkCreateDescriptorPool(device, &dpci, NULL, &descriptor_pool));
   VkDescriptorSetLayout set_layouts[3] = { src_layout, dst_layout, dst_layout };
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 3, .pSetLayouts = set_layouts,
   };
   VkDescriptorSet sets[3]; /* source, copied/live, graphics poison */
   CHECK(vkAllocateDescriptorSets(device, &dsai, sets));

   VkDescriptorBufferInfo good_buffer_info[2] = {
      { good_out.buffer, 0, sizeof(uint32_t) },
      { good_in.buffer, 0, sizeof(uint32_t) },
   };
   VkDescriptorImageInfo image_info = {
      .sampler = VK_NULL_HANDLE, .imageView = image_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkWriteDescriptorSet source_writes[3] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = sets[0], .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &good_buffer_info[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = sets[0], .dstBinding = 19, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &good_buffer_info[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = sets[0], .dstBinding = 47, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &image_info },
   };
   vkUpdateDescriptorSets(device, 3, source_writes, 0, NULL);

   VkCopyDescriptorSet copies[3] = {
      { .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
         .srcSet = sets[0], .srcBinding = 3,
         .dstSet = sets[1], .dstBinding = 3, .descriptorCount = 1 },
      { .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
         .srcSet = sets[0], .srcBinding = 19,
         .dstSet = sets[1], .dstBinding = 19, .descriptorCount = 1 },
      { .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
         .srcSet = sets[0], .srcBinding = 47,
         .dstSet = sets[1], .dstBinding = 47, .descriptorCount = 1 },
   };
   vkUpdateDescriptorSets(device, 0, NULL, 3, copies);

   VkDescriptorBufferInfo poison_buffer_info[2] = {
      { poison_out.buffer, 0, sizeof(uint32_t) },
      { poison_in.buffer, 0, sizeof(uint32_t) },
   };
   VkWriteDescriptorSet poison_writes[3];
   memcpy(poison_writes, source_writes, sizeof(poison_writes));
   for (unsigned i = 0; i < 3; i++)
      poison_writes[i].dstSet = sets[2];
   poison_writes[0].pBufferInfo = &poison_buffer_info[0];
   poison_writes[1].pBufferInfo = &poison_buffer_info[1];
   vkUpdateDescriptorSets(device, 3, poison_writes, 0, NULL);

   /* Rewriting the source after the copy also makes this assert immediate-copy
    * semantics rather than accidentally retaining the source set. */
   for (unsigned i = 0; i < 2; i++) {
      source_writes[i].pBufferInfo = &poison_buffer_info[i];
   }
   vkUpdateDescriptorSets(device, 2, source_writes, 0, NULL);

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_spv), .pCode = comp_spv,
   };
   VkShaderModule shader;
   CHECK(vkCreateShaderModule(device, &smci, NULL, &shader));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = shader, .pName = "main",
      },
      .layout = pipeline_layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL,
                                  &pipeline));

   VkCommandPoolCreateInfo cpci2 = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool command_pool;
   CHECK(vkCreateCommandPool(device, &cpci2, NULL, &command_pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   CHECK(vkAllocateCommandBuffers(device, &cbai, &command_buffer));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(command_buffer, &cbbi));

   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        0, NULL, 0, NULL, 1, &barrier);

   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &sets[1], 0, NULL);
   /* This later graphics bind must not replace the compute set above. */
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &sets[2], 0, NULL);
   vkCmdDispatch(command_buffer, 1, 1, 1);
   CHECK(vkEndCommandBuffer(command_buffer));

   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &command_buffer,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   VkMappedMemoryRange result_ranges[2] = {
      { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = good_out.memory, .offset = 0, .size = VK_WHOLE_SIZE },
      { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = poison_out.memory, .offset = 0, .size = VK_WHOLE_SIZE },
   };
   CHECK(vkInvalidateMappedMemoryRanges(device, 2, result_ranges));

   if (*good_out.map != 1250) {
      fprintf(stderr, "descriptor result: got %u, expected 1250 "
              "(sparse/copy/immutable failure)\n", *good_out.map);
      return 1;
   }
   if (*poison_out.map != 0xcafebabe) {
      fprintf(stderr, "graphics bind contaminated compute state: "
              "poison output is 0x%08x\n", *poison_out.map);
      return 1;
   }

   puts("descriptor semantics: pass");

   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyShaderModule(device, shader, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyDescriptorSetLayout(device, dst_layout, NULL);
   vkDestroyDescriptorSetLayout(device, src_layout, NULL);
   vkDestroySampler(device, linear, NULL);
   vkDestroySampler(device, nearest, NULL);
   vkDestroyImageView(device, image_view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, image_memory, NULL);
   struct buffer *buffers[] = { &good_in, &good_out, &poison_in, &poison_out };
   for (unsigned i = 0; i < 4; i++) {
      vkUnmapMemory(device, buffers[i]->memory);
      vkDestroyBuffer(device, buffers[i]->buffer, NULL);
      vkFreeMemory(device, buffers[i]->memory, NULL);
   }
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
