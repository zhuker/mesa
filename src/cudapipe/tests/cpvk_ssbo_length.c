/*
 * `buffer.length()` on a runtime-sized array.
 *
 * A shader asks for it with nir_intrinsic_get_ssbo_size, which the driver
 * answers out of the descriptor row beside the buffer's base address. Nothing
 * else in this suite, in the samples or in either capture calls it, which is
 * how the native driver reached this point neither lowering the intrinsic nor
 * writing the field it reads: the resource index's (slot, offset) pair
 * survived into the backend and was read there as a 64-bit address.
 *
 * Three buffers of deliberately different sizes are bound to one set, and the
 * shader writes all three lengths plus a value read out of two of them, so a
 * driver that reports a plausible-looking constant is caught as well as one
 * that reports zero.
 *
 * GLSL (glslangValidator -V, SDK 1.4.357.1):
 *
 *   #version 450
 *   layout(local_size_x = 1) in;
 *   layout(set = 0, binding = 0, std430) buffer Small { uint small_data[]; };
 *   layout(set = 0, binding = 1, std430) buffer Big   { uint big_data[]; };
 *   layout(set = 0, binding = 2, std430) buffer Out   { uint results[]; };
 *   void main()
 *   {
 *      results[0] = uint(small_data.length());
 *      results[1] = uint(big_data.length());
 *      results[2] = uint(results.length());
 *      results[3] = small_data[0] + big_data[0];
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)

/* Element counts, not bytes: the shader's arrays are uint. */
#define SMALL_ELEMS 5u
#define BIG_ELEMS   64u
#define OUT_ELEMS   8u

static const uint32_t comp_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000002e, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
   0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
   0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2,
   0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000008,
   0x0074754f, 0x00050006, 0x00000008, 0x00000000, 0x75736572, 0x0073746c,
   0x00030005, 0x0000000a, 0x00000000, 0x00040005, 0x0000000e, 0x6c616d53,
   0x0000006c, 0x00060006, 0x0000000e, 0x00000000, 0x6c616d73, 0x61645f6c,
   0x00006174, 0x00030005, 0x00000010, 0x00000000, 0x00030005, 0x00000018,
   0x00676942, 0x00060006, 0x00000018, 0x00000000, 0x5f676962, 0x61746164,
   0x00000000, 0x00030005, 0x0000001a, 0x00000000, 0x00040047, 0x00000007,
   0x00000006, 0x00000004, 0x00030047, 0x00000008, 0x00000003, 0x00050048,
   0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000a,
   0x00000021, 0x00000002, 0x00040047, 0x0000000a, 0x00000022, 0x00000000,
   0x00040047, 0x0000000d, 0x00000006, 0x00000004, 0x00030047, 0x0000000e,
   0x00000003, 0x00050048, 0x0000000e, 0x00000000, 0x00000023, 0x00000000,
   0x00040047, 0x00000010, 0x00000021, 0x00000000, 0x00040047, 0x00000010,
   0x00000022, 0x00000000, 0x00040047, 0x00000017, 0x00000006, 0x00000004,
   0x00030047, 0x00000018, 0x00000003, 0x00050048, 0x00000018, 0x00000000,
   0x00000023, 0x00000000, 0x00040047, 0x0000001a, 0x00000021, 0x00000001,
   0x00040047, 0x0000001a, 0x00000022, 0x00000000, 0x00040047, 0x0000002d,
   0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x0003001d,
   0x00000007, 0x00000006, 0x0003001e, 0x00000008, 0x00000007, 0x00040020,
   0x00000009, 0x00000002, 0x00000008, 0x0004003b, 0x00000009, 0x0000000a,
   0x00000002, 0x00040015, 0x0000000b, 0x00000020, 0x00000001, 0x0004002b,
   0x0000000b, 0x0000000c, 0x00000000, 0x0003001d, 0x0000000d, 0x00000006,
   0x0003001e, 0x0000000e, 0x0000000d, 0x00040020, 0x0000000f, 0x00000002,
   0x0000000e, 0x0004003b, 0x0000000f, 0x00000010, 0x00000002, 0x00040020,
   0x00000014, 0x00000002, 0x00000006, 0x0004002b, 0x0000000b, 0x00000016,
   0x00000001, 0x0003001d, 0x00000017, 0x00000006, 0x0003001e, 0x00000018,
   0x00000017, 0x00040020, 0x00000019, 0x00000002, 0x00000018, 0x0004003b,
   0x00000019, 0x0000001a, 0x00000002, 0x0004002b, 0x0000000b, 0x0000001f,
   0x00000002, 0x0004002b, 0x0000000b, 0x00000024, 0x00000003, 0x00040017,
   0x0000002b, 0x00000006, 0x00000003, 0x0004002b, 0x00000006, 0x0000002c,
   0x00000001, 0x0006002c, 0x0000002b, 0x0000002d, 0x0000002c, 0x0000002c,
   0x0000002c, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x00050044, 0x00000006, 0x00000011, 0x00000010,
   0x00000000, 0x0004007c, 0x0000000b, 0x00000012, 0x00000011, 0x0004007c,
   0x00000006, 0x00000013, 0x00000012, 0x00060041, 0x00000014, 0x00000015,
   0x0000000a, 0x0000000c, 0x0000000c, 0x0003003e, 0x00000015, 0x00000013,
   0x00050044, 0x00000006, 0x0000001b, 0x0000001a, 0x00000000, 0x0004007c,
   0x0000000b, 0x0000001c, 0x0000001b, 0x0004007c, 0x00000006, 0x0000001d,
   0x0000001c, 0x00060041, 0x00000014, 0x0000001e, 0x0000000a, 0x0000000c,
   0x00000016, 0x0003003e, 0x0000001e, 0x0000001d, 0x00050044, 0x00000006,
   0x00000020, 0x0000000a, 0x00000000, 0x0004007c, 0x0000000b, 0x00000021,
   0x00000020, 0x0004007c, 0x00000006, 0x00000022, 0x00000021, 0x00060041,
   0x00000014, 0x00000023, 0x0000000a, 0x0000000c, 0x0000001f, 0x0003003e,
   0x00000023, 0x00000022, 0x00060041, 0x00000014, 0x00000025, 0x00000010,
   0x0000000c, 0x0000000c, 0x0004003d, 0x00000006, 0x00000026, 0x00000025,
   0x00060041, 0x00000014, 0x00000027, 0x0000001a, 0x0000000c, 0x0000000c,
   0x0004003d, 0x00000006, 0x00000028, 0x00000027, 0x00050080, 0x00000006,
   0x00000029, 0x00000026, 0x00000028, 0x00060041, 0x00000014, 0x0000002a,
   0x0000000a, 0x0000000c, 0x00000024, 0x0003003e, 0x0000002a, 0x00000029,
   0x000100fd, 0x00010038,
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

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t count = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));
   VkPhysicalDeviceProperties pdev_props;
   vkGetPhysicalDeviceProperties(pdev, &pdev_props);
   printf("device: %s\n", pdev_props.deviceName);

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);

   const uint32_t elems[3] = { SMALL_ELEMS, BIG_ELEMS, OUT_ELEMS };
   VkBuffer buffers[3];
   VkDeviceMemory memories[3];
   void *maps[3];
   for (unsigned i = 0; i < 3; i++) {
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = elems[i] * sizeof(uint32_t),
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &buffers[i]));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, buffers[i], &req);
      uint32_t type = pick_memory(pdev, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (type == UINT32_MAX) {
         fprintf(stderr, "no host-visible coherent memory\n");
         return 1;
      }
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size, .memoryTypeIndex = type };
      CHECK(vkAllocateMemory(dev, &mai, NULL, &memories[i]));
      CHECK(vkBindBufferMemory(dev, buffers[i], memories[i], 0));
      CHECK(vkMapMemory(dev, memories[i], 0, VK_WHOLE_SIZE, 0, &maps[i]));
      memset(maps[i], 0, elems[i] * sizeof(uint32_t));
   }
   /* small_data[0] + big_data[0] == 300, so a read that lands on the wrong
    * buffer or on nothing is visible in the same run. */
   ((uint32_t *)maps[0])[0] = 100;
   ((uint32_t *)maps[1])[0] = 200;

   VkDescriptorSetLayoutBinding bindings[3];
   for (unsigned i = 0; i < 3; i++)
      bindings[i] = (VkDescriptorSetLayoutBinding) {
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = bindings };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

   VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));

   VkDescriptorBufferInfo infos[3];
   VkWriteDescriptorSet writes[3];
   for (unsigned i = 0; i < 3; i++) {
      infos[i] = (VkDescriptorBufferInfo) {
         .buffer = buffers[i], .offset = 0, .range = VK_WHOLE_SIZE };
      writes[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &infos[i] };
   }
   vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_spv), .pCode = comp_spv };
   VkShaderModule module;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &module));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = module, .pName = "main" },
      .layout = layout };
   VkPipeline pipeline;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                  &pipeline));

   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0 };
   VkCommandPool cmd_pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &cmd_pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                           &set, 0, NULL);
   vkCmdDispatch(cmd, 1, 1, 1);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   const uint32_t *out = maps[2];
   printf("  small_data.length() = %u  (expected %u)\n", out[0], SMALL_ELEMS);
   printf("  big_data.length()   = %u  (expected %u)\n", out[1], BIG_ELEMS);
   printf("  results.length()    = %u  (expected %u)\n", out[2], OUT_ELEMS);
   printf("  small[0] + big[0]   = %u  (expected 300)\n", out[3]);

   int bad = 0;
   if (out[0] != SMALL_ELEMS) { printf("FAIL small length\n"); bad = 1; }
   if (out[1] != BIG_ELEMS)   { printf("FAIL big length\n"); bad = 1; }
   if (out[2] != OUT_ELEMS)   { printf("FAIL out length\n"); bad = 1; }
   if (out[3] != 300)         { printf("FAIL buffer contents\n"); bad = 1; }
   if (!bad)
      puts("PASS every runtime array reported its own bound length");

   vkDestroyCommandPool(dev, cmd_pool, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, module, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorPool(dev, pool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   for (unsigned i = 0; i < 3; i++) {
      vkUnmapMemory(dev, memories[i]);
      vkDestroyBuffer(dev, buffers[i], NULL);
      vkFreeMemory(dev, memories[i], NULL);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return bad;
}
