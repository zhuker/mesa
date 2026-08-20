/* Native-driver smoke test.
 *
 * Milestone 1: a CUDA device is enumerated as a Vulkan physical device.
 * Milestone 2: a device, a queue, the three memory types, a mapped write that
 *              reads back, and a buffer bound to memory.
 *
 * Deliberately not a conformance test. It is the cheapest thing that fails
 * loudly when the object plumbing is wrong, which is what every milestone
 * here needs before anything larger is pointed at the driver.
 */
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   printf("FAIL %s -> %d\n", #x, _r); return 1; } } while (0)


/*
 * Milestone 3: SPIR-V in, a CUDA kernel out.
 *
 * The compute shader below is
 *     layout(local_size_x = 64) in;
 *     layout(std430, binding = 0) buffer B { uint data[]; };
 *     void main() { data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 2u; }
 * compiled by glslangValidator and embedded so the test needs no files.
 *
 * What it proves is that the 3,277-line NIR-to-PTX backend compiles and runs
 * inside the native driver unchanged -- it never depended on Gallium.
 */
static const uint32_t comp_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000001d, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000f, 0x00060010, 0x00000004,
   0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
   0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
   0x00000008, 0x00000042, 0x00050006, 0x00000008, 0x00000000, 0x61746164,
   0x00000000, 0x00030005, 0x0000000a, 0x00000000, 0x00080005, 0x0000000f,
   0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044,
   0x00040047, 0x00000007, 0x00000006, 0x00000004, 0x00030047, 0x00000008,
   0x00000003, 0x00050048, 0x00000008, 0x00000000, 0x00000023, 0x00000000,
   0x00040047, 0x0000000a, 0x00000021, 0x00000000, 0x00040047, 0x0000000a,
   0x00000022, 0x00000000, 0x00040047, 0x0000000f, 0x0000000b, 0x0000001c,
   0x00040047, 0x0000001c, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
   0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
   0x00000000, 0x0003001d, 0x00000007, 0x00000006, 0x0003001e, 0x00000008,
   0x00000007, 0x00040020, 0x00000009, 0x00000002, 0x00000008, 0x0004003b,
   0x00000009, 0x0000000a, 0x00000002, 0x00040015, 0x0000000b, 0x00000020,
   0x00000001, 0x0004002b, 0x0000000b, 0x0000000c, 0x00000000, 0x00040017,
   0x0000000d, 0x00000006, 0x00000003, 0x00040020, 0x0000000e, 0x00000001,
   0x0000000d, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000001, 0x0004002b,
   0x00000006, 0x00000010, 0x00000000, 0x00040020, 0x00000011, 0x00000001,
   0x00000006, 0x0004002b, 0x00000006, 0x00000016, 0x00000002, 0x00040020,
   0x00000018, 0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x0000001a,
   0x00000040, 0x0004002b, 0x00000006, 0x0000001b, 0x00000001, 0x0006002c,
   0x0000000d, 0x0000001c, 0x0000001a, 0x0000001b, 0x0000001b, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
   0x00050041, 0x00000011, 0x00000012, 0x0000000f, 0x00000010, 0x0004003d,
   0x00000006, 0x00000013, 0x00000012, 0x00050041, 0x00000011, 0x00000014,
   0x0000000f, 0x00000010, 0x0004003d, 0x00000006, 0x00000015, 0x00000014,
   0x00050084, 0x00000006, 0x00000017, 0x00000015, 0x00000016, 0x00060041,
   0x00000018, 0x00000019, 0x0000000a, 0x0000000c, 0x00000013, 0x0003003e,
   0x00000019, 0x00000017, 0x000100fd, 0x00010038
};

/*
 * Milestone 4: record a dispatch and run it.
 *
 * The shader writes data[i] = i * 2 into a storage buffer, so the check is on
 * the values, not on an API return code -- a dispatch that silently does
 * nothing is exactly the failure this driver's history is made of.
 */
static int test_dispatch(VkDevice dev, VkQueue queue, uint32_t memtype)
{
   VkDescriptorSetLayoutBinding b = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_spv), .pCode = comp_spv };
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));

   const uint32_t N = 256;
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = N * 4,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = N * 4, .memoryTypeIndex = memtype };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));
   uint32_t *data;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&data));
   memset(data, 0xff, N * 4);

   VkDescriptorPoolSize psz = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 1 };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &psz };
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
   VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0,
                                  .range = VK_WHOLE_SIZE };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = sm, .pName = "main" },
      .layout = pl };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
   printf("compute pipeline: SPIR-V -> NIR -> PTX -> CUmodule ok\n");

   VkCommandPoolCreateInfo cpci2 = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0 };
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpci2, NULL, &cpool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1,
                           &dset, 0, NULL);
   vkCmdDispatch(cb, N / 64, 1, 1);
   CHECK(vkEndCommandBuffer(cb));

   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cb };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned wrong = 0;
   for (uint32_t i = 0; i < N; i++)
      if (data[i] != i * 2) wrong++;
   printf("dispatch %u threads: data[i] == i*2 on %u of %u\n", N, N - wrong, N);
   if (wrong) {
      printf("FAIL first few: %u %u %u %u\n", data[0], data[1], data[2], data[3]);
      return 1;
   }

   vkFreeCommandBuffers(dev, cpool, 1, &cb);
   vkDestroyCommandPool(dev, cpool, NULL);
   vkUnmapMemory(dev, mem);
   vkDestroyDescriptorPool(dev, pool, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkFreeMemory(dev, mem, NULL);

   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   return 0;
}

/*
 * Milestone 5: images. Layout, memory requirements, binding, a view, and the
 * format table -- which reports what the kernels can actually decode, so a
 * format is refused rather than advertised and then clamped.
 */
static int test_image(VkPhysicalDevice pd, VkDevice dev)
{
   static const struct { VkFormat f; const char *name; bool want; } fmts[] = {
      { VK_FORMAT_B8G8R8A8_UNORM,      "B8G8R8A8_UNORM",  true  },
      { VK_FORMAT_R8G8B8A8_SRGB,       "R8G8B8A8_SRGB",   true  },
      { VK_FORMAT_D32_SFLOAT,          "D32_SFLOAT",      true  },
      { VK_FORMAT_R64G64B64A64_SFLOAT, "R64G64B64A64",    false },
   };
   for (unsigned i = 0; i < sizeof(fmts) / sizeof(fmts[0]); i++) {
      VkFormatProperties p;
      vkGetPhysicalDeviceFormatProperties(pd, fmts[i].f, &p);
      bool got = p.optimalTilingFeatures != 0;
      printf("  format %-16s features 0x%08x %s\n", fmts[i].name,
             p.optimalTilingFeatures, got == fmts[i].want ? "" : "UNEXPECTED");
      if (got != fmts[i].want) return 1;
   }

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
      .extent = { 64, 64, 1 }, .mipLevels = 4, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   CHECK(vkCreateImage(dev, &ici, NULL, &img));

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   printf("  image 64x64 4 mips: size %llu align %llu\n",
          (unsigned long long)req.size, (unsigned long long)req.alignment);

   /* Each level at its own offset, and none of them overlapping. */
   uint64_t prev_end = 0;
   for (uint32_t l = 0; l < 4; l++) {
      VkImageSubresource sub = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .mipLevel = l };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, img, &sub, &sl);
      printf("    level %u offset %6llu pitch %4llu size %6llu\n", l,
             (unsigned long long)sl.offset, (unsigned long long)sl.rowPitch,
             (unsigned long long)sl.size);
      if (sl.offset < prev_end) { printf("FAIL level %u overlaps\n", l); return 1; }
      prev_end = sl.offset + sl.size;
   }

   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = 0 };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, img, mem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 4, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &view));
   printf("  image view created and bound\n");

   vkDestroyImageView(dev, view, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyImage(dev, img, NULL);
   return 0;
}

int main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_0 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   printf("physical devices: %u\n", n);
   if (!n) return 1;

   VkPhysicalDevice pd;
   n = 1;
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) { printf("FAIL enum %d\n", r); return 1; }

   VkPhysicalDeviceProperties p;
   vkGetPhysicalDeviceProperties(pd, &p);
   printf("  %s  api %u.%u.%u\n", p.deviceName, VK_VERSION_MAJOR(p.apiVersion),
          VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   printf("  %u memory types, %u heaps\n", mp.memoryTypeCount,
          mp.memoryHeapCount);

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci };
   VkDevice dev;
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   printf("vkCreateDevice ok (renderer brought up: stream, arenas, raster queues)\n");

   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);
   printf("vkGetDeviceQueue ok (%p)\n", (void *)queue);

   /* Each memory type: allocate, and map the ones that claim host visibility.
    * A device-local allocation must refuse the map rather than stage behind
    * the caller's back. */
   for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
      const VkDeviceSize size = 64 * 1024;
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = size, .memoryTypeIndex = t };
      VkDeviceMemory mem;
      CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));

      bool visible = (mp.memoryTypes[t].propertyFlags &
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
      void *ptr = NULL;
      VkResult mr = vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &ptr);
      if (visible) {
         if (mr != VK_SUCCESS) { printf("FAIL map type %u -> %d\n", t, mr); return 1; }
         memset(ptr, 0xA5, size);
         if (((unsigned char *)ptr)[size - 1] != 0xA5) {
            printf("FAIL readback type %u\n", t); return 1;
         }
         vkUnmapMemory(dev, mem);
      } else if (mr == VK_SUCCESS) {
         printf("FAIL type %u is device-local and mapped anyway\n", t);
         return 1;
      }
      printf("  memory type %u flags 0x%02x %-14s map %s\n", t,
             mp.memoryTypes[t].propertyFlags,
             visible ? "host-visible" : "device-local",
             visible ? "wrote+read 64 KiB" : "refused, as it should be");

      if (t == 0) {
         VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
         VkBuffer buf;
         CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
         VkMemoryRequirements req;
         vkGetBufferMemoryRequirements(dev, buf, &req);
         CHECK(vkBindBufferMemory(dev, buf, mem, 0));
         printf("  buffer 4096 B: requirements size %llu align %llu types 0x%x, bound\n",
                (unsigned long long)req.size,
                (unsigned long long)req.alignment, req.memoryTypeBits);
         vkDestroyBuffer(dev, buf, NULL);
      }
      vkFreeMemory(dev, mem, NULL);
   }

   if (test_dispatch(dev, queue, 2))
      return 1;
   if (test_image(pd, dev))
      return 1;

   CHECK(vkDeviceWaitIdle(dev));
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("ok\n");
   return 0;
}
