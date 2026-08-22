/* D32 dynamic-rendering attachments load from and store to the named image. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 8
#define H 8
#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r); return 1; } } while (0)

static uint32_t
pick_memory(VkPhysicalDevice pdev, uint32_t bits)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         return i;
   return UINT32_MAX;
}

static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

static void
record_scope(VkCommandBuffer cmd, VkImageView view, VkAttachmentLoadOp load,
             float clear)
{
   VkRenderingAttachmentInfo depth = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = load, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil.depth = clear,
   };
   VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea.extent = { W, H }, .layerCount = 1,
      .pDepthAttachment = &depth,
   };
   begin_rendering(cmd, &rendering);
   end_rendering(cmd);
}

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkPhysicalDeviceDynamicRenderingFeatures dynamic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   const char *extension = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dynamic,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &extension };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);
   begin_rendering = (PFN_vkCmdBeginRenderingKHR)
      vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR");
   end_rendering = (PFN_vkCmdEndRenderingKHR)
      vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR");
   if (!begin_rendering || !end_rendering) {
      fprintf(stderr, "dynamic rendering aliases missing\n");
      return 1;
   }

   VkImageCreateInfo image_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_D32_SFLOAT,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT };
   VkImage image;
   CHECK(vkCreateImage(device, &image_ci, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, image, &req);
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = pick_memory(physical,
                                                               req.memoryTypeBits) };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(device, &mai, NULL, &memory));
   CHECK(vkBindImageMemory(device, image, memory, 0));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_D32_SFLOAT,
      .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(device, &vci, NULL, &view));

   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = 0 };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer command;
   CHECK(vkAllocateCommandBuffers(device, &cai, &command));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(command, &begin));
   record_scope(command, view, VK_ATTACHMENT_LOAD_OP_CLEAR, 0.25f);
   CHECK(vkEndCommandBuffer(command));
   VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers = &command };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   float *mapped;
   CHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   for (unsigned i = 0; i < W * H; i++)
      if (fabsf(mapped[i] - 0.25f) > 1e-6f) {
         fprintf(stderr, "store mismatch at %u: %f\n", i, mapped[i]);
         return 1;
      }
   for (unsigned i = 0; i < W * H; i++)
      mapped[i] = 0.75f;

   CHECK(vkResetCommandBuffer(command, 0));
   CHECK(vkBeginCommandBuffer(command, &begin));
   record_scope(command, view, VK_ATTACHMENT_LOAD_OP_LOAD, 0.0f);
   CHECK(vkEndCommandBuffer(command));
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   for (unsigned i = 0; i < W * H; i++)
      if (fabsf(mapped[i] - 0.75f) > 1e-6f) {
         fprintf(stderr, "load mismatch at %u: %f\n", i, mapped[i]);
         return 1;
      }

   vkUnmapMemory(device, memory);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyImageView(device, view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("depth attachment: pass");
   return 0;
}
