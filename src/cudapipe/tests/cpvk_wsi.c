/* Headless WSI is real: surface queries, swapchain images, acquire and present. */
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS && \
                                           r != VK_SUBOPTIMAL_KHR) { \
   fprintf(stderr, "%s failed: %d\n", #x, r); return 1; } } while (0)

int
main(void)
{
   const char *instance_exts[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
   };
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = instance_exts,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   PFN_vkCreateHeadlessSurfaceEXT create_headless =
      (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(
         instance, "vkCreateHeadlessSurfaceEXT");
   if (!create_headless) {
      fprintf(stderr, "headless surface entrypoint missing\n");
      return 1;
   }
   VkHeadlessSurfaceCreateInfoEXT hsci = {
      .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT };
   VkSurfaceKHR surface;
   CHECK(create_headless(instance, &hsci, NULL, &surface));

   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   VkBool32 supported = VK_FALSE;
   CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical, 0, surface,
                                               &supported));
   if (!supported) {
      fprintf(stderr, "queue family cannot present\n");
      return 1;
   }
   VkSurfaceCapabilitiesKHR caps;
   CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));
   count = 0;
   CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, NULL));
   VkSurfaceFormatKHR *formats = calloc(count, sizeof(*formats));
   CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count,
                                               formats));
   if (!count) {
      fprintf(stderr, "no surface formats\n");
      return 1;
   }
   VkSurfaceFormatKHR format = formats[0];
   free(formats);

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   const char *device_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = &device_ext };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);

   VkExtent2D extent = caps.currentExtent;
   if (extent.width == UINT32_MAX)
      extent = (VkExtent2D){ 64, 64 };
   VkSwapchainCreateInfoKHR sci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = caps.minImageCount > 2 ? caps.minImageCount : 2,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = caps.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE,
   };
   VkSwapchainKHR swapchain;
   CHECK(vkCreateSwapchainKHR(device, &sci, NULL, &swapchain));
   count = 0;
   CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, NULL));
   if (count < sci.minImageCount) {
      fprintf(stderr, "swapchain returned only %u images\n", count);
      return 1;
   }
   VkImage *images = calloc(count, sizeof(*images));
   CHECK(vkGetSwapchainImagesKHR(device, swapchain, &count, images));
   free(images);

   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(device, &fci, NULL, &fence));
   uint32_t image_index;
   CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                               VK_NULL_HANDLE, fence, &image_index));
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
   VkPresentInfoKHR present = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .swapchainCount = 1, .pSwapchains = &swapchain,
      .pImageIndices = &image_index,
   };
   CHECK(vkQueuePresentKHR(queue, &present));
   CHECK(vkQueueWaitIdle(queue));

   vkDestroyFence(device, fence, NULL);
   vkDestroySwapchainKHR(device, swapchain, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroySurfaceKHR(instance, surface, NULL);
   vkDestroyInstance(instance, NULL);
   puts("headless WSI: pass");
   return 0;
}
