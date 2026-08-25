/*
 * Negative and positive binary sync state for the native queue, including
 * asynchronous completion and a run of queued event-backed fences.
 */
#include <stdio.h>
#include <vulkan/vulkan.h>

#define CHECK(call) do { \
   VkResult _r = (call); \
   if (_r != VK_SUCCESS) { \
      fprintf(stderr, "%s failed: %d\n", #call, _r); \
      return 1; \
   } \
} while (0)

#define EXPECT(call, expected) do { \
   VkResult _r = (call); \
   if (_r != (expected)) { \
      fprintf(stderr, "%s returned %d, expected %d\n", \
              #call, _r, (expected)); \
      return 1; \
   } \
} while (0)

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

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);

   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(device, &fci, NULL, &fence));
   EXPECT(vkGetFenceStatus(device, fence), VK_NOT_READY);
   EXPECT(vkWaitForFences(device, 1, &fence, VK_TRUE, 0), VK_TIMEOUT);

   VkSubmitInfo empty = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
   CHECK(vkQueueSubmit(queue, 1, &empty, fence));
   VkResult pending = vkGetFenceStatus(device, fence);
   if (pending != VK_SUCCESS && pending != VK_NOT_READY) {
      fprintf(stderr, "submitted fence status: %d\n", pending);
      return 1;
   }
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

   CHECK(vkResetFences(device, 1, &fence));
   EXPECT(vkGetFenceStatus(device, fence), VK_NOT_READY);
   EXPECT(vkWaitForFences(device, 1, &fence, VK_TRUE, 0), VK_TIMEOUT);

   VkSemaphoreCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
   };
   VkSemaphore semaphore;
   CHECK(vkCreateSemaphore(device, &sci, NULL, &semaphore));

   VkSubmitInfo signal = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &semaphore,
   };
   CHECK(vkQueueSubmit(queue, 1, &signal, VK_NULL_HANDLE));

   VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo wait = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &semaphore,
      .pWaitDstStageMask = &wait_stage,
   };
   CHECK(vkQueueSubmit(queue, 1, &wait, fence));
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

   enum { STRESS = 64 };
   VkFence stress[STRESS];
   for (unsigned i = 0; i < STRESS; i++) {
      CHECK(vkCreateFence(device, &fci, NULL, &stress[i]));
      CHECK(vkQueueSubmit(queue, 1, &empty, stress[i]));
   }
   CHECK(vkWaitForFences(device, STRESS, stress, VK_TRUE, UINT64_MAX));
   for (unsigned i = 0; i < STRESS; i++) {
      CHECK(vkGetFenceStatus(device, stress[i]));
      vkDestroyFence(device, stress[i], NULL);
   }

   VkFenceCreateInfo signaled_fci = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
   };
   VkFence initially_signaled;
   CHECK(vkCreateFence(device, &signaled_fci, NULL, &initially_signaled));
   CHECK(vkGetFenceStatus(device, initially_signaled));
   CHECK(vkResetFences(device, 1, &initially_signaled));
   EXPECT(vkGetFenceStatus(device, initially_signaled), VK_NOT_READY);

   vkDestroyFence(device, initially_signaled, NULL);
   vkDestroySemaphore(device, semaphore, NULL);
   vkDestroyFence(device, fence, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("sync state: pass");
   return 0;
}
