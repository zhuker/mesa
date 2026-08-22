/* Recorded event commands execute in queue order and retain their event. */
#include <stdio.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r); return 1; } } while (0)

static VkCommandBuffer
alloc_command(VkDevice device, VkCommandPool pool)
{
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command = VK_NULL_HANDLE;
   vkAllocateCommandBuffers(device, &ai, &command);
   return command;
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
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, 0, 0, &queue);
   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkEventCreateInfo eci = { .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
   VkEvent event;
   CHECK(vkCreateEvent(device, &eci, NULL, &event));

   VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
   };
   VkCommandBuffer set = alloc_command(device, pool);
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(set, &bi));
   vkCmdSetEvent(set, event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
   CHECK(vkEndCommandBuffer(set));
   if (vkGetEventStatus(device, event) != VK_EVENT_RESET) {
      fprintf(stderr, "event changed during recording\n");
      return 1;
   }
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &set,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   if (vkGetEventStatus(device, event) != VK_EVENT_SET) {
      fprintf(stderr, "recorded set did not execute\n");
      return 1;
   }

   VkCommandBuffer reset = alloc_command(device, pool);
   CHECK(vkBeginCommandBuffer(reset, &bi));
   vkCmdResetEvent(reset, event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
   CHECK(vkEndCommandBuffer(reset));
   si.pCommandBuffers = &reset;
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   if (vkGetEventStatus(device, event) != VK_EVENT_RESET) {
      fprintf(stderr, "recorded reset did not execute\n");
      return 1;
   }

   VkCommandBuffer wait = alloc_command(device, pool);
   CHECK(vkBeginCommandBuffer(wait, &bi));
   vkCmdWaitEvents(wait, 1, &event,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   0, NULL, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(wait));
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence wait_fence;
   CHECK(vkCreateFence(device, &fci, NULL, &wait_fence));
   si.pCommandBuffers = &wait;
   CHECK(vkQueueSubmit(queue, 1, &si, wait_fence));
   if (vkWaitForFences(device, 1, &wait_fence, VK_TRUE, 0) != VK_TIMEOUT) {
      fprintf(stderr, "event wait completed before the event was set\n");
      return 1;
   }
   CHECK(vkSetEvent(device, event));
   CHECK(vkWaitForFences(device, 1, &wait_fence, VK_TRUE, UINT64_MAX));
   vkDestroyFence(device, wait_fence, NULL);

   VkEvent retained;
   CHECK(vkCreateEvent(device, &eci, NULL, &retained));
   VkCommandBuffer retain = alloc_command(device, pool);
   CHECK(vkBeginCommandBuffer(retain, &bi));
   vkCmdSetEvent(retain, retained, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
   CHECK(vkEndCommandBuffer(retain));
   vkDestroyEvent(device, retained, NULL);
   si.pCommandBuffers = &retain;
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   vkDestroyEvent(device, event, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("event state: pass");
   return 0;
}
