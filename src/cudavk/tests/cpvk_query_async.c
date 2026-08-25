/* Query availability is published at its ordered CUDA completion point. */
#include <stdint.h>
#include <stdio.h>
#include <vulkan/vulkan.h>
#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)
int main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance instance; CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1; VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci };
   VkDevice device; CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue; vkGetDeviceQueue(device, 0, 0, &queue);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .queueFamilyIndex = 0 };
   VkCommandPool pool; CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
   VkEventCreateInfo eci = { .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
   VkEvent event; CHECK(vkCreateEvent(device, &eci, NULL, &event));
   VkQueryPoolCreateInfo qpi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                 .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                 .queryCount = 1 };
   VkQueryPool query; CHECK(vkCreateQueryPool(device, &qpi, NULL, &query));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   vkCmdResetQueryPool(cmd, query, 0, 1);
   vkCmdWaitEvents(cmd, 1, &event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   0, NULL, 0, NULL, 0, NULL);
   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query, 0);
   CHECK(vkEndCommandBuffer(cmd));
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence; CHECK(vkCreateFence(device, &fci, NULL, &fence));
   VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   uint64_t result[2] = { UINT64_MAX, UINT64_MAX };
   VkResult status = vkGetQueryPoolResults(device, query, 0, 1, sizeof(result),
      result, sizeof(result), VK_QUERY_RESULT_64_BIT |
                              VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   if (status != VK_NOT_READY) {
      fprintf(stderr, "query became available before ordered wait: %d\n", status);
      return 1;
   }
   CHECK(vkSetEvent(device, event));
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
   result[0] = result[1] = 0;
   CHECK(vkGetQueryPoolResults(device, query, 0, 1, sizeof(result), result,
      sizeof(result), VK_QUERY_RESULT_64_BIT |
                      VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   if (!result[1] || !result[0]) {
      fprintf(stderr, "retired query missing value/availability\n"); return 1;
   }
   vkDestroyFence(device, fence, NULL); vkDestroyQueryPool(device, query, NULL);
   vkDestroyEvent(device, event, NULL); vkDestroyCommandPool(device, pool, NULL);
   vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
   puts("async query availability: pass"); return 0;
}
