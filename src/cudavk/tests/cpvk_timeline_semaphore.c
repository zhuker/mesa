/*
 * Timeline semaphores: the counter, the two wait forms, and the queue's half.
 *
 * The driver's contribution to VK_KHR_timeline_semaphore is one vk_sync type,
 * not an entrypoint -- vkWaitSemaphores, vkSignalSemaphore and
 * vkGetSemaphoreCounterValue are the runtime's, over cpvk_sync.c. So what this
 * checks is the type's contract: the value is monotone and readable, a wait
 * for a value that has not arrived times out and one for a value already
 * passed does not, a queue submit can signal a point, and a wait can be
 * registered before the signal exists.
 *
 * The entrypoints are called through their KHR aliases on purpose. This driver
 * advertises Vulkan 1.1 and reaches timeline semaphores by extension, so the
 * core names are not legal here and the aliases are what an application would
 * use. Advertising an extension whose aliases are not wired is a null jump in
 * the loader, and has happened in this tree before.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
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

#define EXPECT_VALUE(what, got, expected) do { \
   if ((got) != (expected)) { \
      fprintf(stderr, "%s: %llu, expected %llu\n", (what), \
              (unsigned long long)(got), (unsigned long long)(expected)); \
      return 1; \
   } \
} while (0)

static PFN_vkGetSemaphoreCounterValueKHR get_counter;
static PFN_vkWaitSemaphoresKHR wait_semaphores;
static PFN_vkSignalSemaphoreKHR signal_semaphore;

static VkDevice device;
static VkQueue queue;

struct late_signal {
   VkSemaphore semaphore;
   uint64_t value;
};

static int
late_signal_thread(void *data)
{
   const struct late_signal *ls = data;
   struct timespec delay = { .tv_nsec = 20 * 1000 * 1000 };
   thrd_sleep(&delay, NULL);

   VkSemaphoreSignalInfo si = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = ls->semaphore,
      .value = ls->value,
   };
   return signal_semaphore(device, &si) == VK_SUCCESS ? 0 : 1;
}

static VkSemaphore
create_timeline(uint64_t initial)
{
   VkSemaphoreTypeCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initial,
   };
   VkSemaphoreCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &tci,
   };
   VkSemaphore semaphore = VK_NULL_HANDLE;
   if (vkCreateSemaphore(device, &sci, NULL, &semaphore) != VK_SUCCESS)
      return VK_NULL_HANDLE;
   return semaphore;
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

   /* The extension has to be enumerated, not assumed. */
   uint32_t ext_count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &ext_count, NULL));
   VkExtensionProperties *exts = calloc(ext_count, sizeof(*exts));
   if (!exts)
      return 1;
   CHECK(vkEnumerateDeviceExtensionProperties(physical, NULL, &ext_count, exts));
   bool found = false;
   for (uint32_t i = 0; i < ext_count; i++) {
      if (!strcmp(exts[i].extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME))
         found = true;
   }
   free(exts);
   if (!found) {
      fprintf(stderr, "VK_KHR_timeline_semaphore not advertised\n");
      return 1;
   }

   /* And the feature bit with it: an extension without its feature is not
    * usable, and the pair is what an application actually tests. */
   VkPhysicalDeviceTimelineSemaphoreFeaturesKHR tl_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &tl_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);
   if (!tl_features.timelineSemaphore) {
      fprintf(stderr, "timelineSemaphore feature is false\n");
      return 1;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *device_exts[] = { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME };
   VkPhysicalDeviceTimelineSemaphoreFeaturesKHR enable_tl = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .timelineSemaphore = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enable_tl,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_exts,
   };
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   vkGetDeviceQueue(device, 0, 0, &queue);

   get_counter = (PFN_vkGetSemaphoreCounterValueKHR)
      vkGetDeviceProcAddr(device, "vkGetSemaphoreCounterValueKHR");
   wait_semaphores = (PFN_vkWaitSemaphoresKHR)
      vkGetDeviceProcAddr(device, "vkWaitSemaphoresKHR");
   signal_semaphore = (PFN_vkSignalSemaphoreKHR)
      vkGetDeviceProcAddr(device, "vkSignalSemaphoreKHR");
   if (!get_counter || !wait_semaphores || !signal_semaphore) {
      fprintf(stderr, "KHR timeline entrypoints not wired: %p %p %p\n",
              (void *)get_counter, (void *)wait_semaphores,
              (void *)signal_semaphore);
      return 1;
   }

   /* Created at a value, and that value is what it reads. */
   VkSemaphore timeline = create_timeline(7);
   if (!timeline) {
      fprintf(stderr, "timeline creation failed\n");
      return 1;
   }
   uint64_t value = 0;
   CHECK(get_counter(device, timeline, &value));
   EXPECT_VALUE("initial counter", value, 7);

   VkSemaphoreWaitInfo wi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &timeline,
      .pValues = &value,
   };

   /* A wait for a value that has not arrived must expire, at zero timeout and
    * at a real one. */
   value = 8;
   EXPECT(wait_semaphores(device, &wi, 0), VK_TIMEOUT);
   EXPECT(wait_semaphores(device, &wi, 20 * 1000 * 1000), VK_TIMEOUT);

   /* A wait for a value already passed returns immediately, and so does one
    * for the current value. */
   value = 3;
   EXPECT(wait_semaphores(device, &wi, 0), VK_SUCCESS);
   value = 7;
   EXPECT(wait_semaphores(device, &wi, 0), VK_SUCCESS);

   /* Signalled from the host. */
   VkSemaphoreSignalInfo si = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = timeline,
      .value = 9,
   };
   CHECK(signal_semaphore(device, &si));
   CHECK(get_counter(device, timeline, &value));
   EXPECT_VALUE("host-signalled counter", value, 9);
   value = 9;
   EXPECT(wait_semaphores(device, &wi, 0), VK_SUCCESS);

   /* Signalled by a queue submit. The submit is empty, so what is being timed
    * is the queue's completion path, not any rendering. */
   uint64_t signal_value = 12;
   VkTimelineSemaphoreSubmitInfo tsi = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &signal_value,
   };
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &tsi,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &timeline,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   value = 12;
   CHECK(wait_semaphores(device, &wi, UINT64_MAX));
   CHECK(get_counter(device, timeline, &value));
   EXPECT_VALUE("queue-signalled counter", value, 12);

   /* A submit that both waits a reached point and signals a later one. */
   uint64_t wait_value = 12, next_value = 15;
   VkTimelineSemaphoreSubmitInfo tsi2 = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = 1,
      .pWaitSemaphoreValues = &wait_value,
      .signalSemaphoreValueCount = 1,
      .pSignalSemaphoreValues = &next_value,
   };
   VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VkSubmitInfo chained = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &tsi2,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &timeline,
      .pWaitDstStageMask = &wait_stage,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &timeline,
   };
   CHECK(vkQueueSubmit(queue, 1, &chained, VK_NULL_HANDLE));
   value = 15;
   CHECK(wait_semaphores(device, &wi, UINT64_MAX));

   /* Two semaphores in one wait, which is the form vkWaitSemaphores exists
    * for. ALL first, then ANY with only one of them reached. */
   VkSemaphore second = create_timeline(0);
   if (!second) {
      fprintf(stderr, "second timeline creation failed\n");
      return 1;
   }
   VkSemaphore both[2] = { timeline, second };
   uint64_t values[2] = { 15, 4 };
   VkSemaphoreWaitInfo pair = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 2,
      .pSemaphores = both,
      .pValues = values,
   };
   EXPECT(wait_semaphores(device, &pair, 0), VK_TIMEOUT);

   VkSemaphoreWaitInfo any = pair;
   any.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
   EXPECT(wait_semaphores(device, &any, 0), VK_SUCCESS);

   si.semaphore = second;
   si.value = 4;
   CHECK(signal_semaphore(device, &si));
   EXPECT(wait_semaphores(device, &pair, 0), VK_SUCCESS);

   /*
    * Wait before signal: a submit whose wait value does not exist yet, made to
    * exist by another thread afterwards. This is the case the driver cannot do
    * on its own -- there is nothing on the stream to wait for -- and which the
    * runtime's ASSISTED timeline mode covers by holding the submit until the
    * value is promised. If that ever stops working this call never returns.
    */
   uint64_t future = 40;
   VkTimelineSemaphoreSubmitInfo tsi3 = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      .waitSemaphoreValueCount = 1,
      .pWaitSemaphoreValues = &future,
   };
   VkSubmitInfo waiting = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &tsi3,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &second,
      .pWaitDstStageMask = &wait_stage,
   };
   struct late_signal ls = { .semaphore = second, .value = 40 };
   thrd_t signaller;
   if (thrd_create(&signaller, late_signal_thread, &ls) != thrd_success) {
      fprintf(stderr, "signal thread creation failed\n");
      return 1;
   }
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   CHECK(vkCreateFence(device, &fci, NULL, &fence));
   CHECK(vkQueueSubmit(queue, 1, &waiting, fence));
   int signal_status = 1;
   thrd_join(signaller, &signal_status);
   if (signal_status != 0) {
      fprintf(stderr, "late host signal failed\n");
      return 1;
   }
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
   CHECK(get_counter(device, second, &value));
   EXPECT_VALUE("late-signalled counter", value, 40);

   /* Monotone: a signal below the current value never moves it backwards. */
   si.semaphore = second;
   si.value = 5;
   signal_semaphore(device, &si);
   CHECK(get_counter(device, second, &value));
   EXPECT_VALUE("counter after a backwards signal", value, 40);

   vkDestroyFence(device, fence, NULL);
   vkDestroySemaphore(device, second, NULL);
   vkDestroySemaphore(device, timeline, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("timeline semaphore: pass");
   return 0;
}
