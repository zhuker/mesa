/* Milestone 1 smoke test: does the native driver enumerate a device? */
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

int main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_0 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   VkResult r = vkCreateInstance(&ici, NULL, &inst);
   if (r != VK_SUCCESS) { printf("vkCreateInstance %d\n", r); return 1; }

   uint32_t n = 0;
   r = vkEnumeratePhysicalDevices(inst, &n, NULL);
   printf("vkEnumeratePhysicalDevices -> %d, count %u\n", r, n);
   if (r != VK_SUCCESS || !n) return 1;

   VkPhysicalDevice pd[4];
   n = n > 4 ? 4 : n;
   vkEnumeratePhysicalDevices(inst, &n, pd);

   for (uint32_t i = 0; i < n; i++) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(pd[i], &p);
      printf("  [%u] %s  api %u.%u.%u  type %u\n", i, p.deviceName,
             VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
             VK_VERSION_PATCH(p.apiVersion), p.deviceType);

      VkPhysicalDeviceMemoryProperties m;
      vkGetPhysicalDeviceMemoryProperties(pd[i], &m);
      printf("      %u memory types, %u heaps, heap0 %.1f GiB\n",
             m.memoryTypeCount, m.memoryHeapCount,
             m.memoryHeaps[0].size / 1073741824.0);

      uint32_t q = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &q, NULL);
      printf("      %u queue famil%s\n", q, q == 1 ? "y" : "ies");
   }

   vkDestroyInstance(inst, NULL);
   printf("ok\n");
   return 0;
}
