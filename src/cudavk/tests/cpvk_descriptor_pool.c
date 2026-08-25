/* Descriptor-pool set and per-type capacities are enforced and retired. */
#include <stdio.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r); return 1; } } while (0)

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
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));

   VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 2, .stageFlags = VK_SHADER_STAGE_ALL };
   VkDescriptorSetLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &binding };
   VkDescriptorSetLayout layout;
   CHECK(vkCreateDescriptorSetLayout(device, &lci, NULL, &layout));

   VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 };
   VkDescriptorPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &size };
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(device, &pci, NULL, &pool));
   VkDescriptorSetAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(device, &ai, &set));
   VkDescriptorSet extra = VK_NULL_HANDLE;
   VkResult result = vkAllocateDescriptorSets(device, &ai, &extra);
   if (result != VK_ERROR_OUT_OF_POOL_MEMORY) {
      fprintf(stderr, "maxSets overflow returned %d\n", result);
      return 1;
   }
   CHECK(vkFreeDescriptorSets(device, pool, 1, &set));
   CHECK(vkAllocateDescriptorSets(device, &ai, &set));
   CHECK(vkResetDescriptorPool(device, pool, 0));
   CHECK(vkAllocateDescriptorSets(device, &ai, &set));
   vkDestroyDescriptorPool(device, pool, NULL);

   size.descriptorCount = 1;
   CHECK(vkCreateDescriptorPool(device, &pci, NULL, &pool));
   ai.descriptorPool = pool;
   result = vkAllocateDescriptorSets(device, &ai, &set);
   if (result != VK_ERROR_OUT_OF_POOL_MEMORY) {
      fprintf(stderr, "descriptor capacity overflow returned %d\n", result);
      return 1;
   }
   vkDestroyDescriptorPool(device, pool, NULL);
   vkDestroyDescriptorSetLayout(device, layout, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts("descriptor pool: pass");
   return 0;
}
