/* ICD entry point for the native cudapipe driver.
 *
 * The runtime already exports vk_icdNegotiateLoaderICDInterfaceVersion and
 * vk_icdGetPhysicalDeviceProcAddr; only the instance proc-addr hook is the
 * driver's own. Deliberately free of the generated entrypoint header so this
 * file takes no build-order dependency on it.
 */
#include "vulkan/vulkan_core.h"
#include "util/macros.h"

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
cpvk_GetInstanceProcAddr(VkInstance instance, const char *pName);

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName);

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return cpvk_GetInstanceProcAddr(instance, pName);
}
