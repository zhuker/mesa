/*
 * Texture-cache fatal/no-replay regression test.
 *
 * The real application fragment shader samples a combined image sampler and
 * performs one SSBO atomicAdd for every covered invocation.  One fullscreen
 * triangle covers a 32x32 single-sample target exactly once, so a successful
 * fragment launch must leave counter == 1024.  Replaying that launch after a
 * later fatal error would instead leave 2048.
 *
 * Both the sampled image and the colour target are optimal-tiled and backed by
 * DEVICE_LOCAL memory.  The 1x1 sampled image contains RGBA (37,89,151,255).
 * The normal control checks every copied-back target pixel as well as the
 * exact counter.  The --fault run accepts VK_ERROR_DEVICE_LOST from either
 * vkQueueSubmit or (if submit succeeded) vkQueueWaitIdle, but requires it
 * exactly once and still reads the counter that was mapped before submission.
 *
 * The outer process enables the hardware texture cache and captures its stats.
 * On cudapipe it requires one hardware/direct fragment launch, no A-buffer
 * launch, and (for --fault) the one-shot injected-fatal diagnostic.  Other
 * Vulkan drivers ignore those environment variables and provide the normal
 * exact-pixel control.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror \
 *      src/cudapipe/tests/cpvk_texture_cache_no_replay.c \
 *      -o /tmp/cpvk_texture_cache_no_replay -lvulkan
 * Run controls:
 *   /tmp/cpvk_texture_cache_no_replay
 *   VK_ICD_FILENAMES=.../cudapipe_native_devenv_icd.x86_64.json \
 *      /tmp/cpvk_texture_cache_no_replay
 * Fault run (cudapipe only):
 *   VK_ICD_FILENAMES=.../cudapipe_native_devenv_icd.x86_64.json \
 *      /tmp/cpvk_texture_cache_no_replay --fault
 *
 * SPIR-V is embedded.  It was generated with glslangValidator 1.4.357.1 from:
 *
 *   #version 450
 *   const vec2 pos[3] = vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
 *   void main() { gl_Position=vec4(pos[gl_VertexIndex],0,1); }
 *
 *   #version 450
 *   layout(set=0,binding=0) uniform sampler2D sampled_image;
 *   layout(std430,set=0,binding=1) buffer Counter { uint count; } counter;
 *   layout(location=0) out vec4 out_colour;
 *   void main() {
 *      vec4 texel=texture(sampled_image,vec2(0.25,0.75));
 *      atomicAdd(counter.count,1u);
 *      out_colour=texel;
 *   }
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define W 32u
#define H 32u
#define EXPECTED (W * H)

static void
vk_ok(VkResult r, const char *what)
{
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s failed: %d\n", what, r);
      exit(2);
   }
}
#define VK_OK(x) vk_ok((x), #x)

static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000028u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0007000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x0000001au, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00060005u, 0x0000000bu,
   0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000000bu, 0x00000000u, 0x505f6c67u,
   0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u,
   0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu,
   0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u,
   0x0000000du, 0x00000000u, 0x00060005u, 0x0000001au, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u,
   0x00050005u, 0x0000001du, 0x65646e69u, 0x6c626178u, 0x00000065u, 0x00030047u, 0x0000000bu, 0x00000002u,
   0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u,
   0x0000000bu, 0x00000001u, 0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
   0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00040047u, 0x0000001au, 0x0000000bu, 0x0000002au,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x00000008u, 0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u,
   0x0006001eu, 0x0000000bu, 0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu,
   0x00000003u, 0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu,
   0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u,
   0x00000006u, 0x00000002u, 0x0004002bu, 0x00000008u, 0x00000011u, 0x00000003u, 0x0004001cu, 0x00000012u,
   0x00000010u, 0x00000011u, 0x0004002bu, 0x00000006u, 0x00000013u, 0xbf800000u, 0x0005002cu, 0x00000010u,
   0x00000014u, 0x00000013u, 0x00000013u, 0x0004002bu, 0x00000006u, 0x00000015u, 0x40400000u, 0x0005002cu,
   0x00000010u, 0x00000016u, 0x00000015u, 0x00000013u, 0x0005002cu, 0x00000010u, 0x00000017u, 0x00000013u,
   0x00000015u, 0x0006002cu, 0x00000012u, 0x00000018u, 0x00000014u, 0x00000016u, 0x00000017u, 0x00040020u,
   0x00000019u, 0x00000001u, 0x0000000eu, 0x0004003bu, 0x00000019u, 0x0000001au, 0x00000001u, 0x00040020u,
   0x0000001cu, 0x00000007u, 0x00000012u, 0x00040020u, 0x0000001eu, 0x00000007u, 0x00000010u, 0x0004002bu,
   0x00000006u, 0x00000021u, 0x00000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u, 0x00040020u,
   0x00000026u, 0x00000003u, 0x00000007u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
   0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000001cu, 0x0000001du, 0x00000007u, 0x0004003du, 0x0000000eu,
   0x0000001bu, 0x0000001au, 0x0003003eu, 0x0000001du, 0x00000018u, 0x00050041u, 0x0000001eu, 0x0000001fu,
   0x0000001du, 0x0000001bu, 0x0004003du, 0x00000010u, 0x00000020u, 0x0000001fu, 0x00050051u, 0x00000006u,
   0x00000023u, 0x00000020u, 0x00000000u, 0x00050051u, 0x00000006u, 0x00000024u, 0x00000020u, 0x00000001u,
   0x00070050u, 0x00000007u, 0x00000025u, 0x00000023u, 0x00000024u, 0x00000021u, 0x00000022u, 0x00050041u,
   0x00000026u, 0x00000027u, 0x0000000du, 0x0000000fu, 0x0003003eu, 0x00000027u, 0x00000025u, 0x000100fdu,
   0x00010038u,
};

static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000022u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000020u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00040005u, 0x00000009u, 0x65786574u, 0x0000006cu, 0x00060005u, 0x0000000du, 0x706d6173u, 0x5f64656cu,
   0x67616d69u, 0x00000065u, 0x00040005u, 0x00000015u, 0x6e756f43u, 0x00726574u, 0x00050006u, 0x00000015u,
   0x00000000u, 0x6e756f63u, 0x00000074u, 0x00040005u, 0x00000017u, 0x6e756f63u, 0x00726574u, 0x00050005u,
   0x00000020u, 0x5f74756fu, 0x6f6c6f63u, 0x00007275u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00030047u, 0x00000015u, 0x00000003u, 0x00050048u,
   0x00000015u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00040047u, 0x00000017u, 0x00000021u, 0x00000001u,
   0x00040047u, 0x00000017u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000020u, 0x0000001eu, 0x00000000u,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u,
   0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu,
   0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3e800000u, 0x0004002bu, 0x00000006u, 0x00000011u, 0x3f400000u,
   0x0005002cu, 0x0000000fu, 0x00000012u, 0x00000010u, 0x00000011u, 0x00040015u, 0x00000014u, 0x00000020u,
   0x00000000u, 0x0003001eu, 0x00000015u, 0x00000014u, 0x00040020u, 0x00000016u, 0x00000002u, 0x00000015u,
   0x0004003bu, 0x00000016u, 0x00000017u, 0x00000002u, 0x00040015u, 0x00000018u, 0x00000020u, 0x00000001u,
   0x0004002bu, 0x00000018u, 0x00000019u, 0x00000000u, 0x00040020u, 0x0000001au, 0x00000002u, 0x00000014u,
   0x0004002bu, 0x00000014u, 0x0000001cu, 0x00000001u, 0x0004002bu, 0x00000014u, 0x0000001du, 0x00000000u,
   0x00040020u, 0x0000001fu, 0x00000003u, 0x00000007u, 0x0004003bu, 0x0000001fu, 0x00000020u, 0x00000003u,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000007u, 0x0004003du, 0x0000000bu, 0x0000000eu, 0x0000000du, 0x00050057u,
   0x00000007u, 0x00000013u, 0x0000000eu, 0x00000012u, 0x0003003eu, 0x00000009u, 0x00000013u, 0x00050041u,
   0x0000001au, 0x0000001bu, 0x00000017u, 0x00000019u, 0x000700eau, 0x00000014u, 0x0000001eu, 0x0000001bu,
   0x0000001cu, 0x0000001du, 0x0000001cu, 0x0004003du, 0x00000007u, 0x00000021u, 0x00000009u, 0x0003003eu,
   0x00000020u, 0x00000021u, 0x000100fdu, 0x00010038u,
};

static uint32_t
pick_memory(VkPhysicalDevice pdev, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static void
make_buffer(VkPhysicalDevice pdev, VkDevice dev, VkDeviceSize size,
            VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags,
            VkBuffer *buffer, VkDeviceMemory *memory)
{
   VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_OK(vkCreateBuffer(dev, &bi, NULL, buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *buffer, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits, memory_flags);
   if (type == UINT32_MAX) {
      fprintf(stderr, "no buffer memory type with flags 0x%x\n", memory_flags);
      exit(2);
   }
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = type,
   };
   VK_OK(vkAllocateMemory(dev, &ai, NULL, memory));
   VK_OK(vkBindBufferMemory(dev, *buffer, *memory, 0));
}

static void
make_image(VkPhysicalDevice pdev, VkDevice dev, uint32_t width, uint32_t height,
           VkImageUsageFlags usage, VkImage *image, VkDeviceMemory *memory)
{
   VkImageCreateInfo ii = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { width, height, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VK_OK(vkCreateImage(dev, &ii, NULL, image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, *image, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (type == UINT32_MAX) {
      fprintf(stderr, "no DEVICE_LOCAL memory for optimal image\n");
      exit(2);
   }
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = type,
   };
   VK_OK(vkAllocateMemory(dev, &ai, NULL, memory));
   VK_OK(vkBindImageMemory(dev, *image, *memory, 0));
}

static VkShaderModule
make_shader(VkDevice dev, const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo si = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule shader;
   VK_OK(vkCreateShaderModule(dev, &si, NULL, &shader));
   return shader;
}

static unsigned
substring_count(const char *text, const char *needle)
{
   unsigned n = 0;
   size_t len = strlen(needle);
   while ((text = strstr(text, needle)) != NULL) {
      n++;
      text += len;
   }
   return n;
}

static int
assert_native_stats(const char *log, int fault)
{
   const char *marker = strstr(log, "cpvk-test-device: ");
   if (!marker || strncmp(marker + strlen("cpvk-test-device: "),
                           "cudapipe (", strlen("cudapipe (")))
      return 0;

   const char *hw = strstr(log, "cudapipe: hardware texture: ");
   const char *fallbacks = hw ? strstr(hw, " fallbacks shader=") : NULL;
   const char *paths = hw ? strstr(hw, " direct=") : NULL;
   const char *modes = hw ? strstr(hw, " modes inline=") : NULL;
   unsigned long long hits = 0, launches = 0, shader_fb = 0, descriptor_fb = 0;
   unsigned long long direct_hits = 0, direct_launches = 0;
   unsigned long long abuf_hits = 0, abuf_launches = 0;
   unsigned long long inline_hits = 0, fused_hits = 0;
   if (!hw || !fallbacks || !paths || !modes ||
       sscanf(hw, "cudapipe: hardware texture: %llu/%llu fragment launches hit",
              &hits, &launches) != 2 ||
       sscanf(fallbacks, " fallbacks shader=%llu descriptor=%llu",
              &shader_fb, &descriptor_fb) != 2 ||
       sscanf(paths, " direct=%llu/%llu, abuffer=%llu/%llu",
              &direct_hits, &direct_launches,
              &abuf_hits, &abuf_launches) != 4 ||
       sscanf(modes, " modes inline=%llu fused=%llu",
              &inline_hits, &fused_hits) != 2) {
      fprintf(stderr, "FAIL native: cannot parse hardware texture stats\n");
      return 1;
   }
   if (hits != 1 || launches != 1 || shader_fb || descriptor_fb ||
       direct_hits != 1 || direct_launches != 1 || abuf_hits || abuf_launches ||
       inline_hits + fused_hits != 1) {
      fprintf(stderr, "FAIL native path: hw=%llu/%llu direct=%llu/%llu "
              "abuffer=%llu/%llu fallback=%llu/%llu modes=%llu/%llu\n",
              hits, launches, direct_hits, direct_launches,
              abuf_hits, abuf_launches, shader_fb, descriptor_fb,
              inline_hits, fused_hits);
      return 1;
   }

   const char *injected =
      "injected fatal after one hardware FS enqueue, fs_attempts=1";
   unsigned injected_count = substring_count(log, injected);
   if ((fault && injected_count != 1) || (!fault && injected_count != 0)) {
      fprintf(stderr, "FAIL native injection diagnostic count=%u expected=%d\n",
              injected_count, fault ? 1 : 0);
      return 1;
   }
   printf("PASS path=native-hardware-direct hw=1/1 direct=1/1 abuffer=0/0 "
          "injected=%u\n", injected_count);
   return 0;
}

static int
run_wrapped(const char *self, int fault)
{
   int fds[2];
   if (pipe(fds)) {
      fprintf(stderr, "pipe failed: %s\n", strerror(errno));
      return 1;
   }
   pid_t pid = fork();
   if (pid < 0) {
      fprintf(stderr, "fork failed: %s\n", strerror(errno));
      return 1;
   }
   if (pid == 0) {
      close(fds[0]);
      if (dup2(fds[1], STDERR_FILENO) < 0)
         _exit(126);
      close(fds[1]);
      setenv("CUDAPIPE_TEXTURE_CACHE", "1", 1);
      setenv("CUDAPIPE_TEXTURE_CACHE_STATS", "1", 1);
      if (fault)
         setenv("CUDAPIPE_TEXTURE_CACHE_FAIL_AFTER_FS_ENQUEUE", "1", 1);
      else
         unsetenv("CUDAPIPE_TEXTURE_CACHE_FAIL_AFTER_FS_ENQUEUE");
      char *const normal_args[] = { (char *)self, (char *)"--child", NULL };
      char *const fault_args[] = {
         (char *)self, (char *)"--child-fault", NULL
      };
      execvp(self, fault ? fault_args : normal_args);
      _exit(127);
   }

   close(fds[1]);
   size_t used = 0, cap = 4096;
   char *log = malloc(cap);
   if (!log)
      return 1;
   for (;;) {
      if (used + 2048 + 1 > cap) {
         cap *= 2;
         char *grown = realloc(log, cap);
         if (!grown) {
            free(log);
            return 1;
         }
         log = grown;
      }
      ssize_t got = read(fds[0], log + used, cap - used - 1);
      if (got > 0) {
         used += (size_t)got;
         continue;
      }
      if (got < 0 && errno == EINTR)
         continue;
      break;
   }
   close(fds[0]);
   log[used] = 0;
   if (used)
      fwrite(log, 1, used, stderr);

   int status = 0;
   if (waitpid(pid, &status, 0) < 0) {
      fprintf(stderr, "waitpid failed: %s\n", strerror(errno));
      free(log);
      return 1;
   }
   if (!WIFEXITED(status) || WEXITSTATUS(status)) {
      int ret = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
      free(log);
      return ret;
   }
   int ret = assert_native_stats(log, fault);
   free(log);
   return ret;
}

static int
run_child(int fault)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "cpvk_texture_cache_no_replay",
      .apiVersion = VK_API_VERSION_1_0,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   VK_OK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t ndev = 0;
   VK_OK(vkEnumeratePhysicalDevices(inst, &ndev, NULL));
   if (!ndev) {
      fprintf(stderr, "no Vulkan physical device\n");
      return 2;
   }
   VkPhysicalDevice *pdevs = calloc(ndev, sizeof(*pdevs));
   if (!pdevs)
      return 2;
   VK_OK(vkEnumeratePhysicalDevices(inst, &ndev, pdevs));
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);

   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceFeatures supported;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceFeatures(pdev, &supported);
   fprintf(stderr, "cpvk-test-device: %s\n", props.deviceName);
   if (!supported.fragmentStoresAndAtomics) {
      fprintf(stderr, "fragmentStoresAndAtomics is not supported\n");
      return 2;
   }

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, NULL);
   VkQueueFamilyProperties *qprops = calloc(nq, sizeof(*qprops));
   if (!qprops)
      return 2;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qprops);
   uint32_t qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++)
      if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qprops[i].queueCount) {
         qfam = i;
         break;
      }
   free(qprops);
   if (qfam == UINT32_MAX) {
      fprintf(stderr, "no graphics queue family\n");
      return 2;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qfam,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceFeatures enabled = {
      .fragmentStoresAndAtomics = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &enabled,
   };
   VkDevice dev;
   VK_OK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfam, 0, &queue);

   VkBuffer upload, readback, counter;
   VkDeviceMemory upload_mem, readback_mem, counter_mem;
   make_buffer(pdev, dev, 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &upload, &upload_mem);
   make_buffer(pdev, dev, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &readback, &readback_mem);
   make_buffer(pdev, dev, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &counter, &counter_mem);

   uint8_t *upload_map;
   uint8_t *readback_map;
   uint32_t *counter_map;
   VK_OK(vkMapMemory(dev, upload_mem, 0, VK_WHOLE_SIZE, 0,
                     (void **)&upload_map));
   VK_OK(vkMapMemory(dev, readback_mem, 0, VK_WHOLE_SIZE, 0,
                     (void **)&readback_map));
   VK_OK(vkMapMemory(dev, counter_mem, 0, VK_WHOLE_SIZE, 0,
                     (void **)&counter_map));
   const uint8_t expected_rgba[4] = { 37, 89, 151, 255 };
   memcpy(upload_map, expected_rgba, sizeof(expected_rgba));
   memset(readback_map, 0xcd, W * H * 4);
   *counter_map = 0;

   VkImage texture, color;
   VkDeviceMemory texture_mem, color_mem;
   make_image(pdev, dev, 1, 1,
              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              &texture, &texture_mem);
   make_image(pdev, dev, W, H,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
              &color, &color_mem);

   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView texture_view, color_view;
   vi.image = texture;
   VK_OK(vkCreateImageView(dev, &vi, NULL, &texture_view));
   vi.image = color;
   VK_OK(vkCreateImageView(dev, &vi, NULL, &color_view));

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod = 0.0f,
      .maxLod = 0.0f,
   };
   VkSampler sampler;
   VK_OK(vkCreateSampler(dev, &sci, NULL, &sampler));

   VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
   VkDescriptorSetLayoutCreateInfo dlci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout dsl;
   VK_OK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
   };
   VkPipelineLayout pipeline_layout;
   VK_OK(vkCreatePipelineLayout(dev, &plci, NULL, &pipeline_layout));

   VkDescriptorPoolSize pool_sizes[2] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   VK_OK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl,
   };
   VkDescriptorSet set;
   VK_OK(vkAllocateDescriptorSets(dev, &dsai, &set));
   VkDescriptorImageInfo di = {
      .sampler = sampler,
      .imageView = texture_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   VkDescriptorBufferInfo db = {
      .buffer = counter,
      .offset = 0,
      .range = sizeof(uint32_t),
   };
   VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &di,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &db,
      },
   };
   vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

   VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass render_pass;
   VK_OK(vkCreateRenderPass(dev, &rpci, NULL, &render_pass));
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &color_view,
      .width = W,
      .height = H,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   VK_OK(vkCreateFramebuffer(dev, &fbci, NULL, &framebuffer));

   VkShaderModule vs = make_shader(dev, vert_spv, sizeof(vert_spv));
   VkShaderModule fs = make_shader(dev, frag_spv, sizeof(frag_spv));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main",
      },
   };
   VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   VkDynamicState dynamic_states[2] = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
   };
   VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamic_states,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &ms,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic,
      .layout = pipeline_layout,
      .renderPass = render_pass,
      .subpass = 0,
   };
   VkPipeline pipeline;
   VK_OK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL,
                                   &pipeline));

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qfam,
   };
   VkCommandPool command_pool;
   VK_OK(vkCreateCommandPool(dev, &cpci, NULL, &command_pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   VK_OK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VK_OK(vkBeginCommandBuffer(cmd, &cbbi));

   VkImageMemoryBarrier initial[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = texture,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = color,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      },
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, 0, NULL, 0, NULL, 2, initial);
   VkBufferImageCopy upload_region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { 1, 1, 1 },
   };
   vkCmdCopyBufferToImage(cmd, upload, texture,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &upload_region);
   VkImageMemoryBarrier texture_ready = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = texture,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0, 0, NULL, 0, NULL, 1, &texture_ready);

   VkClearValue clear = { .color = {{ 0.0f, 0.0f, 0.0f, 0.0f }} };
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = { { 0, 0 }, { W, H } },
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &set, 0, NULL);
   VkViewport viewport = { 0.0f, 0.0f, W, H, 0.0f, 1.0f };
   VkRect2D scissor = { { 0, 0 }, { W, H } };
   vkCmdSetViewport(cmd, 0, 1, &viewport);
   vkCmdSetScissor(cmd, 0, 1, &scissor);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   VkImageMemoryBarrier color_ready = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = color,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkBufferMemoryBarrier counter_ready = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counter,
      .offset = 0,
      .size = sizeof(uint32_t),
   };
   vkCmdPipelineBarrier(cmd,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_HOST_BIT,
                        0, 0, NULL, 1, &counter_ready, 1, &color_ready);
   VkBufferImageCopy read_region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &read_region);
   VkBufferMemoryBarrier readback_ready = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        0, 0, NULL, 1, &readback_ready, 0, NULL);
   VK_OK(vkEndCommandBuffer(cmd));

   VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   VkResult submit_result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
   VkResult wait_result = VK_NOT_READY;
   if (submit_result == VK_SUCCESS)
      wait_result = vkQueueWaitIdle(queue);

   /* counter_map was deliberately mapped before submission.  Read it even if
    * either Vulkan call reported device loss; this is the observable proof
    * that the already successful FS enqueue was not replayed. */
   uint32_t got = *(volatile uint32_t *)counter_map;
   unsigned lost_count = (submit_result == VK_ERROR_DEVICE_LOST) +
                         (wait_result == VK_ERROR_DEVICE_LOST);
   printf("result=%s path=optimal-device-local+hardware-eligible-direct "
          "submit=%d wait=%d counter=%u expected=%u\n",
          fault ? "fault" : "control", submit_result, wait_result,
          got, EXPECTED);
   fflush(stdout);

   int fail = 0;
   if (got != EXPECTED) {
      fprintf(stderr, "FAIL counter=%u expected=%u (0=lost, 2048=replayed)\n",
              got, EXPECTED);
      fail = 1;
   }
   if (fault) {
      if (lost_count != 1 ||
          (submit_result != VK_SUCCESS &&
           submit_result != VK_ERROR_DEVICE_LOST) ||
          (submit_result == VK_SUCCESS && wait_result != VK_ERROR_DEVICE_LOST)) {
         fprintf(stderr, "FAIL fault result: submit=%d wait=%d lost_count=%u\n",
                 submit_result, wait_result, lost_count);
         fail = 1;
      }
   } else if (submit_result != VK_SUCCESS || wait_result != VK_SUCCESS) {
      fprintf(stderr, "FAIL control result: submit=%d wait=%d\n",
              submit_result, wait_result);
      fail = 1;
   }

   if (!fault && !fail) {
      for (uint32_t p = 0; p < EXPECTED; p++) {
         if (memcmp(readback_map + p * 4, expected_rgba, 4)) {
            fprintf(stderr, "FAIL pixel %u got=(%u,%u,%u,%u) "
                    "expected=(%u,%u,%u,%u)\n", p,
                    readback_map[p * 4 + 0], readback_map[p * 4 + 1],
                    readback_map[p * 4 + 2], readback_map[p * 4 + 3],
                    expected_rgba[0], expected_rgba[1],
                    expected_rgba[2], expected_rgba[3]);
            fail = 1;
            break;
         }
      }
   }
   if (!fail)
      printf("PASS %s exact counter=%u%s\n", fault ? "fault-no-replay" : "control",
             got, fault ? "" : " pixels=1024/1024");

   vkUnmapMemory(dev, counter_mem);
   vkUnmapMemory(dev, readback_mem);
   vkUnmapMemory(dev, upload_mem);
   vkDestroyCommandPool(dev, command_pool, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyFramebuffer(dev, framebuffer, NULL);
   vkDestroyRenderPass(dev, render_pass, NULL);
   vkDestroyDescriptorPool(dev, pool, NULL);
   vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyImageView(dev, color_view, NULL);
   vkDestroyImageView(dev, texture_view, NULL);
   vkDestroyImage(dev, color, NULL);
   vkFreeMemory(dev, color_mem, NULL);
   vkDestroyImage(dev, texture, NULL);
   vkFreeMemory(dev, texture_mem, NULL);
   vkDestroyBuffer(dev, counter, NULL);
   vkFreeMemory(dev, counter_mem, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, readback_mem, NULL);
   vkDestroyBuffer(dev, upload, NULL);
   vkFreeMemory(dev, upload_mem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--child"))
      return run_child(0);
   if (argc == 2 && !strcmp(argv[1], "--child-fault"))
      return run_child(1);
   if (argc == 1)
      return run_wrapped(argv[0], 0);
   if (argc == 2 && !strcmp(argv[1], "--fault"))
      return run_wrapped(argv[0], 1);
   fprintf(stderr, "usage: %s [--fault]\n", argv[0]);
   return 2;
}
