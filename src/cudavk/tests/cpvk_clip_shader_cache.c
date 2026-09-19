/* Native graphics compile/cache integration; NOT a CPU test.
 * Link the driver's static archive, not the loader/shared ICD, and use
 * -Wl,--wrap=cp_compile_nir_to_ptx. No production instrumentation is needed.
 * Running creates a CUDA context and JITs real shaders. It submits no draws.
 * Build-only is safe; execution requires separate device authorization.
 */
#include "cpvk_private.h"
#include "cp_clip_order.h"
#include "cp_shader_clip_metadata.h"
#include "nir.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c11/threads.h"

#define REQUIRE(c) do { if (!(c)) { \
   fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
   exit(EXIT_FAILURE); } } while (0)
#define VK_CHECK(c) REQUIRE((c) == VK_SUCCESS)

static atomic_uint backend_calls[MESA_SHADER_STAGES];
static bool race_enabled;
static mtx_t race_lock;
static cnd_t race_changed;
static unsigned race_arrived;

/* GNU ld redirects only the real call boundary in cpvk_pipeline.c. The
 * wrapper forwards all arguments unchanged, and never fabricates a binary.
 * Waiting AFTER both real compiles finish forces the publication-race path:
 * neither caller can publish before the other has already missed and built.
 */
struct cp_shader_binary *__real_cp_compile_nir_to_ptx(
   struct nir_shader *, int, int, const char *, const char *, const char *,
   bool, bool, bool, bool);
struct cp_shader_binary *__wrap_cp_compile_nir_to_ptx(
   struct nir_shader *, int, int, const char *, const char *, const char *,
   bool, bool, bool, bool);

struct cp_shader_binary *
__wrap_cp_compile_nir_to_ptx(struct nir_shader *nir, int major, int minor,
                           const char *sampler, const char *math,
                           const char *fs_helper, bool no_inline,
                           bool inline_fs, bool force_fused, bool hw_texture)
{
   mesa_shader_stage stage = nir->info.stage;
   atomic_fetch_add(&backend_calls[stage], 1);
   struct cp_shader_binary *bin = __real_cp_compile_nir_to_ptx(
      nir, major, minor, sampler, math, fs_helper, no_inline, inline_fs,
      force_fused, hw_texture);
   if (race_enabled && stage == MESA_SHADER_FRAGMENT) {
      REQUIRE(bin && cp_shader_has_standalone_exec(bin));
      REQUIRE(mtx_lock(&race_lock) == thrd_success);
      race_arrived++;
      REQUIRE(race_arrived <= 2);
      REQUIRE(cnd_broadcast(&race_changed) == thrd_success);
      while (race_arrived != 2)
         REQUIRE(cnd_wait(&race_changed, &race_lock) == thrd_success);
      REQUIRE(mtx_unlock(&race_lock) == thrd_success);
   }
   return bin;
}

static VkShaderModule
load_shader(struct cpvk_device *dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   REQUIRE(f);
   REQUIRE(fseek(f, 0, SEEK_END) == 0);
   long bytes = ftell(f);
   REQUIRE(bytes > 0 && bytes % sizeof(uint32_t) == 0);
   REQUIRE(fseek(f, 0, SEEK_SET) == 0);
   uint32_t *code = malloc(bytes);
   REQUIRE(code);
   REQUIRE(fread(code, 1, bytes, f) == (size_t)bytes);
   REQUIRE(fclose(f) == 0);
   VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes, .pCode = code,
   };
   VkShaderModule module;
   VK_CHECK(dev->vk.dispatch_table.CreateShaderModule(
      cpvk_device_to_handle(dev), &info, NULL, &module));
   free(code);
   return module;
}

struct build_job {
   VkDevice device;
   const VkGraphicsPipelineCreateInfo *info;
   VkPipeline pipeline;
   VkResult result;
};

static int
build_pipeline(void *data)
{
   struct build_job *job = data;
   job->result = cpvk_CreateGraphicsPipelines(job->device, VK_NULL_HANDLE, 1,
                                             job->info, NULL, &job->pipeline);
   return 0;
}

static void
check_binary(struct cp_shader_binary *bin, unsigned mask)
{
   REQUIRE(bin && bin->is_fragment && cp_shader_has_standalone_exec(bin));
   REQUIRE(bin->clip_order_exclusions == mask);
   /* These fixtures have no unrelated effects. Retained reasons must not
    * broaden the existing side-effect/discard metadata. */
   REQUIRE(!bin->writes_memory && !bin->uses_discard);
   for (unsigned draws = 1; draws <= 3; draws++) {
      struct cp_clip_order_policy p = {
         .batch_draws = draws,
         .depth_test = true, .depth_write = true, .inclusive = true,
         .unsupported_fragment_semantics = cp_shader_clip_excluded(bin->clip_order_exclusions),
      };
      REQUIRE(cp_clip_order_stable(true, &p) == (mask == 0));
      REQUIRE(!cp_clip_order_stable(false, &p));
      p.blend = true;
      REQUIRE(cp_clip_order_stable(true, &p));
      REQUIRE(cp_clip_order_stable(false, &p));
      p.blend = false;
      p.reads_const_bufs = true;
      REQUIRE(cp_clip_order_stable(true, &p) == (draws > 1 || mask == 0));
      REQUIRE(cp_clip_order_stable(false, &p) == (draws > 1));
   }
}

int
main(int argc, char **argv)
{
   if ((argc != 5 && argc != 6) ||
       strcmp(argv[1], "--allow-device-compilation") ||
       (argc == 6 && strcmp(argv[5], "--race"))) {
      fprintf(stderr, "Usage: %s --allow-device-compilation VS.spv FS.spv "
                      "MASK [--race]\n"
                      "DEVICE-TOUCHING test: CUDA initialization/JIT; no draws.\n",
              argv[0]);
      return 2;
   }
   char *end;
   unsigned long mask = strtoul(argv[4], &end, 0);
   REQUIRE(*argv[4] && !*end && mask <= 7);
   REQUIRE(mtx_init(&race_lock, mtx_plain) == thrd_success);
   REQUIRE(cnd_init(&race_changed) == thrd_success);
   race_enabled = argc == 6;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance instance;
   VK_CHECK(cpvk_CreateInstance(&ici, NULL, &instance));
   struct cpvk_instance *inst = cpvk_instance_from_handle(instance);
   uint32_t count = 0;
   VK_CHECK(inst->vk.dispatch_table.EnumeratePhysicalDevices(instance, &count, NULL));
   REQUIRE(count > 0);
   VkPhysicalDevice *physical = calloc(count, sizeof(*physical));
   REQUIRE(physical);
   VK_CHECK(inst->vk.dispatch_table.EnumeratePhysicalDevices(instance, &count, physical));
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
   };
   VkDevice device;
   VK_CHECK(cpvk_CreateDevice(physical[0], &dci, NULL, &device));
   free(physical);
   struct cpvk_device *dev = cpvk_device_from_handle(device);
   REQUIRE(dev->num_shaders == 0);
   VkShaderModule vs = load_shader(dev, argv[2]);
   VkShaderModule fs = load_shader(dev, argv[3]);
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   VK_CHECK(dev->vk.dispatch_table.CreatePipelineLayout(device, &lci, NULL, &layout));

   /* A complete conventional render-pass pipeline, without allocating any
    * image, command buffer or queue work. */
   VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference color = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &color,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &attachment,
      .subpassCount = 1, .pSubpasses = &subpass,
   };
   VkRenderPass render_pass;
   VK_CHECK(dev->vk.dispatch_table.CreateRenderPass(device, &rpci, NULL, &render_pass));
   VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport viewport = {0, 0, 16, 16, 0, 1};
   VkRect2D scissor = {{0, 0}, {16, 16}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = true, .depthWriteEnable = true,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
   };
   VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend,
   };
   VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pDepthStencilState = &ds, .pColorBlendState = &cb,
      .layout = layout, .renderPass = render_pass,
   };
   struct build_job jobs[2] = {
      {.device = device, .info = &info}, {.device = device, .info = &info},
   };
   if (race_enabled) {
      thrd_t threads[2];
      REQUIRE(thrd_create(&threads[0], build_pipeline, &jobs[0]) == thrd_success);
      REQUIRE(thrd_create(&threads[1], build_pipeline, &jobs[1]) == thrd_success);
      for (unsigned i = 0; i < 2; i++) {
         int result;
         REQUIRE(thrd_join(threads[i], &result) == thrd_success && result == 0);
      }
      REQUIRE(race_arrived == 2);
   } else {
      build_pipeline(&jobs[0]);
      VK_CHECK(jobs[0].result);
      REQUIRE(atomic_load(&backend_calls[MESA_SHADER_FRAGMENT]) == 1);
      REQUIRE(dev->num_shaders == 2);
      build_pipeline(&jobs[1]);
   }
   for (unsigned i = 0; i < 2; i++)
      VK_CHECK(jobs[i].result);
   unsigned expected_calls = race_enabled ? 2 : 1;
   REQUIRE(atomic_load(&backend_calls[MESA_SHADER_FRAGMENT]) == expected_calls);
   REQUIRE(dev->num_shaders == 2);
   struct cpvk_pipeline *first = cpvk_pipeline_from_handle(jobs[0].pipeline);
   struct cpvk_pipeline *second = cpvk_pipeline_from_handle(jobs[1].pipeline);
   REQUIRE(first && second && first != second && first->fs == second->fs);
   REQUIRE(first->vs == second->vs && first->vs->clip_order_exclusions == 0);
   struct cp_shader_binary *owner = first->fs;
   check_binary(owner, mask);
   unsigned fs_entries = 0;
   for (unsigned i = 0; i < dev->num_shaders; i++)
      fs_entries += dev->shader_cache[i].bin == owner;
   REQUIRE(fs_entries == 1);
   cpvk_DestroyPipeline(device, jobs[0].pipeline, NULL);
   cpvk_DestroyPipeline(device, jobs[1].pipeline, NULL);
   /* No pipeline retains the shader now; only the real device cache does. */
   check_binary(owner, mask);
   build_pipeline(&jobs[0]);
   VK_CHECK(jobs[0].result);
   REQUIRE(cpvk_pipeline_from_handle(jobs[0].pipeline)->fs == owner);
   REQUIRE(atomic_load(&backend_calls[MESA_SHADER_FRAGMENT]) == expected_calls);
   REQUIRE(dev->num_shaders == 2);
   check_binary(owner, mask);
   printf("native clip cache: PASS mask=%lu fs_backend_calls=%u entries=%u race=%u\n",
          mask, expected_calls, dev->num_shaders, race_enabled);
   for (unsigned i = 0; i < CP_SHADER_EXEC_COUNT; i++)
      printf("execution mode %u: %s\n", i, owner->exec[i].kernel ?
             "present on verified owner" : "SKIP unavailable/not requested");
   puts("No draws. No shader-output correctness or runtime fallback coverage claimed.");
   cpvk_DestroyPipeline(device, jobs[0].pipeline, NULL);
   dev->vk.dispatch_table.DestroyRenderPass(device, render_pass, NULL);
   dev->vk.dispatch_table.DestroyPipelineLayout(device, layout, NULL);
   dev->vk.dispatch_table.DestroyShaderModule(device, vs, NULL);
   dev->vk.dispatch_table.DestroyShaderModule(device, fs, NULL);
   cpvk_DestroyDevice(device, NULL);
   cpvk_DestroyInstance(instance, NULL);
   cnd_destroy(&race_changed);
   mtx_destroy(&race_lock);
   return 0;
}
