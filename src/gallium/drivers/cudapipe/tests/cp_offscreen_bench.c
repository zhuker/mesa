/*
 * Offscreen scene benchmark for cudapipe.
 *
 * Renders a textured, depth-tested, instanced scene into a VkImage, times it,
 * reads the result back and writes a PNG. No swapchain and no display server:
 * the driver targets headless rendering, so the harness does too.
 *
 * dEQP covers features one at a time on small images. This covers what an
 * application actually does — many instances, a real depth buffer, perspective,
 * a sampled texture — at a size where timings mean something, and leaves an
 * image behind that can be diffed between runs.
 *
 * Build:
 *   cc -O2 cp_offscreen_bench.c -o cp_offscreen_bench -lvulkan -lz -lm
 *   glslangValidator -V cp_bench.vert -o cp_bench.vert.spv
 *   glslangValidator -V cp_bench.frag -o cp_bench.frag.spv
 *
 * Run:
 *   VK_DRIVER_FILES=<icd>.json ./cp_offscreen_bench [frames] [out.png]
 */

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#define WIDTH  512
#define HEIGHT 512
#define TEX_SIZE 64
#define GRID 8                        /* GRID * GRID instanced cubes */
#define NUM_INSTANCES (GRID * GRID)

#define VK_CHECK(x) do {                                                  \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS) {                                                \
      fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, _r); \
      exit(1);                                                            \
   }                                                                      \
} while (0)

struct vertex {
   float pos[3];
   float uv[2];
};

/* A unit cube: 6 faces, 2 triangles each, with per-face UVs. */
static const struct vertex cube_verts[] = {
   {{-1,-1,-1},{0,0}}, {{ 1,-1,-1},{1,0}}, {{ 1, 1,-1},{1,1}},
   {{-1,-1,-1},{0,0}}, {{ 1, 1,-1},{1,1}}, {{-1, 1,-1},{0,1}},
   {{-1,-1, 1},{0,0}}, {{ 1, 1, 1},{1,1}}, {{ 1,-1, 1},{1,0}},
   {{-1,-1, 1},{0,0}}, {{-1, 1, 1},{0,1}}, {{ 1, 1, 1},{1,1}},
   {{-1,-1,-1},{0,0}}, {{-1, 1, 1},{1,1}}, {{-1,-1, 1},{1,0}},
   {{-1,-1,-1},{0,0}}, {{-1, 1,-1},{0,1}}, {{-1, 1, 1},{1,1}},
   {{ 1,-1,-1},{0,0}}, {{ 1,-1, 1},{1,0}}, {{ 1, 1, 1},{1,1}},
   {{ 1,-1,-1},{0,0}}, {{ 1, 1, 1},{1,1}}, {{ 1, 1,-1},{0,1}},
   {{-1,-1,-1},{0,0}}, {{-1,-1, 1},{0,1}}, {{ 1,-1, 1},{1,1}},
   {{-1,-1,-1},{0,0}}, {{ 1,-1, 1},{1,1}}, {{ 1,-1,-1},{1,0}},
   {{-1, 1,-1},{0,0}}, {{ 1, 1, 1},{1,1}}, {{-1, 1, 1},{0,1}},
   {{-1, 1,-1},{0,0}}, {{ 1, 1,-1},{1,0}}, {{ 1, 1, 1},{1,1}},
};
#define NUM_VERTS ((unsigned)(sizeof(cube_verts) / sizeof(cube_verts[0])))

struct instance {
   float offset[4];
   float tint[4];
};

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static uint32_t *
read_spv(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "cannot open %s (run glslangValidator first)\n", path);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long len = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *data = malloc(len);
   if (fread(data, 1, len, f) != (size_t)len) {
      fprintf(stderr, "short read on %s\n", path);
      exit(1);
   }
   fclose(f);
   *size = len;
   return data;
}

static void
png_chunk(FILE *f, const char *type, const uint8_t *data, size_t len)
{
   uint8_t be[4] = { len >> 24, len >> 16, len >> 8, len };
   fwrite(be, 1, 4, f);
   uLong crc = crc32(0, (const Bytef *)type, 4);
   fwrite(type, 1, 4, f);
   if (len) {
      crc = crc32(crc, data, len);
      fwrite(data, 1, len, f);
   }
   uint8_t cb[4] = { crc >> 24, crc >> 16, crc >> 8, crc };
   fwrite(cb, 1, 4, f);
}

/* Minimal PNG writer, so the harness needs no image library. */
static void
write_png(const char *path, const uint8_t *rgba, unsigned w, unsigned h)
{
   size_t stride = (size_t)w * 4 + 1;
   size_t raw_size = (size_t)h * stride;
   uint8_t *raw = malloc(raw_size);
   for (unsigned y = 0; y < h; y++) {
      raw[y * stride] = 0;  /* filter type: none */
      memcpy(raw + y * stride + 1, rgba + (size_t)y * w * 4, (size_t)w * 4);
   }

   uLongf comp_size = compressBound(raw_size);
   uint8_t *comp = malloc(comp_size);
   compress2(comp, &comp_size, raw, raw_size, 6);
   free(raw);

   FILE *f = fopen(path, "wb");
   if (!f) {
      free(comp);
      return;
   }
   static const uint8_t sig[8] = { 137, 'P', 'N', 'G', 13, 10, 26, 10 };
   fwrite(sig, 1, 8, f);

   uint8_t ihdr[13] = {
      w >> 24, w >> 16, w >> 8, w,
      h >> 24, h >> 16, h >> 8, h,
      8, 6, 0, 0, 0,   /* 8-bit RGBA */
   };
   png_chunk(f, "IHDR", ihdr, sizeof(ihdr));
   png_chunk(f, "IDAT", comp, comp_size);
   png_chunk(f, "IEND", NULL, 0);
   fclose(f);
   free(comp);
}

static uint32_t
find_memory_type(VkPhysicalDevice pdev, uint32_t type_bits,
                 VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mem;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mem);
   for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
      if ((type_bits & (1u << i)) &&
          (mem.memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   fprintf(stderr, "no memory type with 0x%x\n", want);
   exit(1);
}

struct buffer {
   VkBuffer buf;
   VkDeviceMemory mem;
};

static struct buffer
make_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size,
            VkBufferUsageFlags usage, const void *data)
{
   struct buffer b;
   VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));

   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b.buf, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_memory_type(pdev, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
   VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));

   if (data) {
      void *ptr;
      VK_CHECK(vkMapMemory(dev, b.mem, 0, size, 0, &ptr));
      memcpy(ptr, data, size);
      vkUnmapMemory(dev, b.mem);
   }
   return b;
}

struct image {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
};

static struct image
make_image(VkDevice dev, VkPhysicalDevice pdev, unsigned w, unsigned h,
           VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
   struct image im;
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = { w, h, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VK_CHECK(vkCreateImage(dev, &ici, NULL, &im.img));

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.img, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_memory_type(pdev, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &im.mem));
   VK_CHECK(vkBindImageMemory(dev, im.img, im.mem, 0));

   VkImageViewCreateInfo iv = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = im.img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = { aspect, 0, 1, 0, 1 },
   };
   VK_CHECK(vkCreateImageView(dev, &iv, NULL, &im.view));
   return im;
}

/* Perspective projection times a rotating view, written column-major. */
static void
build_mvp(float *m, float angle)
{
   const float fov = 1.0f;          /* radians */
   const float aspect = (float)WIDTH / HEIGHT;
   const float znear = 1.0f, zfar = 100.0f;
   float f = 1.0f / tanf(fov * 0.5f);

   float proj[16] = {
      f / aspect, 0, 0, 0,
      0, -f, 0, 0,                  /* negated: Vulkan's Y points down */
      0, 0, zfar / (znear - zfar), -1,
      0, 0, (znear * zfar) / (znear - zfar), 0,
   };

   float c = cosf(angle), s = sinf(angle);
   float dist = 34.0f;
   float view[16] = {
      c, 0, -s, 0,
      0, 1,  0, 0,
      s, 0,  c, 0,
      0, 0, -dist, 1,
   };

   for (int col = 0; col < 4; col++) {
      for (int row = 0; row < 4; row++) {
         float sum = 0.0f;
         for (int k = 0; k < 4; k++)
            sum += proj[k * 4 + row] * view[col * 4 + k];
         m[col * 4 + row] = sum;
      }
   }
}

int
main(int argc, char **argv)
{
   unsigned frames = argc > 1 ? (unsigned)atoi(argv[1]) : 10;
   const char *out_path = argc > 2 ? argv[2] : "cp_bench.png";
   const char *shader_dir = getenv("CP_BENCH_SHADERS");
   if (!shader_dir)
      shader_dir = ".";

   VkInstance instance;
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "cp_offscreen_bench",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VK_CHECK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t n = 0;
   vkEnumeratePhysicalDevices(instance, &n, NULL);
   VkPhysicalDevice *pdevs = malloc(n * sizeof(*pdevs));
   vkEnumeratePhysicalDevices(instance, &n, pdevs);
   VkPhysicalDevice pdev = pdevs[0];

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   printf("device : %s\n", props.deviceName);

   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n, NULL);
   VkQueueFamilyProperties *qfs = malloc(n * sizeof(*qfs));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n, qfs);
   uint32_t qf = 0;
   for (uint32_t i = 0; i < n; i++)
      if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qf = i; break; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice dev;
   VK_CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   struct image color = make_image(dev, pdev, WIDTH, HEIGHT,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT);
   struct image depth = make_image(dev, pdev, WIDTH, HEIGHT,
      VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_IMAGE_ASPECT_DEPTH_BIT);
   struct image tex = make_image(dev, pdev, TEX_SIZE, TEX_SIZE,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT);

   /*
    * A smooth gradient, deliberately not a checkerboard.
    *
    * The cubes are small on screen, so the texture is minified with no mip
    * levels. A high-frequency pattern aliases hard under minification, and two
    * correct implementations that pick even slightly different sample points
    * then disagree wildly — which would drown out real bugs when diffing
    * against another driver. A smooth texture keeps sampling differences
    * proportional to the error that caused them.
    */
   uint8_t *texels = malloc(TEX_SIZE * TEX_SIZE * 4);
   for (unsigned y = 0; y < TEX_SIZE; y++) {
      for (unsigned x = 0; x < TEX_SIZE; x++) {
         uint8_t *p = texels + (y * TEX_SIZE + x) * 4;
         p[0] = (uint8_t)(x * 255 / (TEX_SIZE - 1));
         p[1] = (uint8_t)(y * 255 / (TEX_SIZE - 1));
         p[2] = (uint8_t)(255 - (x + y) * 255 / (2 * (TEX_SIZE - 1)));
         p[3] = 255;
      }
   }
   struct buffer tex_stage = make_buffer(dev, pdev, TEX_SIZE * TEX_SIZE * 4,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT, texels);

   VkSampler sampler;
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
   };
   VK_CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

   struct instance instances[NUM_INSTANCES];
   for (unsigned i = 0; i < NUM_INSTANCES; i++) {
      unsigned gx = i % GRID, gy = i / GRID;
      instances[i].offset[0] = ((float)gx - (GRID - 1) * 0.5f) * 3.0f;
      instances[i].offset[1] = ((float)gy - (GRID - 1) * 0.5f) * 3.0f;
      instances[i].offset[2] = 0.0f;
      instances[i].offset[3] = 0.0f;
      instances[i].tint[0] = 0.4f + 0.6f * (float)gx / GRID;
      instances[i].tint[1] = 0.4f + 0.6f * (float)gy / GRID;
      instances[i].tint[2] = 1.0f - 0.5f * (float)(gx + gy) / (2 * GRID);
      instances[i].tint[3] = 1.0f;
   }

   struct buffer vbo = make_buffer(dev, pdev, sizeof(cube_verts),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cube_verts);
   struct buffer inst = make_buffer(dev, pdev, sizeof(instances),
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, instances);
   struct buffer ubo = make_buffer(dev, pdev, 16 * sizeof(float),
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, NULL);
   struct buffer readback = make_buffer(dev, pdev, WIDTH * HEIGHT * 4,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL);

   printf("scene  : %u instances x %u verts = %u triangles at %ux%u\n",
          NUM_INSTANCES, NUM_VERTS, NUM_INSTANCES * NUM_VERTS / 3, WIDTH, HEIGHT);

   /* ---- descriptors ---- */
   VkDescriptorSetLayoutBinding bindings[2] = {
      { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
        VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout dsl;
   VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

   VkDescriptorPoolSize pool_sizes[2] = {
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));

   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl,
   };
   VkDescriptorSet dset;
   VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

   VkDescriptorBufferInfo dbi = { ubo.buf, 0, 16 * sizeof(float) };
   VkDescriptorImageInfo dii = {
      sampler, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
   };
   VkWriteDescriptorSet writes[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
        .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &dbi },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
        .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii },
   };
   vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

   /* ---- render pass ---- */
   VkAttachmentDescription attachments[2] = {
      { .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL },
      { .format = VK_FORMAT_D32_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference color_ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference depth_ref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pDepthStencilAttachment = &depth_ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass rp;
   VK_CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkImageView fb_views[2] = { color.view, depth.view };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp,
      .attachmentCount = 2,
      .pAttachments = fb_views,
      .width = WIDTH, .height = HEIGHT, .layers = 1,
   };
   VkFramebuffer fb;
   VK_CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   /* ---- pipeline ---- */
   char path[512];
   size_t vs_size, fs_size;
   snprintf(path, sizeof(path), "%s/cp_bench.vert.spv", shader_dir);
   uint32_t *vs_code = read_spv(path, &vs_size);
   snprintf(path, sizeof(path), "%s/cp_bench.frag.spv", shader_dir);
   uint32_t *fs_code = read_spv(path, &fs_size);

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = vs_size, .pCode = vs_code,
   };
   VkShaderModule vs, fs;
   VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &vs));
   smci.codeSize = fs_size;
   smci.pCode = fs_code;
   VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &fs));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };

   /* Binding 1 advances per instance — this is what exercises instancing. */
   VkVertexInputBindingDescription vbind[2] = {
      { 0, sizeof(struct vertex), VK_VERTEX_INPUT_RATE_VERTEX },
      { 1, sizeof(struct instance), VK_VERTEX_INPUT_RATE_INSTANCE },
   };
   VkVertexInputAttributeDescription vattr[4] = {
      { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(struct vertex, pos) },
      { 1, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(struct vertex, uv) },
      { 2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct instance, offset) },
      { 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(struct instance, tint) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 2, .pVertexBindingDescriptions = vbind,
      .vertexAttributeDescriptionCount = 4, .pVertexAttributeDescriptions = vattr,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport viewport = { 0, 0, WIDTH, HEIGHT, 0.0f, 1.0f };
   VkRect2D scissor = { { 0, 0 }, { WIDTH, HEIGHT } };
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
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
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
   };
   VkPipelineLayout layout;
   VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = layout, .renderPass = rp,
   };
   VkPipeline pipeline;
   VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline));

   /* ---- command buffer ---- */
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qf,
   };
   VkCommandPool cmd_pool;
   VK_CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cmd_pool));

   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };

   /* Upload the texture once. */
   VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkImageMemoryBarrier tb = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = tex.img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &tb);
   VkBufferImageCopy bic = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { TEX_SIZE, TEX_SIZE, 1 },
   };
   vkCmdCopyBufferToImage(cmd, tex_stage.buf, tex.img,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
   tb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   tb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   tb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   tb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &tb);
   VK_CHECK(vkEndCommandBuffer(cmd));
   VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_CHECK(vkQueueWaitIdle(queue));

   /* ---- frames ---- */
   double best = 1e30, total = 0.0;
   for (unsigned f = 0; f < frames; f++) {
      float *mvp;
      VK_CHECK(vkMapMemory(dev, ubo.mem, 0, 16 * sizeof(float), 0, (void **)&mvp));
      build_mvp(mvp, f * 0.05f);
      vkUnmapMemory(dev, ubo.mem);

      double t0 = now_ms();

      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));

      VkClearValue clears[2];
      clears[0].color = (VkClearColorValue){ { 0.08f, 0.10f, 0.16f, 1.0f } };
      clears[1].depthStencil = (VkClearDepthStencilValue){ 1.0f, 0 };
      VkRenderPassBeginInfo rpbi = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = rp, .framebuffer = fb,
         .renderArea = { { 0, 0 }, { WIDTH, HEIGHT } },
         .clearValueCount = 2, .pClearValues = clears,
      };
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &dset, 0, NULL);
      VkBuffer vbufs[2] = { vbo.buf, inst.buf };
      VkDeviceSize offsets[2] = { 0, 0 };
      vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, offsets);
      vkCmdDraw(cmd, NUM_VERTS, NUM_INSTANCES, 0, 0);
      vkCmdEndRenderPass(cmd);

      VK_CHECK(vkEndCommandBuffer(cmd));
      VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));

      double dt = now_ms() - t0;
      total += dt;
      if (dt < best)
         best = dt;
   }

   printf("frames : %u   best %.1f ms   mean %.1f ms   (%.1f fps at best)\n",
          frames, best, total / frames, 1000.0 / best);

   /* ---- read back ---- */
   VK_CHECK(vkResetCommandBuffer(cmd, 0));
   VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
   VkBufferImageCopy rb = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { WIDTH, HEIGHT, 1 },
   };
   vkCmdCopyImageToBuffer(cmd, color.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buf, 1, &rb);
   VK_CHECK(vkEndCommandBuffer(cmd));
   VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_CHECK(vkQueueWaitIdle(queue));

   void *pixels;
   VK_CHECK(vkMapMemory(dev, readback.mem, 0, WIDTH * HEIGHT * 4, 0, &pixels));
   write_png(out_path, pixels, WIDTH, HEIGHT);
   vkUnmapMemory(dev, readback.mem);
   printf("wrote  : %s\n", out_path);

   return 0;
}
