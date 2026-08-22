/*
 * Clipped geometry, merged. The native driver's eighth test: two draws whose descriptor sets name
 * different textures.
 *
 * cpvk_batch showed that two draws differing only in a uniform buffer merge
 * correctly. gltfscenerendering's materials differ in a texture and its
 * merged frame is wrong, so this is the same test with the one structural
 * difference that separates them.
 *
 * The original comment follows.
 *
 * The native driver's fourth test: sampling a texture.
 *
 * The same geometry, but the fragment shader replaces its colour with a
 * sample from a 4x4 checkerboard, magnified by a linear filter so that the
 * result depends on the sampler state and not only on the fetch. A texture
 * handle is a computed offset into the descriptor set's buffer, which is a
 * different mechanism from a uniform buffer's address, so this is the first
 * test of that path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* Draws in the batch. gltfscenerendering's are nine; three passed. */
#define NDRAW 9
/*
 * Triangles per draw. The sample's draws carry 796 to 67,763 of them and this
 * test carried one, which is the last difference between them that has not
 * been ruled out -- and the one this driver already documents as making the
 * clipper's primitive order visible.
 */
#ifndef TPD
#define TPD 6000   /* ~48,000 triangles a batch, past gltfscenerendering's 45,184 */
#endif

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

static VkShaderModule
load_spv(VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "cannot open %s\n", path); return VK_NULL_HANDLE; }
   fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(n);
   if (fread(code, 1, n, f) != (size_t)n) { fclose(f); return VK_NULL_HANDLE; }
   fclose(f);
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = n, .pCode = code,
   };
   VkShaderModule mod = VK_NULL_HANDLE;
   vkCreateShaderModule(dev, &ci, NULL, &mod);
   free(code);
   return mod;
}

/*
 * gltfscenerendering's vertex: four attributes at 0, 12, 24 and 32 in a
 * 60-byte stride, which CUDAPIPE_DEBUG_DRAW showed is the last thing this
 * test did not have.
 */
struct vertex {
   float x, y, z;        /* 0  */
   float nx, ny, nz;     /* 12 */
   float u, v;           /* 24 */
   float cr, cg, cb;     /* 32 */
   float pad[3];         /* 44, to a 60-byte stride */
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

int
main(int argc, char **argv)
{
   const char *vs_path = argc > 1 ? argv[1] : "/tmp/lat/tex.vert.spv";
   const char *fs_path = argc > 2 ? argv[2] : "/tmp/lat/tex.frag.spv";
   const char *out = argc > 3 ? argv[3] : "/tmp/lat/native_batchtex.ppm";

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &prio };
   const char *dev_exts[] = { VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME };
   VkPhysicalDeviceDynamicRenderingFeatures dyn = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = &dyn,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qi,
                              .enabledExtensionCount = 1,
                              .ppEnabledExtensionNames = dev_exts };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);

   /* Colour target, host-visible so it can be read back. */
   VkImageCreateInfo imgi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_B8G8R8A8_UNORM,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
   VkImage img;
   CHECK(vkCreateImage(dev, &imgi, NULL, &img));

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (type == UINT32_MAX) { fprintf(stderr, "no host-visible type\n"); return 1; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, img, mem, 0));

   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = imgi.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));

   /* Depth target. Never read back; its existence is what turns the depth
    * test on, and the renderer keeps the depth values itself. */
   VkImageCreateInfo dimgi = imgi;
   dimgi.format = VK_FORMAT_D32_SFLOAT;
   dimgi.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage dimg;
   CHECK(vkCreateImage(dev, &dimgi, NULL, &dimg));
   VkMemoryRequirements dreq;
   vkGetImageMemoryRequirements(dev, dimg, &dreq);
   VkMemoryAllocateInfo dmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = dreq.size,
                                 .memoryTypeIndex =
                                    pick_memory(pdev, dreq.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
   VkDeviceMemory dmem;
   CHECK(vkAllocateMemory(dev, &dmai, NULL, &dmem));
   CHECK(vkBindImageMemory(dev, dimg, dmem, 0));
   VkImageViewCreateInfo dvci = vci;
   dvci.image = dimg;
   dvci.format = dimgi.format;
   dvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   VkImageView dview;
   CHECK(vkCreateImageView(dev, &dvci, NULL, &dview));

   /*
    * Two overlapping triangles. The red one is nearer on the left, the blue
    * one nearer on the right, so neither wins outright and a depth test that
    * does nothing is visibly a different picture.
    */
   /*
    * Three overlapping triangles at three depths, which is the one thing the
    * side-by-side version never tested: within a batch the depth-keyed
    * visibility buffer has to pick a winner *across* draws, and the winning
    * fragment has to be shaded from its own draw's binding row rather than
    * from whichever draw was last. A scene of meshes overlaps constantly;
    * three triangles in a row never do.
    */
   /*
    * NDRAW overlapping bands, each of TPD small triangles, so a batch carries
    * the triangle counts a scene's draws do.
    */
   const int nvert = NDRAW * TPD * 3;
   struct vertex *verts = calloc(nvert, sizeof(*verts));
   uint32_t *indices = calloc(nvert, sizeof(*indices));
   if (!verts || !indices)
      return 1;
   for (int n = 0; n < NDRAW; n++) {
      float x0 = -0.8f + n * 0.12f;
      float z = 0.8f - n * 0.06f;
      for (int k = 0; k < TPD; k++) {
         /*
          * Deliberately outside the clip volume. cpvk_batchtex keeps every
          * triangle on screen, which means its batches never take the stable
          * clip path -- and that path is where a merged draw's primitive
          * index has to name the input triangle for cp_write_batch_rows() to
          * find the right material. Spreading the same geometry over three
          * times the volume makes most of these triangles straddle an edge.
          */
         float fx = 3.0f * (x0 + 0.6f * ((float)(k % 40) / 40.0f));
         float fy = 3.0f * (-0.6f + 1.3f * ((float)(k / 40) / (float)((TPD + 39) / 40)));
         struct vertex *v = &verts[(n * TPD + k) * 3];
         v[0] = (struct vertex){ fx,         fy,         z,
                                 0,0,1, 0.0f,0.0f, 1,1,1 };
         v[1] = (struct vertex){ fx + 0.02f, fy,         z,
                                 0,0,1, 1.0f,0.0f, 1,1,1 };
         v[2] = (struct vertex){ fx + 0.01f, fy + 0.05f, z,
                                 0,0,1, 0.5f,1.0f, 1,1,1 };
         for (int e = 0; e < 3; e++)
            indices[(n * TPD + k) * 3 + e] = (uint32_t)((n * TPD + k) * 3 + e);
      }
   }
   const size_t verts_bytes = (size_t)nvert * sizeof(*verts);
   const size_t indices_bytes = (size_t)nvert * sizeof(*indices);

   VkBuffer vbuf, ibuf;
   VkDeviceMemory vmem, imem;
   {
      VkBufferCreateInfo bi = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = verts_bytes,
         .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT };
      CHECK(vkCreateBuffer(dev, &bi, NULL, &vbuf));
      VkMemoryRequirements br;
      vkGetBufferMemoryRequirements(dev, vbuf, &br);
      VkMemoryAllocateInfo bmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = br.size,
         .memoryTypeIndex = pick_memory(pdev, br.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &vmem));
      CHECK(vkBindBufferMemory(dev, vbuf, vmem, 0));
      void *p;
      CHECK(vkMapMemory(dev, vmem, 0, VK_WHOLE_SIZE, 0, &p));
      memcpy(p, verts, verts_bytes);

      bi.size = indices_bytes;
      bi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      CHECK(vkCreateBuffer(dev, &bi, NULL, &ibuf));
      vkGetBufferMemoryRequirements(dev, ibuf, &br);
      bmai.allocationSize = br.size;
      bmai.memoryTypeIndex = pick_memory(pdev, br.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &imem));
      CHECK(vkBindBufferMemory(dev, ibuf, imem, 0));
      CHECK(vkMapMemory(dev, imem, 0, VK_WHOLE_SIZE, 0, &p));
      memcpy(p, indices, indices_bytes);
   }

   VkShaderModule vs = load_spv(dev, vs_path), fs = load_spv(dev, fs_path);
   if (!vs || !fs) return 1;

   /* Two uniform buffers: binding 0 read by the vertex shader, binding 1 by
    * the fragment shader. */
   struct { float dx, dy, pad0, pad1; } vs_ubo = { 0.15f, -0.1f, 0, 0 };
   struct { float r, g, b, a; } fs_ubo = { 0.5f, 1.0f, 0.75f, 1.0f };
   VkBuffer ubuf[2];
   VkDeviceMemory umem[2];
   const void *usrc[2] = { &vs_ubo, &fs_ubo };
   const size_t usize[2] = { sizeof(vs_ubo), sizeof(fs_ubo) };
   for (int u = 0; u < 2; u++) {
      VkBufferCreateInfo bi = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = usize[u], .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
      CHECK(vkCreateBuffer(dev, &bi, NULL, &ubuf[u]));
      VkMemoryRequirements br;
      vkGetBufferMemoryRequirements(dev, ubuf[u], &br);
      VkMemoryAllocateInfo bmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = br.size,
         .memoryTypeIndex = pick_memory(pdev, br.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &umem[u]));
      CHECK(vkBindBufferMemory(dev, ubuf[u], umem[u], 0));
      void *p;
      CHECK(vkMapMemory(dev, umem[u], 0, VK_WHOLE_SIZE, 0, &p));
      memcpy(p, usrc[u], usize[u]);
   }

   /*
    * Two 4x4 textures, one red-ish and one blue-ish. The two draws differ in
    * which one their descriptor set names, and in nothing else.
    */
   VkImage timg[NDRAW];
   VkDeviceMemory tmem[NDRAW];
   VkImageView tview[NDRAW];
   VkImageCreateInfo timgi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { 4, 4, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };

   for (int n = 0; n < NDRAW; n++) {
      CHECK(vkCreateImage(dev, &timgi, NULL, &timg[n]));
      VkMemoryRequirements treq;
      vkGetImageMemoryRequirements(dev, timg[n], &treq);
      VkMemoryAllocateInfo tmai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = treq.size,
         .memoryTypeIndex = pick_memory(pdev, treq.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
      CHECK(vkAllocateMemory(dev, &tmai, NULL, &tmem[n]));
      CHECK(vkBindImageMemory(dev, timg[n], tmem[n], 0));

      VkImageSubresource sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
      VkSubresourceLayout lay;
      vkGetImageSubresourceLayout(dev, timg[n], &sub, &lay);
      void *p;
      CHECK(vkMapMemory(dev, tmem[n], 0, VK_WHOLE_SIZE, 0, &p));
      unsigned char *t = (unsigned char *)p + lay.offset;
      for (int y = 0; y < 4; y++)
         for (int x = 0; x < 4; x++) {
            unsigned char *px = t + y * lay.rowPitch + x * 4;
            /* A distinct colour per draw, so a batch shaded from one row
             * shows the wrong count of colours. */
            px[0] = (unsigned char)(30 + (n % 3) * 100);
            px[1] = (unsigned char)(30 + ((n / 3) % 3) * 100);
            px[2] = (unsigned char)(30 + (n * 23) % 200);
            px[3] = 255;
         }

      VkImageViewCreateInfo tvci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = timg[n], .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = timgi.format,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      CHECK(vkCreateImageView(dev, &tvci, NULL, &tview[n]));
   }

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .maxLod = 0.25f };
   VkSampler samp;
   CHECK(vkCreateSampler(dev, &sci, NULL, &samp));

   /*
    * Two descriptor set layouts, which is what a scene renderer has: set 0
    * for the frame's uniforms, set 1 for a material. Only set 1 changes
    * between the draws.
    */
   VkDescriptorSetLayoutBinding dslb0 = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
   VkDescriptorSetLayoutBinding dslb1 = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   VkDescriptorSetLayoutCreateInfo dsli0 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb0 };
   VkDescriptorSetLayoutCreateInfo dsli1 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb1 };
   VkDescriptorSetLayout dsl, dsl_mat;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli0, NULL, &dsl));
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli1, NULL, &dsl_mat));

   VkDescriptorPoolSize dps[2] = {
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NDRAW },
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NDRAW } };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = NDRAW + 1, .poolSizeCount = 2, .pPoolSizes = dps };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet dset[NDRAW], scene_set;
   VkDescriptorSetLayout dsls[NDRAW];
   for (int n = 0; n < NDRAW; n++)
      dsls[n] = dsl_mat;
   VkDescriptorSetAllocateInfo sai = dsai;
   sai.descriptorSetCount = 1;
   sai.pSetLayouts = &dsl;
   CHECK(vkAllocateDescriptorSets(dev, &sai, &scene_set));
   dsai.descriptorSetCount = NDRAW;
   dsai.pSetLayouts = dsls;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, dset));

   VkDescriptorBufferInfo dbi = { ubuf[0], 0, VK_WHOLE_SIZE };
   VkDescriptorImageInfo dii[NDRAW];
   VkWriteDescriptorSet writes[NDRAW * 2];
   for (int n = 0; n < NDRAW; n++) {
      dii[n] = (VkDescriptorImageInfo){
         .sampler = samp, .imageView = tview[n],
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
      writes[n] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset[n],
         .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &dii[n] };
   }
   writes[NDRAW] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_set,
      .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(dev, NDRAW + 1, writes, 0, NULL);

   VkDescriptorSetLayout both[2] = { dsl, dsl_mat };
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2, .pSetLayouts = both };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkVertexInputBindingDescription vbind = {
      .binding = 0, .stride = sizeof(struct vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
   VkVertexInputAttributeDescription vattr[4] = {
      { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(struct vertex, x) },
      { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(struct vertex, nx) },
      { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(struct vertex, u) },
      { .location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(struct vertex, cr) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vbind,
      .vertexAttributeDescriptionCount = 4,
      .pVertexAttributeDescriptions = vattr };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkViewport vp = { 0, 0, W, H, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS };
   VkFormat cfmt = imgi.format;
   VkPipelineRenderingCreateInfo pri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt,
      .depthAttachmentFormat = dimgi.format };
   VkGraphicsPipelineCreateInfo gpi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &pri, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = layout };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));

   VkRenderingAttachmentInfo at = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = { 0.1f, 0.1f, 0.15f, 1.0f } };
   VkRenderingAttachmentInfo dat = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = dview,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil.depth = 1.0f };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &at,
      .pDepthAttachment = &dat };

   PFN_vkCmdBeginRenderingKHR beginRendering =
      (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
   PFN_vkCmdEndRenderingKHR endRendering =
      (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");
   if (!beginRendering || !endRendering) {
      fprintf(stderr, "dynamic rendering entrypoints missing\n");
      return 1;
   }

   VkDeviceSize zero = 0;
   beginRendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &sc);
   vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &zero);
   vkCmdBindIndexBuffer(cmd, ibuf, 0, VK_INDEX_TYPE_UINT32);

   /* Two draws, two index ranges, two sets naming two different textures. */
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                           &scene_set, 0, NULL);
   for (int n = 0; n < NDRAW; n++) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1,
                              1, &dset[n], 0, NULL);
      vkCmdDrawIndexed(cmd, TPD * 3, 1, n * TPD * 3, 0, 0);
   }
   endRendering(cmd);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   void *map = NULL;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &map));

   FILE *f = fopen(out, "wb");
   fprintf(f, "P6\n%d %d\n255\n", W, H);
   const unsigned char *px = map;
   for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
         const unsigned char *p = px + (y * W + x) * 4;
         fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);
      }
   }
   fclose(f);
   printf("  %s written\n", out);
   return 0;
}
