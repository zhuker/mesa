/*
 * The native driver's second test: everything the triangle did not have.
 *
 * An indexed draw out of a vertex buffer with two attributes of different
 * formats, a depth attachment with a depth test, and both attachments cleared
 * by loadOp rather than by the test writing over the memory first. Two
 * overlapping triangles at different depths, so a broken depth test is a
 * different picture and not a subtle one.
 *
 * The colour attribute is R8G8B8A8_UNORM on purpose: it is the case the
 * hardcoded four-float vertex element got wrong, reading all four bytes as a
 * single number.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

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

struct vertex {
   float x, y, z;
   unsigned char r, g, b, a;
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
   const char *vs_path = argc > 1 ? argv[1] : "/tmp/lat/vtx.vert.spv";
   const char *fs_path = argc > 2 ? argv[2] : "/tmp/lat/vtx.frag.spv";
   const char *out = argc > 3 ? argv[3] : "/tmp/lat/native_draw.ppm";

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
   const struct vertex verts[6] = {
      { -0.8f, -0.7f, 0.3f, 255,  32,  32, 255 },
      {  0.4f, -0.7f, 0.3f, 255,  32,  32, 255 },
      { -0.2f,  0.8f, 0.3f, 255,  32,  32, 255 },

      { -0.4f, -0.7f, 0.6f,  32,  64, 255, 255 },
      {  0.8f, -0.7f, 0.6f,  32,  64, 255, 255 },
      {  0.2f,  0.8f, 0.6f,  32,  64, 255, 255 },
   };
   const uint16_t indices[6] = { 3, 4, 5, 0, 1, 2 };   /* far one first */

   VkBuffer vbuf, ibuf;
   VkDeviceMemory vmem, imem;
   {
      VkBufferCreateInfo bi = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = sizeof(verts),
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
      memcpy(p, verts, sizeof(verts));

      bi.size = sizeof(indices);
      bi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      CHECK(vkCreateBuffer(dev, &bi, NULL, &ibuf));
      vkGetBufferMemoryRequirements(dev, ibuf, &br);
      bmai.allocationSize = br.size;
      bmai.memoryTypeIndex = pick_memory(pdev, br.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      CHECK(vkAllocateMemory(dev, &bmai, NULL, &imem));
      CHECK(vkBindBufferMemory(dev, ibuf, imem, 0));
      CHECK(vkMapMemory(dev, imem, 0, VK_WHOLE_SIZE, 0, &p));
      memcpy(p, indices, sizeof(indices));
   }

   VkShaderModule vs = load_spv(dev, vs_path), fs = load_spv(dev, fs_path);
   if (!vs || !fs) return 1;

   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
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
   VkVertexInputAttributeDescription vattr[2] = {
      { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = 0 },
      { .location = 1, .binding = 0, .format = VK_FORMAT_R8G8B8A8_UNORM,
        .offset = offsetof(struct vertex, r) },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vbind,
      .vertexAttributeDescriptionCount = 2,
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
   vkCmdBindIndexBuffer(cmd, ibuf, 0, VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
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
