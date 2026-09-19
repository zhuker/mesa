/* Equal-depth clipping order fixture. Based on cpvk_depth_convention.c's
 * portable optimal-image/readback setup. No renderer is used as its oracle.
 * One draw submits red (clipped at x=-1) BEFORE green (inside), both z=0.5.
 * LEQUAL and GEQUAL must leave green in overlap, regardless of clip allocation.
 * Check interior samples against original input triangles, not output fan IDs.
 * Shader sources: cpvk_clip_order.vert / .frag; embedded SPIR-V below was made
 * with glslangValidator -V. This executable is opt-in: compiling does NOT run it.
 * cc -std=c11 -Wall -Wextra -Werror cpvk_clip_order.c -lvulkan -lm -o /tmp/cpvk_clip_order
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "cp_clip_order_oracle.h"
#define W 64
#define H 64
#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, r_); return 1; } } while (0)

static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000038u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0008000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x00000021u, 0x0000002eu,
   0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00060005u,
   0x0000000bu, 0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000000bu, 0x00000000u,
   0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u, 0x746e696fu,
   0x657a6953u, 0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u,
   0x0065636eu, 0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu,
   0x00030005u, 0x0000000du, 0x00000000u, 0x00060005u, 0x00000021u, 0x565f6c67u, 0x65747265u, 0x646e4978u,
   0x00007865u, 0x00050005u, 0x00000024u, 0x65646e69u, 0x6c626178u, 0x00000065u, 0x00040005u, 0x0000002eu,
   0x6f6c6f63u, 0x00007275u, 0x00030047u, 0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu, 0x00000000u,
   0x0000000bu, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u, 0x0000000bu, 0x00000001u, 0x00050048u,
   0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u, 0x0000000bu, 0x00000003u, 0x0000000bu,
   0x00000004u, 0x00040047u, 0x00000021u, 0x0000000bu, 0x0000002au, 0x00030047u, 0x0000002eu, 0x0000000eu,
   0x00040047u, 0x0000002eu, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
   0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u,
   0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u, 0x00000009u, 0x00000001u,
   0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u, 0x0006001eu, 0x0000000bu, 0x00000007u, 0x00000006u,
   0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000003u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu,
   0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u, 0x00000006u, 0x00000002u, 0x0004002bu, 0x00000008u,
   0x00000011u, 0x00000006u, 0x0004001cu, 0x00000012u, 0x00000010u, 0x00000011u, 0x0004002bu, 0x00000006u,
   0x00000013u, 0xc0000000u, 0x0004002bu, 0x00000006u, 0x00000014u, 0xbf400000u, 0x0005002cu, 0x00000010u,
   0x00000015u, 0x00000013u, 0x00000014u, 0x0004002bu, 0x00000006u, 0x00000016u, 0x3f400000u, 0x0005002cu,
   0x00000010u, 0x00000017u, 0x00000016u, 0x00000014u, 0x0004002bu, 0x00000006u, 0x00000018u, 0x00000000u,
   0x0005002cu, 0x00000010u, 0x00000019u, 0x00000018u, 0x00000016u, 0x0004002bu, 0x00000006u, 0x0000001au,
   0xbf000000u, 0x0005002cu, 0x00000010u, 0x0000001bu, 0x0000001au, 0x0000001au, 0x0004002bu, 0x00000006u,
   0x0000001cu, 0x3f000000u, 0x0005002cu, 0x00000010u, 0x0000001du, 0x0000001cu, 0x0000001au, 0x0005002cu,
   0x00000010u, 0x0000001eu, 0x00000018u, 0x0000001cu, 0x0009002cu, 0x00000012u, 0x0000001fu, 0x00000015u,
   0x00000017u, 0x00000019u, 0x0000001bu, 0x0000001du, 0x0000001eu, 0x00040020u, 0x00000020u, 0x00000001u,
   0x0000000eu, 0x0004003bu, 0x00000020u, 0x00000021u, 0x00000001u, 0x00040020u, 0x00000023u, 0x00000007u,
   0x00000012u, 0x00040020u, 0x00000025u, 0x00000007u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000028u,
   0x3f800000u, 0x00040020u, 0x0000002cu, 0x00000003u, 0x00000007u, 0x0004003bu, 0x0000002cu, 0x0000002eu,
   0x00000003u, 0x0004002bu, 0x0000000eu, 0x00000030u, 0x00000003u, 0x00020014u, 0x00000031u, 0x0007002cu,
   0x00000007u, 0x00000033u, 0x00000028u, 0x00000018u, 0x00000018u, 0x00000028u, 0x0007002cu, 0x00000007u,
   0x00000034u, 0x00000018u, 0x00000028u, 0x00000018u, 0x00000028u, 0x00040017u, 0x00000035u, 0x00000031u,
   0x00000004u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x0004003bu, 0x00000023u, 0x00000024u, 0x00000007u, 0x0004003du, 0x0000000eu, 0x00000022u, 0x00000021u,
   0x0003003eu, 0x00000024u, 0x0000001fu, 0x00050041u, 0x00000025u, 0x00000026u, 0x00000024u, 0x00000022u,
   0x0004003du, 0x00000010u, 0x00000027u, 0x00000026u, 0x00050051u, 0x00000006u, 0x00000029u, 0x00000027u,
   0x00000000u, 0x00050051u, 0x00000006u, 0x0000002au, 0x00000027u, 0x00000001u, 0x00070050u, 0x00000007u,
   0x0000002bu, 0x00000029u, 0x0000002au, 0x0000001cu, 0x00000028u, 0x00050041u, 0x0000002cu, 0x0000002du,
   0x0000000du, 0x0000000fu, 0x0003003eu, 0x0000002du, 0x0000002bu, 0x0004003du, 0x0000000eu, 0x0000002fu,
   0x00000021u, 0x000500b1u, 0x00000031u, 0x00000032u, 0x0000002fu, 0x00000030u, 0x00070050u, 0x00000035u,
   0x00000036u, 0x00000032u, 0x00000032u, 0x00000032u, 0x00000032u, 0x000600a9u, 0x00000007u, 0x00000037u,
   0x00000036u, 0x00000033u, 0x00000034u, 0x0003003eu, 0x0000002eu, 0x00000037u, 0x000100fdu, 0x00010038u,
};

static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000000du, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x0000000bu, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x00030005u, 0x00000009u, 0x0000006fu, 0x00040005u, 0x0000000bu, 0x6f6c6f63u, 0x00007275u,
   0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00030047u, 0x0000000bu, 0x0000000eu, 0x00040047u,
   0x0000000bu, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
   0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x00040020u,
   0x0000000au, 0x00000001u, 0x00000007u, 0x0004003bu, 0x0000000au, 0x0000000bu, 0x00000001u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x00000007u,
   0x0000000cu, 0x0000000bu, 0x0003003eu, 0x00000009u, 0x0000000cu, 0x000100fdu, 0x00010038u,
};
static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static VkQueue queue;

static uint32_t
pick_memory(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(physical, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static int
make_image(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
           VkImage *image, VkDeviceMemory *memory, VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   CHECK(vkCreateImage(device, &ici, NULL, image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, *image, &req);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = pick_memory(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   CHECK(vkAllocateMemory(device, &mai, NULL, memory));
   CHECK(vkBindImageMemory(device, *image, *memory, 0));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { aspect, 0, 1, 0, 1 } };
   CHECK(vkCreateImageView(device, &vci, NULL, view));
   return 0;
}

static int
make_buffer(VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = size,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   CHECK(vkCreateBuffer(device, &bci, NULL, buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, *buffer, &req);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = pick_memory(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   CHECK(vkAllocateMemory(device, &mai, NULL, memory));
   CHECK(vkBindBufferMemory(device, *buffer, *memory, 0));
   return 0;
}

/* One draw contains clipped red first, inside green second, both z=0.5. */
static int
render(VkCompareOp compare,
       float *depth, float *colour)
{
   VkImage cimage, dimage;
   VkDeviceMemory cmemory, dmemory;
   VkImageView cview, dview;
   if (make_image(VK_FORMAT_R32G32B32A32_SFLOAT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, &cimage, &cmemory, &cview))
      return 1;
   if (make_image(VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, &dimage, &dmemory, &dview))
      return 1;

   VkAttachmentDescription att[2] = {
      { .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
      { .format = VK_FORMAT_D32_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
   };
   VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkAttachmentReference dref = { 1,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &cref,
      .pDepthStencilAttachment = &dref };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2, .pAttachments = att,
      .subpassCount = 1, .pSubpasses = &sub };
   VkRenderPass pass;
   CHECK(vkCreateRenderPass(device, &rpci, NULL, &pass));

   VkImageView views[2] = { cview, dview };
   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass,
      .attachmentCount = 2, .pAttachments = views,
      .width = W, .height = H, .layers = 1 };
   VkFramebuffer framebuffer;
   CHECK(vkCreateFramebuffer(device, &fbci, NULL, &framebuffer));

   VkShaderModuleCreateInfo vsci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof vert_spv, .pCode = vert_spv };
   VkShaderModuleCreateInfo fsci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof frag_spv, .pCode = frag_spv };
   VkShaderModule vs, fs;
   CHECK(vkCreateShaderModule(device, &vsci, NULL, &vs));
   CHECK(vkCreateShaderModule(device, &fsci, NULL, &fs));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkViewport viewport = { 0, 0, W, H, 0.0f, 1.0f };
   VkRect2D scissor = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = compare, .maxDepthBounds = 1.0f };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(device, &plci, NULL, &layout));
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pDepthStencilState = &ds, .pColorBlendState = &cb,
      .layout = layout, .renderPass = pass, .subpass = 0 };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, NULL,
                                   &pipeline));

   VkBuffer cbuf, dbuf;
   VkDeviceMemory cbmem, dbmem;
   if (make_buffer((VkDeviceSize)W * H * 16, &cbuf, &cbmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 4, &dbuf, &dbmem)) return 1;

   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0 };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
   VkCommandBufferBeginInfo cbbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   CHECK(vkBeginCommandBuffer(cmd, &cbbi));

   VkClearValue clears[2];
   memset(clears, 0, sizeof clears);
   clears[1].depthStencil.depth = compare == VK_COMPARE_OP_LESS_OR_EQUAL ? 1.0f : 0.0f;
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass,
      .framebuffer = framebuffer, .renderArea = { { 0, 0 }, { W, H } },
      .clearValueCount = 2, .pClearValues = clears };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdDraw(cmd, 6, 1, 0, 0);
   vkCmdEndRenderPass(cmd);

   VkImageMemoryBarrier barriers[2] = {
      { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = cimage,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
      { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = dimage,
        .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 } },
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, barriers);
   VkBufferImageCopy ccopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, cimage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          cbuf, 1, &ccopy);
   VkBufferImageCopy dcopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, dimage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dbuf, 1, &dcopy);
   VkMemoryBarrier host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   void *mapped;
   CHECK(vkMapMemory(device, dbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(depth, mapped, (size_t)W * H * 4);
   vkUnmapMemory(device, dbmem);
   CHECK(vkMapMemory(device, cbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(colour, mapped, (size_t)W * H * 16);
   vkUnmapMemory(device, cbmem);

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, vs, NULL);
   vkDestroyShaderModule(device, fs, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, pass, NULL);
   vkDestroyImageView(device, cview, NULL);
   vkDestroyImageView(device, dview, NULL);
   vkDestroyImage(device, cimage, NULL);
   vkDestroyImage(device, dimage, NULL);
   vkFreeMemory(device, cmemory, NULL);
   vkFreeMemory(device, dmemory, NULL);
   vkDestroyBuffer(device, cbuf, NULL);
   vkDestroyBuffer(device, dbuf, NULL);
   vkFreeMemory(device, cbmem, NULL);
   vkFreeMemory(device, dbmem, NULL);
   return 0;
}

static int
check(VkCompareOp compare)
{
   float depth[W*H], colour[W*H*4];
   if (render(compare, depth, colour)) return 1;
   unsigned counts[3] = {0}, errors = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++) {
         int expected = order_expected(x,y,W,H);
         if (expected == -2) continue;
         counts[expected+1]++;
         unsigned i = y*W+x;
         float want[4] = {expected == 0, expected == 1, 0, expected >= 0};
         float z = expected >= 0 ? .5f :
                   compare == VK_COMPARE_OP_LESS_OR_EQUAL ? 1.0f : 0.0f;
         int bad = !isfinite(depth[i]) || fabsf(depth[i]-z) > 1e-5f;
         for (int c = 0; c < 4; c++)
            bad |= !isfinite(colour[4*i+c]) || fabsf(colour[4*i+c]-want[c]) > 1e-5f;
         if (bad && errors++ < 8)
            fprintf(stderr,"compare=%d pixel=%u,%u winner=%d rgba=%g,%g,%g,%g depth=%g\n",
                    compare,x,y,expected,colour[4*i],colour[4*i+1],
                    colour[4*i+2],colour[4*i+3],depth[i]);
      }
   printf("clip order Vulkan: compare=%d background=%u red=%u green=%u errors=%u\n",
          compare,counts[0],counts[1],counts[2],errors);
   return errors || !counts[0] || !counts[1] || !counts[2];
}

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_0 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci };
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   vkGetDeviceQueue(device, 0, 0, &queue);

   int failed = check(VK_COMPARE_OP_LESS_OR_EQUAL);
   failed |= check(VK_COMPARE_OP_GREATER_OR_EQUAL);

   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   puts(failed ? "clip order Vulkan: FAIL" : "clip order Vulkan: PASS");
   return failed;
}
