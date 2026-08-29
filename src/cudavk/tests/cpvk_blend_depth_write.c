/*
 * A blended draw that also writes depth must commit its fragment's depth.
 *
 * This driver renders a blended draw with the ordered-blending *peel* loop
 * (`cp_renderer.c`, `blend_peel`), which composites one layer per pass in
 * submission order. To make an atomicMin pick the lowest not-yet-composited
 * primitive, `emit_fragment` (`kernels/cp_rasterize.cu:754`) keys the
 * visibility buffer on the **primitive index** instead of on the depth:
 *
 *     atomicMin(&visbuf[at], PACK_VISBUF(tri_id, tri_id));
 *
 * `cp_fs_writeback` (`kernels/cp_fs.cu:1119`) is the only writer of
 * `cp->depthbuf`, and it committed depth by reading that same high word and
 * calling it the depth key. So a blended draw with `depthWriteEnable` stored
 * a *primitive index* into the depth buffer. The depth buffer holds a
 * sortable uint32 (`float_to_sortable_uint`), and a small integer decodes to
 * a negative denormal -- or, under a GREATER/GEQUAL depth function where the
 * key is stored inverted, to 0xFFFFxxxx, which is NaN as a float and the
 * nearest possible value as a sortable uint. Every later depth test at those
 * pixels then fails.
 *
 * That is what hides the ferris-wheel glass in `~/favorite2.gfxr`: draw
 * 762487 of frame 1800 is blended and writes depth, and the draw after it
 * (same geometry, VK_COMPARE_OP_GREATER_OR_EQUAL, depth write off) is
 * rejected at all 2,241 of its pixels. See `.audit/favorite2_group2.md`.
 *
 * Four arms, because two things could have been to blame and only one is:
 *
 *   - blend off and blend on. The blend-off arms are the control and pass
 *     before the fix; the blend-on arms are the test.
 *   - LESS_OR_EQUAL and GREATER_OR_EQUAL, because the key is stored inverted
 *     for the second family and the two produce different wrong values --
 *     a negative denormal and a NaN. Neither is a depth.
 *
 * The geometry is one full-screen triangle at a constant window depth of
 * 0.25, so the expected value is exact and every texel carries it. The
 * colour is checked too, so an arm cannot pass by drawing nothing.
 *
 * Dynamic rendering, optimal tiling, device-local images copied back through
 * a buffer, so it runs unchanged on lavapipe as the oracle and on NVIDIA.
 *
 * SPIR-V generated with glslangValidator from:
 *
 *   -- tri.vert --
 *   #version 450
 *   void main(){
 *     vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
 *     gl_Position = vec4(p * 2.0 - 1.0, 0.25, 1.0);
 *   }
 *
 *   -- red.frag --
 *   #version 450
 *   layout(location=0) out vec4 o;
 *   void main(){ o = vec4(1.0, 0.0, 0.0, 1.0); }
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64
#define CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){fprintf(stderr,"%s failed: %d\n",#x,_r); exit(1);} } while(0)

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static uint32_t family;
static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

static const uint32_t vert_spv[] = {
	0x07230203,0x00010000,0x0008000b,0x0000002b,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x0000001d,0x00030003,
	0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,
	0x00000070,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00060005,
	0x0000001b,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x0000001b,0x00000000,
	0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001b,0x00000001,0x505f6c67,0x746e696f,
	0x657a6953,0x00000000,0x00070006,0x0000001b,0x00000002,0x435f6c67,0x4470696c,0x61747369,
	0x0065636e,0x00070006,0x0000001b,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,
	0x00030005,0x0000001d,0x00000000,0x00040047,0x0000000c,0x0000000b,0x0000002a,0x00030047,
	0x0000001b,0x00000002,0x00050048,0x0000001b,0x00000000,0x0000000b,0x00000000,0x00050048,
	0x0000001b,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001b,0x00000002,0x0000000b,
	0x00000003,0x00050048,0x0000001b,0x00000003,0x0000000b,0x00000004,0x00020013,0x00000002,
	0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,
	0x00000006,0x00000002,0x00040020,0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,
	0x00000020,0x00000001,0x00040020,0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,
	0x0000000c,0x00000001,0x0004002b,0x0000000a,0x0000000e,0x00000001,0x0004002b,0x0000000a,
	0x00000010,0x00000002,0x00040017,0x00000017,0x00000006,0x00000004,0x00040015,0x00000018,
	0x00000020,0x00000000,0x0004002b,0x00000018,0x00000019,0x00000001,0x0004001c,0x0000001a,
	0x00000006,0x00000019,0x0006001e,0x0000001b,0x00000017,0x00000006,0x0000001a,0x0000001a,
	0x00040020,0x0000001c,0x00000003,0x0000001b,0x0004003b,0x0000001c,0x0000001d,0x00000003,
	0x0004002b,0x0000000a,0x0000001e,0x00000000,0x0004002b,0x00000006,0x00000020,0x40000000,
	0x0004002b,0x00000006,0x00000022,0x3f800000,0x0004002b,0x00000006,0x00000025,0x3e800000,
	0x00040020,0x00000029,0x00000003,0x00000017,0x00050036,0x00000002,0x00000004,0x00000000,
	0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,
	0x0000000a,0x0000000d,0x0000000c,0x000500c4,0x0000000a,0x0000000f,0x0000000d,0x0000000e,
	0x000500c7,0x0000000a,0x00000011,0x0000000f,0x00000010,0x0004006f,0x00000006,0x00000012,
	0x00000011,0x0004003d,0x0000000a,0x00000013,0x0000000c,0x000500c7,0x0000000a,0x00000014,
	0x00000013,0x00000010,0x0004006f,0x00000006,0x00000015,0x00000014,0x00050050,0x00000007,
	0x00000016,0x00000012,0x00000015,0x0003003e,0x00000009,0x00000016,0x0004003d,0x00000007,
	0x0000001f,0x00000009,0x0005008e,0x00000007,0x00000021,0x0000001f,0x00000020,0x00050050,
	0x00000007,0x00000023,0x00000022,0x00000022,0x00050083,0x00000007,0x00000024,0x00000021,
	0x00000023,0x00050051,0x00000006,0x00000026,0x00000024,0x00000000,0x00050051,0x00000006,
	0x00000027,0x00000024,0x00000001,0x00070050,0x00000017,0x00000028,0x00000026,0x00000027,
	0x00000025,0x00000022,0x00050041,0x00000029,0x0000002a,0x0000001d,0x0000001e,0x0003003e,
	0x0000002a,0x00000028,0x000100fd,0x00010038
};

static const uint32_t frag_red_spv[] = {
	0x07230203,0x00010000,0x0008000b,0x0000000d,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
	0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
	0x00030005,0x00000009,0x0000006f,0x00040047,0x00000009,0x0000001e,0x00000000,0x00020013,
	0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,
	0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,
	0x00000008,0x00000009,0x00000003,0x0004002b,0x00000006,0x0000000a,0x3f800000,0x0004002b,
	0x00000006,0x0000000b,0x00000000,0x0007002c,0x00000007,0x0000000c,0x0000000a,0x0000000b,
	0x0000000b,0x0000000a,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
	0x00000005,0x0003003e,0x00000009,0x0000000c,0x000100fd,0x00010038
};

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n"); exit(1);
}

static VkShaderModule mod(const uint32_t *code, size_t bytes)
{
   VkShaderModuleCreateInfo ci = { .sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize=bytes, .pCode=code };
   VkShaderModule m; CHECK(vkCreateShaderModule(dev,&ci,NULL,&m)); return m;
}

static int failures;

/* one arm: blending on or off, and which depth function */
static void arm(const char *name, int blend, VkCompareOp op)
{
   const VkFormat dfmt = VK_FORMAT_D32_SFLOAT;
   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   /* The clear has to be the far value for the function under test, or the
    * triangle never passes the depth test and the arm proves nothing. */
   const float clear_depth = (op == VK_COMPARE_OP_GREATER_OR_EQUAL) ? 0.0f : 1.0f;

   VkImage dimg, cimg;
   VkDeviceMemory dmem, cmem;
   VkImageCreateInfo ii = { .sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType=VK_IMAGE_TYPE_2D, .format=dfmt, .extent={W,H,1}, .mipLevels=1,
      .arrayLayers=1, .samples=VK_SAMPLE_COUNT_1_BIT, .tiling=VK_IMAGE_TILING_OPTIMAL,
      .usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED };
   CHECK(vkCreateImage(dev,&ii,NULL,&dimg));
   VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev,dimg,&mr);
   VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize=mr.size,.memoryTypeIndex=mem_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   CHECK(vkAllocateMemory(dev,&ai,NULL,&dmem)); CHECK(vkBindImageMemory(dev,dimg,dmem,0));
   VkImageViewCreateInfo vi={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=dimg,
      .viewType=VK_IMAGE_VIEW_TYPE_2D,.format=dfmt,
      .subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1}};
   VkImageView dview; CHECK(vkCreateImageView(dev,&vi,NULL,&dview));

   VkImageCreateInfo ci2 = ii;
   ci2.format = cfmt;
   ci2.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   CHECK(vkCreateImage(dev,&ci2,NULL,&cimg));
   vkGetImageMemoryRequirements(dev,cimg,&mr);
   VkMemoryAllocateInfo ai2={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize=mr.size,.memoryTypeIndex=mem_type(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   CHECK(vkAllocateMemory(dev,&ai2,NULL,&cmem)); CHECK(vkBindImageMemory(dev,cimg,cmem,0));
   VkImageViewCreateInfo vi2={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=cimg,
      .viewType=VK_IMAGE_VIEW_TYPE_2D,.format=cfmt,
      .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
   VkImageView cview; CHECK(vkCreateImageView(dev,&vi2,NULL,&cview));

   VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout; CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&layout));

   VkShaderModule vs = mod(vert_spv,sizeof(vert_spv));
   VkShaderModule fs = mod(frag_red_spv,sizeof(frag_red_spv));
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vs,.pName="main"},
      {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fs,.pName="main"}};
   VkPipelineVertexInputStateCreateInfo vin={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo iaa={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport vp={0,0,W,H,0,1}; VkRect2D sc={{0,0},{W,H}};
   VkPipelineViewportStateCreateInfo vps={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount=1,.pViewports=&vp,.scissorCount=1,.pScissors=&sc};
   VkPipelineRasterizationStateCreateInfo rs={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,
      .frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1.0f};
   VkPipelineMultisampleStateCreateInfo ms={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds={.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable=VK_TRUE,.depthWriteEnable=VK_TRUE,.depthCompareOp=op};
   /* Ordinary source-alpha blending. The fragment is opaque, so the blended
    * colour is the unblended one and the two families can be compared. */
   VkPipelineColorBlendAttachmentState cba={
      .blendEnable=blend?VK_TRUE:VK_FALSE,
      .srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp=VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp=VK_BLEND_OP_ADD,
      .colorWriteMask=0xf};
   VkPipelineColorBlendStateCreateInfo cb={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount=1,.pAttachments=&cba};
   VkPipelineRenderingCreateInfo prc={.sType=VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount=1,.pColorAttachmentFormats=&cfmt,
      .depthAttachmentFormat=dfmt};
   VkGraphicsPipelineCreateInfo gp={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.pNext=&prc,
      .stageCount=2,.pStages=st,.pVertexInputState=&vin,.pInputAssemblyState=&iaa,
      .pViewportState=&vps,.pRasterizationState=&rs,.pMultisampleState=&ms,
      .pDepthStencilState=&ds,.pColorBlendState=&cb,.layout=layout};
   VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gp,NULL,&pipe));

   VkDeviceSize dsz = (VkDeviceSize)W*H*4;
   VkBuffer dbuf, cbuf; VkDeviceMemory dbmem, cbmem;
   VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=dsz,
      .usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   CHECK(vkCreateBuffer(dev,&bi,NULL,&dbuf));
   CHECK(vkCreateBuffer(dev,&bi,NULL,&cbuf));
   VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev,dbuf,&br);
   VkMemoryAllocateInfo bai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=br.size,
      .memoryTypeIndex=mem_type(br.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CHECK(vkAllocateMemory(dev,&bai,NULL,&dbmem)); CHECK(vkBindBufferMemory(dev,dbuf,dbmem,0));
   CHECK(vkAllocateMemory(dev,&bai,NULL,&cbmem)); CHECK(vkBindBufferMemory(dev,cbuf,cbmem,0));

   VkCommandPoolCreateInfo pi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family};
   VkCommandPool pool; CHECK(vkCreateCommandPool(dev,&pi,NULL,&pool));
   VkCommandBufferAllocateInfo cai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
   VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev,&cai,&cmd));
   VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CHECK(vkBeginCommandBuffer(cmd,&cbi));

   VkImageMemoryBarrier b={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .image=dimg,.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1},
      .dstAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
   vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,0,0,NULL,0,NULL,1,&b);
   VkImageMemoryBarrier b2=b; b2.image=cimg; b2.newLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   b2.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
   b2.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,0,0,NULL,0,NULL,1,&b2);

   VkRenderingAttachmentInfo dat={.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView=dview,.imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue={.depthStencil={clear_depth,0}}};
   VkRenderingAttachmentInfo cat={.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView=cview,.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue={.color={{0,0,0,1}}}};
   VkRenderingInfo ri={.sType=VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea={{0,0},{W,H}},.layerCount=1,
      .colorAttachmentCount=1,.pColorAttachments=&cat,.pDepthAttachment=&dat};
   begin_rendering(cmd,&ri);
   vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
   vkCmdDraw(cmd,3,1,0,0);
   end_rendering(cmd);

   VkImageMemoryBarrier b3={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .image=dimg,.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1},
      .srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT};
   vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&b3);
   VkImageMemoryBarrier b4=b3; b4.image=cimg;
   b4.oldLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   b4.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
   b4.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&b4);

   VkBufferImageCopy dr={.imageSubresource={VK_IMAGE_ASPECT_DEPTH_BIT,0,0,1},.imageExtent={W,H,1}};
   vkCmdCopyImageToBuffer(cmd,dimg,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dbuf,1,&dr);
   VkBufferImageCopy cr={.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={W,H,1}};
   vkCmdCopyImageToBuffer(cmd,cimg,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,cbuf,1,&cr);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
   CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   void *dmap, *cmap;
   CHECK(vkMapMemory(dev,dbmem,0,VK_WHOLE_SIZE,0,&dmap));
   CHECK(vkMapMemory(dev,cbmem,0,VK_WHOLE_SIZE,0,&cmap));
   const float *dp=(const float*)dmap;
   const uint8_t *cp8=(const uint8_t*)cmap;
   int depth_ok = 1, nan = 0;
   float mn=1e30f, mx=-1e30f;
   for (int i=0;i<W*H;i++) {
      float v = dp[i];
      if (isnan(v)) { nan++; depth_ok = 0; continue; }
      if (v<mn) mn=v; if (v>mx) mx=v;
      if (!(v > 0.2f && v < 0.3f)) depth_ok = 0;
   }
   int red = 0;
   for (int i=0;i<W*H;i++)
      if (cp8[i*4+0]>200 && cp8[i*4+1]<50 && cp8[i*4+2]<50) red++;
   printf("  %-30s depth min=%.6g max=%.6g nan=%d  red=%d/%d  (expect 0.25 everywhere)\n",
          name, mn, mx, nan, red, W*H);
   /* A draw that never ran would leave the clear, which is also not 0.25, so
    * the colour count is what separates "wrong depth" from "no draw". */
   int ok = depth_ok && red == W*H;
   printf("      -> %s\n", ok ? "PASS" : (red != W*H ? "FAIL: the draw did not paint"
                                                     : "FAIL: depth is not the fragment's"));
   if (!ok)
      failures++;
   vkUnmapMemory(dev,dbmem); vkUnmapMemory(dev,cbmem);
   vkDestroyCommandPool(dev,pool,NULL);
   vkDestroyBuffer(dev,dbuf,NULL); vkFreeMemory(dev,dbmem,NULL);
   vkDestroyBuffer(dev,cbuf,NULL); vkFreeMemory(dev,cbmem,NULL);
   vkDestroyPipeline(dev,pipe,NULL); vkDestroyPipelineLayout(dev,layout,NULL);
   vkDestroyShaderModule(dev,vs,NULL); vkDestroyShaderModule(dev,fs,NULL);
   vkDestroyImageView(dev,dview,NULL); vkDestroyImage(dev,dimg,NULL); vkFreeMemory(dev,dmem,NULL);
   vkDestroyImageView(dev,cview,NULL); vkDestroyImage(dev,cimg,NULL); vkFreeMemory(dev,cmem,NULL);
}

int main(void)
{
   VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
   VkInstance inst; CHECK(vkCreateInstance(&ici,NULL,&inst));
   uint32_t n=1; CHECK(vkEnumeratePhysicalDevices(inst,&n,&pdev));
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pdev,&pp);
   printf("device: %s (api %u.%u.%u)\n",pp.deviceName,VK_VERSION_MAJOR(pp.apiVersion),
          VK_VERSION_MINOR(pp.apiVersion),VK_VERSION_PATCH(pp.apiVersion));
   uint32_t fc=0; vkGetPhysicalDeviceQueueFamilyProperties(pdev,&fc,NULL);
   VkQueueFamilyProperties *fp=calloc(fc,sizeof(*fp));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev,&fc,fp);
   family=UINT32_MAX;
   for(uint32_t i=0;i<fc;i++) if(fp[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){family=i;break;}
   free(fp);
   int core = pp.apiVersion >= VK_API_VERSION_1_3;
   const char *ext[1]={VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME};
   float prio=1.0f;
   VkDeviceQueueCreateInfo qi={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&prio};
   VkPhysicalDeviceDynamicRenderingFeatures dyn={
      .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,.dynamicRendering=VK_TRUE};
   VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&dyn,
      .queueCreateInfoCount=1,.pQueueCreateInfos=&qi,
      .enabledExtensionCount=core?0u:1u,.ppEnabledExtensionNames=core?NULL:ext};
   CHECK(vkCreateDevice(pdev,&dci,NULL,&dev));
   vkGetDeviceQueue(dev,family,0,&queue);
   begin_rendering=(PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev,core?"vkCmdBeginRendering":"vkCmdBeginRenderingKHR");
   end_rendering=(PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev,core?"vkCmdEndRendering":"vkCmdEndRenderingKHR");
   if(!begin_rendering||!end_rendering){fprintf(stderr,"no dynamic rendering\n");return 1;}

   arm("opaque  LESS_OR_EQUAL",    0, VK_COMPARE_OP_LESS_OR_EQUAL);
   arm("blended LESS_OR_EQUAL",    1, VK_COMPARE_OP_LESS_OR_EQUAL);
   arm("opaque  GREATER_OR_EQUAL", 0, VK_COMPARE_OP_GREATER_OR_EQUAL);
   arm("blended GREATER_OR_EQUAL", 1, VK_COMPARE_OP_GREATER_OR_EQUAL);

   printf("%s: %d of 4 arms failed\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
