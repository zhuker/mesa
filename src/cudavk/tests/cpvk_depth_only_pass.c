/*
 * A render pass with a depth attachment and no colour attachment must still
 * write depth.
 *
 * This driver's fragment writeback -- cp_fs_writeback in kernels/cp_fs.cu --
 * is the only thing that ever writes cp->depthbuf: the rasterizer reads it to
 * test against and never advances it. The writeback was launched only when
 * there was a colour attachment to blend into, and the whole fragment stage
 * was skipped for a draw with nowhere to put a colour, so a depth-only pass
 * committed nothing. Every draw in such a pass tested against the clear value
 * and the attachment kept it.
 *
 * That is what a shadow map is: `~/favorite2.gfxr` renders a 2080x2080
 * VK_FORMAT_D16_UNORM shadow map in a pass whose framebuffer has one
 * attachment, the depth one, and on this driver the whole image came back
 * uniformly 65535 -- the loadOp clear -- while NVIDIA's had 107,639 texels of
 * real depth in it. See `.audit/favorite2_shading.md`.
 *
 * Four arms, because two things could have been to blame and only one was:
 *
 *   - D16 and D32_SFLOAT, so that this is not read as a D16 problem. It is
 *     not: both formats fail the same way and for the same reason.
 *   - with and without a colour attachment. The colour arms are the control
 *     and passed before this was fixed; the depth-only arms are the test.
 *
 * The geometry is one full-screen triangle at a constant window depth, so the
 * expected value is exact and every texel carries it: a depth-only pass that
 * did nothing leaves the 1.0 clear, and one that worked leaves 0.25. Nothing
 * here interpolates, so there is no tolerance to argue about beyond the
 * sixteen-bit quantisation of the constant.
 *
 * Dynamic rendering, optimal tiling, device-local images copied back through a
 * buffer, so it runs unchanged on lavapipe as the oracle and on NVIDIA.
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
 *   -- empty.frag --
 *   #version 450
 *   void main(){}
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

static const uint32_t frag_empty_spv[] = {
	0x07230203,0x00010000,0x0008000b,0x00000006,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0005000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00030010,0x00000004,0x00000007,
	0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00020013,
	0x00000002,0x00030021,0x00000003,0x00000002,0x00050036,0x00000002,0x00000004,0x00000000,
	0x00000003,0x000200f8,0x00000005,0x000100fd,0x00010038
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

/* one arm: depth format, whether a colour attachment exists */
static void arm(const char *name, VkFormat dfmt, int with_color, int cull_back)
{
   VkImage dimg, cimg = VK_NULL_HANDLE;
   VkDeviceMemory dmem, cmem = VK_NULL_HANDLE;
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
   VkImageView cview = VK_NULL_HANDLE;
   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   if (with_color) {
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
      CHECK(vkCreateImageView(dev,&vi2,NULL,&cview));
   }

   VkPipelineLayoutCreateInfo pli={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout; CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&layout));

   VkShaderModule vs = mod(vert_spv,sizeof(vert_spv));
   VkShaderModule fs = with_color ? mod(frag_red_spv,sizeof(frag_red_spv))
                                  : mod(frag_empty_spv,sizeof(frag_empty_spv));
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
      .polygonMode=VK_POLYGON_MODE_FILL,
      .cullMode= cull_back?VK_CULL_MODE_BACK_BIT:VK_CULL_MODE_NONE,
      .frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1.0f};
   VkPipelineMultisampleStateCreateInfo ms={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds={.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable=VK_TRUE,.depthWriteEnable=VK_TRUE,.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL};
   VkPipelineColorBlendAttachmentState cba={.colorWriteMask=0xf};
   VkPipelineColorBlendStateCreateInfo cb={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount=with_color?1u:0u,.pAttachments=with_color?&cba:NULL};
   VkPipelineRenderingCreateInfo prc={.sType=VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount=with_color?1u:0u,.pColorAttachmentFormats=with_color?&cfmt:NULL,
      .depthAttachmentFormat=dfmt};
   VkGraphicsPipelineCreateInfo gp={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.pNext=&prc,
      .stageCount=2,.pStages=st,.pVertexInputState=&vin,.pInputAssemblyState=&iaa,
      .pViewportState=&vps,.pRasterizationState=&rs,.pMultisampleState=&ms,
      .pDepthStencilState=&ds,.pColorBlendState=&cb,.layout=layout};
   VkPipeline pipe; CHECK(vkCreateGraphicsPipelines(dev,VK_NULL_HANDLE,1,&gp,NULL,&pipe));

   /* readback buffer */
   VkDeviceSize dsz = (VkDeviceSize)W*H*(dfmt==VK_FORMAT_D16_UNORM?2:4);
   VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=dsz,
      .usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buf; CHECK(vkCreateBuffer(dev,&bi,NULL,&buf));
   VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev,buf,&br);
   VkMemoryAllocateInfo bai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=br.size,
      .memoryTypeIndex=mem_type(br.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory bmem; CHECK(vkAllocateMemory(dev,&bai,NULL,&bmem));
   CHECK(vkBindBufferMemory(dev,buf,bmem,0));

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
   if (with_color) {
      VkImageMemoryBarrier b2=b; b2.image=cimg; b2.newLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      b2.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
      b2.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,0,0,NULL,0,NULL,1,&b2);
   }

   VkRenderingAttachmentInfo dat={.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView=dview,.imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue={.depthStencil={1.0f,0}}};
   VkRenderingAttachmentInfo cat={.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView=cview,.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue={.color={{0,0,0,1}}}};
   VkRenderingInfo ri={.sType=VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea={{0,0},{W,H}},.layerCount=1,
      .colorAttachmentCount=with_color?1u:0u,.pColorAttachments=with_color?&cat:NULL,
      .pDepthAttachment=&dat};
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
   VkBufferImageCopy region={.imageSubresource={VK_IMAGE_ASPECT_DEPTH_BIT,0,0,1},.imageExtent={W,H,1}};
   vkCmdCopyImageToBuffer(cmd,dimg,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buf,1,&region);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
   CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   void *map; CHECK(vkMapMemory(dev,bmem,0,VK_WHOLE_SIZE,0,&map));
   double got;
   if (dfmt==VK_FORMAT_D16_UNORM) {
      uint16_t *p=(uint16_t*)map;
      unsigned mn=65535,mx=0; for(int i=0;i<W*H;i++){ if(p[i]<mn)mn=p[i]; if(p[i]>mx)mx=p[i]; }
      printf("  %-26s D16   min=%u max=%u  centre=%u  (expect ~16384, clear 65535)\n",name,mn,mx,p[(H/2)*W+W/2]);
      got = p[(H/2)*W+W/2]/65535.0;
   } else {
      float *p=(float*)map;
      float mn=1e30f,mx=-1e30f; for(int i=0;i<W*H;i++){ if(p[i]<mn)mn=p[i]; if(p[i]>mx)mx=p[i]; }
      printf("  %-26s D32F  min=%.5f max=%.5f centre=%.5f (expect 0.25, clear 1.0)\n",name,mn,mx,p[(H/2)*W+W/2]);
      got = p[(H/2)*W+W/2];
   }
   /* The triangle covers the whole attachment at a constant window depth of
    * 0.25, so anything else means the pass committed no depth at all and the
    * loadOp clear of 1.0 is what is left. */
   int ok = got > 0.2 && got < 0.3;
   printf("      -> %s\n", ok ? "PASS: depth written" : "FAIL: depth not written");
   if (!ok)
      failures++;
   vkUnmapMemory(dev,bmem);
   vkDestroyCommandPool(dev,pool,NULL);
   vkDestroyBuffer(dev,buf,NULL); vkFreeMemory(dev,bmem,NULL);
   vkDestroyPipeline(dev,pipe,NULL); vkDestroyPipelineLayout(dev,layout,NULL);
   vkDestroyShaderModule(dev,vs,NULL); vkDestroyShaderModule(dev,fs,NULL);
   vkDestroyImageView(dev,dview,NULL); vkDestroyImage(dev,dimg,NULL); vkFreeMemory(dev,dmem,NULL);
   if (with_color){ vkDestroyImageView(dev,cview,NULL); vkDestroyImage(dev,cimg,NULL); vkFreeMemory(dev,cmem,NULL);}
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

   arm("D16 + colour",        VK_FORMAT_D16_UNORM,  1, 0);
   arm("D16 depth-only",      VK_FORMAT_D16_UNORM,  0, 0);
   arm("D32F + colour",       VK_FORMAT_D32_SFLOAT, 1, 0);
   arm("D32F depth-only",     VK_FORMAT_D32_SFLOAT, 0, 0);

   printf("%s: %d of 4 arms failed\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
