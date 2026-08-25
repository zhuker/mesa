/*
 * cpvk_interop_render -- the Vulkan producer half of the CUDA/PyTorch interop
 * sample.  See PROTOCOL.md in this directory for the wire contract, and the
 * interop design document in this repository's docs for why it is shaped
 * this way.
 *
 * This program never links or calls libcuda.  It is pure Vulkan and libc.  One
 * integer -- a file descriptor -- crosses the process boundary, and everything
 * CUDA-shaped lives on the far side of it.
 *
 * It is one binary for every driver that implements the contract it uses.
 * Nothing in here branches on which driver answered: no vendor ID test, no
 * device name test, no driver ID test, no ICD path.  It asks Vulkan what is
 * supported, reports what it found, and runs or declines on the answer.
 * Driver selection belongs to whoever sets the loader's environment.
 *
 * Build: gcc -O2 -o cpvk_interop_render cpvk_interop_render.c -lvulkan -lz -lm
 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#include <vulkan/vulkan.h>

/* ------------------------------------------------------------------ */
/* Exit codes (PROTOCOL.md, "Exit codes")                             */

#define EXIT_PASS        0
#define EXIT_VERIFY      1
#define EXIT_PROTO       2
#define EXIT_UNSUPPORTED 3

/* ------------------------------------------------------------------ */
/* The wire protocol.  All little-endian, all packed, all fixed size.  */

#define CPIO_MAGIC 0x4F495043u   /* 'CPIO' */

enum cpio_type {
   MSG_OFFER   = 1,
   MSG_REQUEST = 2,
   MSG_SLOT    = 3,
   MSG_SEMS    = 4,
   MSG_HELLO   = 5,
   MSG_READY   = 6,
   MSG_RESULT  = 7,
   MSG_BYE     = 8,
   MSG_ERROR   = 9,
};

#define SYNC_HOST     (1u << 0)
#define SYNC_TIMELINE (1u << 1)

#pragma pack(push, 1)

struct msg_offer {
   uint32_t magic, type;
   uint32_t version;
   uint32_t slots;
   uint32_t width, height;
   uint32_t frame_format;
   uint32_t frame_bytes;
   uint32_t sync_modes;
   uint8_t  device_uuid[16];
   uint32_t scene;
   uint32_t pad;
};

struct msg_request {
   uint32_t magic, type;
   uint64_t result_bytes;
   uint32_t result_dtype;
   uint32_t result_ndim;
   uint32_t result_shape[4];
   uint32_t sync_mode;
   uint32_t pad;
};

struct msg_slot {
   uint32_t magic, type;
   uint32_t slot;
   uint32_t pad;
   uint64_t alloc_size;
   uint64_t frame_offset, frame_bytes;
   uint64_t result_offset, result_bytes;
};

struct msg_sems {
   uint32_t magic, type;
   uint32_t pad0, pad1;
};

struct msg_hello {
   uint32_t magic, type;
   uint32_t status;
   uint32_t pad;
};

struct msg_ready {
   uint32_t magic, type;
   uint32_t slot, seq;
   uint64_t timeline_value;
};

struct msg_result {
   uint32_t magic, type;
   uint32_t slot, seq;
   uint32_t status;
   uint32_t result_ndim;
   uint32_t result_shape[4];
   uint64_t result_bytes;
   uint64_t timeline_value;
};

struct msg_bye {
   uint32_t magic, type;
   uint32_t frames, status;
};

struct msg_error {
   uint32_t magic, type;
   uint32_t code, text_len;
   char     text[256];
};

union any_msg {
   struct { uint32_t magic, type; } hdr;
   struct msg_offer   offer;
   struct msg_request request;
   struct msg_slot    slot;
   struct msg_sems    sems;
   struct msg_hello   hello;
   struct msg_ready   ready;
   struct msg_result  result;
   struct msg_bye     bye;
   struct msg_error   error;
};

#pragma pack(pop)

_Static_assert(sizeof(struct msg_offer)   == 60, "offer layout");
_Static_assert(sizeof(struct msg_request) == 48, "request layout");
_Static_assert(sizeof(struct msg_slot)    == 56, "slot layout");
_Static_assert(sizeof(struct msg_sems)    == 16, "sems layout");
_Static_assert(sizeof(struct msg_hello)   == 16, "hello layout");
_Static_assert(sizeof(struct msg_ready)   == 24, "ready layout");
_Static_assert(sizeof(struct msg_result)  == 56, "result layout");
_Static_assert(sizeof(struct msg_bye)     == 16, "bye layout");
_Static_assert(sizeof(struct msg_error)   == 272, "error layout");

/* ------------------------------------------------------------------ */
/* Options                                                            */

struct options {
   unsigned frames;
   unsigned slots;
   unsigned width, height;
   unsigned sync;             /* SYNC_HOST or SYNC_TIMELINE */
   const char *op;
   const char *exec_cmd;
   const char *socket_path;
   unsigned png_every;
   const char *png_dir;
   bool no_producer_wait;
   unsigned seed;
   bool self_test;
   int  fake_consumer_fd;     /* >= 0 in fake-consumer mode */
   bool fake_no_write;        /* fake consumer: never write the result */
   bool fake_overrun;         /* fake consumer: write past its own declaration */
   const char *argv0;
};

/* ------------------------------------------------------------------ */
/* State line, timing, logging                                        */

static char g_state[256] = "starting";
static const char *g_who = "renderer";

static void
set_state(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(g_state, sizeof(g_state), fmt, ap);
   va_end(ap);
}

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void
logf_(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fprintf(stdout, "%s: ", g_who);
   vfprintf(stdout, fmt, ap);
   fputc('\n', stdout);
   va_end(ap);
   fflush(stdout);
}

static pid_t g_child_pid = -1;

static void
kill_child(void)
{
   if (g_child_pid > 0) {
      /* One process group, so this reaches grandchildren too.  Ignore the
       * signal first: we are in that group as well. */
      signal(SIGTERM, SIG_IGN);
      kill(-getpgrp(), SIGTERM);
      g_child_pid = -1;
   }
}

/* Every fatal path goes through here so that the state line is always
 * printed and the peer is never left running. */
static void __attribute__((noreturn))
fatal(int code, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fprintf(stdout, "%s: FATAL: ", g_who);
   vfprintf(stdout, fmt, ap);
   fputc('\n', stdout);
   va_end(ap);
   fprintf(stdout, "%s: state was: %s\n", g_who, g_state);
   fflush(stdout);
   kill_child();
   _exit(code);
}

#define VK_CHECK(x) do {                                                  \
   VkResult _r = (x);                                                     \
   if (_r != VK_SUCCESS)                                                  \
      fatal(EXIT_PROTO, "%s failed: VkResult %d (%s:%d)",                 \
            #x, (int)_r, __FILE__, __LINE__);                             \
} while (0)

/* ------------------------------------------------------------------ */
/* The pattern.  PROTOCOL.md, "The frame".  This is the C copy of the  */
/* function that also exists in interop.frag and in the consumer's     */
/* Python.  Additions only, no subtraction, so there is no sign        */
/* question in any of the three.                                      */

static inline void
pattern_pixel(uint32_t x, uint32_t y, uint32_t seq, uint8_t px[4])
{
   px[0] = (uint8_t)((x + seq) & 255u);
   px[1] = (uint8_t)((y + 2u * seq) & 255u);
   px[2] = (uint8_t)((x ^ y) & 255u);
   px[3] = 255u;
}

/* The stamp: (0,0) = seq, (1,0) = ~seq, u32 little-endian. */
static inline void
stamp_bytes(uint32_t seq, uint8_t out[8])
{
   uint32_t a = seq, b = ~seq;
   out[0] = a & 255u; out[1] = (a >> 8) & 255u;
   out[2] = (a >> 16) & 255u; out[3] = (a >> 24) & 255u;
   out[4] = b & 255u; out[5] = (b >> 8) & 255u;
   out[6] = (b >> 16) & 255u; out[7] = (b >> 24) & 255u;
}

/* The whole frame as the shader must have written it. */
static void
pattern_frame(uint8_t *dst, uint32_t w, uint32_t h, uint32_t seq)
{
   for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
         pattern_pixel(x, y, seq, dst + ((size_t)y * w + x) * 4);
   stamp_bytes(seq, dst);
}

/* check mode: result[i] = (frame[i] + 1) & 255, except the first 8 bytes,
 * which the consumer overwrites with the same stamp shape. */
static void
expected_result(uint8_t *dst, uint32_t w, uint32_t h, uint32_t seq)
{
   pattern_frame(dst, w, h, seq);
   size_t n = (size_t)w * h * 4;
   for (size_t i = 8; i < n; i++)
      dst[i] = (uint8_t)((dst[i] + 1) & 255u);
   stamp_bytes(seq, dst);
}

/* ------------------------------------------------------------------ */
/* SOCK_SEQPACKET messaging.  One sendmsg is one message; there is no  */
/* framing and a short read is a protocol error, not a partial read.   */

static int
send_msg(int fd, const void *buf, size_t len, const int *fds, int nfds)
{
   struct iovec iov = { .iov_base = (void *)buf, .iov_len = len };
   struct msghdr mh = { .msg_iov = &iov, .msg_iovlen = 1 };
   union {
      char buf[CMSG_SPACE(sizeof(int) * 8)];
      struct cmsghdr align;
   } cmsgu;

   if (nfds > 0) {
      memset(&cmsgu, 0, sizeof(cmsgu));
      mh.msg_control = cmsgu.buf;
      mh.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
      struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
      c->cmsg_level = SOL_SOCKET;
      c->cmsg_type = SCM_RIGHTS;
      c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
      memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
   }

   ssize_t n;
   do {
      n = sendmsg(fd, &mh, MSG_NOSIGNAL);
   } while (n < 0 && errno == EINTR);

   if (n < 0)
      return -errno;
   if ((size_t)n != len)
      return -EPROTO;
   return 0;
}

/* Wait for the socket to be readable, or the deadline to pass.
 * Returns 1 readable, 0 timeout, -1 hangup/error. */
static int
wait_readable(int fd, double deadline_ms)
{
   for (;;) {
      double left = deadline_ms - now_ms();
      if (left <= 0)
         return 0;
      struct pollfd p = { .fd = fd, .events = POLLIN };
      int r = poll(&p, 1, (int)(left < 1 ? 1 : left));
      if (r < 0) {
         if (errno == EINTR)
            continue;
         return -1;
      }
      if (r == 0)
         return 0;
      if (p.revents & POLLIN)
         return 1;
      if (p.revents & (POLLHUP | POLLERR | POLLNVAL))
         return -1;
   }
}

/* Receive exactly one message.  Returns the byte count, 0 on EOF,
 * -1 on error.  Any file descriptors arrive in fds[]. */
static ssize_t
recv_msg(int fd, void *buf, size_t cap, int *fds, int *nfds_out)
{
   struct iovec iov = { .iov_base = buf, .iov_len = cap };
   struct msghdr mh = { .msg_iov = &iov, .msg_iovlen = 1 };
   union {
      char buf[CMSG_SPACE(sizeof(int) * 8)];
      struct cmsghdr align;
   } cmsgu;
   memset(&cmsgu, 0, sizeof(cmsgu));
   mh.msg_control = cmsgu.buf;
   mh.msg_controllen = sizeof(cmsgu.buf);

   ssize_t n;
   do {
      n = recvmsg(fd, &mh, MSG_CMSG_CLOEXEC);
   } while (n < 0 && errno == EINTR);

   if (nfds_out)
      *nfds_out = 0;
   if (n <= 0)
      return n;

   if (mh.msg_flags & MSG_TRUNC)
      return -1;

   for (struct cmsghdr *c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c)) {
      if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
         int cnt = (int)((c->cmsg_len - CMSG_LEN(0)) / sizeof(int));
         if (fds && nfds_out) {
            memcpy(fds, CMSG_DATA(c), sizeof(int) * cnt);
            *nfds_out = cnt;
         }
      }
   }
   return n;
}

static void
send_error(int fd, uint32_t code, const char *text)
{
   struct msg_error e;
   memset(&e, 0, sizeof(e));
   e.magic = CPIO_MAGIC;
   e.type = MSG_ERROR;
   e.code = code;
   snprintf(e.text, sizeof(e.text), "%s", text);
   e.text_len = (uint32_t)strlen(e.text);
   (void)send_msg(fd, &e, sizeof(e), NULL, 0);
}

/* Receive a message of an expected type, with a deadline.  Fatal on
 * anything else: a wrong magic or type is fatal on both sides, and so is
 * a dead peer. */
static ssize_t
expect_msg(int fd, uint32_t want_type, union any_msg *m, size_t want_size,
           double timeout_ms, int *fds, int *nfds)
{
   int r = wait_readable(fd, now_ms() + timeout_ms);
   if (r == 0)
      fatal(EXIT_PROTO, "timeout after %.0f ms waiting for message type %u",
            timeout_ms, want_type);
   if (r < 0)
      fatal(EXIT_PROTO, "peer hung up while waiting for message type %u",
            want_type);

   ssize_t n = recv_msg(fd, m, sizeof(*m), fds, nfds);
   if (n == 0)
      fatal(EXIT_PROTO, "EOF from peer while waiting for message type %u",
            want_type);
   if (n < 0)
      fatal(EXIT_PROTO, "recvmsg: %s", strerror(errno));

   if (m->hdr.magic != CPIO_MAGIC)
      fatal(EXIT_PROTO, "bad magic 0x%08x from peer", m->hdr.magic);

   if (m->hdr.type == MSG_ERROR && want_type != MSG_ERROR) {
      m->error.text[sizeof(m->error.text) - 1] = 0;
      fatal(EXIT_PROTO, "peer sent MSG_ERROR code %u: %s",
            m->error.code, m->error.text);
   }
   if (m->hdr.type != want_type)
      fatal(EXIT_PROTO, "expected message type %u, got %u",
            want_type, m->hdr.type);
   if ((size_t)n != want_size)
      fatal(EXIT_PROTO, "message type %u is %zd bytes, expected %zu",
            want_type, n, want_size);
   return n;
}

/* ------------------------------------------------------------------ */
/* PNG.  Adapted from the offscreen benchmark in this repository.      */

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

static void
write_png(const char *path, const uint8_t *rgba, unsigned w, unsigned h)
{
   size_t stride = (size_t)w * 4 + 1;
   size_t raw_size = (size_t)h * stride;
   uint8_t *raw = malloc(raw_size);
   if (!raw)
      return;
   for (unsigned y = 0; y < h; y++) {
      raw[y * stride] = 0;  /* filter type: none */
      memcpy(raw + y * stride + 1, rgba + (size_t)y * w * 4, (size_t)w * 4);
   }

   uLongf comp_size = compressBound(raw_size);
   uint8_t *comp = malloc(comp_size);
   if (!comp) {
      free(raw);
      return;
   }
   compress2(comp, &comp_size, raw, raw_size, 6);
   free(raw);

   FILE *f = fopen(path, "wb");
   if (!f) {
      fprintf(stdout, "%s: cannot write %s: %s\n", g_who, path, strerror(errno));
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

/* ------------------------------------------------------------------ */
/* Vulkan context.  API version 1.1 core plus the few extensions this  */
/* sample genuinely needs, each checked for before it is enabled, so   */
/* that the same binary runs on any driver that implements them.       */

struct vk {
   VkInstance inst;
   VkPhysicalDevice pdev;
   VkDevice dev;
   VkQueue queue;
   uint32_t qfam;

   VkPhysicalDeviceProperties props;
   uint8_t uuid[16];
   char driver_name[VK_MAX_DRIVER_NAME_SIZE];
   char driver_info[VK_MAX_DRIVER_INFO_SIZE];
   uint32_t driver_id;

   bool has_ext_mem_fd;
   bool has_ext_sem_fd;
   bool has_timeline;
   bool has_driver_props;
   bool timeline_ok;          /* both timeline extensions are present */

   VkCommandPool pool;

   PFN_vkGetMemoryFdKHR             GetMemoryFdKHR;
   PFN_vkGetMemoryFdPropertiesKHR   GetMemoryFdPropertiesKHR;
   PFN_vkGetSemaphoreFdKHR          GetSemaphoreFdKHR;
   PFN_vkImportSemaphoreFdKHR       ImportSemaphoreFdKHR;
   PFN_vkWaitSemaphoresKHR          WaitSemaphoresKHR;
   PFN_vkSignalSemaphoreKHR         SignalSemaphoreKHR;
   PFN_vkGetSemaphoreCounterValueKHR GetSemaphoreCounterValueKHR;
};

static bool
have_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
   for (uint32_t i = 0; i < n; i++)
      if (!strcmp(e[i].extensionName, name))
         return true;
   return false;
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
   return UINT32_MAX;
}

/* Does this physical device export a VkBuffer as an OPAQUE_FD?  This is the
 * question the whole sample rests on, and it is asked before anything is
 * allocated.  A driver that answers no is declined, not worked around. */
static bool
buffer_export_supported(VkPhysicalDevice pdev, VkBufferUsageFlags usage,
                        VkExternalMemoryFeatureFlags *feat_out)
{
   VkPhysicalDeviceExternalBufferInfo info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
      .usage = usage,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };
   VkExternalBufferProperties props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
   };
   vkGetPhysicalDeviceExternalBufferProperties(pdev, &info, &props);
   if (feat_out)
      *feat_out = props.externalMemoryProperties.externalMemoryFeatures;
   return (props.externalMemoryProperties.externalMemoryFeatures &
           VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
}

#define SLOT_BUF_USAGE (VK_BUFFER_USAGE_TRANSFER_DST_BIT | \
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT)

static void
vk_init(struct vk *vk, const uint8_t *want_uuid, bool want_timeline,
        bool check_export)
{
   memset(vk, 0, sizeof(*vk));

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "cpvk_interop_render",
      .apiVersion = VK_API_VERSION_1_1,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkResult r = vkCreateInstance(&ici, NULL, &vk->inst);
   if (r == VK_ERROR_INCOMPATIBLE_DRIVER) {
      /* No ICD at all: VK_DRIVER_FILES names a file that is missing or
       * unusable.  That is the same verdict as a driver that cannot
       * export -- unsupported, not a protocol failure. */
      printf("%s: the loader found no usable Vulkan driver: vkCreateInstance "
             "returned VK_ERROR_INCOMPATIBLE_DRIVER\n", g_who);
      printf("%s: driver selection is the environment's business; "
             "VK_DRIVER_FILES is currently %s\n", g_who,
             getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "unset");
      fflush(stdout);
      kill_child();
      _exit(EXIT_UNSUPPORTED);
   }
   if (r != VK_SUCCESS)
      fatal(EXIT_PROTO, "vkCreateInstance failed: %d (VK_DRIVER_FILES=%s)",
            (int)r, getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "(unset)");

   uint32_t ndev = 0;
   VK_CHECK(vkEnumeratePhysicalDevices(vk->inst, &ndev, NULL));
   if (!ndev)
      fatal(EXIT_UNSUPPORTED, "no Vulkan physical device (VK_DRIVER_FILES=%s)",
            getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "(unset)");
   VkPhysicalDevice *devs = calloc(ndev, sizeof(*devs));
   VK_CHECK(vkEnumeratePhysicalDevices(vk->inst, &ndev, devs));

   int chosen = -1;
   for (uint32_t i = 0; i < ndev; i++) {
      VkPhysicalDeviceIDProperties idp = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
      };
      VkPhysicalDeviceProperties2 p2 = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &idp,
      };
      vkGetPhysicalDeviceProperties2(devs[i], &p2);
      if (want_uuid) {
         if (!memcmp(idp.deviceUUID, want_uuid, 16)) {
            chosen = (int)i;
            break;
         }
         continue;
      }
      if (chosen < 0)
         chosen = (int)i;
      if (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
         chosen = (int)i;
         break;
      }
   }
   if (chosen < 0)
      fatal(EXIT_PROTO, "no physical device matched the offered device UUID");
   vk->pdev = devs[chosen];
   free(devs);

   uint32_t next = 0;
   vkEnumerateDeviceExtensionProperties(vk->pdev, NULL, &next, NULL);
   VkExtensionProperties *exts = calloc(next ? next : 1, sizeof(*exts));
   vkEnumerateDeviceExtensionProperties(vk->pdev, NULL, &next, exts);

   vk->has_ext_mem_fd    = have_ext(exts, next, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
   vk->has_ext_sem_fd    = have_ext(exts, next, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
   vk->has_timeline      = have_ext(exts, next, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
   vk->has_driver_props  = have_ext(exts, next, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);

   VkPhysicalDeviceIDProperties idp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
   };
   VkPhysicalDeviceDriverPropertiesKHR dp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
   };
   if (vk->has_driver_props)
      idp.pNext = &dp;
   VkPhysicalDeviceProperties2 p2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &idp,
   };
   vkGetPhysicalDeviceProperties2(vk->pdev, &p2);
   vk->props = p2.properties;
   memcpy(vk->uuid, idp.deviceUUID, 16);
   if (vk->has_driver_props) {
      snprintf(vk->driver_name, sizeof(vk->driver_name), "%s", dp.driverName);
      snprintf(vk->driver_info, sizeof(vk->driver_info), "%s", dp.driverInfo);
      vk->driver_id = dp.driverID;
   } else {
      snprintf(vk->driver_name, sizeof(vk->driver_name), "%s", vk->props.deviceName);
      snprintf(vk->driver_info, sizeof(vk->driver_info), "(no VK_KHR_driver_properties)");
   }

   /* Ask the export question before allocating anything.  This is the one
    * capability the sample cannot do without, and asking it is what any
    * portable application does in place of knowing the driver. */
   if (check_export) {
      VkExternalMemoryFeatureFlags feat = 0;
      if (!vk->has_ext_mem_fd || !buffer_export_supported(vk->pdev, SLOT_BUF_USAGE, &feat)) {
         printf("%s: this driver does not report OPAQUE_FD as exportable for "
                "this buffer usage\n", g_who);
         printf("%s:   VK_KHR_external_memory_fd : %s\n", g_who,
                vk->has_ext_mem_fd ? "present" : "absent");
         printf("%s:   externalMemoryFeatures for OPAQUE_FD : 0x%x "
                "(EXPORTABLE bit 0x%x is not set)\n",
                g_who, feat, VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
         printf("%s:   buffer usage asked about : 0x%x\n", g_who,
                (unsigned)SLOT_BUF_USAGE);
         fflush(stdout);
         kill_child();
         _exit(EXIT_UNSUPPORTED);
      }

      /* The frame format has to work as a colour attachment and as a copy
       * source.  Ask; do not assume. */
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(vk->pdev, VK_FORMAT_R8G8B8A8_UINT, &fp);
      VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                  VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
      if ((fp.optimalTilingFeatures & need) != need) {
         printf("%s: this driver does not support VK_FORMAT_R8G8B8A8_UINT as a "
                "colour attachment and transfer source in optimal tiling\n", g_who);
         printf("%s:   optimalTilingFeatures : 0x%x, needed 0x%x\n",
                g_who, fp.optimalTilingFeatures, need);
         fflush(stdout);
         kill_child();
         _exit(EXIT_UNSUPPORTED);
      }
   }

   /* Timeline mode is optional.  If either extension is missing, say so and
    * let the caller run the host handshake instead: PROTOCOL.md has host mode
    * as the oracle, and it needs nothing beyond 1.1 core. */
   vk->timeline_ok = vk->has_ext_sem_fd && vk->has_timeline;
   if (want_timeline && !vk->timeline_ok) {
      logf_("timeline mode is unavailable here: VK_KHR_external_semaphore_fd "
            "%s, VK_KHR_timeline_semaphore %s; running host mode",
            vk->has_ext_sem_fd ? "present" : "absent",
            vk->has_timeline ? "present" : "absent");
      want_timeline = false;
   }

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(vk->pdev, &nq, NULL);
   VkQueueFamilyProperties *qs = calloc(nq, sizeof(*qs));
   vkGetPhysicalDeviceQueueFamilyProperties(vk->pdev, &nq, qs);
   /* Any graphics family supports transfer operations, whether or not it
    * advertises VK_QUEUE_TRANSFER_BIT, so graphics is the whole test. */
   vk->qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++) {
      if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         vk->qfam = i;
         break;
      }
   }
   free(qs);
   if (vk->qfam == UINT32_MAX)
      fatal(EXIT_UNSUPPORTED, "no queue family reports VK_QUEUE_GRAPHICS_BIT");

   const char *dev_exts[8];
   uint32_t ndev_exts = 0;
   if (vk->has_ext_mem_fd)
      dev_exts[ndev_exts++] = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
   if (want_timeline) {
      dev_exts[ndev_exts++] = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
      dev_exts[ndev_exts++] = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
   }
   free(exts);

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = vk->qfam,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceTimelineSemaphoreFeaturesKHR tsf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
      .timelineSemaphore = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = want_timeline ? (void *)&tsf : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = ndev_exts,
      .ppEnabledExtensionNames = dev_exts,
   };
   VK_CHECK(vkCreateDevice(vk->pdev, &dci, NULL, &vk->dev));
   vkGetDeviceQueue(vk->dev, vk->qfam, 0, &vk->queue);

#define GET(name) vk->name = (PFN_vk##name)vkGetDeviceProcAddr(vk->dev, "vk" #name)
   GET(GetMemoryFdKHR);
   GET(GetMemoryFdPropertiesKHR);
   if (want_timeline) {
      GET(GetSemaphoreFdKHR);
      GET(ImportSemaphoreFdKHR);
      GET(WaitSemaphoresKHR);
      GET(SignalSemaphoreKHR);
      GET(GetSemaphoreCounterValueKHR);
   }
#undef GET
   if (vk->has_ext_mem_fd && !vk->GetMemoryFdKHR)
      fatal(EXIT_UNSUPPORTED, "vkGetMemoryFdKHR missing on %s", vk->driver_name);

   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = vk->qfam,
   };
   VK_CHECK(vkCreateCommandPool(vk->dev, &pci, NULL, &vk->pool));
}

static VkCommandBuffer
alloc_cmd(struct vk *vk)
{
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = vk->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   VK_CHECK(vkAllocateCommandBuffers(vk->dev, &ai, &cb));
   return cb;
}

static VkFence
alloc_fence(struct vk *vk, bool signaled)
{
   VkFenceCreateInfo fi = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0,
   };
   VkFence f;
   VK_CHECK(vkCreateFence(vk->dev, &fi, NULL, &f));
   return f;
}

static void
wait_fence(struct vk *vk, VkFence f, const char *what)
{
   VkResult r = vkWaitForFences(vk->dev, 1, &f, VK_TRUE, 5ull * 1000 * 1000 * 1000);
   if (r == VK_TIMEOUT)
      fatal(EXIT_PROTO, "GPU fence timed out after 5 s (%s)", what);
   VK_CHECK(r);
   VK_CHECK(vkResetFences(vk->dev, 1, &f));
}

/* A plain host-visible buffer, not exported: staging on both sides. */
struct hostbuf {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *map;
   VkDeviceSize size;
};

static void
hostbuf_create(struct vk *vk, struct hostbuf *hb, VkDeviceSize size,
               VkBufferUsageFlags usage)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_CHECK(vkCreateBuffer(vk->dev, &bci, NULL, &hb->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(vk->dev, hb->buf, &req);
   /* HOST_CACHED first.  The plain HOST_VISIBLE|HOST_COHERENT type on this
    * driver is write-combined: reading a megabyte back through it costs
    * tens of milliseconds, and verification reads every byte. */
   uint32_t type = find_memory_type(vk->pdev, req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
   if (type == UINT32_MAX)
      type = find_memory_type(vk->pdev, req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX)
      fatal(EXIT_PROTO, "no HOST_VISIBLE|HOST_COHERENT memory type for staging");
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = type,
   };
   VK_CHECK(vkAllocateMemory(vk->dev, &mai, NULL, &hb->mem));
   VK_CHECK(vkBindBufferMemory(vk->dev, hb->buf, hb->mem, 0));
   VK_CHECK(vkMapMemory(vk->dev, hb->mem, 0, VK_WHOLE_SIZE, 0, &hb->map));
   hb->size = size;
}

/* Ownership transfer to and from VK_QUEUE_FAMILY_EXTERNAL.  The buffer is
 * handed to an external consumer and handed back, and Vulkan requires the
 * release and the acquire to be written even where a given implementation
 * happens to need nothing for them. */
static void
buffer_barrier(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize offset,
               VkDeviceSize size, VkAccessFlags src_access,
               VkAccessFlags dst_access, uint32_t src_family,
               uint32_t dst_family, VkPipelineStageFlags src_stage,
               VkPipelineStageFlags dst_stage)
{
   VkBufferMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .srcQueueFamilyIndex = src_family,
      .dstQueueFamilyIndex = dst_family,
      .buffer = buf,
      .offset = offset,
      .size = size,
   };
   vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 1, &b, 0, NULL);
}

/* ------------------------------------------------------------------ */
/* The producer                                                       */

struct slot {
   VkDeviceMemory mem;
   VkBuffer frame_buf, result_buf;
   VkDeviceSize alloc_size;
   VkDeviceSize frame_offset, frame_bytes;
   VkDeviceSize result_offset, result_bytes;

   VkImage image;
   VkImageView view;
   VkDeviceMemory image_mem;
   VkFramebuffer fb;

   VkCommandBuffer cmd_render, cmd_verify;
   VkFence fence_render, fence_verify;

   bool busy;                 /* rendered, MSG_RESULT not yet seen */
   bool owned_external;       /* released to VK_QUEUE_FAMILY_EXTERNAL */
   bool render_pending;       /* fence_render not yet waited */
   uint32_t seq;
};

struct producer {
   struct vk vk;
   struct options *opt;
   int sock;

   uint32_t frame_bytes;
   uint64_t result_bytes;
   uint32_t result_ndim;
   uint32_t result_shape[4];
   uint32_t result_dtype;

   struct slot *slots;

   VkRenderPass rp;
   VkPipelineLayout layout;
   VkPipeline pipe;
   VkShaderModule vs, fs;

   struct hostbuf stage_result, stage_frame;

   VkSemaphore sem_render, sem_consume;

   /* running totals for the summary line */
   uint64_t total_bytes, total_bad, total_poison, total_stamp_err, total_tail_bad;
   unsigned frames_failed;
};

static uint8_t *
read_file(const char *path, size_t *len_out)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      return NULL;
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint8_t *buf = malloc(n > 0 ? (size_t)n : 1);
   if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
      free(buf);
      fclose(f);
      return NULL;
   }
   fclose(f);
   *len_out = (size_t)n;
   return buf;
}

/* The .spv files sit next to the binary; the Makefile builds them there. */
static VkShaderModule
load_shader(struct producer *p, const char *name)
{
   char path[1024];
   const char *dirs[3];
   char argv0dir[512] = ".";
   const char *env = getenv("CPVK_INTEROP_SHADER_DIR");
   if (p->opt->argv0) {
      snprintf(argv0dir, sizeof(argv0dir), "%s", p->opt->argv0);
      char *slash = strrchr(argv0dir, '/');
      if (slash)
         *slash = 0;
      else
         snprintf(argv0dir, sizeof(argv0dir), ".");
   }
   dirs[0] = env;
   dirs[1] = argv0dir;
   dirs[2] = ".";

   uint8_t *code = NULL;
   size_t len = 0;
   for (int i = 0; i < 3 && !code; i++) {
      if (!dirs[i])
         continue;
      snprintf(path, sizeof(path), "%s/%s", dirs[i], name);
      code = read_file(path, &len);
   }
   if (!code)
      fatal(EXIT_PROTO, "cannot find %s (looked in $CPVK_INTEROP_SHADER_DIR, "
            "%s and .); run make", name, argv0dir);

   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = len,
      .pCode = (const uint32_t *)code,
   };
   VkShaderModule m;
   VK_CHECK(vkCreateShaderModule(p->vk.dev, &ci, NULL, &m));
   free(code);
   return m;
}

static void
build_pipeline(struct producer *p)
{
   VkAttachmentDescription att = {
      .format = VK_FORMAT_R8G8B8A8_UINT,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   VkAttachmentReference ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ref,
   };
   VkSubpassDependency deps[2] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att,
      .subpassCount = 1,
      .pSubpasses = &sub,
      .dependencyCount = 2,
      .pDependencies = deps,
   };
   VK_CHECK(vkCreateRenderPass(p->vk.dev, &rpci, NULL, &p->rp));

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = 4,
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   VK_CHECK(vkCreatePipelineLayout(p->vk.dev, &plci, NULL, &p->layout));

   p->vs = load_shader(p, "interop.vert.spv");
   p->fs = load_shader(p, "interop.frag.spv");

   VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = p->vs,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = p->fs,
         .pName = "main",
      },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkViewport vp = {
      .x = 0, .y = 0,
      .width = (float)p->opt->width, .height = (float)p->opt->height,
      .minDepth = 0, .maxDepth = 1,
   };
   VkRect2D sc = { .offset = { 0, 0 },
                   .extent = { p->opt->width, p->opt->height } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &sc,
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
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gp = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vps,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .layout = p->layout,
      .renderPass = p->rp,
      .subpass = 0,
   };
   VK_CHECK(vkCreateGraphicsPipelines(p->vk.dev, VK_NULL_HANDLE, 1, &gp, NULL, &p->pipe));
}

/* One VkDeviceMemory per slot, two VkBuffers bound into it, one fd.
 * No dedicated allocation: dedicated is per resource and two buffers share
 * this allocation (PROTOCOL.md, "Memory layout"). */
static void
slot_create(struct producer *p, struct slot *s)
{
   struct vk *vk = &p->vk;

   VkExternalMemoryBufferCreateInfo ext = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &ext,
      .size = p->frame_bytes,
      .usage = SLOT_BUF_USAGE,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VK_CHECK(vkCreateBuffer(vk->dev, &bci, NULL, &s->frame_buf));
   /* Rounded up to 4 so that one vkCmdFillBuffer poisons all of it. */
   bci.size = (p->result_bytes + 3) & ~3ull;
   VK_CHECK(vkCreateBuffer(vk->dev, &bci, NULL, &s->result_buf));

   VkMemoryRequirements rf, rr;
   vkGetBufferMemoryRequirements(vk->dev, s->frame_buf, &rf);
   vkGetBufferMemoryRequirements(vk->dev, s->result_buf, &rr);

   VkDeviceSize align = rf.alignment > rr.alignment ? rf.alignment : rr.alignment;
   if (align < 256)
      align = 256;                      /* PROTOCOL.md: 256 is not cosmetic */

   s->frame_offset = 0;
   s->frame_bytes = p->frame_bytes;
   s->result_offset = (rf.size + align - 1) / align * align;
   s->result_bytes = p->result_bytes;
   s->alloc_size = (s->result_offset + rr.size + align - 1) / align * align;

   uint32_t bits = rf.memoryTypeBits & rr.memoryTypeBits;
   uint32_t type = find_memory_type(vk->pdev, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (type == UINT32_MAX)
      type = find_memory_type(vk->pdev, bits, 0);
   if (type == UINT32_MAX)
      fatal(EXIT_PROTO, "no memory type serves both slot buffers");

   VkExportMemoryAllocateInfo emai = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &emai,
      .allocationSize = s->alloc_size,
      .memoryTypeIndex = type,
   };
   VK_CHECK(vkAllocateMemory(vk->dev, &mai, NULL, &s->mem));
   VK_CHECK(vkBindBufferMemory(vk->dev, s->frame_buf, s->mem, s->frame_offset));
   VK_CHECK(vkBindBufferMemory(vk->dev, s->result_buf, s->mem, s->result_offset));

   /* One offscreen image per slot, so two frames in flight never share it. */
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UINT,
      .extent = { p->opt->width, p->opt->height, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &s->image));
   VkMemoryRequirements ri;
   vkGetImageMemoryRequirements(vk->dev, s->image, &ri);
   uint32_t itype = find_memory_type(vk->pdev, ri.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (itype == UINT32_MAX)
      itype = find_memory_type(vk->pdev, ri.memoryTypeBits, 0);
   VkMemoryAllocateInfo imai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = ri.size,
      .memoryTypeIndex = itype,
   };
   VK_CHECK(vkAllocateMemory(vk->dev, &imai, NULL, &s->image_mem));
   VK_CHECK(vkBindImageMemory(vk->dev, s->image, s->image_mem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = s->image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UINT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VK_CHECK(vkCreateImageView(vk->dev, &ivci, NULL, &s->view));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = p->rp,
      .attachmentCount = 1,
      .pAttachments = &s->view,
      .width = p->opt->width,
      .height = p->opt->height,
      .layers = 1,
   };
   VK_CHECK(vkCreateFramebuffer(vk->dev, &fbci, NULL, &s->fb));

   s->cmd_render = alloc_cmd(vk);
   s->cmd_verify = alloc_cmd(vk);
   s->fence_render = alloc_fence(vk, false);
   s->fence_verify = alloc_fence(vk, false);
   s->busy = false;
   s->owned_external = false;
   s->render_pending = false;
}

static int
slot_export_fd(struct producer *p, struct slot *s)
{
   VkMemoryGetFdInfoKHR gfi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = s->mem,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };
   int fd = -1;
   VK_CHECK(p->vk.GetMemoryFdKHR(p->vk.dev, &gfi, &fd));
   if (fd < 0)
      fatal(EXIT_PROTO, "vkGetMemoryFdKHR returned a negative fd");
   return fd;
}

/* ------------------------------------------------------------------ */
/* Recording                                                          */

static void
record_render(struct producer *p, struct slot *s, uint32_t seq)
{
   VkCommandBuffer cb = s->cmd_render;
   VK_CHECK(vkResetCommandBuffer(cb, 0));
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VK_CHECK(vkBeginCommandBuffer(cb, &bi));

   /* Acquire the exported buffers back from the external consumer.  Only
    * needed if they are still released -- with --no-producer-wait a slot is
    * re-rendered before the acquire in the verify pass has happened. */
   if (s->owned_external) {
      buffer_barrier(cb, s->frame_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, p->vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      buffer_barrier(cb, s->result_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, p->vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      s->owned_external = false;
   }

   /* Poison the whole result range before the handover.  A byte that was
    * never written must be visible, not plausible. */
   vkCmdFillBuffer(cb, s->result_buf, 0, VK_WHOLE_SIZE, 0xCDCDCDCDu);

   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = p->rp,
      .framebuffer = s->fb,
      .renderArea = { { 0, 0 }, { p->opt->width, p->opt->height } },
   };
   vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipe);
   vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &seq);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRenderPass(cb);

   VkBufferImageCopy region = {
      .bufferOffset = 0,
      .bufferRowLength = 0,        /* tightly packed */
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1,
      },
      .imageOffset = { 0, 0, 0 },
      .imageExtent = { p->opt->width, p->opt->height, 1 },
   };
   vkCmdCopyImageToBuffer(cb, s->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          s->frame_buf, 1, &region);

   /* Release both buffers to the external consumer. */
   buffer_barrier(cb, s->frame_buf, 0, VK_WHOLE_SIZE,
                  VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                  p->vk.qfam, VK_QUEUE_FAMILY_EXTERNAL,
                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
   buffer_barrier(cb, s->result_buf, 0, VK_WHOLE_SIZE,
                  VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                  p->vk.qfam, VK_QUEUE_FAMILY_EXTERNAL,
                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

   VK_CHECK(vkEndCommandBuffer(cb));
}

static void
record_verify(struct producer *p, struct slot *s)
{
   VkCommandBuffer cb = s->cmd_verify;
   VK_CHECK(vkResetCommandBuffer(cb, 0));
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VK_CHECK(vkBeginCommandBuffer(cb, &bi));

   if (s->owned_external) {
      buffer_barrier(cb, s->result_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_READ_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, p->vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      buffer_barrier(cb, s->frame_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_READ_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, p->vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      s->owned_external = false;
   }

   VkBufferCopy c1 = { .srcOffset = 0, .dstOffset = 0, .size = s->result_bytes };
   vkCmdCopyBuffer(cb, s->result_buf, p->stage_result.buf, 1, &c1);
   VkBufferCopy c2 = { .srcOffset = 0, .dstOffset = 0, .size = s->frame_bytes };
   vkCmdCopyBuffer(cb, s->frame_buf, p->stage_frame.buf, 1, &c2);

   VkMemoryBarrier mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

   VK_CHECK(vkEndCommandBuffer(cb));
}

static void
submit(struct producer *p, VkCommandBuffer cb, VkFence fence,
       VkSemaphore wait_sem, uint64_t wait_val,
       VkSemaphore signal_sem, uint64_t signal_val)
{
   VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
   VkTimelineSemaphoreSubmitInfoKHR tsi = {
      .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
      .waitSemaphoreValueCount = wait_sem ? 1 : 0,
      .pWaitSemaphoreValues = &wait_val,
      .signalSemaphoreValueCount = signal_sem ? 1 : 0,
      .pSignalSemaphoreValues = &signal_val,
   };
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = (wait_sem || signal_sem) ? &tsi : NULL,
      .waitSemaphoreCount = wait_sem ? 1 : 0,
      .pWaitSemaphores = &wait_sem,
      .pWaitDstStageMask = &wait_stage,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
      .signalSemaphoreCount = signal_sem ? 1 : 0,
      .pSignalSemaphores = &signal_sem,
   };
   VK_CHECK(vkQueueSubmit(p->vk.queue, 1, &si, fence));
}

/* ------------------------------------------------------------------ */
/* Verification                                                       */

struct verdict {
   uint64_t checked;
   uint64_t bad;
   uint64_t poison;        /* still 0xCD inside the written range */
   uint64_t tail_bad;      /* not 0xCD beyond result_bytes: an overrun */
   int64_t  first_bad;
   int stamp_err;
   uint32_t got_seq, got_not_seq;
};

static void
verify_frame(struct producer *p, struct slot *s, uint32_t seq,
             uint64_t written, struct verdict *v, uint8_t *expected)
{
   const uint8_t *got = p->stage_result.map;
   memset(v, 0, sizeof(*v));
   v->first_bad = -1;

   /* (a) the check-mode stamp */
   memcpy(&v->got_seq, got, 4);
   memcpy(&v->got_not_seq, got + 4, 4);
   if (v->got_seq != seq)
      v->stamp_err++;
   if (v->got_not_seq != ~seq)
      v->stamp_err++;

   /* (b) bytes 8.. equal (pattern + 1) & 255 */
   expected_result(expected, p->opt->width, p->opt->height, seq);
   uint64_t n = written < s->result_bytes ? written : s->result_bytes;
   uint64_t cmp = n < (uint64_t)p->frame_bytes ? n : (uint64_t)p->frame_bytes;
   v->checked = cmp > 8 ? cmp - 8 : 0;
   for (uint64_t i = 8; i < cmp; i++) {
      if (got[i] != expected[i]) {
         v->bad++;
         if (v->first_bad < 0)
            v->first_bad = (int64_t)i;
      }
      if (got[i] == 0xCD && expected[i] != 0xCD)
         v->poison++;
   }

   /* (c) the tail beyond MSG_RESULT.result_bytes is still poison */
   for (uint64_t i = n; i < s->result_bytes; i++) {
      if (got[i] != 0xCD) {
         v->tail_bad++;
         if (v->first_bad < 0)
            v->first_bad = (int64_t)i;
      }
   }
}

/* Only the tail check is content-independent, so it is the one that still
 * runs for --op sobel and --op blur. */
static void
verify_tail_only(struct producer *p, struct slot *s, uint64_t written,
                 struct verdict *v)
{
   const uint8_t *got = p->stage_result.map;
   memset(v, 0, sizeof(*v));
   v->first_bad = -1;
   uint64_t n = written < s->result_bytes ? written : s->result_bytes;
   for (uint64_t i = n; i < s->result_bytes; i++) {
      if (got[i] != 0xCD) {
         v->tail_bad++;
         if (v->first_bad < 0)
            v->first_bad = (int64_t)i;
      }
   }
}

static void
dump_pngs(struct producer *p, struct slot *s, unsigned slot_index,
          uint32_t seq, const uint8_t *expected)
{
   unsigned w = p->opt->width, h = p->opt->height;
   char path[1024];
   const uint8_t *got = p->stage_result.map;

   if (strcmp(p->opt->png_dir, ".") && mkdir(p->opt->png_dir, 0755) < 0 &&
       errno != EEXIST)
      logf_("cannot create %s: %s", p->opt->png_dir, strerror(errno));

   snprintf(path, sizeof(path), "%s/f%06u_slot%u_in.png",
            p->opt->png_dir, seq, slot_index);
   write_png(path, p->stage_frame.map, w, h);

   snprintf(path, sizeof(path), "%s/f%06u_slot%u_out.png",
            p->opt->png_dir, seq, slot_index);
   write_png(path, got, w, h);

   snprintf(path, sizeof(path), "%s/f%06u_slot%u_expected.png",
            p->opt->png_dir, seq, slot_index);
   write_png(path, expected, w, h);

   uint8_t *diff = malloc((size_t)w * h * 4);
   if (diff) {
      for (size_t i = 0; i < (size_t)w * h; i++) {
         bool bad = false;
         for (int c = 0; c < 4; c++) {
            size_t o = i * 4 + c;
            if (o < 8)
               continue;                       /* the stamp */
            if (o < (size_t)w * h * 4 && got[o] != expected[o])
               bad = true;
         }
         bool poison = got[i * 4] == 0xCD && got[i * 4 + 1] == 0xCD &&
                       got[i * 4 + 2] == 0xCD && got[i * 4 + 3] == 0xCD;
         diff[i * 4 + 0] = bad ? 255 : 0;
         diff[i * 4 + 1] = poison ? 255 : 0;    /* never written */
         diff[i * 4 + 2] = 0;
         diff[i * 4 + 3] = 255;
      }
      snprintf(path, sizeof(path), "%s/f%06u_slot%u_diff.png",
               p->opt->png_dir, seq, slot_index);
      write_png(path, diff, w, h);
      free(diff);
   }
   logf_("wrote %s/f%06u_slot%u_{in,out,expected,diff}.png",
         p->opt->png_dir, seq, slot_index);
}

/* ------------------------------------------------------------------ */
/* Handshake and steady state                                         */

static void
uuid_str(const uint8_t u[16], char out[40])
{
   snprintf(out, 40,
            "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
            u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
}

#define HANDSHAKE_MS 60000.0
#define FRAME_MS      2000.0

static int
producer_run(struct options *opt, int sock)
{
   struct producer P;
   memset(&P, 0, sizeof(P));
   P.opt = opt;
   P.sock = sock;

   set_state("bringing up Vulkan");
   vk_init(&P.vk, NULL, opt->sync == SYNC_TIMELINE, true);

   /* If the driver has no timeline semaphores, the run continues in host
    * mode.  vk_init has already said so. */
   if (opt->sync == SYNC_TIMELINE && !P.vk.timeline_ok)
      opt->sync = SYNC_HOST;

   char uuid[40];
   uuid_str(P.vk.uuid, uuid);
   /* Reported, never acted on. */
   if (P.vk.has_driver_props)
      printf("renderer: driverName=%s driverID=%u driverInfo=%s "
             "driverVersion=%u.%u.%u device=%s uuid=%s "
             "sync=%s slots=%u frames=%u %ux%u op=%s\n",
             P.vk.driver_name, P.vk.driver_id, P.vk.driver_info,
             VK_VERSION_MAJOR(P.vk.props.driverVersion),
             VK_VERSION_MINOR(P.vk.props.driverVersion),
             VK_VERSION_PATCH(P.vk.props.driverVersion),
             P.vk.props.deviceName, uuid,
             opt->sync == SYNC_TIMELINE ? "timeline" : "host",
             opt->slots, opt->frames, opt->width, opt->height, opt->op);
   else
      printf("renderer: device=%s driverVersion=%u.%u.%u (raw %u, no "
             "VK_KHR_driver_properties) uuid=%s "
             "sync=%s slots=%u frames=%u %ux%u op=%s\n",
             P.vk.props.deviceName,
             VK_VERSION_MAJOR(P.vk.props.driverVersion),
             VK_VERSION_MINOR(P.vk.props.driverVersion),
             VK_VERSION_PATCH(P.vk.props.driverVersion),
             P.vk.props.driverVersion, uuid,
             opt->sync == SYNC_TIMELINE ? "timeline" : "host",
             opt->slots, opt->frames, opt->width, opt->height, opt->op);
   fflush(stdout);

   P.frame_bytes = opt->width * opt->height * 4;

   build_pipeline(&P);

   /* Phase 1: OFFER */
   struct msg_offer offer;
   memset(&offer, 0, sizeof(offer));
   offer.magic = CPIO_MAGIC;
   offer.type = MSG_OFFER;
   offer.version = 1;
   offer.slots = opt->slots;
   offer.width = opt->width;
   offer.height = opt->height;
   offer.frame_format = 0;                  /* VK_FORMAT_R8G8B8A8_UINT */
   offer.frame_bytes = P.frame_bytes;
   /* Offer exactly the mode this process was started in.  PROTOCOL.md lets
    * the consumer pick one bit out of sync_modes; making that a single bit
    * removes the only way the two halves can disagree about which mode the
    * run is in, and the producer's CLI is what the test matrix varies. */
   offer.sync_modes = opt->sync;
   memcpy(offer.device_uuid, P.vk.uuid, 16);
   offer.scene = 0;
   set_state("sending MSG_OFFER");
   {
      int r = send_msg(sock, &offer, sizeof(offer), NULL, 0);
      if (r < 0)
         fatal(EXIT_PROTO, "sendmsg(MSG_OFFER): %s", strerror(-r));
   }

   /* Phase 2: REQUEST */
   set_state("waiting MSG_REQUEST");
   union any_msg m;
   expect_msg(sock, MSG_REQUEST, &m, sizeof(struct msg_request),
              HANDSHAKE_MS, NULL, NULL);
   P.result_bytes = m.request.result_bytes;
   P.result_dtype = m.request.result_dtype;
   P.result_ndim = m.request.result_ndim;
   memcpy(P.result_shape, m.request.result_shape, sizeof(P.result_shape));

   if (P.result_bytes == 0 || P.result_bytes > (1ull << 32))
      fatal(EXIT_PROTO, "MSG_REQUEST asks for %llu result bytes",
            (unsigned long long)P.result_bytes);
   if (m.request.sync_mode != opt->sync)
      fatal(EXIT_PROTO, "consumer chose sync_mode 0x%x; MSG_OFFER advertised "
            "only 0x%x (%s), which is how this producer was started",
            m.request.sync_mode, opt->sync,
            opt->sync == SYNC_TIMELINE ? "timeline" : "host");
   if (!(m.request.sync_mode & offer.sync_modes))
      fatal(EXIT_PROTO, "consumer chose a sync mode that was not offered");
   if (!strcmp(opt->op, "add1") && P.result_bytes != P.frame_bytes)
      logf_("warning: op add1 but the consumer asked for %llu result bytes, "
            "frame is %u; content is verified over the overlap only",
            (unsigned long long)P.result_bytes, P.frame_bytes);
   logf_("MSG_REQUEST: %llu bytes, dtype %u, ndim %u, shape [%u %u %u %u]",
         (unsigned long long)P.result_bytes, P.result_dtype, P.result_ndim,
         P.result_shape[0], P.result_shape[1], P.result_shape[2], P.result_shape[3]);

   /* Phase 3: one VkDeviceMemory per slot, one fd each. */
   P.slots = calloc(opt->slots, sizeof(*P.slots));
   hostbuf_create(&P.vk, &P.stage_result, (P.result_bytes + 3) & ~3ull,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   hostbuf_create(&P.vk, &P.stage_frame, P.frame_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   for (unsigned i = 0; i < opt->slots; i++) {
      struct slot *s = &P.slots[i];
      slot_create(&P, s);
      int fd = slot_export_fd(&P, s);

      struct msg_slot ms;
      memset(&ms, 0, sizeof(ms));
      ms.magic = CPIO_MAGIC;
      ms.type = MSG_SLOT;
      ms.slot = i;
      ms.alloc_size = s->alloc_size;
      ms.frame_offset = s->frame_offset;
      ms.frame_bytes = s->frame_bytes;
      ms.result_offset = s->result_offset;
      ms.result_bytes = s->result_bytes;
      set_state("sending MSG_SLOT %u", i);
      int r = send_msg(sock, &ms, sizeof(ms), &fd, 1);
      if (r < 0) {
         close(fd);
         fatal(EXIT_PROTO, "sendmsg(MSG_SLOT %u): %s", i, strerror(-r));
      }
      /* CUDA takes ownership of the receiver's copy; ours is done. */
      close(fd);
      if (i == 0)
         logf_("slot layout: alloc=%llu frame@%llu+%llu result@%llu+%llu",
               (unsigned long long)s->alloc_size,
               (unsigned long long)s->frame_offset,
               (unsigned long long)s->frame_bytes,
               (unsigned long long)s->result_offset,
               (unsigned long long)s->result_bytes);
   }

   /* Phase 3b: the two timeline semaphores. */
   if (opt->sync == SYNC_TIMELINE) {
      VkExportSemaphoreCreateInfo esci = {
         .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
         .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
      };
      VkSemaphoreTypeCreateInfoKHR stci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
         .pNext = &esci,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR,
         .initialValue = 0,
      };
      VkSemaphoreCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
         .pNext = &stci,
      };
      VK_CHECK(vkCreateSemaphore(P.vk.dev, &sci, NULL, &P.sem_render));
      VK_CHECK(vkCreateSemaphore(P.vk.dev, &sci, NULL, &P.sem_consume));

      int fds[2];
      VkSemaphoreGetFdInfoKHR gi = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
         .semaphore = P.sem_render,
         .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
      };
      VK_CHECK(P.vk.GetSemaphoreFdKHR(P.vk.dev, &gi, &fds[0]));
      gi.semaphore = P.sem_consume;
      VK_CHECK(P.vk.GetSemaphoreFdKHR(P.vk.dev, &gi, &fds[1]));

      struct msg_sems msems;
      memset(&msems, 0, sizeof(msems));
      msems.magic = CPIO_MAGIC;
      msems.type = MSG_SEMS;
      set_state("sending MSG_SEMS");
      int r = send_msg(sock, &msems, sizeof(msems), fds, 2);
      if (r < 0) {
         close(fds[0]);
         close(fds[1]);
         fatal(EXIT_PROTO, "sendmsg(MSG_SEMS): %s", strerror(-r));
      }
      close(fds[0]);
      close(fds[1]);
   }

   /* Phase 4: HELLO.  Frame counting starts here, so torch import, CUDA
    * init and cuDNN autotune do not land in frame 0. */
   set_state("waiting MSG_HELLO (consumer importing)");
   double t_handshake = now_ms();
   expect_msg(sock, MSG_HELLO, &m, sizeof(struct msg_hello),
              HANDSHAKE_MS, NULL, NULL);
   if (m.hello.status != 0)
      fatal(EXIT_PROTO, "consumer MSG_HELLO status %u", m.hello.status);
   logf_("MSG_HELLO after %.0f ms; starting %u frames",
         now_ms() - t_handshake, opt->frames);

   uint8_t *expected = malloc(P.frame_bytes);
   if (!expected)
      fatal(EXIT_PROTO, "out of memory");

   unsigned issued = 0, done = 0, outstanding = 0, rr = 0;
   int verify_failures = 0;
   bool check_content = !strcmp(opt->op, "add1");
   double t0 = now_ms();

   while (done < opt->frames) {
      /* Issue as many frames as the ring allows. */
      for (;;) {
         if (issued >= opt->frames)
            break;
         unsigned cap = opt->no_producer_wait ? opt->slots + 1 : opt->slots;
         if (outstanding >= cap)
            break;

         struct slot *s = NULL;
         unsigned idx = 0;
         if (opt->no_producer_wait) {
            /* The negative control: reuse the next slot round robin whether
             * or not its MSG_RESULT has come back. */
            idx = rr++ % opt->slots;
            s = &P.slots[idx];
         } else {
            for (unsigned i = 0; i < opt->slots; i++) {
               if (!P.slots[i].busy) {
                  idx = i;
                  s = &P.slots[i];
                  break;
               }
            }
            if (!s)
               break;
         }

         uint32_t seq = ++issued;          /* sequence numbers start at 1 */
         if (s->render_pending) {
            set_state("waiting GPU for the previous render on slot %u", idx);
            wait_fence(&P.vk, s->fence_render, "render");
            s->render_pending = false;
         }
         s->seq = seq;
         s->busy = true;
         set_state("rendering slot=%u seq=%u", idx, seq);
         record_render(&P, s, seq);

         if (opt->sync == SYNC_TIMELINE) {
            uint64_t wait_val = seq > opt->slots ? seq - opt->slots : 0;
            submit(&P, s->cmd_render, s->fence_render,
                   P.sem_consume, wait_val, P.sem_render, seq);
            s->owned_external = true;
            s->render_pending = true;
         } else {
            submit(&P, s->cmd_render, s->fence_render,
                   VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0);
            s->owned_external = true;
            set_state("waiting fence slot=%u seq=%u", idx, seq);
            wait_fence(&P.vk, s->fence_render, "render");
            s->render_pending = false;
         }

         struct msg_ready ready = {
            .magic = CPIO_MAGIC, .type = MSG_READY,
            .slot = idx, .seq = seq,
            .timeline_value = opt->sync == SYNC_TIMELINE ? seq : 0,
         };
         set_state("sending MSG_READY slot=%u seq=%u", idx, seq);
         int rr_send = send_msg(sock, &ready, sizeof(ready), NULL, 0);
         if (rr_send < 0)
            fatal(EXIT_PROTO, "sendmsg(MSG_READY): %s", strerror(-rr_send));
         outstanding++;
      }

      /* Wait for one MSG_RESULT, naming the oldest frame still out. */
      {
         unsigned wslot = opt->slots;
         uint32_t wseq = 0;
         for (unsigned i = 0; i < opt->slots; i++) {
            if (P.slots[i].busy && (wseq == 0 || P.slots[i].seq < wseq)) {
               wseq = P.slots[i].seq;
               wslot = i;
            }
         }
         set_state("waiting MSG_RESULT slot=%u seq=%u", wslot, wseq);
      }

      int r = wait_readable(sock, now_ms() + FRAME_MS);
      if (r == 0)
         fatal(EXIT_PROTO, "no MSG_RESULT within %.0f ms", FRAME_MS);
      if (r < 0)
         fatal(EXIT_PROTO, "peer hung up (%s)", strerror(errno));

      ssize_t n = recv_msg(sock, &m, sizeof(m), NULL, NULL);
      if (n == 0)
         fatal(verify_failures ? EXIT_VERIFY : EXIT_PROTO,
               "EOF from consumer after %u of %u frames", done, opt->frames);
      if (n < 0)
         fatal(EXIT_PROTO, "recvmsg: %s", strerror(errno));
      if (m.hdr.magic != CPIO_MAGIC)
         fatal(EXIT_PROTO, "bad magic 0x%08x", m.hdr.magic);
      if (m.hdr.type == MSG_ERROR) {
         m.error.text[sizeof(m.error.text) - 1] = 0;
         logf_("consumer MSG_ERROR code %u: %s", m.error.code, m.error.text);
         /* code 1 is the consumer's verification verdict; anything else is
          * a protocol failure.  --no-producer-wait is expected to make the
          * consumer reject a frame, and that is a verification failure. */
         fatal((m.error.code == EXIT_VERIFY || opt->no_producer_wait) ?
               EXIT_VERIFY : EXIT_PROTO, "consumer stopped the run");
      }
      if (m.hdr.type != MSG_RESULT)
         fatal(EXIT_PROTO, "expected MSG_RESULT, got type %u", m.hdr.type);
      if ((size_t)n != sizeof(struct msg_result))
         fatal(EXIT_PROTO, "MSG_RESULT is %zd bytes, expected %zu",
               n, sizeof(struct msg_result));
      if (m.result.slot >= opt->slots)
         fatal(EXIT_PROTO, "MSG_RESULT names slot %u of %u",
               m.result.slot, opt->slots);

      struct slot *s = &P.slots[m.result.slot];
      uint32_t seq = m.result.seq;
      outstanding--;
      s->busy = false;
      done++;

      if (m.result.status != 0) {
         logf_("frame %u slot %u: consumer reported status %u",
               seq, m.result.slot, m.result.status);
         verify_failures++;
         P.frames_failed++;
      }
      if (m.result.result_bytes > s->result_bytes)
         fatal(EXIT_PROTO, "MSG_RESULT claims %llu bytes written into a %llu "
               "byte result range",
               (unsigned long long)m.result.result_bytes,
               (unsigned long long)s->result_bytes);

      /* Read the result back and check it. */
      set_state("verifying slot=%u seq=%u", m.result.slot, seq);
      record_verify(&P, s);
      if (opt->sync == SYNC_TIMELINE) {
         uint64_t wv = m.result.timeline_value ? m.result.timeline_value : seq;
         submit(&P, s->cmd_verify, s->fence_verify,
                P.sem_consume, wv, VK_NULL_HANDLE, 0);
      } else {
         submit(&P, s->cmd_verify, s->fence_verify,
                VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0);
      }
      wait_fence(&P.vk, s->fence_verify, "verify readback");

      struct verdict v;
      if (check_content)
         verify_frame(&P, s, seq, m.result.result_bytes, &v, expected);
      else
         verify_tail_only(&P, s, m.result.result_bytes, &v);

      P.total_bytes += v.checked;
      P.total_bad += v.bad;
      P.total_poison += v.poison;
      P.total_stamp_err += v.stamp_err;
      P.total_tail_bad += v.tail_bad;

      bool bad = v.bad || v.tail_bad || v.stamp_err;
      if (bad) {
         if (P.frames_failed < 8 || (P.frames_failed % 100) == 0) {
            logf_("frame %u slot %u FAILED: %llu/%llu bytes wrong, %llu poison, "
                  "%d stamp errors, %llu poison-tail bytes overwritten, "
                  "first bad offset %lld",
                  seq, m.result.slot,
                  (unsigned long long)v.bad, (unsigned long long)v.checked,
                  (unsigned long long)v.poison, v.stamp_err,
                  (unsigned long long)v.tail_bad, (long long)v.first_bad);
            if (v.stamp_err)
               logf_("   stamp: got seq %u (~seq %u), expected %u (~%u)",
                     v.got_seq, v.got_not_seq, seq, ~seq);
            if (check_content)
               dump_pngs(&P, s, m.result.slot, seq, expected);
         }
         verify_failures++;
         P.frames_failed++;
      } else if (opt->png_every && (seq % opt->png_every) == 0) {
         if (check_content) {
            expected_result(expected, opt->width, opt->height, seq);
            dump_pngs(&P, s, m.result.slot, seq, expected);
         }
      }
   }

   double dt = now_ms() - t0;
   vkDeviceWaitIdle(P.vk.dev);

   /* Phase 7: BYE */
   struct msg_bye bye = {
      .magic = CPIO_MAGIC, .type = MSG_BYE,
      .frames = done, .status = verify_failures ? 1u : 0u,
   };
   set_state("sending MSG_BYE");
   (void)send_msg(sock, &bye, sizeof(bye), NULL, 0);

   printf("renderer: %u frames in %.1f ms (%.3f ms/frame)\n", done, dt,
          done ? dt / done : 0.0);
   printf("return path : %llu / %llu bytes wrong, %llu poison, %llu stamp errors\n",
          (unsigned long long)(P.total_bad + P.total_tail_bad),
          (unsigned long long)P.total_bytes,
          (unsigned long long)P.total_poison,
          (unsigned long long)P.total_stamp_err);
   if (P.total_tail_bad)
      printf("renderer: %llu bytes past MSG_RESULT.result_bytes were overwritten "
             "(the poison tail caught an overrun)\n",
             (unsigned long long)P.total_tail_bad);
   if (!check_content)
      printf("renderer: content verification skipped for op %s; only the "
             "poison tail was checked\n", opt->op);
   fflush(stdout);

   free(expected);
   return verify_failures ? EXIT_VERIFY : EXIT_PASS;
}

/* ------------------------------------------------------------------ */
/* Fake consumer.                                                     */
/*                                                                    */
/* This is the producer binary in a second role, used by --self-test.  */
/* It speaks PROTOCOL.md exactly as the PyTorch consumer does -- it    */
/* receives the fds over SCM_RIGHTS, imports the memory, waits and     */
/* signals the timeline semaphores, and writes the result -- but it    */
/* does the arithmetic with Vulkan copies and the host CPU instead of  */
/* CUDA.  It links no CUDA either.  It exists so that the protocol,    */
/* the fd passing, the queue-family handover, the poison, the timeouts */
/* and the verification path can all be exercised without torch.       */

struct fake_slot {
   VkDeviceMemory mem;
   VkBuffer frame_buf, result_buf;
   uint64_t alloc_size, frame_offset, frame_bytes, result_offset, result_bytes;
};

static int
fake_consumer_run(struct options *opt, int sock)
{
   g_who = "fake-consumer";
   struct vk vk;
   union any_msg m;

   set_state("waiting MSG_OFFER");
   expect_msg(sock, MSG_OFFER, &m, sizeof(struct msg_offer),
              HANDSHAKE_MS, NULL, NULL);
   struct msg_offer offer = m.offer;
   if (offer.version != 1)
      fatal(EXIT_PROTO, "MSG_OFFER version %u, expected 1", offer.version);

   /* Take what was offered: the producer may have fallen back to host mode
    * because the driver has no timeline semaphores. */
   bool timeline = (opt->sync == SYNC_TIMELINE) &&
                   (offer.sync_modes & SYNC_TIMELINE) != 0;
   if (!(offer.sync_modes & (timeline ? SYNC_TIMELINE : SYNC_HOST)))
      fatal(EXIT_PROTO, "MSG_OFFER advertised sync_modes 0x%x, which contains "
            "neither mode this consumer can run", offer.sync_modes);

   vk_init(&vk, offer.device_uuid, timeline, false);
   logf_("imported onto %s via %s", vk.props.deviceName, vk.driver_name);

   struct msg_request req;
   memset(&req, 0, sizeof(req));
   req.magic = CPIO_MAGIC;
   req.type = MSG_REQUEST;
   req.result_bytes = offer.frame_bytes;      /* check mode: same shape */
   req.result_dtype = 0;                      /* u8 */
   req.result_ndim = 3;
   req.result_shape[0] = offer.height;
   req.result_shape[1] = offer.width;
   req.result_shape[2] = 4;
   req.sync_mode = timeline ? SYNC_TIMELINE : SYNC_HOST;
   set_state("sending MSG_REQUEST");
   if (send_msg(sock, &req, sizeof(req), NULL, 0) < 0)
      fatal(EXIT_PROTO, "sendmsg(MSG_REQUEST): %s", strerror(errno));

   struct fake_slot *slots = calloc(offer.slots, sizeof(*slots));
   for (unsigned i = 0; i < offer.slots; i++) {
      int fds[8], nfds = 0;
      set_state("waiting MSG_SLOT %u", i);
      expect_msg(sock, MSG_SLOT, &m, sizeof(struct msg_slot),
                 HANDSHAKE_MS, fds, &nfds);
      if (nfds != 1)
         fatal(EXIT_PROTO, "MSG_SLOT carried %d fds, expected 1", nfds);
      struct fake_slot *fs = &slots[m.slot.slot];
      fs->alloc_size = m.slot.alloc_size;
      fs->frame_offset = m.slot.frame_offset;
      fs->frame_bytes = m.slot.frame_bytes;
      fs->result_offset = m.slot.result_offset;
      fs->result_bytes = m.slot.result_bytes;

      /* An OPAQUE_FD is opaque: vkGetMemoryFdPropertiesKHR is not allowed
       * for it, and the importer is required to pick a memory type itself.
       * Same physical device, so the buffers' own requirements name it.
       * (CUDA does not have to do this; it takes alloc_size and nothing
       * else.  This is the Vulkan-importer detail only.) */
      VkExternalMemoryBufferCreateInfo ext = {
         .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
      };
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .pNext = &ext,
         .size = fs->frame_bytes,
         .usage = SLOT_BUF_USAGE,
      };
      VK_CHECK(vkCreateBuffer(vk.dev, &bci, NULL, &fs->frame_buf));
      bci.size = (fs->result_bytes + 3) & ~3ull;
      VK_CHECK(vkCreateBuffer(vk.dev, &bci, NULL, &fs->result_buf));

      VkMemoryRequirements rf, rr;
      vkGetBufferMemoryRequirements(vk.dev, fs->frame_buf, &rf);
      vkGetBufferMemoryRequirements(vk.dev, fs->result_buf, &rr);
      uint32_t bits = rf.memoryTypeBits & rr.memoryTypeBits;
      uint32_t type = find_memory_type(vk.pdev, bits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (type == UINT32_MAX)
         type = find_memory_type(vk.pdev, bits, 0);
      if (type == UINT32_MAX) {
         close(fds[0]);
         fatal(EXIT_PROTO, "no memory type accepts the imported fd");
      }

      VkImportMemoryFdInfoKHR imp = {
         .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
         .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
         .fd = fds[0],
      };
      VkMemoryAllocateInfo mai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .pNext = &imp,
         .allocationSize = fs->alloc_size,
         .memoryTypeIndex = type,
      };
      VkResult r = vkAllocateMemory(vk.dev, &mai, NULL, &fs->mem);
      if (r != VK_SUCCESS) {
         /* The importer closes an fd whose import failed. */
         close(fds[0]);
         fatal(EXIT_PROTO, "importing slot %u failed: %d", i, (int)r);
      }
      /* Import succeeded: Vulkan owns the fd now, exactly as CUDA would. */

      VK_CHECK(vkBindBufferMemory(vk.dev, fs->frame_buf, fs->mem, fs->frame_offset));
      VK_CHECK(vkBindBufferMemory(vk.dev, fs->result_buf, fs->mem, fs->result_offset));
   }

   VkSemaphore sem_render = VK_NULL_HANDLE, sem_consume = VK_NULL_HANDLE;
   if (timeline) {
      int fds[8], nfds = 0;
      set_state("waiting MSG_SEMS");
      expect_msg(sock, MSG_SEMS, &m, sizeof(struct msg_sems),
                 HANDSHAKE_MS, fds, &nfds);
      if (nfds != 2)
         fatal(EXIT_PROTO, "MSG_SEMS carried %d fds, expected 2", nfds);

      VkSemaphoreTypeCreateInfoKHR stci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR,
         .initialValue = 0,
      };
      VkSemaphoreCreateInfo sci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
         .pNext = &stci,
      };
      VK_CHECK(vkCreateSemaphore(vk.dev, &sci, NULL, &sem_render));
      VK_CHECK(vkCreateSemaphore(vk.dev, &sci, NULL, &sem_consume));

      VkImportSemaphoreFdInfoKHR isi = {
         .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
         .semaphore = sem_render,
         .flags = 0,
         .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
         .fd = fds[0],
      };
      VkResult r = vk.ImportSemaphoreFdKHR(vk.dev, &isi);
      if (r != VK_SUCCESS) {
         close(fds[0]);
         close(fds[1]);
         fatal(EXIT_PROTO, "vkImportSemaphoreFdKHR(sem_render): %d", (int)r);
      }
      isi.semaphore = sem_consume;
      isi.fd = fds[1];
      r = vk.ImportSemaphoreFdKHR(vk.dev, &isi);
      if (r != VK_SUCCESS) {
         close(fds[1]);
         fatal(EXIT_PROTO, "vkImportSemaphoreFdKHR(sem_consume): %d", (int)r);
      }
   }

   /* Staging: the frame comes here, the answer goes back from here. */
   struct hostbuf stage_in, stage_out;
   hostbuf_create(&vk, &stage_in, offer.frame_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   hostbuf_create(&vk, &stage_out, (req.result_bytes + 3) & ~3ull,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

   VkCommandBuffer cb_in = alloc_cmd(&vk), cb_out = alloc_cmd(&vk);
   VkFence fence = alloc_fence(&vk, false);
   uint8_t *expect_frame = malloc(offer.frame_bytes);
   if (!expect_frame)
      fatal(EXIT_PROTO, "out of memory");

   struct msg_hello hello = {
      .magic = CPIO_MAGIC, .type = MSG_HELLO, .status = 0,
   };
   set_state("sending MSG_HELLO");
   if (send_msg(sock, &hello, sizeof(hello), NULL, 0) < 0)
      fatal(EXIT_PROTO, "sendmsg(MSG_HELLO): %s", strerror(errno));

   unsigned frames = 0;
   for (;;) {
      set_state("waiting MSG_READY (%u frames done)", frames);
      int r = wait_readable(sock, now_ms() + 30000.0);
      if (r == 0)
         fatal(EXIT_PROTO, "no MSG_READY within 30 s");
      if (r < 0)
         fatal(EXIT_PROTO, "producer hung up");
      ssize_t n = recv_msg(sock, &m, sizeof(m), NULL, NULL);
      if (n == 0)
         fatal(EXIT_PROTO, "EOF from producer after %u frames", frames);
      if (n < 0)
         fatal(EXIT_PROTO, "recvmsg: %s", strerror(errno));
      if (m.hdr.magic != CPIO_MAGIC)
         fatal(EXIT_PROTO, "bad magic 0x%08x", m.hdr.magic);
      if (m.hdr.type == MSG_BYE) {
         logf_("MSG_BYE after %u frames, status %u", m.bye.frames, m.bye.status);
         break;
      }
      if (m.hdr.type == MSG_ERROR) {
         m.error.text[sizeof(m.error.text) - 1] = 0;
         fatal(EXIT_PROTO, "producer MSG_ERROR %u: %s", m.error.code, m.error.text);
      }
      if (m.hdr.type != MSG_READY)
         fatal(EXIT_PROTO, "expected MSG_READY, got %u", m.hdr.type);

      uint32_t slot = m.ready.slot, seq = m.ready.seq;
      uint64_t tv = m.ready.timeline_value;
      if (slot >= offer.slots)
         fatal(EXIT_PROTO, "MSG_READY names slot %u", slot);
      struct fake_slot *fs = &slots[slot];
      set_state("reading slot=%u seq=%u", slot, seq);

      /* Acquire from the producer, copy the frame out. */
      VK_CHECK(vkResetCommandBuffer(cb_in, 0));
      VkCommandBufferBeginInfo bi = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      };
      VK_CHECK(vkBeginCommandBuffer(cb_in, &bi));
      buffer_barrier(cb_in, fs->frame_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_READ_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      buffer_barrier(cb_in, fs->result_buf, 0, VK_WHOLE_SIZE, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_QUEUE_FAMILY_EXTERNAL, vk.qfam,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);
      VkBufferCopy c = { 0, 0, offer.frame_bytes };
      vkCmdCopyBuffer(cb_in, fs->frame_buf, stage_in.buf, 1, &c);
      VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(cb_in, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
      VK_CHECK(vkEndCommandBuffer(cb_in));

      {
         VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
         VkTimelineSemaphoreSubmitInfoKHR tsi = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &tv,
         };
         VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = timeline ? &tsi : NULL,
            .waitSemaphoreCount = timeline ? 1 : 0,
            .pWaitSemaphores = &sem_render,
            .pWaitDstStageMask = &ws,
            .commandBufferCount = 1,
            .pCommandBuffers = &cb_in,
         };
         VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, fence));
      }
      wait_fence(&vk, fence, "consumer read");

      /* Each side owns half the verdict: the consumer checks the frame it
       * was given.  With --no-producer-wait this is what sees the torn
       * frame or the wrong stamp. */
      const uint8_t *in = stage_in.map;
      pattern_frame(expect_frame, offer.width, offer.height, seq);
      uint64_t fbad = 0;
      int64_t ffirst = -1;
      for (uint64_t i = 0; i < offer.frame_bytes; i++) {
         if (in[i] != expect_frame[i]) {
            fbad++;
            if (ffirst < 0)
               ffirst = (int64_t)i;
         }
      }
      if (fbad) {
         char t[256];
         uint32_t got_stamp;
         memcpy(&got_stamp, in, 4);
         snprintf(t, sizeof(t), "frame seq %u slot %u: %llu of %u bytes wrong, "
                  "first at %lld, stamp says %u", seq, slot,
                  (unsigned long long)fbad, offer.frame_bytes,
                  (long long)ffirst, got_stamp);
         logf_("FAILED: %s", t);
         send_error(sock, EXIT_VERIFY, t);
         return EXIT_VERIFY;
      }

      /* The arithmetic: result[i] = (frame[i] + 1) & 255, and the stamp. */
      uint8_t *out = stage_out.map;
      uint64_t n_out = req.result_bytes;
      for (uint64_t i = 0; i < n_out; i++)
         out[i] = (uint8_t)((in[i] + 1) & 255u);
      stamp_bytes(seq, out);

      VK_CHECK(vkResetCommandBuffer(cb_out, 0));
      VK_CHECK(vkBeginCommandBuffer(cb_out, &bi));
      if (!opt->fake_no_write) {
         VkBufferCopy c2 = { 0, 0, n_out };
         vkCmdCopyBuffer(cb_out, stage_out.buf, fs->result_buf, 1, &c2);
      }
      /* Release both back to the producer. */
      buffer_barrier(cb_out, fs->result_buf, 0, VK_WHOLE_SIZE,
                     VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                     vk.qfam, VK_QUEUE_FAMILY_EXTERNAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      buffer_barrier(cb_out, fs->frame_buf, 0, VK_WHOLE_SIZE,
                     VK_ACCESS_TRANSFER_READ_BIT, 0,
                     vk.qfam, VK_QUEUE_FAMILY_EXTERNAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      VK_CHECK(vkEndCommandBuffer(cb_out));

      {
         uint64_t sv = seq;
         VkTimelineSemaphoreSubmitInfoKHR tsi = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &sv,
         };
         VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = timeline ? &tsi : NULL,
            .commandBufferCount = 1,
            .pCommandBuffers = &cb_out,
            .signalSemaphoreCount = timeline ? 1 : 0,
            .pSignalSemaphores = &sem_consume,
         };
         VK_CHECK(vkQueueSubmit(vk.queue, 1, &si, fence));
      }
      /* host mode: the equivalent of ev.synchronize() before replying. */
      wait_fence(&vk, fence, "consumer write");

      struct msg_result res;
      memset(&res, 0, sizeof(res));
      res.magic = CPIO_MAGIC;
      res.type = MSG_RESULT;
      res.slot = slot;
      res.seq = seq;
      res.status = 0;
      res.result_ndim = 3;
      res.result_shape[0] = offer.height;
      res.result_shape[1] = offer.width;
      res.result_shape[2] = 4;
      /* --fake-overrun writes the whole result but declares half of it, so
       * the bytes above the declaration must still be poison and are not.
       * That is what the poison tail is for. */
      res.result_bytes = opt->fake_overrun ? n_out / 2 : n_out;
      res.timeline_value = timeline ? seq : 0;
      set_state("sending MSG_RESULT slot=%u seq=%u", slot, seq);
      if (send_msg(sock, &res, sizeof(res), NULL, 0) < 0)
         fatal(EXIT_PROTO, "sendmsg(MSG_RESULT): %s", strerror(errno));
      frames++;
   }

   vkDeviceWaitIdle(vk.dev);
   logf_("done, %u frames%s", frames,
         opt->fake_no_write ? " (wrote nothing on purpose)" : "");
   return EXIT_PASS;
}

/* ------------------------------------------------------------------ */
/* Process plumbing                                                   */

/* Split a command line the way a shell would for the simple cases:
 * whitespace separated, single and double quotes, backslash escapes. */
static char **
split_cmd(const char *cmd, int *argc_out)
{
   size_t len = strlen(cmd);
   char *buf = malloc(len + 1);
   char **argv = calloc(len / 2 + 4, sizeof(char *));
   int argc = 0;
   size_t o = 0;
   size_t i = 0;

   while (i < len) {
      while (i < len && (cmd[i] == ' ' || cmd[i] == '\t'))
         i++;
      if (i >= len)
         break;
      argv[argc++] = buf + o;
      while (i < len && cmd[i] != ' ' && cmd[i] != '\t') {
         if (cmd[i] == '\'' || cmd[i] == '"') {
            char q = cmd[i++];
            while (i < len && cmd[i] != q)
               buf[o++] = cmd[i++];
            if (i < len)
               i++;
         } else if (cmd[i] == '\\' && i + 1 < len) {
            i++;
            buf[o++] = cmd[i++];
         } else {
            buf[o++] = cmd[i++];
         }
      }
      buf[o++] = 0;
   }
   argv[argc] = NULL;
   *argc_out = argc;
   return argv;
}

static int
spawn_peer(struct options *opt, const char *cmd)
{
   int sv[2];
   if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) < 0)
      fatal(EXIT_PROTO, "socketpair(SOCK_SEQPACKET): %s", strerror(errno));

   int argc = 0;
   char **cargv = split_cmd(cmd, &argc);
   if (argc == 0)
      fatal(EXIT_PROTO, "--exec was given an empty command");

   char fdstr[16];
   snprintf(fdstr, sizeof(fdstr), "%d", sv[1]);
   char **full = calloc(argc + 2, sizeof(char *));
   for (int i = 0; i < argc; i++)
      full[i] = cargv[i];
   full[argc] = fdstr;            /* the child fd, appended last */
   full[argc + 1] = NULL;

   /* Both processes in one process group, so one kill ends the run. */
   setpgid(0, 0);

   pid_t pid = fork();
   if (pid < 0)
      fatal(EXIT_PROTO, "fork: %s", strerror(errno));
   if (pid == 0) {
      close(sv[0]);
      int flags = fcntl(sv[1], F_GETFD);
      if (flags >= 0)
         fcntl(sv[1], F_SETFD, flags & ~FD_CLOEXEC);
      execvp(full[0], full);
      fprintf(stderr, "exec %s: %s\n", full[0], strerror(errno));
      _exit(127);
   }
   close(sv[1]);
   g_child_pid = pid;
   logf_("exec: %s %s (pid %d)", full[0], fdstr, (int)pid);
   return sv[0];
}

static int
listen_socket(struct options *opt)
{
   int ls = socket(AF_UNIX, SOCK_SEQPACKET, 0);
   if (ls < 0)
      fatal(EXIT_PROTO, "socket(AF_UNIX, SOCK_SEQPACKET): %s", strerror(errno));
   struct sockaddr_un sa;
   memset(&sa, 0, sizeof(sa));
   sa.sun_family = AF_UNIX;
   snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", opt->socket_path);
   unlink(opt->socket_path);
   if (bind(ls, (struct sockaddr *)&sa, sizeof(sa)) < 0)
      fatal(EXIT_PROTO, "bind(%s): %s", opt->socket_path, strerror(errno));
   if (listen(ls, 1) < 0)
      fatal(EXIT_PROTO, "listen: %s", strerror(errno));
   logf_("listening on %s, waiting up to 60 s for the consumer",
         opt->socket_path);
   set_state("waiting for a connection on %s", opt->socket_path);
   int r = wait_readable(ls, now_ms() + HANDSHAKE_MS);
   if (r <= 0)
      fatal(EXIT_PROTO, "no consumer connected to %s within 60 s",
            opt->socket_path);
   int s = accept(ls, NULL, NULL);
   if (s < 0)
      fatal(EXIT_PROTO, "accept: %s", strerror(errno));
   close(ls);
   return s;
}

static void
usage(const char *argv0)
{
   printf(
"usage: %s [options]\n"
"  --frames N            frames to run after MSG_HELLO (default 1000)\n"
"  --slots N             ring slots, 1..8 (default 3)\n"
"  --width W --height H  frame size (default 512 512)\n"
"  --sync host|timeline  synchronisation mode (default timeline)\n"
"  --op add1|sobel|blur  passed to the consumer; only add1 is verified\n"
"  --exec \"CMD\"          fork/exec CMD with the socket fd appended last\n"
"  --socket PATH         bind and listen instead, for debugging\n"
"  --png-every N         write PNGs every N frames (default 0, off)\n"
"  --png-dir DIR         where the PNGs go (default .)\n"
"  --no-producer-wait    negative control: reuse a slot early, expect failure\n"
"  --seed N              reserved, accepted and ignored\n"
"  --self-test           run against this binary in fake-consumer mode\n"
"  --self-test-nowrite   the same, with a consumer that never writes\n"
"  --self-test-overrun   the same, with a consumer that writes past its\n"
"                        own declared result_bytes\n"
"exit: 0 pass, 1 verification failure, 2 protocol/timeout/peer died,\n"
"      3 the driver does not report OPAQUE_FD\n", argv0);
}

int
main(int argc, char **argv)
{
   struct options opt = {
      .frames = 1000,
      .slots = 3,
      .width = 512,
      .height = 512,
      .sync = SYNC_TIMELINE,
      .op = "add1",
      .png_dir = ".",
      .fake_consumer_fd = -1,
      .argv0 = argv[0],
   };
   bool fake = false, self_test_nowrite = false, self_test_overrun = false;

   signal(SIGPIPE, SIG_IGN);
   setvbuf(stdout, NULL, _IOLBF, 0);

   for (int i = 1; i < argc; i++) {
      const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fatal(EXIT_PROTO, "%s needs a value", a), ""))
      if (!strcmp(a, "--frames"))            opt.frames = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--slots"))        opt.slots = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--width"))        opt.width = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--height"))       opt.height = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--sync")) {
         const char *v = NEXT();
         if (!strcmp(v, "host"))
            opt.sync = SYNC_HOST;
         else if (!strcmp(v, "timeline"))
            opt.sync = SYNC_TIMELINE;
         else
            fatal(EXIT_PROTO, "--sync takes host or timeline, not %s", v);
      }
      else if (!strcmp(a, "--op"))           opt.op = NEXT();
      else if (!strcmp(a, "--exec"))         opt.exec_cmd = NEXT();
      else if (!strcmp(a, "--socket"))       opt.socket_path = NEXT();
      else if (!strcmp(a, "--png-every"))    opt.png_every = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--png-dir"))      opt.png_dir = NEXT();
      else if (!strcmp(a, "--no-producer-wait")) opt.no_producer_wait = true;
      else if (!strcmp(a, "--seed"))         opt.seed = strtoul(NEXT(), NULL, 0);
      else if (!strcmp(a, "--self-test"))    opt.self_test = true;
      else if (!strcmp(a, "--self-test-nowrite")) { opt.self_test = true; self_test_nowrite = true; }
      else if (!strcmp(a, "--self-test-overrun")) { opt.self_test = true; self_test_overrun = true; }
      else if (!strcmp(a, "--fake-consumer")) fake = true;
      else if (!strcmp(a, "--fake-no-write")) opt.fake_no_write = true;
      else if (!strcmp(a, "--fake-overrun")) opt.fake_overrun = true;
      else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
      else if (a[0] >= '0' && a[0] <= '9')   opt.fake_consumer_fd = atoi(a);
      else fatal(EXIT_PROTO, "unknown option %s (try --help)", a);
#undef NEXT
   }

   if (opt.slots < 1 || opt.slots > 8)
      fatal(EXIT_PROTO, "--slots must be 1..8, got %u", opt.slots);
   if (opt.width < 2 || opt.height < 1)
      fatal(EXIT_PROTO, "--width/--height too small (%ux%u)", opt.width, opt.height);
   if (strcmp(opt.op, "add1") && strcmp(opt.op, "sobel") && strcmp(opt.op, "blur"))
      fatal(EXIT_PROTO, "--op takes add1, sobel or blur, not %s", opt.op);

   if (fake) {
      if (opt.fake_consumer_fd < 0)
         fatal(EXIT_PROTO, "--fake-consumer needs the socket fd as the last argument");
      return fake_consumer_run(&opt, opt.fake_consumer_fd);
   }

   int sock;
   char selfcmd[1024];
   if (opt.self_test) {
      snprintf(selfcmd, sizeof(selfcmd), "%s --fake-consumer --sync %s%s%s",
               argv[0], opt.sync == SYNC_TIMELINE ? "timeline" : "host",
               self_test_nowrite ? " --fake-no-write" : "",
               self_test_overrun ? " --fake-overrun" : "");
      sock = spawn_peer(&opt, selfcmd);
   } else if (opt.exec_cmd) {
      /* The op is passed through to the consumer, unless the caller already
       * put one in the command. */
      char cmd[4096];
      if (strstr(opt.exec_cmd, "--op"))
         snprintf(cmd, sizeof(cmd), "%s", opt.exec_cmd);
      else
         snprintf(cmd, sizeof(cmd), "%s --op %s", opt.exec_cmd, opt.op);
      sock = spawn_peer(&opt, cmd);
   } else if (opt.socket_path) {
      sock = listen_socket(&opt);
   } else {
      fatal(EXIT_PROTO, "one of --exec, --socket or --self-test is required");
   }

   int rc = producer_run(&opt, sock);

   /* Give the peer a moment to exit, then make sure nothing is left. */
   if (g_child_pid > 0) {
      int status = 0;
      double deadline = now_ms() + 10000.0;
      pid_t r = 0;
      while (now_ms() < deadline) {
         r = waitpid(g_child_pid, &status, WNOHANG);
         if (r == g_child_pid)
            break;
         struct timespec ts = { 0, 5 * 1000 * 1000 };
         nanosleep(&ts, NULL);
      }
      if (r == g_child_pid) {
         if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
            logf_("consumer exited %d", WEXITSTATUS(status));
            if (rc == EXIT_PASS)
               rc = WEXITSTATUS(status) == EXIT_VERIFY ? EXIT_VERIFY : EXIT_PROTO;
         } else if (WIFSIGNALED(status)) {
            logf_("consumer died on signal %d", WTERMSIG(status));
            if (rc == EXIT_PASS)
               rc = EXIT_PROTO;
         }
      } else {
         logf_("consumer did not exit within 10 s; killing the process group");
         kill(-getpgrp(), SIGTERM);
      }
      g_child_pid = -1;
   }

   printf("renderer: exit %d (%s)\n", rc,
          rc == EXIT_PASS ? "pass" :
          rc == EXIT_VERIFY ? "verification failure" :
          rc == EXIT_UNSUPPORTED ? "unsupported" : "protocol error");
   return rc;
}
