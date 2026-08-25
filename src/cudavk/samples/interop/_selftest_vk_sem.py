#!/usr/bin/env python3
"""_selftest_vk_sem.py -- exported Vulkan TIMELINE semaphores, in ctypes.

TEST SCAFFOLD.  Not part of the sample: nothing in the shipped path imports
it, and the runner does not use it.  It exists so the consumer can be tested
without the producer.

The self-test producer has no Vulkan renderer, but the consumer's timeline path
cannot be tested at all without a real VK_SEMAPHORE_TYPE_TIMELINE exported over
VK_KHR_external_semaphore_fd: CUDA has no way to create one.  So this creates
two of them through libvulkan.so.1 directly, exports one fd each and drives them
from the host with vkSignalSemaphore / vkWaitSemaphores.

That is enough to test everything the consumer does with them:
cuImportExternalSemaphore as TIMELINE_SEMAPHORE_FD,
cuWaitExternalSemaphoresAsync before the op and
cuSignalExternalSemaphoresAsync after it.  What it does not test is a semaphore
signalled by a real vkQueueSubmit, which needs the renderer.

Test scaffold.  Not part of the sample's shipped path.
"""

import ctypes

vk = ctypes.CDLL("libvulkan.so.1")

VkInstance = VkPhysicalDevice = VkDevice = ctypes.c_void_p
VkSemaphore = ctypes.c_uint64

ST_APPLICATION_INFO = 0
ST_INSTANCE_CREATE_INFO = 1
ST_DEVICE_QUEUE_CREATE_INFO = 2
ST_DEVICE_CREATE_INFO = 3
ST_SEMAPHORE_CREATE_INFO = 9
ST_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
ST_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004
ST_EXPORT_SEMAPHORE_CREATE_INFO = 1000077000
ST_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES = 1000207000
ST_SEMAPHORE_TYPE_CREATE_INFO = 1000207002
ST_SEMAPHORE_WAIT_INFO = 1000207004
ST_SEMAPHORE_SIGNAL_INFO = 1000207005
ST_SEMAPHORE_GET_FD_INFO_KHR = 1000079001

VK_SEMAPHORE_TYPE_TIMELINE = 1
VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT = 1
VK_API_VERSION_1_2 = (1 << 22) | (2 << 12)


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
                ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class VkPhysicalDeviceIDProperties(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("deviceUUID", ctypes.c_uint8 * 16),
                ("driverUUID", ctypes.c_uint8 * 16),
                ("deviceLUID", ctypes.c_uint8 * 8),
                ("deviceNodeMask", ctypes.c_uint32),
                ("deviceLUIDValid", ctypes.c_uint32)]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32), ("queueFamilyIndex", ctypes.c_uint32),
                ("queueCount", ctypes.c_uint32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("queueCreateInfoCount", ctypes.c_uint32),
                ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
                ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p),
                ("pEnabledFeatures", ctypes.c_void_p)]


class VkPhysicalDeviceTimelineSemaphoreFeatures(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("timelineSemaphore", ctypes.c_uint32)]


class VkExportSemaphoreCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("handleTypes", ctypes.c_uint32)]


class VkSemaphoreTypeCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("semaphoreType", ctypes.c_int), ("initialValue", ctypes.c_uint64)]


class VkSemaphoreCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32)]


class VkSemaphoreGetFdInfoKHR(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("semaphore", VkSemaphore), ("handleType", ctypes.c_uint32)]


class VkSemaphoreSignalInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("semaphore", VkSemaphore), ("value", ctypes.c_uint64)]


class VkSemaphoreWaitInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("semaphoreCount", ctypes.c_uint32),
                ("pSemaphores", ctypes.POINTER(VkSemaphore)),
                ("pValues", ctypes.POINTER(ctypes.c_uint64))]


class VkError(RuntimeError):
    pass


def _ck(res, what):
    if res != 0:
        raise VkError("%s -> VkResult %d" % (what, res))


class VkTimeline:
    """One Vulkan device and two exported timeline semaphores."""

    def __init__(self, want_uuid=None):
        app = VkApplicationInfo(ST_APPLICATION_INFO, None, b"cpvk_interop_selftest",
                                1, b"none", 1, VK_API_VERSION_1_2)
        ici = VkInstanceCreateInfo()
        ctypes.memset(ctypes.byref(ici), 0, ctypes.sizeof(ici))
        ici.sType = ST_INSTANCE_CREATE_INFO
        ici.pApplicationInfo = ctypes.pointer(app)
        self.instance = VkInstance()
        _ck(vk.vkCreateInstance(ctypes.byref(ici), None, ctypes.byref(self.instance)),
            "vkCreateInstance")

        n = ctypes.c_uint32(0)
        _ck(vk.vkEnumeratePhysicalDevices(self.instance, ctypes.byref(n), None),
            "vkEnumeratePhysicalDevices")
        pds = (VkPhysicalDevice * n.value)()
        _ck(vk.vkEnumeratePhysicalDevices(self.instance, ctypes.byref(n), pds),
            "vkEnumeratePhysicalDevices")

        self.phys = None
        for i in range(n.value):
            uuid = self.device_uuid(pds[i])
            if want_uuid is None or uuid == want_uuid:
                self.phys, self.uuid = pds[i], uuid
                break
        if self.phys is None:
            raise VkError("no VkPhysicalDevice has UUID %s" % want_uuid.hex())

        prio = (ctypes.c_float * 1)(1.0)
        qci = VkDeviceQueueCreateInfo(ST_DEVICE_QUEUE_CREATE_INFO, None, 0, 0, 1,
                                      prio)
        feat = VkPhysicalDeviceTimelineSemaphoreFeatures(
            ST_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, None, 1)
        ext = (ctypes.c_char_p * 1)(b"VK_KHR_external_semaphore_fd")
        dci = VkDeviceCreateInfo()
        ctypes.memset(ctypes.byref(dci), 0, ctypes.sizeof(dci))
        dci.sType = ST_DEVICE_CREATE_INFO
        dci.pNext = ctypes.cast(ctypes.pointer(feat), ctypes.c_void_p)
        dci.queueCreateInfoCount = 1
        dci.pQueueCreateInfos = ctypes.pointer(qci)
        dci.enabledExtensionCount = 1
        dci.ppEnabledExtensionNames = ctypes.cast(ext, ctypes.c_void_p)
        self.device = VkDevice()
        _ck(vk.vkCreateDevice(self.phys, ctypes.byref(dci), None,
                              ctypes.byref(self.device)), "vkCreateDevice")

        addr = vk.vkGetDeviceProcAddr
        addr.restype = ctypes.c_void_p
        p = addr(self.device, b"vkGetSemaphoreFdKHR")
        if not p:
            raise VkError("vkGetSemaphoreFdKHR is not available")
        self._get_fd = ctypes.CFUNCTYPE(
            ctypes.c_int, VkDevice, ctypes.POINTER(VkSemaphoreGetFdInfoKHR),
            ctypes.POINTER(ctypes.c_int))(p)
        self.semaphores = []

    @staticmethod
    def device_uuid(phys):
        idp = VkPhysicalDeviceIDProperties()
        ctypes.memset(ctypes.byref(idp), 0, ctypes.sizeof(idp))
        idp.sType = ST_PHYSICAL_DEVICE_ID_PROPERTIES
        # VkPhysicalDeviceProperties2 is large and only its first two fields are
        # written by this caller; the driver fills the rest of the buffer.
        buf = (ctypes.c_uint8 * 4096)()
        ctypes.memmove(buf, ctypes.byref(ctypes.c_int(ST_PHYSICAL_DEVICE_PROPERTIES_2)), 4)
        pnext = ctypes.cast(ctypes.pointer(idp), ctypes.c_void_p)
        ctypes.memmove(ctypes.byref(buf, 8), ctypes.byref(pnext), 8)
        vk.vkGetPhysicalDeviceProperties2(phys, buf)
        return bytes(bytearray(idp.deviceUUID))

    def create_timeline(self):
        """A timeline semaphore starting at 0, exportable as an OPAQUE_FD."""
        exp = VkExportSemaphoreCreateInfo(ST_EXPORT_SEMAPHORE_CREATE_INFO, None,
                                          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
        typ = VkSemaphoreTypeCreateInfo(ST_SEMAPHORE_TYPE_CREATE_INFO,
                                        ctypes.cast(ctypes.pointer(exp), ctypes.c_void_p),
                                        VK_SEMAPHORE_TYPE_TIMELINE, 0)
        sci = VkSemaphoreCreateInfo(ST_SEMAPHORE_CREATE_INFO,
                                    ctypes.cast(ctypes.pointer(typ), ctypes.c_void_p), 0)
        sem = VkSemaphore(0)
        _ck(vk.vkCreateSemaphore(self.device, ctypes.byref(sci), None,
                                 ctypes.byref(sem)), "vkCreateSemaphore")
        self.semaphores.append(sem)
        return sem

    def export_fd(self, sem):
        info = VkSemaphoreGetFdInfoKHR(ST_SEMAPHORE_GET_FD_INFO_KHR, None, sem,
                                       VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
        fd = ctypes.c_int(-1)
        _ck(self._get_fd(self.device, ctypes.byref(info), ctypes.byref(fd)),
            "vkGetSemaphoreFdKHR")
        return fd.value

    def signal(self, sem, value):
        info = VkSemaphoreSignalInfo(ST_SEMAPHORE_SIGNAL_INFO, None, sem, value)
        _ck(vk.vkSignalSemaphore(self.device, ctypes.byref(info)),
            "vkSignalSemaphore")

    def counter(self, sem):
        v = ctypes.c_uint64(0)
        _ck(vk.vkGetSemaphoreCounterValue(self.device, sem, ctypes.byref(v)),
            "vkGetSemaphoreCounterValue")
        return v.value

    def wait(self, sem, value, timeout_ns):
        sems = (VkSemaphore * 1)(sem)
        vals = (ctypes.c_uint64 * 1)(value)
        info = VkSemaphoreWaitInfo(ST_SEMAPHORE_WAIT_INFO, None, 0, 1, sems, vals)
        r = vk.vkWaitSemaphores(self.device, ctypes.byref(info),
                                ctypes.c_uint64(timeout_ns))
        if r == 2:            # VK_TIMEOUT
            return False
        _ck(r, "vkWaitSemaphores")
        return True

    def close(self):
        for s in self.semaphores:
            vk.vkDestroySemaphore(self.device, s, None)
        self.semaphores = []
        if self.device:
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None


if __name__ == "__main__":
    t = VkTimeline()
    print("physical device uuid", t.uuid.hex())
    s = t.create_timeline()
    fd = t.export_fd(s)
    print("timeline semaphore exported as fd", fd, "counter", t.counter(s))
    t.signal(s, 7)
    print("after signal 7, counter", t.counter(s), "wait(7):",
          t.wait(s, 7, 10 ** 9))
    import os
    os.close(fd)
    t.close()
