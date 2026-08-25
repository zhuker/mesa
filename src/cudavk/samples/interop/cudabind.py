"""cudabind -- a thin ctypes binding to libcuda.so.1 for the interop consumer.

PyTorch has no external-memory API (CUDA_INTEROP.md in this tree's docs, "Consuming the
frame in PyTorch"), so the driver API is called directly here and the resulting
CUdeviceptr is wrapped with __cuda_array_interface__.

Symbol naming.  cuda.h #defines a handful of base names onto _v2 entry points;
plain dlsym of such a base name gives the OLD v1 ABI.  The remap list below is
EXPLICIT on purpose.  libcuda 580 also exports symbols that are not the public
API at all -- cuCtxSynchronize_v2 is the recorded example: it returns 709
CUDA_ERROR_CONTEXT_IS_DESTROYED on a perfectly healthy context.  Nothing is
resolved by guessing a suffix.
"""

import ctypes

_lib = ctypes.CDLL("libcuda.so.1")

# Names that cuda.h #defines onto their _v2 entry point.  Checked against
# /usr/local/cuda/include/cuda.h, not guessed.  cuCtxSynchronize and
# cuDeviceGetUuid are deliberately NOT here: cuda.h leaves both alone.
_V2 = {
    "cuCtxCreate", "cuCtxDestroy", "cuCtxPushCurrent", "cuCtxPopCurrent",
    "cuDevicePrimaryCtxRelease",
    "cuMemAlloc", "cuMemFree", "cuMemcpyHtoD", "cuMemcpyDtoH",
    "cuMemcpyDtoD", "cuMemcpyDtoDAsync", "cuMemsetD8", "cuMemsetD32",
    "cuStreamDestroy",
}
# NOT in that set, checked in cuda.h rather than assumed: cuMemsetD8Async and
# cuStreamCreate have no _v2 at all, cuCtxSynchronize has an exported _v2 that
# is not the public API, and cuDeviceGetUuid_v2 is a different function (MIG),
# not a newer ABI for the same one.


def _sym(name):
    real = name + "_v2" if name in _V2 else name
    return getattr(_lib, real)


# ---------------------------------------------------------------- types
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUdeviceptr = ctypes.c_ulonglong
CUcontext = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUexternalMemory = ctypes.c_void_p
CUexternalSemaphore = ctypes.c_void_p
CUmemGenericAllocationHandle = ctypes.c_ulonglong

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0

CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9

CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_char * 16)]


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocationProp(ctypes.Structure):
    class _Flags(ctypes.Structure):
        _fields_ = [("compressionType", ctypes.c_ubyte),
                    ("gpuDirectRDMACapable", ctypes.c_ubyte),
                    ("usage", ctypes.c_ushort),
                    ("reserved", ctypes.c_ubyte * 4)]
    _fields_ = [("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _Flags)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _Win32Handle),
                ("nvSciBufObject", ctypes.c_void_p)]


class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _HandleUnion),
                ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _HandleUnion),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class _Fence(ctypes.Structure):
    _fields_ = [("value", ctypes.c_ulonglong)]


class _NvSciSync(ctypes.Union):
    _fields_ = [("fence", ctypes.c_void_p), ("reserved", ctypes.c_ulonglong)]


class _SignalParams(ctypes.Structure):
    class _KeyedMutex(ctypes.Structure):
        _fields_ = [("key", ctypes.c_ulonglong)]
    _fields_ = [("fence", _Fence), ("nvSciSync", _NvSciSync),
                ("keyedMutex", _KeyedMutex), ("reserved", ctypes.c_uint * 12)]


class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS(ctypes.Structure):
    _fields_ = [("params", _SignalParams), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


class _WaitParams(ctypes.Structure):
    class _KeyedMutex(ctypes.Structure):
        _fields_ = [("key", ctypes.c_ulonglong), ("timeoutMs", ctypes.c_uint)]
    _fields_ = [("fence", _Fence), ("nvSciSync", _NvSciSync),
                ("keyedMutex", _KeyedMutex), ("reserved", ctypes.c_uint * 10)]


class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(ctypes.Structure):
    _fields_ = [("params", _WaitParams), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


# ------------------------------------------------------------- prototypes
# Every function used is given argtypes.  Without them ctypes passes a Python
# int as a C int and silently truncates every 64-bit size and pointer.
_P = {
    "cuInit": [ctypes.c_uint],
    "cuGetErrorName": [CUresult, ctypes.POINTER(ctypes.c_char_p)],
    "cuGetErrorString": [CUresult, ctypes.POINTER(ctypes.c_char_p)],
    "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
    "cuDeviceGet": [ctypes.POINTER(CUdevice), ctypes.c_int],
    "cuDeviceGetUuid": [ctypes.POINTER(CUuuid), CUdevice],
    "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, CUdevice],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(CUcontext), CUdevice],
    "cuDevicePrimaryCtxRelease": [CUdevice],
    "cuCtxGetCurrent": [ctypes.POINTER(CUcontext)],
    "cuCtxSetCurrent": [CUcontext],
    "cuCtxSynchronize": [],
    "cuPointerGetAttribute": [ctypes.c_void_p, ctypes.c_int, CUdeviceptr],
    "cuImportExternalMemory": [ctypes.POINTER(CUexternalMemory),
                               ctypes.POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)],
    "cuExternalMemoryGetMappedBuffer": [ctypes.POINTER(CUdeviceptr), CUexternalMemory,
                                        ctypes.POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)],
    "cuDestroyExternalMemory": [CUexternalMemory],
    "cuImportExternalSemaphore": [ctypes.POINTER(CUexternalSemaphore),
                                  ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)],
    "cuDestroyExternalSemaphore": [CUexternalSemaphore],
    "cuSignalExternalSemaphoresAsync": [ctypes.POINTER(CUexternalSemaphore),
                                        ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS),
                                        ctypes.c_uint, CUstream],
    "cuWaitExternalSemaphoresAsync": [ctypes.POINTER(CUexternalSemaphore),
                                      ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS),
                                      ctypes.c_uint, CUstream],
    # VMM -- used by the self test's fake producer only.
    "cuMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(CUmemAllocationProp), ctypes.c_int],
    "cuMemCreate": [ctypes.POINTER(CUmemGenericAllocationHandle), ctypes.c_size_t,
                    ctypes.POINTER(CUmemAllocationProp), ctypes.c_ulonglong],
    "cuMemAddressReserve": [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t, ctypes.c_size_t,
                            CUdeviceptr, ctypes.c_ulonglong],
    "cuMemMap": [CUdeviceptr, ctypes.c_size_t, ctypes.c_size_t,
                 CUmemGenericAllocationHandle, ctypes.c_ulonglong],
    "cuMemSetAccess": [CUdeviceptr, ctypes.c_size_t, ctypes.POINTER(CUmemAccessDesc),
                       ctypes.c_size_t],
    "cuMemExportToShareableHandle": [ctypes.c_void_p, CUmemGenericAllocationHandle,
                                     ctypes.c_int, ctypes.c_ulonglong],
    "cuMemUnmap": [CUdeviceptr, ctypes.c_size_t],
    "cuMemRelease": [CUmemGenericAllocationHandle],
    "cuMemAddressFree": [CUdeviceptr, ctypes.c_size_t],
    "cuMemcpyHtoD": [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t],
    "cuMemcpyDtoH": [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t],
    "cuMemsetD8": [CUdeviceptr, ctypes.c_ubyte, ctypes.c_size_t],
    "cuMemsetD8Async": [CUdeviceptr, ctypes.c_ubyte, ctypes.c_size_t, CUstream],
    "cuMemcpyDtoDAsync": [CUdeviceptr, CUdeviceptr, ctypes.c_size_t, CUstream],
    "cuMemAlloc": [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t],
    "cuMemFree": [CUdeviceptr],
    "cuStreamCreate": [ctypes.POINTER(CUstream), ctypes.c_uint],
    "cuStreamSynchronize": [CUstream],
    "cuStreamDestroy": [CUstream],
}

_fn = {}
for _name, _args in _P.items():
    _f = _sym(_name)
    _f.restype = CUresult
    _f.argtypes = _args
    _fn[_name] = _f


def _get(name):
    return _fn[name]


# ---------------------------------------------------------------- errors
class CudaError(RuntimeError):
    def __init__(self, call, res):
        self.res = res
        super().__init__("%s -> %s" % (call, err_str(res)))


def err_str(res):
    name = ctypes.c_char_p()
    text = ctypes.c_char_p()
    _fn["cuGetErrorName"](res, ctypes.byref(name))
    _fn["cuGetErrorString"](res, ctypes.byref(text))
    return "%d %s (%s)" % (res,
                           name.value.decode() if name.value else "?",
                           text.value.decode() if text.value else "?")


def check(res, what):
    if res != 0:
        raise CudaError(what, res)
    return res


# ---------------------------------------------------------------- device
def init():
    check(_fn["cuInit"](0), "cuInit")


def device_count():
    n = ctypes.c_int()
    check(_fn["cuDeviceGetCount"](ctypes.byref(n)), "cuDeviceGetCount")
    return n.value


def device_get(ordinal):
    d = CUdevice()
    check(_fn["cuDeviceGet"](ctypes.byref(d), ordinal), "cuDeviceGet")
    return d


def device_uuid(dev):
    u = CUuuid()
    check(_fn["cuDeviceGetUuid"](ctypes.byref(u), dev), "cuDeviceGetUuid")
    return bytes(bytearray(u.bytes[:16]))


def device_name(dev):
    buf = ctypes.create_string_buffer(256)
    check(_fn["cuDeviceGetName"](buf, 256, dev), "cuDeviceGetName")
    return buf.value.decode()


def find_device_by_uuid(uuid16):
    """Return (ordinal, CUdevice) whose cuDeviceGetUuid matches, else (None, None).

    Matching by index is forbidden: the Vulkan physical device order and the
    CUDA device order are unrelated in general.
    """
    for i in range(device_count()):
        dev = device_get(i)
        if device_uuid(dev) == uuid16:
            return i, dev
    return None, None


def ctx_get_current():
    c = CUcontext()
    check(_fn["cuCtxGetCurrent"](ctypes.byref(c)), "cuCtxGetCurrent")
    return c.value


def primary_ctx_retain(dev):
    c = CUcontext()
    check(_fn["cuDevicePrimaryCtxRetain"](ctypes.byref(c), dev),
          "cuDevicePrimaryCtxRetain")
    return c.value


def primary_ctx_release(dev):
    check(_fn["cuDevicePrimaryCtxRelease"](dev), "cuDevicePrimaryCtxRelease")


def ctx_synchronize():
    check(_fn["cuCtxSynchronize"](), "cuCtxSynchronize")


def pointer_range(ptr):
    """(range_start, range_size) of an allocation, for a bounds cross-check."""
    start = ctypes.c_ulonglong(0)
    size = ctypes.c_ulonglong(0)
    r1 = _fn["cuPointerGetAttribute"](ctypes.byref(start),
                                      CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                      CUdeviceptr(ptr))
    r2 = _fn["cuPointerGetAttribute"](ctypes.byref(size),
                                      CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                                      CUdeviceptr(ptr))
    if r1 != 0 or r2 != 0:
        return None, None
    return start.value, size.value


def pointer_context(ptr):
    c = CUcontext()
    r = _fn["cuPointerGetAttribute"](ctypes.byref(c), CU_POINTER_ATTRIBUTE_CONTEXT,
                                     CUdeviceptr(ptr))
    return c.value if r == 0 else None


# ------------------------------------------------------- external memory
def import_external_memory(fd, size):
    """cuImportExternalMemory on an OPAQUE_FD.

    CUDA takes ownership of `fd` on success; the caller must not close it.
    On failure the fd is still the caller's and must be closed.
    """
    hd = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
    ctypes.memset(ctypes.byref(hd), 0, ctypes.sizeof(hd))
    hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
    hd.handle.fd = fd
    hd.size = size
    hd.flags = 0          # not DEDICATED: two buffers share this allocation
    em = CUexternalMemory()
    check(_fn["cuImportExternalMemory"](ctypes.byref(em), ctypes.byref(hd)),
          "cuImportExternalMemory")
    return em


def external_memory_get_mapped_buffer(em, offset, size):
    bd = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    ctypes.memset(ctypes.byref(bd), 0, ctypes.sizeof(bd))
    bd.offset = offset
    bd.size = size
    bd.flags = 0
    p = CUdeviceptr()
    check(_fn["cuExternalMemoryGetMappedBuffer"](ctypes.byref(p), em, ctypes.byref(bd)),
          "cuExternalMemoryGetMappedBuffer")
    return p.value


def destroy_external_memory(em):
    check(_fn["cuDestroyExternalMemory"](em), "cuDestroyExternalMemory")


# ---------------------------------------------------- external semaphores
def import_external_semaphore(fd, handle_type):
    hd = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
    ctypes.memset(ctypes.byref(hd), 0, ctypes.sizeof(hd))
    hd.type = handle_type
    hd.handle.fd = fd
    hd.flags = 0
    sem = CUexternalSemaphore()
    check(_fn["cuImportExternalSemaphore"](ctypes.byref(sem), ctypes.byref(hd)),
          "cuImportExternalSemaphore")
    return sem


def signal_external_semaphore(sem, value, stream):
    p = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.params.fence.value = value
    arr = (CUexternalSemaphore * 1)(sem)
    check(_fn["cuSignalExternalSemaphoresAsync"](arr, ctypes.byref(p), 1,
                                                 CUstream(stream)),
          "cuSignalExternalSemaphoresAsync")


def wait_external_semaphore(sem, value, stream):
    p = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.params.fence.value = value
    arr = (CUexternalSemaphore * 1)(sem)
    check(_fn["cuWaitExternalSemaphoresAsync"](arr, ctypes.byref(p), 1,
                                               CUstream(stream)),
          "cuWaitExternalSemaphoresAsync")


def destroy_external_semaphore(sem):
    check(_fn["cuDestroyExternalSemaphore"](sem), "cuDestroyExternalSemaphore")


# ----------------------------------------------------------- copies
def memcpy_htod(dptr, host_buf, size):
    check(_fn["cuMemcpyHtoD"](CUdeviceptr(dptr), host_buf, size), "cuMemcpyHtoD")


def memcpy_dtoh(host_buf, dptr, size):
    check(_fn["cuMemcpyDtoH"](host_buf, CUdeviceptr(dptr), size), "cuMemcpyDtoH")


def memset_d8(dptr, value, size):
    check(_fn["cuMemsetD8"](CUdeviceptr(dptr), value, size), "cuMemsetD8")


def memset_d8_async(dptr, value, size, stream):
    check(_fn["cuMemsetD8Async"](CUdeviceptr(dptr), value, size, CUstream(stream)),
          "cuMemsetD8Async")


def memcpy_dtod_async(dst, src, size, stream):
    check(_fn["cuMemcpyDtoDAsync"](CUdeviceptr(dst), CUdeviceptr(src), size,
                                   CUstream(stream)), "cuMemcpyDtoDAsync")


def mem_alloc(size):
    p = CUdeviceptr()
    check(_fn["cuMemAlloc"](ctypes.byref(p), size), "cuMemAlloc")
    return p.value


def mem_free(ptr):
    check(_fn["cuMemFree"](CUdeviceptr(ptr)), "cuMemFree")


def stream_create():
    s = CUstream()
    check(_fn["cuStreamCreate"](ctypes.byref(s), 1), "cuStreamCreate")   # NON_BLOCKING
    return s.value


def stream_synchronize(stream):
    check(_fn["cuStreamSynchronize"](CUstream(stream)), "cuStreamSynchronize")


def stream_destroy(stream):
    check(_fn["cuStreamDestroy"](CUstream(stream)), "cuStreamDestroy")


# ------------------------------------------------------------- VMM export
class VmmExport:
    """A VMM allocation exportable as a POSIX fd, mapped in the current context.

    Only the self test uses this; the real producer is Vulkan.  Allocation
    granularity is 2 MiB on this machine, so `size` is rounded up.
    """

    def __init__(self, min_size, dev_ordinal=0):
        prop = CUmemAllocationProp()
        ctypes.memset(ctypes.byref(prop), 0, ctypes.sizeof(prop))
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev_ordinal
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        gran = ctypes.c_size_t()
        check(_fn["cuMemGetAllocationGranularity"](ctypes.byref(gran), ctypes.byref(prop),
                                                   CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "cuMemGetAllocationGranularity")
        self.granularity = gran.value
        size = (min_size + gran.value - 1) // gran.value * gran.value
        self.size = size
        h = CUmemGenericAllocationHandle()
        check(_fn["cuMemCreate"](ctypes.byref(h), size, ctypes.byref(prop), 0),
              "cuMemCreate")
        self.handle = h
        p = CUdeviceptr()
        check(_fn["cuMemAddressReserve"](ctypes.byref(p), size, 0, CUdeviceptr(0), 0),
              "cuMemAddressReserve")
        self.ptr = p.value
        check(_fn["cuMemMap"](CUdeviceptr(self.ptr), size, 0, h, 0), "cuMemMap")
        acc = CUmemAccessDesc()
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = dev_ordinal
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        check(_fn["cuMemSetAccess"](CUdeviceptr(self.ptr), size, ctypes.byref(acc), 1),
              "cuMemSetAccess")

    def export_fd(self):
        fd = ctypes.c_int(-1)
        check(_fn["cuMemExportToShareableHandle"](ctypes.byref(fd), self.handle,
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                                  0),
              "cuMemExportToShareableHandle")
        return fd.value

    def close(self):
        if self.ptr:
            _fn["cuMemUnmap"](CUdeviceptr(self.ptr), self.size)
            _fn["cuMemAddressFree"](CUdeviceptr(self.ptr), self.size)
            _fn["cuMemRelease"](self.handle)
            self.ptr = 0


# ------------------------------------------------- __cuda_array_interface__
_TYPESTR_ITEMSIZE = {"|u1": 1, "|i1": 1, "<u2": 2, "<i2": 2, "<f2": 2,
                     "<u4": 4, "<i4": 4, "<f4": 4, "<u8": 8, "<i8": 8, "<f8": 8}


class CudaArrayIface:
    """Minimal __cuda_array_interface__ v3 exporter over a raw CUdeviceptr.

    torch.as_tensor() on this is zero copy: the tensor's data_ptr() is exactly
    `ptr`.  "data": (ptr, False) -- False means writable, and True is a hard
    TypeError in torch, not a read-only tensor.

    torch does NOT bounds-check the shape against anything, so nbytes is
    checked here against the caller-supplied limit.  That check is the only one
    there is.
    """

    def __init__(self, ptr, shape, typestr, limit_bytes, strides=None):
        itemsize = _TYPESTR_ITEMSIZE[typestr]
        nbytes = itemsize
        for s in shape:
            nbytes *= int(s)
        if nbytes > limit_bytes:
            raise ValueError("wrap of %d bytes exceeds the %d-byte range at 0x%x"
                             % (nbytes, limit_bytes, ptr))
        self.ptr = int(ptr)
        self.nbytes = nbytes
        self.__cuda_array_interface__ = {
            "shape": tuple(int(s) for s in shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
            "strides": strides,
            "stream": None,
        }
