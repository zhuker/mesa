#!/usr/bin/env python3
"""CPU-only NVRTC compile of actual production source; never initializes CUDA."""
import argparse, ctypes as C, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'kernels'
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
n=C.CDLL('/usr/local/cuda/lib64/libnvrtc.so')
n.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)]
n.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
n.nvrtcGetProgramLogSize.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
n.nvrtcGetProgramLog.argtypes=[C.c_void_p,C.c_char_p]
n.nvrtcGetPTXSize.argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
n.nvrtcGetPTX.argtypes=[C.c_void_p,C.c_char_p]
n.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
headers=['cp_rast_types.h','cp_fs_interp.h','cp_vf_lane.h','cp_fs_write.h']
src=(root/'cp_rasterize.cu').read_bytes()
hnames=(C.c_char_p*len(headers))(*(x.encode() for x in headers))
hsrcs=(C.c_char_p*len(headers))(*((root/x).read_bytes() for x in headers))
results=[]
for mode in ['off','on','combined']:
    prog=C.c_void_p()
    assert n.nvrtcCreateProgram(C.byref(prog),src,b'cp_rasterize.cu',len(headers),hsrcs,hnames)==0
    opts=[b'--gpu-architecture=compute_90',b'--std=c++14']
    if mode!='off': opts.append(b'-DCP_PIXEL_TRACE=1')
    if mode=='combined': opts += [b'-DCP_CLIP_COMMON_REJECT=1',b'-DCP_CLIP_COPY_BYPASS=1']
    status=n.nvrtcCompileProgram(prog,len(opts),(C.c_char_p*len(opts))(*opts))
    size=C.c_size_t(); n.nvrtcGetProgramLogSize(prog,C.byref(size)); log=C.create_string_buffer(size.value); n.nvrtcGetProgramLog(prog,log)
    (out/(mode+'.log')).write_bytes(log.value)
    result={'mode':mode,'status':status,'options':[x.decode() for x in opts],'source_sha256':hashlib.sha256(src).hexdigest()}
    if status==0:
        n.nvrtcGetPTXSize(prog,C.byref(size)); ptx=C.create_string_buffer(size.value); n.nvrtcGetPTX(prog,ptx)
        (out/(mode+'.ptx')).write_bytes(ptx.value)
        result['ptx_sha256']=hashlib.sha256(ptx.value).hexdigest()
    print(json.dumps(result),flush=True); results.append(result)
    n.nvrtcDestroyProgram(C.byref(prog))
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(any(x['status'] for x in results))
