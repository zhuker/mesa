import os, subprocess, pathlib, json, time, threading, fcntl, hashlib
root=pathlib.Path('/home/builder/b200-retry-validation'); root.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('CUDAVK','CPVK','VK_','DUMP_','SUBMIT_')) and k not in ('LD_PRELOAD','CTRL_ICD','CAND_ICD','CTRL_WRAP','CAND_WRAP','OUT','CTRLFRAMES')}
env.update(PATH='/home/builder/bin:/home/builder/mesa-three-ms-baseline/.build-venv/bin:/usr/lib/llvm-18/bin:/usr/local/cuda/bin:/usr/bin:/bin', LD_LIBRARY_PATH='/home/builder/lib:/usr/local/cuda/lib64', MESA='/home/builder/mesa-three-ms-vfetch-retry', SHIM='/home/builder/favorite-cpp/submit_shim.so')
env['CUDAVK_VFETCH_REGCAP_RETRY']='1'
icd=env['MESA']+'/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json'; env.update(ICD=icd,VK_DRIVER_FILES=icd)
def run(cmd, log, env=env):
 with open(root/log,'w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  while p.poll() is None:
   snap=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
   with open(root/'gpu-exclusivity.log','a') as g: g.write(str(time.time())+' '+snap+'\n')
   if 'FAILED' in (root/log).read_text():
    os.killpg(p.pid,15); raise RuntimeError('GATE FAILURE '+log)
   unexpected=[]
   for line in snap.splitlines():
    try:
     pid=int(line.split(',')[0]); original=pid
     while pid not in (0,1,p.pid):
      stat=pathlib.Path('/proc/'+str(pid)+'/stat').read_text(); pid=int(stat[stat.rfind(')')+2:].split()[1])
     if pid!=p.pid: unexpected.append(line)
    except FileNotFoundError: pass
    except ValueError: unexpected.append(line)
   if unexpected:
    os.killpg(p.pid,15); raise RuntimeError('UNRELATED/UNVERIFIABLE GPU WORKLOAD: '+repr(unexpected))
   try: p.wait(timeout=10)
   except subprocess.TimeoutExpired: pass
  if p.returncode: raise RuntimeError('failed '+log+' rc='+str(p.returncode))
binary=pathlib.Path(env['MESA']+'/build-cudavk/src/cudavk/libvulkan_cudavk.so')
sha=hashlib.sha256(binary.read_bytes()).hexdigest()
assert sha == 'b48f4cc9dc03411bfba3a51697b7fcac11435ac68734327ce7b23359e3ec9a6b'
(root/'binary-before.sha256').write_text(sha+'\n')
lock=open('/tmp/cudavk-gpu.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX)
snap=subprocess.check_output(['nvidia-smi'],text=True); (root/'baseline-gpu-before.txt').write_text(snap)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
run(['python3',env['MESA']+'/src/cudavk/tests/cpvk_vfetch_regcap_retry_gate.py',env['MESA']+'/build-cudavk/src/cudavk/cpvk_batchbig','--log-dir',str(root/'repaired-presence')],'repaired-presence-gate.log')
(root/'PRESENCE_PASS').write_text('source-grounded cap32 and actual 9-to-1 fused launch, exact pixels\n')
run(['meson','test','-C',env['MESA']+'/build-cudavk','--no-rebuild','--maxfail','1','--suite','cudavk','--print-errorlogs'],'retry-on-suite.log')
(root/'SUITE_PASS').write_text('unchanged native suite retry on passed\n')
fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
for cap,ref in [('favorite3','ctrl'),('favorite2','f2-current')]:
 e=env.copy(); e.update(CTRLFRAMES='/home/builder/sentinels/'+ref, OUT=str(root/('retry-'+cap)))
 run(['bash',env['MESA']+'/src/cudavk/tests/cp_harness_ab.sh',cap,'retry-'+cap,'CUDAVK_VFETCH_REGCAP_RETRY=0','CUDAVK_VFETCH_REGCAP_RETRY=1','3'],'retry-'+cap+'.log',e)
assert hashlib.sha256(binary.read_bytes()).hexdigest()==sha
(root/'binary-after.sha256').write_text(sha+'\n')
(root/'RETRY_PASS').write_text('presence, unchanged retry-on suite and both three-round off/on captures passed\n')
