#!/usr/bin/python3
import os, subprocess, pathlib, sys
arm=os.environ.get('CUDAVK_VFETCH_REGCAP_RETRY','default')
p=subprocess.run(['/home/builder/mesa-three-ms-vfetch-retry/build-cudavk/src/cudavk/cpvk_batchbig'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
pathlib.Path('/home/builder/b200-retry-validation/presence-'+arm+'.log').write_bytes(p.stdout)
sys.stdout.buffer.write(p.stdout)
sys.exit(p.returncode)
