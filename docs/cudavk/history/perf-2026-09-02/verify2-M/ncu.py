import csv,collections
def load(p):
    d=collections.OrderedDict()
    for row in csv.reader(l for l in open(p) if l.startswith('"')):
        if row[0]=='ID': continue
        k=(int(row[0]),row[4],row[8],row[7])
        d.setdefault(k,{})[row[12]]=float(row[14].replace(',',''))
    return d
r=load('/tmp/ncu-rtx/v2.csv'); b=load('/tmp/ncu-b200-v2.csv')
assert [k for k in r]==[k for k in b], "IDs/kernels differ"
print("n launches",len(r))
print("metrics:",sorted(next(iter(r.values())).keys()))
agg=collections.defaultdict(lambda:[0,0,0,0,0,0])
print("%3s %-26s %-14s %9s %9s %6s %6s %6s %6s %5s %5s %6s %6s"%("ID","kernel","grid","cycR","cycB","ratio","ipcR","ipcB","inst","regR","regB","wavR","wavB"))
for k in r:
    cr=r[k]['sm__cycles_active.avg']; cb=b[k]['sm__cycles_active.avg']
    ir=r[k]['smsp__inst_executed.avg.per_cycle_active']; ib=b[k]['smsp__inst_executed.avg.per_cycle_active']
    print("%3d %-26s %-14s %9.0f %9.0f %6.3f %6.3f %6.3f %6.3f %5.0f %5.0f %6.2f %6.2f"%(k[0],k[1],k[2],cr,cb,cb/cr if cr else 0,ir,ib,(cb*ib)/(cr*ir) if cr*ir else 0,r[k]['launch__registers_per_thread'],b[k]['launch__registers_per_thread'],r[k]['launch__waves_per_multiprocessor'],b[k]['launch__waves_per_multiprocessor']))
    a=agg[k[1]]; a[0]+=1;a[1]+=cr;a[2]+=cb;a[3]+=cr*ir;a[4]+=cb*ib
tR=tB=0
for kn,a in agg.items():
    print("%-28s n=%2d cycR=%8.0f cycB=%8.0f ratio=%.3f instproxy=%.3f"%(kn,a[0],a[1],a[2],a[2]/a[1],a[4]/a[3])); tR+=a[1]; tB+=a[2]
print("ALL ratio",tB/tR, tR,tB)
# elapsed-based? check for sm__throughput and dram
for m in ['sm__throughput.avg.pct_of_peak_sustained_elapsed','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active']:
    sr=sum(r[k][m] for k in r)/len(r); sb=sum(b[k][m] for k in b)/len(b)
    print(m, "meanR=%.2f meanB=%.2f"%(sr,sb))
