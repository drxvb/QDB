"""hnswlib reference on a texmex data set (fvecs/ivecs): M=16, ef_construction=200, recall@10 at several ef.
Usage: reference_ann.py <dir> <prefix, e.g. gist> <threads> [limit] [--build-only] [--print-flags]   (truth is valid for the full set only)"""
import importlib.metadata, json, os, re, statistics, subprocess, sys, sysconfig, threading, time
import hnswlib
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
flags = {a for a in sys.argv[1:] if a.startswith('--')}
d, prefix, threads = args[0], args[1], int(args[2])
limit = int(args[3]) if len(args) > 3 else None

def read_loadavg():
    return open('/proc/loadavg').read().split()[:3]

def read_cpuset():
    for line in open('/proc/self/status'):
        if line.startswith('Cpus_allowed_list:'):
            return line.split(':', 1)[1].strip()
    return 'unknown'

def read_proc_threads():
    for line in open('/proc/self/status'):
        if line.startswith('Threads:'):
            return int(line.split()[1])
    return 1

def inspect_hnswlib_flags():
    so_path = getattr(hnswlib, '__file__', 'unknown')
    version = 'unknown'
    wheel_tag = 'unknown'
    try:
        dist = importlib.metadata.distribution('hnswlib')
        version = dist.version
        w = dist.read_text('WHEEL') or ''
        for line in w.splitlines():
            if line.startswith('Tag:'):
                wheel_tag = line.split(':', 1)[1].strip()
                break
    except Exception:
        pass
    compiler_comment = ''
    try:
        out = subprocess.check_output(['readelf', '-p', '.comment', so_path], text=True, stderr=subprocess.DEVNULL)
        comments = [line.split(']', 1)[1].strip() for line in out.splitlines() if ']' in line and ('GCC' in line or 'clang' in line)]
        compiler_comment = '; '.join(comments)
    except Exception:
        pass
    simd_symbols = []
    try:
        with open(so_path, 'rb') as f:
            raw_bytes = f.read()
        for sym in (b'L2SqrSIMD16ExtAVX512', b'L2SqrSIMD16ExtAVX', b'L2SqrSIMD16ExtSSE', b'L2SqrSIMD4Ext', b'InnerProductSIMD16ExtAVX512', b'InnerProductSIMD16ExtAVX', b'InnerProductSIMD4ExtSSE'):
            if sym in raw_bytes:
                simd_symbols.append(sym.decode('ascii'))
    except Exception:
        pass
    isa_counts = {}
    try:
        out = subprocess.check_output(['objdump', '-d', so_path], text=True, stderr=subprocess.DEVNULL)
        zmm = len(re.findall(r'%zmm', out))
        ymm = len(re.findall(r'%ymm', out))
        xmm = len(re.findall(r'%xmm', out))
        fma = len(re.findall(r'vfmadd', out))
        isa_counts = dict(zmm=zmm, ymm=ymm, xmm=xmm, fma=fma)
    except Exception:
        pass
    affinity_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return dict(
        version=version,
        so_path=so_path,
        wheel_tag=wheel_tag,
        compiler=compiler_comment,
        python_cflags=sysconfig.get_config_var('CFLAGS') or '',
        simd_strings=simd_symbols,
        isa_insns=isa_counts,
        os_cpu_count=os.cpu_count(),
        sched_affinity_cpus=affinity_count,
        cpuset=read_cpuset(),
    )

if '--print-flags' in flags:
    print('REFFLAGS ' + json.dumps(inspect_hnswlib_flags()), flush=True)

def vecs(path, dtype):
    raw = np.memmap(path, dtype=dtype, mode='r')
    dim = int(raw[:1].view(np.int32)[0])
    return raw.reshape(-1, dim + 1)[:, 1:]

x = vecs(f'{d}/{prefix}_base.fvecs', np.float32)
if limit:
    x = x[:limit]
x = np.ascontiguousarray(x)
q = np.ascontiguousarray(vecs(f'{d}/{prefix}_query.fvecs', np.float32))
truth = np.array(vecs(f'{d}/{prefix}_groundtruth.ivecs', np.int32)[:, :10])
index = hnswlib.Index(space='l2', dim=x.shape[1])
index.init_index(max_elements=len(x), ef_construction=200, M=16, random_seed=100)

base_proc_threads = read_proc_threads()
peak_proc_threads = base_proc_threads
stop_monitor = False
def monitor_threads():
    global peak_proc_threads
    while not stop_monitor:
        try:
            t = read_proc_threads()
            if t > peak_proc_threads:
                peak_proc_threads = t
        except Exception:
            pass
        time.sleep(0.02)

mon = threading.Thread(target=monitor_threads, daemon=True)
mon.start()
time.sleep(0.03)
threads_with_mon = read_proc_threads()
load_before = read_loadavg()
start = time.perf_counter()
index.add_items(x, np.arange(len(x)), num_threads=threads)
build = time.perf_counter() - start
load_after = read_loadavg()
stop_monitor = True
mon.join(timeout=1.0)
# hnswlib ParallelFor spawns `num_threads` std::threads while the calling thread waits on join().
spawned_worker_threads = max(1, peak_proc_threads - threads_with_mon)

print('REFBUILD ' + json.dumps(dict(
    prefix=prefix,
    threads=threads,
    spawned_worker_threads=spawned_worker_threads,
    proc_threads_before=base_proc_threads,
    proc_threads_peak=peak_proc_threads,
    cpuset=read_cpuset(),
    n=len(x),
    dim=int(x.shape[1]),
    build_s=round(build, 1),
    build_sec=round(build, 4),
    load_before=load_before,
    load=load_after,
    load_after=load_after,
)), flush=True)

if '--build-only' not in flags:
    for ef in (32, 64, 128, 256):
        index.set_ef(ef)
        elapsed, found = [], 0
        for query, expected in zip(q, truth):
            s = time.perf_counter_ns()
            labels, _ = index.knn_query(query, k=10, num_threads=1)
            elapsed.append((time.perf_counter_ns() - s) / 1000)
            found += len(set(labels[0].tolist()) & set(expected.tolist()))
        print('REFRESULT ' + json.dumps(dict(ef=ef, recall10=found / (len(q) * 10) if not limit else None,
              median_us=round(statistics.median(elapsed), 1))), flush=True)
