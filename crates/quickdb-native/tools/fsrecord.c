/* LD_PRELOAD recorder for the power-loss emulation (tests/powerloss.rs, examples/pl_rebuild.rs).
 *
 * Build: gcc -shared -fPIC -O2 -o fsrecord.so fsrecord.c -ldl
 * Env:   FSREC_DIR  absolute directory to watch (files directly inside it, and the directory itself)
 *        FSREC_LOG  output log path (binary, appended)
 *
 * Every successful call that changes or syncs a watched file is appended to the log, in completion
 * order (one mutex around the real call and the log write). Writes to fd 1 are recorded as MARK so the
 * writer's progress lines land at the exact point between file operations where the call returned.
 *
 * Record: u8 op, i32 fd (-1 for rename and unlink), u16 len, name, u16 len, name2, u64 a, u64 b, then b bytes of data for WRITE and MARK.
 *   OPEN a=flags | WRITE a=offset b=len | TRUNC a=length | RENAME name->name2 | UNLINK
 *   FSYNC | FDATASYNC | CLOSE | MARK b=len | PWRITE (as WRITE, from the positional calls)
 * The directory itself has the name "" (its fsync makes its entries durable).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

enum { OP_OPEN = 1, OP_WRITE, OP_TRUNC, OP_RENAME, OP_UNLINK, OP_FSYNC, OP_FDATASYNC, OP_CLOSE, OP_MARK, OP_PWRITE };
#define MAXFD 65536

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static char *names[MAXFD];
static char watch[4096];
static size_t watch_len;
static int log_fd = -1;

static int (*r_open)(const char *, int, ...);
static int (*r_open64)(const char *, int, ...);
static int (*r_openat)(int, const char *, int, ...);
static int (*r_openat64)(int, const char *, int, ...);
static int (*r_creat)(const char *, mode_t);
static int (*r_creat64)(const char *, mode_t);
static ssize_t (*r_write)(int, const void *, size_t);
static ssize_t (*r_writev)(int, const struct iovec *, int);
static ssize_t (*r_pwrite)(int, const void *, size_t, off_t);
static ssize_t (*r_pwrite64)(int, const void *, size_t, off_t);
static ssize_t (*r_pwritev)(int, const struct iovec *, int, off_t);
static ssize_t (*r_pwritev64)(int, const struct iovec *, int, off_t);
static int (*r_ftruncate)(int, off_t);
static int (*r_ftruncate64)(int, off_t);
static int (*r_rename)(const char *, const char *);
static int (*r_renameat)(int, const char *, int, const char *);
static int (*r_renameat2)(int, const char *, int, const char *, unsigned);
static int (*r_unlink)(const char *);
static int (*r_unlinkat)(int, const char *, int);
static int (*r_fsync)(int);
static int (*r_fdatasync)(int);
static int (*r_close)(int);

#define LOAD(v, s) v = dlsym(RTLD_NEXT, s)

static void init(void) {
    LOAD(r_open, "open"); LOAD(r_open64, "open64"); LOAD(r_openat, "openat"); LOAD(r_openat64, "openat64");
    LOAD(r_creat, "creat"); LOAD(r_creat64, "creat64");
    LOAD(r_write, "write"); LOAD(r_writev, "writev"); LOAD(r_pwrite, "pwrite"); LOAD(r_pwrite64, "pwrite64");
    LOAD(r_pwritev, "pwritev"); LOAD(r_pwritev64, "pwritev64");
    LOAD(r_ftruncate, "ftruncate"); LOAD(r_ftruncate64, "ftruncate64");
    LOAD(r_rename, "rename"); LOAD(r_renameat, "renameat"); LOAD(r_renameat2, "renameat2");
    LOAD(r_unlink, "unlink"); LOAD(r_unlinkat, "unlinkat");
    LOAD(r_fsync, "fsync"); LOAD(r_fdatasync, "fdatasync"); LOAD(r_close, "close");
    const char *d = getenv("FSREC_DIR"), *l = getenv("FSREC_LOG");
    if (d && l && strlen(d) < sizeof watch) {
        strcpy(watch, d);
        watch_len = strlen(watch);
        while (watch_len > 1 && watch[watch_len - 1] == '/') watch[--watch_len] = 0;
        log_fd = r_open(l, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    }
}
static void setup(void) { pthread_once(&once, init); }

/* Returns the name inside the watched directory ("" for the directory), or NULL. Caller frees. */
static char *watched(int dirfd, const char *p) {
    if (log_fd < 0 || !p) return NULL;
    char buf[8192];
    if (p[0] == '/') {
        snprintf(buf, sizeof buf, "%s", p);
    } else if (dirfd == AT_FDCWD || dirfd < 0) {
        char cwd[4096];
        if (!getcwd(cwd, sizeof cwd)) return NULL;
        snprintf(buf, sizeof buf, "%s/%s", cwd, p);
    } else if (dirfd < MAXFD && names[dirfd] && names[dirfd][0] == 0) {
        snprintf(buf, sizeof buf, "%s/%s", watch, p);
    } else {
        return NULL;
    }
    size_t n = strlen(buf);
    while (n > 1 && buf[n - 1] == '/') buf[--n] = 0;
    if (strncmp(buf, watch, watch_len) != 0) return NULL;
    const char *rest = buf + watch_len;
    if (*rest == 0) return strdup("");
    if (*rest != '/' || strchr(rest + 1, '/')) return NULL;
    return strdup(rest + 1);
}

static void put(const void *b, size_t n) {
    const char *c = b;
    while (n > 0) {
        ssize_t w = r_write(log_fd, c, n);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return; }
        c += w; n -= (size_t)w;
    }
}
static void rec(uint8_t op, int fd, const char *a, const char *b, uint64_t x, uint64_t len, const struct iovec *iov, int cnt) {
    uint16_t la = a ? (uint16_t)strlen(a) : 0, lb = b ? (uint16_t)strlen(b) : 0;
    put(&op, 1); put(&fd, 4); put(&la, 2); if (la) put(a, la); put(&lb, 2); if (lb) put(b, lb);
    put(&x, 8); put(&len, 8);
    uint64_t left = len;
    for (int i = 0; i < cnt && left > 0; i++) {
        size_t take = iov[i].iov_len < left ? iov[i].iov_len : (size_t)left;
        put(iov[i].iov_base, take); left -= take;
    }
}
static const char *fdname(int fd) { return (fd >= 0 && fd < MAXFD) ? names[fd] : NULL; }

static int after_open(int fd, int dirfd, const char *p, int flags) {
    if (fd < 0 || fd >= MAXFD) return fd;
    char *n = watched(dirfd, p);
    free(names[fd]);
    names[fd] = n;
    if (n) rec(OP_OPEN, fd, n, NULL, (uint64_t)flags, 0, NULL, 0);
    return fd;
}

#define MODE_ARG mode_t mode = 0; if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }
int open(const char *p, int flags, ...) { setup(); MODE_ARG pthread_mutex_lock(&mu); int fd = after_open(r_open(p, flags, mode), AT_FDCWD, p, flags); pthread_mutex_unlock(&mu); return fd; }
int open64(const char *p, int flags, ...) { setup(); MODE_ARG pthread_mutex_lock(&mu); int fd = after_open(r_open64(p, flags, mode), AT_FDCWD, p, flags); pthread_mutex_unlock(&mu); return fd; }
int openat(int d, const char *p, int flags, ...) { setup(); MODE_ARG pthread_mutex_lock(&mu); int fd = after_open(r_openat(d, p, flags, mode), d, p, flags); pthread_mutex_unlock(&mu); return fd; }
int openat64(int d, const char *p, int flags, ...) { setup(); MODE_ARG pthread_mutex_lock(&mu); int fd = after_open(r_openat64(d, p, flags, mode), d, p, flags); pthread_mutex_unlock(&mu); return fd; }
int creat(const char *p, mode_t m) { setup(); pthread_mutex_lock(&mu); int fl = O_CREAT | O_WRONLY | O_TRUNC; int fd = after_open(r_creat(p, m), AT_FDCWD, p, fl); pthread_mutex_unlock(&mu); return fd; }
int creat64(const char *p, mode_t m) { setup(); pthread_mutex_lock(&mu); int fl = O_CREAT | O_WRONLY | O_TRUNC; int fd = after_open(r_creat64(p, m), AT_FDCWD, p, fl); pthread_mutex_unlock(&mu); return fd; }

/* The write family: run the real call under the lock, then log what was actually written. */
#define WRITE_BODY(CALL, IOV, CNT, OFFEXPR, OP)                                            \
    setup();                                                                             \
    if (log_fd < 0 || (fd != 1 && !fdname(fd))) return CALL;                             \
    pthread_mutex_lock(&mu);                                                             \
    ssize_t r = CALL;                                                                    \
    if (r > 0) {                                                                         \
        if (fd == 1 && !fdname(fd)) rec(OP_MARK, fd, NULL, NULL, 0, (uint64_t)r, IOV, CNT); \
        else rec(OP, fd, fdname(fd), NULL, (uint64_t)(OFFEXPR), (uint64_t)r, IOV, CNT); \
    }                                                                                    \
    pthread_mutex_unlock(&mu);                                                           \
    return r;

ssize_t write(int fd, const void *b, size_t n) {
    struct iovec v = {(void *)b, n};
    WRITE_BODY(r_write(fd, b, n), &v, 1, lseek(fd, 0, SEEK_CUR) - r, OP_WRITE)
}
ssize_t writev(int fd, const struct iovec *iov, int cnt) {
    WRITE_BODY(r_writev(fd, iov, cnt), iov, cnt, lseek(fd, 0, SEEK_CUR) - r, OP_WRITE)
}
ssize_t pwrite(int fd, const void *b, size_t n, off_t o) {
    struct iovec v = {(void *)b, n};
    WRITE_BODY(r_pwrite(fd, b, n, o), &v, 1, o, OP_PWRITE)
}
ssize_t pwrite64(int fd, const void *b, size_t n, off_t o) {
    struct iovec v = {(void *)b, n};
    WRITE_BODY(r_pwrite64(fd, b, n, o), &v, 1, o, OP_PWRITE)
}
ssize_t pwritev(int fd, const struct iovec *iov, int cnt, off_t o) {
    WRITE_BODY(r_pwritev(fd, iov, cnt, o), iov, cnt, o, OP_PWRITE)
}
ssize_t pwritev64(int fd, const struct iovec *iov, int cnt, off_t o) {
    WRITE_BODY(r_pwritev64(fd, iov, cnt, o), iov, cnt, o, OP_PWRITE)
}

static int simple(int r, uint8_t op, int fd, uint64_t x) {
    if (r == 0 && fdname(fd)) rec(op, fd, fdname(fd), NULL, x, 0, NULL, 0);
    return r;
}
int ftruncate(int fd, off_t l) { setup(); pthread_mutex_lock(&mu); int r = simple(r_ftruncate(fd, l), OP_TRUNC, fd, (uint64_t)l); pthread_mutex_unlock(&mu); return r; }
int ftruncate64(int fd, off_t l) { setup(); pthread_mutex_lock(&mu); int r = simple(r_ftruncate64(fd, l), OP_TRUNC, fd, (uint64_t)l); pthread_mutex_unlock(&mu); return r; }
/* FSREC_NOSYNC (the negative control) hides every sync of a watched file from the log, as if the
 * program never called it; the real call still runs so the process behaves the same. */
static int nosync(void) { return getenv("FSREC_NOSYNC") != NULL; }
int fsync(int fd) { setup(); pthread_mutex_lock(&mu); int r = r_fsync(fd); if (!nosync()) simple(r, OP_FSYNC, fd, 0); pthread_mutex_unlock(&mu); return r; }
int fdatasync(int fd) { setup(); pthread_mutex_lock(&mu); int r = r_fdatasync(fd); if (!nosync()) simple(r, OP_FDATASYNC, fd, 0); pthread_mutex_unlock(&mu); return r; }
int close(int fd) {
    setup();
    pthread_mutex_lock(&mu);
    int r = r_close(fd);
    if (fd >= 0 && fd < MAXFD && names[fd]) {
        if (r == 0) rec(OP_CLOSE, fd, names[fd], NULL, 0, 0, NULL, 0);
        free(names[fd]);
        names[fd] = NULL;
    }
    pthread_mutex_unlock(&mu);
    return r;
}

static int ren(int r, int d1, const char *a, int d2, const char *b) {
    if (r != 0) return r;
    char *x = watched(d1, a), *y = watched(d2, b);
    if (x && y) rec(OP_RENAME, -1, x, y, 0, 0, NULL, 0);
    else if (x) rec(OP_UNLINK, -1, x, NULL, 0, 0, NULL, 0);
    free(x); free(y);
    return r;
}
int rename(const char *a, const char *b) { setup(); pthread_mutex_lock(&mu); int r = ren(r_rename(a, b), AT_FDCWD, a, AT_FDCWD, b); pthread_mutex_unlock(&mu); return r; }
int renameat(int d1, const char *a, int d2, const char *b) { setup(); pthread_mutex_lock(&mu); int r = ren(r_renameat(d1, a, d2, b), d1, a, d2, b); pthread_mutex_unlock(&mu); return r; }
int renameat2(int d1, const char *a, int d2, const char *b, unsigned f) { setup(); pthread_mutex_lock(&mu); int r = ren(r_renameat2(d1, a, d2, b, f), d1, a, d2, b); pthread_mutex_unlock(&mu); return r; }

static int unl(int r, int d, const char *p) {
    if (r != 0) return r;
    char *x = watched(d, p);
    if (x) rec(OP_UNLINK, -1, x, NULL, 0, 0, NULL, 0);
    free(x);
    return r;
}
int unlink(const char *p) { setup(); pthread_mutex_lock(&mu); int r = unl(r_unlink(p), AT_FDCWD, p); pthread_mutex_unlock(&mu); return r; }
int unlinkat(int d, const char *p, int f) { setup(); pthread_mutex_lock(&mu); int r = unl(r_unlinkat(d, p, f), d, p); pthread_mutex_unlock(&mu); return r; }
