#ifndef MY_CONFIG_H
#define MY_CONFIG_H

/* Type sizes */
#define SIZEOF_VOIDP     8
#define SIZEOF_CHARP     8
#define SIZEOF_LONG      8
#define SIZEOF_SHORT     2
#define SIZEOF_INT       4
#define SIZEOF_LONG_LONG 8
#define SIZEOF_OFF_T     8
#define SIZEOF_TIME_T    8
#define HAVE_UINT 1
#define HAVE_ULONG 1
#define HAVE_U_INT32_T 1
#define HAVE_STRUCT_TIMESPE

#define MAX_INDEXES 64U

#define MACHINE_TYPE "x86_64"
#define HAVE_LINUX_LARGE_PAGES 1
/* #undef HAVE_SOLARIS_LARGE_PAGES */
/* #undef HAVE_SOLARIS_ATOMIC */
/* #undef HAVE_SOLARIS_STYLE_GETHOST */
#define SYSTEM_TYPE "Linux"

#define DEFAULT_TMPDIR P_tmpdir
#endif