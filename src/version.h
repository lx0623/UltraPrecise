#pragma once

#if defined(BUILD_WITH_CMAKE)
#include "version_hash.h"
#else
#define BUILD_HASH "fakebuildhash"
#define BUILD_DATE "20191111101010"
#endif

#if !defined(BUILD_VERSION_NAME)
#define BUILD_VERSION_NAME "1.0.1"
#endif

#if !defined(BUILD_VERSION_CODE)
#define BUILD_VERSION_CODE 1
#endif

#define SHOW_VERSION() \
do {    \
    printf("build version: \t%s\n", BUILD_VERSION_NAME);\
    printf("build number: \t%d\n", BUILD_VERSION_CODE);\
    printf("build hash: \t%s\n", BUILD_HASH);\
    printf("built at: \t%s\n", BUILD_DATE);\
} while (0)

#define SHOW_VERSION_SIMPLE() \
do { \
    printf("version: %s-%d(%s), built at: %s\n", BUILD_VERSION_NAME, BUILD_VERSION_CODE, BUILD_HASH, BUILD_DATE); \
} while(0)

#define STRINGFY_HELPER(x) #x
#define STRINGFY_IT(x) STRINGFY_HELPER(x)

#define VERSION_INFO_STRING BUILD_VERSION_NAME "-" STRINGFY_IT(BUILD_VERSION_CODE) "(" BUILD_HASH "), built at: " BUILD_DATE
