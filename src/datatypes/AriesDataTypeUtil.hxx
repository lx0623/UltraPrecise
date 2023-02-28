//
// Created by david.shen on 2019/12/18.
//

#ifndef DATATYPELIB_ARIESDATATYPEUTIL_HXX
#define DATATYPELIB_ARIESDATATYPEUTIL_HXX

#include "AriesDefinition.h"


BEGIN_ARIES_ACC_NAMESPACE

#define aries_max(a,b) ( ((a)>(b)) ? (a):(b) )
#define aries_min(a,b) ( ((a)>(b)) ? (b):(a) )
#define aries_abs(a) ( ((a)<(0)) ? (-a):(a) )
#define aries_is_digit(c) ((c) >= '0' && (c) <= '9')

    ARIES_HOST_DEVICE_NO_INLINE int aries_is_space(int ch);
    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi( const char *str, const char *end );
    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi(const char *str);
    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol( const char *str, const char *end );
    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol(const char *str);
    ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char *str);
    ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char *str, int len);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcpy(char *strDest, const char *strSrc);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncpy(char *strDest, const char *strSrc, unsigned int count);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcat(char *strDes, const char *strSrc);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncat(char *strDes, const char *strSrc, unsigned int count);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strchr(const char *str, int ch);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_sprintf(char *dst, const char *fmt, int v);
    ARIES_HOST_DEVICE_NO_INLINE void *aries_memset(void *dst, int val, unsigned long ulcount);
    ARIES_HOST_DEVICE_NO_INLINE void *aries_memcpy(void *dst, const void *src, unsigned long ulcount);
    ARIES_HOST_DEVICE_NO_INLINE int aries_strcmp(const char *source, const char *dest);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strstr(const char *strSrc, const char *str);
    //NEW ADD
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strerase(char *strDes ,char *strSrc ,int n);
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strinsert(char *strDes ,char *strSrc ,unsigned int n ,char c);
    ARIES_HOST_DEVICE_NO_INLINE int32_t abs_cmp(int32_t *a,const int32_t *b);
    ARIES_HOST_DEVICE_NO_INLINE void abs_add(int32_t *a,const int32_t *b, int32_t *res);
    ARIES_HOST_DEVICE_NO_INLINE void abs_sub(int32_t *a,const int32_t *b, int32_t *res);
    ARIES_HOST_DEVICE_NO_INLINE void abs_lshift(int32_t *a, int len, int n, int32_t *res);
    ARIES_HOST_DEVICE_NO_INLINE void abs_rshift(int32_t *a, int len, int n, int32_t *res);
    ARIES_HOST_DEVICE_NO_INLINE void abs_mul(int32_t *a,const int32_t *b, int32_t *res);
    ARIES_HOST_DEVICE_NO_INLINE int getDecimalLen(const int32_t *a);

    ARIES_HOST_DEVICE_NO_INLINE char *aries_int10_to_str(long int val,char *dst,int radix);
END_ARIES_ACC_NAMESPACE

#endif //DATATYPELIB_ARIESDATATYPEUTIL_HXX
