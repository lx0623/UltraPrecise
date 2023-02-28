//
// Created by david.shen on 2019/12/18.
//

#include <cassert>
#include <cstdio>

#include "AriesDataTypeUtil.hxx"

BEGIN_ARIES_ACC_NAMESPACE
    ARIES_HOST_DEVICE_NO_INLINE int aries_is_space(int ch) {
        return (unsigned long) (ch - 9) < 5u || ' ' == ch;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi( const char *str, const char *end )
    {
        int sign;
        int n = 0;
        const char *p = str;

        while( p != end && aries_is_space( *p ) )
            p++;
        if( p != end )
        {
            sign = ( '-' == *p ) ? -1 : 1;
            if( '+' == *p || '-' == *p )
                p++;

            for( n = 0; p != end && aries_is_digit( *p ); p++ )
                n = 10 * n + ( *p - '0' );

            if( sign == -1 )
                n = -n;
        }
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_atoi( const char *str )
    {
        int sign;
        int n = 0;
        const char *p = str;

        while( aries_is_space( *p ) )
            p++;

        sign = ( '-' == *p ) ? -1 : 1;
        if( '+' == *p || '-' == *p )
            p++;

        for( n = 0; aries_is_digit( *p ); p++ )
            n = 10 * n + ( *p - '0' );

        if( sign == -1 )
            n = -n;
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol( const char *str, const char *end )
    {
        int sign;
        int64_t n = 0;
        const char *p = str;

        while( p != end && aries_is_space( *p ) )
            p++;
        if( p != end )
        {
            sign = ( '-' == *p ) ? -1 : 1;
            if( '+' == *p || '-' == *p )
                p++;

            for( n = 0; p != end && aries_is_digit( *p ); p++ )
                n = 10 * n + ( *p - '0' );

            if( sign == -1 )
                n = -n;
        }
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol( const char *str )
    {
        int sign;
        int64_t n = 0;
        const char *p = str;

        while( aries_is_space( *p ) )
            p++;

        sign = ( '-' == *p ) ? -1 : 1;
        if( '+' == *p || '-' == *p )
            p++;

        for( n = 0; aries_is_digit( *p ); p++ )
            n = 10 * n + ( *p - '0' );

        if( sign == -1 )
            n = -n;
        return n;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char *str) {
        const char *p = str;
        while (*p++);

        return (int) (p - str - 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char *str, int len) {
        return *( str + len - 1 ) ? len : aries_strlen( str );
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcpy(char *strDest, const char *strSrc) {
        if (strDest == strSrc) {
            return strDest;
        }
        assert((strDest != NULL) && (strSrc != NULL));
        char *address = strDest;
        while ((*strDest++ = *strSrc++));
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncpy(char *strDest, const char *strSrc, unsigned int count) {
        if (strDest == strSrc) {
            return strDest;
        }
        assert((strDest != NULL) && (strSrc != NULL));
        char *address = strDest;
        while (count-- && *strSrc)
            *strDest++ = *strSrc++;
        *strDest = 0;
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strcat(char *strDes, const char *strSrc) {
        assert((strDes != NULL) && (strSrc != NULL));
        char *address = strDes;
        while (*strDes)
            ++strDes;
        while ((*strDes++ = *strSrc++));
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strncat(char *strDes, const char *strSrc, unsigned int count) {
        assert((strDes != NULL) && (strSrc != NULL));
        char *address = strDes;
        while (*strDes)
            ++strDes;
        while (count-- && *strSrc)
            *strDes++ = *strSrc++;
        *strDes = 0;
        return address;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strchr(const char *str, int ch) {
        while (*str && *str != (char) ch)
            str++;

        if (*str == (char) ch)
            return ((char *) str);

        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_sprintf(char *dst, const char *fmt, int v) {
        int startPos = 0;
        int len = aries_strlen(fmt);
        //only support format : %d, %010d
        if (fmt[startPos++] != '%' || fmt[len - 1] != 'd') {
            assert(0);
            return dst;
        }

        int outLen = -1;
        bool fillwithz = false;
        if (fmt[startPos] == '0') {
            fillwithz = true;
            ++startPos;
        }
        char tmp[128];
        if (startPos + 1 < len) {
            aries_strncpy(tmp, fmt + startPos, len - startPos - 1);
            outLen = aries_atoi(tmp);
        }
        //no out
        if (outLen == 0) {
            dst[0] = '0';
            dst[1] = 0;
            return dst;
        }
        int negsign = 0;
        int val = v;
        startPos = 0;
        if (val < 0) {
            negsign = 1;
            val = -val;
        }
        do {
            tmp[startPos++] = char('0' + val % 10);
            val /= 10;
        } while (val > 0);

        len = startPos;
        startPos = 0;
        if (negsign) {
            dst[startPos++] = '-';
        }
        if (outLen == -1) {
            if (len == 0) {
                dst[startPos++] = '0';
            } else {
                for (int i = len - 1; i >= 0; i--) {
                    dst[startPos++] = tmp[i];
                }
            }
            dst[startPos] = 0;
        } else {
            int realLen = len + negsign;
            if (fillwithz) {
                int rep0 = outLen - realLen;
                if (rep0 > 0) {
                    for (int i = 0; i < rep0; i++) {
                        dst[startPos++] = '0';
                    }
                }
            }
            int cpylen = outLen - startPos;
            cpylen = cpylen > len ? len : cpylen;
            for (int i = cpylen - 1; i >= 0; i--) {
                dst[startPos++] = tmp[i];
            }
            dst[startPos] = 0;
        }
        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE void *aries_memset(void *dst, int val, unsigned long ulcount) {
        if (!dst)
            return 0;
        char *pchdst = (char *) dst;
        while (ulcount--)
            *pchdst++ = (char) val;

        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE void *aries_memcpy(void *dst, const void *src, unsigned long ulcount) {
        if (!(dst && src))
            return 0;
        if (!ulcount)
            return dst;
        char *pchdst = (char *) dst;
        char *pchsrc = (char *) src;
        while (ulcount--)
            *pchdst++ = *pchsrc++;

        return dst;
    }

    ARIES_HOST_DEVICE_NO_INLINE int aries_strcmp(const char *source, const char *dest) {
        int ret = 0;
        if (!source || !dest)
            return -2;
        while (!(ret = *(unsigned char *) source - *(unsigned char *) dest) && *dest) {
            source++;
            dest++;
        }

        if (ret < 0)
            ret = -1;
        else if (ret > 0)
            ret = 1;

        return (ret);
    }

    ARIES_HOST_DEVICE_NO_INLINE char *aries_strstr(const char *strSrc, const char *str) {
        assert(strSrc != NULL && str != NULL);
        const char *s = strSrc;
        const char *t = str;
        for (; *strSrc; ++strSrc) {
            for (s = strSrc, t = str; *t && *s == *t; ++s, ++t);
            if (!*t)
                return (char *) strSrc;
        }
        return 0;
    }

    //  NewAdd
    //lixin  增加字符串减去指定位置字符函数
	ARIES_HOST_DEVICE_NO_INLINE char *aries_strerase(char *strDes ,char *strSrc ,int n) {

		char *address = strDes;
		for(int i = 0; i < n ; i++){
			*strDes++ = *strSrc++;
        }
		strSrc++;
		while ((*strDes++ = *strSrc++));
		
		return address;
		
    }
	
	//lixin 0719 在字符串 的倒数第 n 个位置上加上 字符 c
    ARIES_HOST_DEVICE_NO_INLINE char *aries_strinsert(char *strDes ,char *strSrc ,unsigned int n ,char c){

		char *address = strDes;
		
		//获取正数的位置
		int len = aries_strlen(strSrc);
		if( len < n ){
			//补上 n-1en 个 0
			*strDes++ = c;
			for(int i = 0; i < n-len ; i++){
				*strDes++ = '0';
			}
			for(int i = 0; i < len ; i++){
				*strDes++ = *strSrc++;
			}
		}
		else{
			int pos = len - n;
			for(int i = 0; i < len ; i++){
				if(i == pos){
					*strDes++ = c;
				}
				*strDes++ = *strSrc++;
			}
			if( pos == len){
				*strDes++ = c;
			}
		}
		*strDes++ = '\0';
		// strSrc = address;
		// printf("添加后:: address = %s\n",strSrc);
        return address;
	}
	
	
	//lixin  增加绝对比较函数，要求比较前已对齐
	ARIES_HOST_DEVICE_NO_INLINE int32_t abs_cmp(int32_t *a,const int32_t *b)
	{
		int32_t res = 0;
		#pragma unroll
		for (int i = 5 - 1; i >= 0 && res == 0; i--) {
			res = a[i] - b[i];
		}
		return res;
	}
	
	//lixin  增加绝对加法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_add(int32_t *a,const int32_t *b, int32_t *res){
		
		//进位
		int overflow = 0;
		for(int i = 0; i < 5; i++){
			res[i] = a[i] + b[i] + overflow; 
			//进位
			overflow = res[i] / 1000000000;
			//剩余
			res[i] = res[i] % 1000000000;
		}
	}
	
	//lixin  增加绝对减法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_sub(int32_t *a,const int32_t *b, int32_t *res){
		//将 const a,b化为可变
		const int32_t *sub1, *sub2;
		int32_t r = abs_cmp(a, b);
		
		//将绝对值大的值 赋值到sub1 小的值赋值到sub2
		if(r >= 0){
			sub1 = a;
			sub2 = b;
		}else{
			sub1 = b;
			sub2 = a;
		}
		
		//借位
		int32_t carry = 0;
		//从高位开始减
		for(int i = 0; i < 5; i++){
			res[i] = sub1[i] + 1000000000 - sub2[i] - carry;
			carry = !(res[i] / 1000000000);
			res[i] = res[i] % 1000000000;
		}
		
		// for(int i = 0; i < 5; i++){
		// 	printf("abs_sub::res::%d\n",res[i]);
		// }
	}
	
	//lixin  增加左移函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_lshift(int32_t *a, int len, int n, int32_t *res){
		
	}
	
	
	//lixin 0720 增加右移函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_rshift(int32_t *a, int len, int n, int32_t *res){
		 int32_t rword = n / 9;
		int32_t rbit = n % 9;
		int32_t rd = 1;
		int32_t rl = 1;

		for(int i = 0; i < rbit; i++) rd *= 10;
		for(int i = 0; i < 9 - rbit; i++) rl *= 10;
		for(int i = 0; i < len - rword - 1; i++){
			res[i] = a[rword + i] / rd + a[rword + i + 1] % rd * rl;
		}
		res[len - rword - 1] = a[len - 1] / rd;
		for(int i = len - rword; i < len; i++)
			res[i] = 0;
	}
	
	//lixin  增加绝对乘法函数
	ARIES_HOST_DEVICE_NO_INLINE void abs_mul(int32_t *a,const int32_t *b, int32_t *res){
		
		//printf("**********************abs_mul start****************\n");
		int64_t temp;
		int32_t carry;
		
		//x位 * y位 需要找一个 x+y位的数 来记录中间结果
		for(int i = 0; i < 5 * 2; i++)
			res[i] = 0;
		
		//a * b 从a的最低位开始 乘以b的每一位
		for(int i = 0; i < 5; i++){
            if( a[i] == 0){
				continue;
			}
			//进位
			carry = 0;
			for(int j = 0; j < 5; j++){
				//i*j 位的结果放在 i+j
				temp = (int64_t)a[i] * b[j] + res[i+j] + carry;
			//	printf("abs_mul::temp = %d * %d + %d + %d = %ld\n",a[i],b[j],res[i+j],carry,temp);
				carry = temp / 1000000000;
				//判断是否溢出
			//   printf("abs_mul::carry = %ld / %d = %d\n",temp,1000000000,carry);
				res[i+j] = temp % 1000000000;
			//    printf("abs_mul::res[%d] = %ld mod %d = %d\n",i+j,temp,1000000000,res[i+j]);
			//    printf("\n");
				
			}
			//a的i位已经 乘以了 b的每一位 ，将最后溢出的结果放在 i+5 即 i+5位上
			res[i+5] = carry;
			//printf("abs_mul::res[%d]  = %d\n",i+5,carry);
		}
	}
		
	//lixin  增加获取数字真实长度
	ARIES_HOST_DEVICE_NO_INLINE int getDecimalLen(const int32_t *a){
		int num=0;
		int i;
		int temp;
		for( i=5-1; i>=0; i--){
			if(a[i]!=0){
				temp = a[i];
				num += i*9;
				break;
			}
		}
		while(temp>0){
			temp /= 10;
			num++;
		}
		return num;
		//printf("**********************abs_mul end****************\n");
	}

    /*
      Converts integer to its string representation in decimal notation.

    SYNOPSIS
    aries_int10_to_str()
            val     - value to convert
            dst     - points to buffer where string representation should be stored
            radix   - flag that shows whenever val should be taken as signed or not

    DESCRIPTION
            This is version of int2str() (in file int2str.cc ) function which is optimized for normal case
    of radix 10/-10. It takes only sign of radix parameter into account and
    not its absolute value.

    RETURN VALUE
    Pointer to ending NUL character.
    */

    ARIES_HOST_DEVICE_NO_INLINE char *aries_int10_to_str(long int val,char *dst,int radix)
    {
        char buffer[64];
        char *p;
        long int new_val;
        auto uval = (unsigned long int) val;

        if (radix < 0)				/* -10 */
        {
            if (val < 0)
            {
                *dst++ = '-';
                /* Avoid integer overflow in (-val) for LONGLONG_MIN (BUG#31799). */
                uval = (unsigned long int)0 - uval;
            }
        }

        p = &buffer[sizeof(buffer)-1];
        *p = '\0';
        new_val= (long) (uval / 10);
        *--p = '0'+ (char) (uval - (unsigned long) new_val * 10);
        val = new_val;

        while (val != 0)
        {
            new_val=val/10;
            *--p = '0' + (char) (val-new_val*10);
            val= new_val;
        }
        while ((*dst++ = *p++) != 0) ;
        return dst-1;
    }

END_ARIES_ACC_NAMESPACE
