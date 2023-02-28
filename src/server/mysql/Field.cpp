//
// Created by tengjp on 19-7-29.
//

#include <cstring>
#include <algorithm>
#include "Field.h"
Field_num::Field_num(const string& val, uint32 len_arg, bool unsigned_arg):
        Field(val, len_arg),
        unsigned_flag(unsigned_arg){

}
Field_int::Field_int(const string& val, uint32 len_arg, bool unsigned_arg) :
        Field_num(val, len_arg, unsigned_arg) {

}
void Field_int::make_sort_key(uchar *to, uint length)
{
    DBUG_ASSERT(length >= 4);
#ifdef WORDS_BIGENDIAN
  //   if (!table->s->db_low_byte_first)
  // {
  //   if (unsigned_flag)
  //     to[0] = ptr[0];
  //   else
  //     to[0] = (char) (ptr[0] ^ 128);		/* Revers signbit */
  //   to[1]   = ptr[1];
  //   to[2]   = ptr[2];
  //   to[3]   = ptr[3];
  // }
  // else
#endif
    {
        uchar* ptr = (uchar*)(int*)val.data();
        if (unsigned_flag)
            to[0] = ptr[3];
        else
            to[0] = (char) (ptr[3] ^ 128);		/* Revers signbit */
        to[1]   = ptr[2];
        to[2]   = ptr[1];
        to[3]   = ptr[0];
    }
}
#define FLT_EXP_DIG (sizeof(float)*8-FLT_MANT_DIG)
void Field_float::make_sort_key(uchar *to, uint length)
{
    DBUG_ASSERT(length >= 4);
    float nr;
#ifdef WORDS_BIGENDIAN
  //   if (table->s->db_low_byte_first)
  // {
  //   float4get(nr,ptr);
  // }
  // else
#endif

    uchar* ptr = (uchar*)(float*)val.data();
    memcpy(&nr, ptr, std::min<uint>(length, sizeof(float)));

    uchar *tmp= to;
    if (nr == (float) 0.0)
    {						/* Change to zero string */
        tmp[0]=(uchar) 128;
        memset(tmp + 1, 0, std::min<uint>(length, sizeof(nr) - 1));
    }
    else
    {
#ifdef WORDS_BIGENDIAN
        memcpy(tmp, &nr, sizeof(nr));
#else
        tmp[0]= ptr[3]; tmp[1]=ptr[2]; tmp[2]= ptr[1]; tmp[3]=ptr[0];
#endif
        if (tmp[0] & 128)				/* Negative */
        {						/* make complement */
            uint i;
            for (i=0 ; i < sizeof(nr); i++)
                tmp[i]= (uchar) (tmp[i] ^ (uchar) 255);
        }
        else
        {
            ushort exp_part=(((ushort) tmp[0] << 8) | (ushort) tmp[1] |
                             (ushort) 32768);
            exp_part+= (ushort) 1 << (16-1-FLT_EXP_DIG);
            tmp[0]= (uchar) (exp_part >> 8);
            tmp[1]= (uchar) exp_part;
        }
    }
}
// void Field_string::make_sort_key(uchar *to, uint length) {
//     uint tmp MY_ATTRIBUTE((unused))=
//             field_charset->coll->strnxfrm(field_charset,
//                                           to, length, char_length(),
//                                           (const uchar*)val.data(), field_length,
//                                           MY_STRXFRM_PAD_WITH_SPACE |
//                                           MY_STRXFRM_PAD_TO_MAXLEN);
//     DBUG_ASSERT(tmp == length);
// 
// }

