/* Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

/* This file is originally from the mysql distribution. Coded by monty */

#include "./include/my_global.h"
// #include <my_sys.h>
#include "./include/m_string.h"
#include "./include/m_ctype.h"
#include "./include/mysql_com.h"

#include "./include/sql_string.h"
#include "server/mysql/include/mysql_def.h"

using namespace mysql;

#include <algorithm>
#include <limits>

using std::min;
using std::max;


#define my_b_read(info,Buffer,Count) \
  ((info)->read_pos + (Count) <= (info)->read_end ?\
   (memcpy(Buffer,(info)->read_pos,(size_t) (Count)), \
    ((info)->read_pos+=(Count)),0) :\
   (*(info)->read_function)((info),Buffer,Count))

size_t copy_and_convert(char *to, size_t to_length,
                        const CHARSET_INFO *to_cs,
                        const char *from, size_t from_length,
                        const CHARSET_INFO *from_cs, uint *errors)
{
    return my_convert(to, to_length, to_cs, from, from_length, from_cs, errors);
}
/*
  Convert a string between two character sets.
  'to' must be large enough to store (form_length * to_cs->mbmaxlen) bytes.

  @param  to[OUT]       Store result here
  @param  to_length     Size of "to" buffer
  @param  to_cs         Character set of result string
  @param  from          Copy from here
  @param  from_length   Length of the "from" string
  @param  from_cs       Character set of the "from" string
  @param  errors[OUT]   Number of conversion errors

  @return Number of bytes copied to 'to' string
*/

static size_t
my_convert_internal(char *to, size_t to_length,
                    const CHARSET_INFO *to_cs,
                    const char *from, size_t from_length,
                    const CHARSET_INFO *from_cs, uint *errors)
{
    int         cnvres;
    my_wc_t     wc;
    const uchar *from_end= (const uchar*) from + from_length;
    char *to_start= to;
    uchar *to_end= (uchar*) to + to_length;
    my_charset_conv_mb_wc mb_wc= from_cs->cset->mb_wc;
    my_charset_conv_wc_mb wc_mb= to_cs->cset->wc_mb;
    uint error_count= 0;

    while (1)
    {
        if ((cnvres= (*mb_wc)(from_cs, &wc, (uchar*) from, from_end)) > 0)
            from+= cnvres;
        else if (cnvres == MY_CS_ILSEQ)
        {
            error_count++;
            from++;
            wc= '?';
        }
        else if (cnvres > MY_CS_TOOSMALL)
        {
            /*
              A correct multibyte sequence detected
              But it doesn't have Unicode mapping.
            */
            error_count++;
            from+= (-cnvres);
            wc= '?';
        }
        else
            break;  // Not enough characters

        outp:
        if ((cnvres= (*wc_mb)(to_cs, wc, (uchar*) to, to_end)) > 0)
            to+= cnvres;
        else if (cnvres == MY_CS_ILUNI && wc != '?')
        {
            error_count++;
            wc= '?';
            goto outp;
        }
        else
            break;
    }
    *errors= error_count;
    return (uint32) (to - to_start);
}


/*
  Convert a string between two character sets.
   Optimized for quick copying of ASCII characters in the range 0x00..0x7F.
  'to' must be large enough to store (form_length * to_cs->mbmaxlen) bytes.

  @param  to[OUT]       Store result here
  @param  to_length     Size of "to" buffer
  @param  to_cs         Character set of result string
  @param  from          Copy from here
  @param  from_length   Length of the "from" string
  @param  from_cs       Character set of the "from" string
  @param  errors[OUT]   Number of conversion errors

  @return Number of bytes copied to 'to' string
*/

size_t
my_convert(char *to, size_t to_length, const CHARSET_INFO *to_cs,
           const char *from, size_t from_length,
           const CHARSET_INFO *from_cs, uint *errors)
{
    size_t length, length2;
    /*
      If any of the character sets is not ASCII compatible,
      immediately switch to slow mb_wc->wc_mb method.
    */
    if ((to_cs->state | from_cs->state) & MY_CS_NONASCII)
        return my_convert_internal(to, to_length, to_cs,
                                   from, from_length, from_cs, errors);

    length= length2= MY_MIN(to_length, from_length);

#if defined(__i386__)
    /*
    Special loop for i386, it allows to refer to a
    non-aligned memory block as UINT32, which makes
    it possible to copy four bytes at once. This
    gives about 10% performance improvement comparing
    to byte-by-byte loop.
  */
  for ( ; length >= 4; length-= 4, from+= 4, to+= 4)
  {
    if ((*(uint32*)from) & 0x80808080)
      break;
    *((uint32*) to)= *((const uint32*) from);
  }
#endif /* __i386__ */

    for (; ; *to++= *from++, length--)
    {
        if (!length)
        {
            *errors= 0;
            return length2;
        }
        if (*((unsigned char*) from) > 0x7F) /* A non-ASCII character */
        {
            size_t copied_length= length2 - length;
            to_length-= copied_length;
            from_length-= copied_length;
            return copied_length + my_convert_internal(to, to_length, to_cs,
                                                       from, from_length, from_cs,
                                                       errors);
        }
    }

    DBUG_ASSERT(FALSE); // Should never get to here
    return 0;           // Make compiler happy
}
