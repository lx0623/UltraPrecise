#ifndef SQL_STRING_INCLUDED
#define SQL_STRING_INCLUDED

/* Copyright (c) 2000, 2016, Oracle and/or its affiliates. All rights reserved.

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

#include "m_ctype.h"                            /* my_charset_bin */

typedef struct charset_info_st CHARSET_INFO;
size_t well_formed_copy_nchars(const CHARSET_INFO *to_cs,
                               char *to, size_t to_length,
                               const CHARSET_INFO *from_cs,
                               const char *from, size_t from_length,
                               size_t nchars,
                               const char **well_formed_error_pos,
                               const char **cannot_convert_error_pos,
                               const char **from_end_pos);
size_t convert_to_printable(char *to, size_t to_len,
                            const char *from, size_t from_len,
                            const CHARSET_INFO *from_cs, size_t nbytes= 0);

size_t bin_to_hex_str(char *to, size_t to_len, char *from, size_t from_len);



static inline bool check_if_only_end_space(const CHARSET_INFO *cs, char *str,
                                           char *end)
{
  // return str+ cs->cset->scan(cs, str, end, MY_SEQ_SPACES) == end;
  return true;
}


inline LEX_CSTRING to_lex_cstring(const LEX_STRING &s)
{
  LEX_CSTRING cstr= { s.str, s.length };
  return cstr;
}


inline LEX_STRING to_lex_string(const LEX_CSTRING &s)
{
  LEX_STRING str= { const_cast<char *>(s.str),  s.length };
  return str;
}

inline LEX_CSTRING to_lex_cstring(const char *s)
{
  LEX_CSTRING cstr= { s, s != NULL ? strlen(s) : 0 };
  return cstr;
}

bool
validate_string(const CHARSET_INFO *cs, const char *str, uint32 length,
                size_t *valid_length, bool *length_error);
#endif /* SQL_STRING_INCLUDED */
