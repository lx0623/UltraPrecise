/* Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is also distributed with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have included with MySQL.

   Without limiting anything contained in the foregoing, this file,
   which is part of C Driver for MySQL (Connector/C), is also subject to the
   Universal FOSS Exception, version 1.0, a copy of which can be found at
   http://oss.oracle.com/licenses/universal-foss-exception.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

/**
  @file mysys/mf_tempfile.cc
*/


#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <server/mysql/include/my_thread_local.h>
#include <AriesAssert.h>
#include <server/mysql/include/my_sys.h>
#include "./include/my_global.h"
#include "./include/m_string.h"
#include "./include/mysys_err.h"

/*
  @brief
  Create a temporary file with unique name in a given directory

  @details
  create_temp_file
    to             pointer to buffer where temporary filename will be stored
    dir            directory where to create the file
    prefix         prefix the filename with this
    mode           Flags to use for my_create/my_open
    MyFlags        Magic flags

  @return
    File descriptor of opened file if success
    -1 and sets errno if fails.

  @note
    The behaviour of this function differs a lot between
    implementation, it's main use is to generate a file with
    a name that does not already exist.

    The implementation using mkstemp should be considered the
    reference implementation when adding a new or modifying an
    existing one

*/

char *convert_dirname(char *to, const char *from, const char *from_end);
int create_temp_file(char *to, const char *dir, const char *prefix) {
  int file = -1;

  DBUG_ENTER("create_temp_file");
  // DBUG_PRINT("enter", ("dir: %s, prefix: %s", dir, prefix));
  char prefix_buff[30];
  uint pfx_len;

  pfx_len = (uint)(my_stpcpy(my_stpnmov(prefix_buff, prefix ? prefix : "tmp.",
                                        sizeof(prefix_buff) - 7),
                             "XXXXXX") -
                   prefix_buff);
  if (!dir && !(dir = getenv("TMPDIR"))) dir = DEFAULT_TMPDIR;
  if (strlen(dir) + pfx_len > FN_REFLEN - 2) {
      errno = ENAMETOOLONG;
      set_my_errno(ENAMETOOLONG);
      char errbuf[MYSYS_STRERROR_SIZE];
      ARIES_EXCEPTION( EE_CANTCREATEFILE, dir, my_errno(), strerror_r( my_errno(), errbuf, sizeof(errbuf) ) );
  }
  my_stpcpy(convert_dirname(to, dir, NullS), prefix_buff);
  file = mkstemp(to);
  // if (file >= 0) {
  //   mysql_mutex_lock(&THR_LOCK_open);
  //   my_tmp_file_created++;
  //   mysql_mutex_unlock(&THR_LOCK_open);
  // }
  DBUG_RETURN(file);
}
