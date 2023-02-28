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

#include <errno.h>
#include <glog/logging.h>
#include <server/mysql/include/my_sys.h>
#include <server/mysql/include/sql_class.h>
#include <server/mysql/include/mysys_err.h>
#include "./include/my_thread_local.h"

std::string filename( int fd );
/**
  Write a chunk of bytes to a file

  if (MyFlags & (MY_NABP | MY_FNABP))
  @returns
    0  if Count == 0
    On succes, 0
    On failure, (size_t)-1 == MY_FILE_ERROR

  otherwise
  @returns
    0  if Count == 0
    On success, the number of bytes written.
    On partial success (if less than Count bytes could be written),
       the actual number of bytes written.
    On failure, (size_t)-1 == MY_FILE_ERROR
*/
size_t my_write(File Filedes, const uchar *Buffer, size_t Count, myf MyFlags)
{
  size_t writtenbytes;
  size_t sum_written= 0;
  uint errors= 0;
  const size_t initial_count= Count;
  // char msgBuff[1024] = {0};

  DBUG_ENTER("my_write");
  // snprintf(msgBuff, 1024, "write file, fd: %d  Buffer: %p  Count: %lu  MyFlags: %d",
  //          Filedes, Buffer, (ulong) Count, MyFlags);
  // LOG(INFO) << msgBuff;

  /* The behavior of write(fd, buf, 0) is not portable */
  if (unlikely(!Count))
    DBUG_RETURN(0);
  
  for (;;)
  {
    errno= 0;
    writtenbytes= write(Filedes, Buffer, Count);
    if (writtenbytes == Count)
    {
      sum_written+= writtenbytes;
      break;
    }
    if (writtenbytes != (size_t) -1)
    {						/* Safeguard */
      sum_written+= writtenbytes;
      Buffer+= writtenbytes;
      Count-= writtenbytes;
    }
    set_my_errno(errno);
    // snprintf(msgBuff, 1024, "Write only %ld bytes, error: %d",
		//	(long) writtenbytes, my_errno());
    if (thd_killed(NULL))
      MyFlags&= ~ MY_WAIT_IF_FULL;		/* End if aborted by user */

      /*
    if ((my_errno() == ENOSPC || my_errno() == EDQUOT) &&
        (MyFlags & MY_WAIT_IF_FULL))
    {
      wait_for_free_space(my_filename(Filedes), errors);
      errors++;
      DBUG_EXECUTE_IF("simulate_no_free_space_error",
                      { DBUG_SET("-d,simulate_file_write_error");});
      continue;
    }
       */

    if (writtenbytes != 0 && writtenbytes != (size_t) -1)
      continue;                                 /* Retry if something written */
    else if (my_errno() == EINTR)
    {
      LOG(INFO) << "my_write() was interrupted and returned %ld" << (long) writtenbytes;
      continue;                                 /* Interrupted, retry */
    }
    else if (writtenbytes == 0 && !errors++)    /* Retry once */
    {
      /* We may come here if the file quota is exeeded */
      continue;
    }
    break;
  } // end of for

  if (MyFlags & (MY_NABP | MY_FNABP))
  {
      if (sum_written == initial_count)
          DBUG_RETURN(0);        /* Want only errors, not bytes written */
      if (MyFlags & (MY_WME | MY_FAE | MY_FNABP))
      {
          char errbuf[MYSYS_STRERROR_SIZE];
          ARIES_EXCEPTION(EE_WRITE, filename(Filedes).data(),
                          my_errno(), strerror_r(my_errno(), errbuf, sizeof(errbuf)));
      }
      DBUG_RETURN(MY_FILE_ERROR);
  }

  if (sum_written == 0)
      DBUG_RETURN(MY_FILE_ERROR);

  DBUG_RETURN(sum_written);
} /* my_write */
