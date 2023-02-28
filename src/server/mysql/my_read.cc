/* Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.

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
#include <string>
#include <glog/logging.h>
#include <server/mysql/include/derror.h>
#include <AriesAssert.h>
#include "./include/my_sys.h"
#include "./include/mysys_err.h"
#include "./include/my_thread_local.h"

std::string filename( int fd )
{
    char buf[1024] = {0};
    char path[1024] = {0};
    snprintf( buf, sizeof(buf), "/proc/%d/fd/%d", getpid(), fd);
    if ( -1 == readlink( buf, path, sizeof(path) - 1 ) )
    {
        set_my_errno( errno );
        char errbuf[MYSYS_STRERROR_SIZE];
        LOG(ERROR) << "Get file name error: " << my_errno() << ", " << strerror_r( my_errno(), errbuf, sizeof(errbuf) );
        strcpy( path, "file name unknown" );
    }

    return std::string( path );
}

/*
  Read a chunk of bytes from a file with retry's if needed

  The parameters are:
    File descriptor
    Buffer to hold at least Count bytes
    Bytes to read
    Flags on what to do on error

    Return:
      -1 on error
      0  if flag has bits MY_NABP or MY_FNABP set
      N  number of bytes read.
*/

size_t my_read(File Filedes, uchar *Buffer, size_t Count, myf MyFlags)
{
  size_t readbytes, save_count;
  DBUG_ENTER("my_read");
  char logBuf[1024] = {0};
  // snprintf(logBuf, sizeof(logBuf), "fd: %d  Buffer: %p  Count: %lu  MyFlags: %d",
  //          Filedes, Buffer, (ulong) Count, MyFlags);
  // LOG(INFO) << logBuf;
  save_count= Count;

  for (;;)
  {
    errno= 0;					/* Linux, Windows don't reset this on EOF/success */
    readbytes= read(Filedes, Buffer, Count);
    // DBUG_EXECUTE_IF ("simulate_file_read_error",
    //                  {
    //                    errno= ENOSPC;
    //                    readbytes= (size_t) -1;
    //                    DBUG_SET("-d,simulate_file_read_error");
    //                    DBUG_SET("-d,simulate_my_b_fill_error");
    //                  });

    if (readbytes != Count)
    {
      set_my_errno(errno);
      // if (errno == 0 || (readbytes != (size_t) -1 &&
      //                    (MyFlags & (MY_NABP | MY_FNABP))))
      //   set_my_errno(HA_ERR_FILE_TOO_SHORT);
      snprintf( logBuf, sizeof(logBuf), "Read only %d bytes off %lu from %d, errno: %d",
                            (int) readbytes, (ulong) Count, Filedes,
                            my_errno());
      LOG(INFO) << logBuf;

      if ((readbytes == 0 || (int) readbytes == -1) && errno == EINTR)
      {  
        LOG(INFO) << "my_read() was interrupted and returned " << (long) readbytes;
        continue;                              /* Interrupted */
      }

      if (MyFlags & (MY_WME | MY_FAE | MY_FNABP))
      {
        char errbuf[MYSYS_STRERROR_SIZE];
        if (readbytes == (size_t) -1)
            ARIES_EXCEPTION( EE_READ, filename(Filedes).data(),
                             my_errno(), strerror_r(my_errno(), errbuf, sizeof(errbuf)) );
        else if (MyFlags & (MY_NABP | MY_FNABP))
            ARIES_EXCEPTION( EE_EOFERR, filename(Filedes).data(),
                             my_errno(), strerror_r(my_errno(), errbuf, sizeof(errbuf)));
      }
      if (readbytes == (size_t) -1 ||
          ((MyFlags & (MY_FNABP | MY_NABP)) && !(MyFlags & MY_FULL_IO)))
        DBUG_RETURN(MY_FILE_ERROR);	/* Return with error */
      if (readbytes != (size_t) -1 && (MyFlags & MY_FULL_IO))
      {
        Buffer+= readbytes;
        Count-= readbytes;
        continue;
      }
    }

    if (MyFlags & (MY_NABP | MY_FNABP))
      readbytes= 0;			/* Ok on read */
    else if (MyFlags & MY_FULL_IO)
      readbytes= save_count;
    break;
  }
  DBUG_RETURN(readbytes);
} /* my_read */
