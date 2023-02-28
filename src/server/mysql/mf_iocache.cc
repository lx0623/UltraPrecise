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

/*
  Cashing of files with only does (sequential) read or writes of fixed-
  length records. A read isn't allowed to go over file-length. A read is ok
  if it ends at file-length and next read can try to read after file-length
  (and get a EOF-error).
  Possibly use of asyncronic io.
  macros for read and writes for faster io.
  Used instead of FILE when reading or writing whole files.
  This code makes mf_rec_cache obsolete (currently only used by ISAM)
  One can change info->pos_in_file to a higher value to skip bytes in file if
  also info->read_pos is set to info->read_end.
  If called through open_cached_file(), then the temporary file will
  only be created if a write exeeds the file buffer or if one calls
  my_b_flush_io_cache().

  If one uses SEQ_READ_APPEND, then two buffers are allocated, one for
  reading and another for writing.  Reads are first done from disk and
  then done from the write buffer.  This is an efficient way to read
  from a log file when one is writing to it at the same time.
  For this to work, the file has to be opened in append mode!
  Note that when one uses SEQ_READ_APPEND, one MUST write using
  my_b_append !  This is needed because we need to lock the mutex
  every time we access the write buffer.

TODO:
  When one SEQ_READ_APPEND and we are reading and writing at the same time,
  each time the write buffer gets full and it's written to disk, we will
  always do a disk read to read a part of the buffer from disk to the
  read buffer.
  This should be fixed so that when we do a my_b_flush_io_cache() and
  we have been reading the write buffer, we should transfer the rest of the
  write buffer to the read buffer before we start to reuse it.
*/

#include <errno.h>
#include <glog/logging.h>
// #include "utils/cpu_timer.h"
#include <server/mysql/include/mysql_com.h>
#include <server/mysql/include/sql_class.h>
#include "./include/my_sys.h"
#include "./include/m_string.h"
#include "./include/my_thread_local.h"
#include "CpuTimer.h"

#define lock_append_buffer(info) \
  mysql_mutex_lock(&(info)->append_buffer_lock)
#define unlock_append_buffer(info) \
  mysql_mutex_unlock(&(info)->append_buffer_lock)

#define IO_ROUND_UP(X) (((X)+IO_SIZE-1) & ~(IO_SIZE-1))
#define IO_ROUND_DN(X) ( (X)            & ~(IO_SIZE-1))

/* from mf_reccache.c */
ulong my_default_record_cache_size=RECORD_CACHE_SIZE;

/*
  Setup internal pointers inside IO_CACHE

  SYNOPSIS
    setup_io_cache()
    info		IO_CACHE handler

  NOTES
    This is called on automaticly on init or reinit of IO_CACHE
    It must be called externally if one moves or copies an IO_CACHE
    object.
*/

void setup_io_cache(IO_CACHE* info)
{
  /* Ensure that my_b_tell() and my_b_bytes_in_cache works */
  if (info->type == WRITE_CACHE)
  {
    info->current_pos= &info->write_pos;
    info->current_end= &info->write_end;
  }
  else
  {
    info->current_pos= &info->read_pos;
    info->current_end= &info->read_end;
  }
}


static void
init_functions(IO_CACHE* info)
{
  enum cache_type type= info->type;
  switch (type) {
  case READ_NET:
    /*
      Must be initialized by the caller. The problem is that
      _my_b_net_read has to be defined in sql directory because of
      the dependency on THD, and therefore cannot be visible to
      programs that link against mysys but know nothing about THD, such
      as myisamchk
    */
    break;
  // case SEQ_READ_APPEND:
  //   info->read_function = _my_b_seq_read;
  //   info->write_function = 0;			/* Force a core if used */
  //   break;
  default:
    info->read_function = /* info->share ? _my_b_read_r : */_my_b_read;
    // info->write_function = _my_b_write;
  }

  setup_io_cache(info);
}


/*
  Initialize an IO_CACHE object

  SYNOPSIS
    init_io_cache_ext()
    info               cache handler to initialize
    file               File that should be associated to to the handler
                       If == -1 then real_open_cached_file()
                       will be called when it's time to open file.
    cachesize          Size of buffer to allocate for read/write
                       If == 0 then use my_default_record_cache_size
    type               Type of cache
    seek_offset        Where cache should start reading/writing
    use_async_io       Set to 1 of we should use async_io (if avaiable)
    cache_myflags      Bitmap of different flags
                       MY_WME | MY_FAE | MY_NABP | MY_FNABP |
                       MY_DONT_CHECK_FILESIZE
    file_key           Instrumented file key for temporary cache file

  RETURN
    0  ok
    #  error
*/

int init_io_cache_ext(IO_CACHE *info, File file, size_t cachesize,
                      enum cache_type type, my_off_t seek_offset,
                      pbool use_async_io, myf cache_myflags/*,
                      PSI_file_key file_key*/)
{
  size_t min_cache;
  my_off_t pos;
  my_off_t end_of_file= ~(my_off_t) 0;
  DBUG_ENTER("init_io_cache_ext");
  // DBUG_PRINT("enter",("cache: 0x%lx  type: %d  pos: %ld",
  //            (ulong) info, (int) type, (ulong) seek_offset));

  info->read_time = 0;
  info->file= file;
  // info->file_key= file_key;
  info->type= TYPE_NOT_SET;	    /* Don't set it until mutex are created */
  info->pos_in_file= seek_offset;
  info->pre_close = info->pre_read = info->post_read = 0;
  info->arg = 0;
  info->alloced_buffer = 0;
  info->buffer=0;
  info->seek_not_done= 0;

  if (file >= 0)
  {
    pos= inline_mysql_file_tell(file, MYF(0));
    if ((pos == (my_off_t) -1) && (my_errno() == ESPIPE))
    {
      /*
         This kind of object doesn't support seek() or tell(). Don't set a
         flag that will make us again try to seek() later and fail.
      */
      info->seek_not_done= 0;
      /*
        Additionally, if we're supposed to start somewhere other than the
        the beginning of whatever this file is, then somebody made a bad
        assumption.
      */
      DBUG_ASSERT(seek_offset == 0);
    }
    else
      info->seek_not_done= MY_TEST(seek_offset != pos);
  }

  info->disk_writes= 0;
  info->share=0;

  if (!cachesize && !(cachesize= my_default_record_cache_size))
    DBUG_RETURN(1);        /* No cache requested */
  min_cache=use_async_io ? IO_SIZE*4 : IO_SIZE*2;
  if (type == READ_CACHE || type == SEQ_READ_APPEND)
  {                       /* Assume file isn't growing */
    if (!(cache_myflags & MY_DONT_CHECK_FILESIZE))
    {
      /* Calculate end of file to avoid allocating oversized buffers */
      end_of_file= inline_mysql_file_seek(file, 0L, MY_SEEK_END, MYF(0));
      /* Need to reset seek_not_done now that we just did a seek. */
      info->seek_not_done= end_of_file == seek_offset ? 0 : 1;
      if (end_of_file < seek_offset)
        end_of_file=seek_offset;
      /* Trim cache size if the file is very small */
      if ((my_off_t) cachesize > end_of_file-seek_offset+IO_SIZE*2-1)
      {
        cachesize= (size_t) (end_of_file-seek_offset)+IO_SIZE*2-1;
        use_async_io=0;    /* No need to use async */
      }
    }
  }
  cache_myflags &= ~MY_DONT_CHECK_FILESIZE;
  if (type != READ_NET && type != WRITE_NET)
  {
    /* Retry allocating memory in smaller blocks until we get one */
    cachesize= ((cachesize + min_cache-1) & ~(min_cache-1));
    for (;;)
    {
      size_t buffer_block;
      /*
        Unset MY_WAIT_IF_FULL bit if it is set, to prevent conflict with
        MY_ZEROFILL.
      */
      myf flags= (myf) (cache_myflags & ~(MY_WME | MY_WAIT_IF_FULL));

      if (cachesize < min_cache)
        cachesize = min_cache;
      buffer_block= cachesize;
      if (type == SEQ_READ_APPEND)
        buffer_block *= 2;
      if (cachesize == min_cache)
        flags|= (myf) MY_WME;

      if ((info->buffer= (uchar*) malloc(buffer_block)) != 0)
      {
        info->write_buffer=info->buffer;
        if (type == SEQ_READ_APPEND)
          info->write_buffer = info->buffer + cachesize;
        info->alloced_buffer=1;
        break;    /* Enough memory found */
      }
      if (cachesize == min_cache)
        DBUG_RETURN(2);    /* Can't alloc cache */
      /* Try with less memory */
      cachesize= (cachesize*3/4 & ~(min_cache-1));
    }
  }

  // DBUG_PRINT("info",("init_io_cache: cachesize = %lu", (ulong) cachesize));
  LOG( INFO ) << "Read cache size: " << cachesize;
  info->read_length=info->buffer_length=cachesize;
  info->myflags=cache_myflags & ~(MY_NABP | MY_FNABP);
  info->request_pos= info->read_pos= info->write_pos = info->buffer;
//   if (type == SEQ_READ_APPEND)
//   {
//     info->append_read_pos = info->write_pos = info->write_buffer;
//     info->write_end = info->write_buffer + info->buffer_length;
//     mysql_mutex_init(key_IO_CACHE_append_buffer_lock,
//                      &info->append_buffer_lock, MY_MUTEX_INIT_FAST);
//   }
// #if defined(SAFE_MUTEX)
//   else
//   {
//     /* Clear mutex so that safe_mutex will notice that it's not initialized */
//     memset(&info->append_buffer_lock, 0, sizeof(info->append_buffer_lock));
//   }
// #endif

  if (type == WRITE_CACHE)
    info->write_end=
      info->buffer+info->buffer_length- (seek_offset & (IO_SIZE-1));
  else
    info->read_end=info->buffer;		/* Nothing in cache */

  /* End_of_file may be changed by user later */
  info->end_of_file= end_of_file;
  info->error=0;
  info->type= type;
  init_functions(info);
  DBUG_RETURN(0);
}  /* init_io_cache_ext */

/*
  Initialize an IO_CACHE object

  SYNOPSIS
    init_io_cache() - Wrapper for init_io_cache_ext()

  NOTE
    This function should be used if the IO_CACHE tempfile is not instrumented.
*/

int init_io_cache(IO_CACHE *info, File file, size_t cachesize,
                  enum cache_type type, my_off_t seek_offset,
                  pbool use_async_io, myf cache_myflags)
{
  return init_io_cache_ext(info, file, cachesize, type, seek_offset,
                           use_async_io, cache_myflags/*, key_file_io_cache*/);
}

/*
  Use this to reset cache to re-start reading or to change the type
  between READ_CACHE <-> WRITE_CACHE
  If we are doing a reinit of a cache where we have the start of the file
  in the cache, we are reusing this memory without flushing it to disk.
*/

my_bool reinit_io_cache(IO_CACHE *info, enum cache_type type,
			my_off_t seek_offset,
			pbool use_async_io MY_ATTRIBUTE((unused)),
			pbool clear_cache)
{
  DBUG_ENTER("reinit_io_cache");
  // DBUG_PRINT("enter",("cache: 0x%lx type: %d  seek_offset: %lu  clear_cache: %d",
// 		      (ulong) info, type, (ulong) seek_offset,
// 		      (int) clear_cache));

  /* One can't do reinit with the following types */
  DBUG_ASSERT(type != READ_NET && info->type != READ_NET &&
	      type != WRITE_NET && info->type != WRITE_NET &&
	      type != SEQ_READ_APPEND && info->type != SEQ_READ_APPEND);

  /* If the whole file is in memory, avoid flushing to disk */
  if (! clear_cache &&
      seek_offset >= info->pos_in_file &&
      seek_offset <= my_b_tell(info))
  {
    /* Reuse current buffer without flushing it to disk */
    uchar *pos;
    if (info->type == WRITE_CACHE && type == READ_CACHE)
    {
      info->read_end=info->write_pos;
      info->end_of_file=my_b_tell(info);
      /*
        Trigger a new seek only if we have a valid
        file handle.
      */
      info->seek_not_done= (info->file != -1);
    }
    else if (type == WRITE_CACHE)
    {
      if (info->type == READ_CACHE)
      {
	info->write_end=info->write_buffer+info->buffer_length;
	info->seek_not_done=1;
      }
      info->end_of_file = ~(my_off_t) 0;
    }
    pos=info->request_pos+(seek_offset-info->pos_in_file);
    if (type == WRITE_CACHE)
      info->write_pos=pos;
    else
      info->read_pos= pos;
  }
  else
  {
    /*
      If we change from WRITE_CACHE to READ_CACHE, assume that everything
      after the current positions should be ignored
    */
    if (info->type == WRITE_CACHE && type == READ_CACHE)
      info->end_of_file=my_b_tell(info);
    /* flush cache if we want to reuse it */
    if (!clear_cache && my_b_flush_io_cache(info,1))
      DBUG_RETURN(1);
    info->pos_in_file=seek_offset;
    /* Better to do always do a seek */
    info->seek_not_done=1;
    info->request_pos=info->read_pos=info->write_pos=info->buffer;
    if (type == READ_CACHE)
    {
      info->read_end=info->buffer;		/* Nothing in cache */
    }
    else
    {
      info->write_end=(info->buffer + info->buffer_length -
		       (seek_offset & (IO_SIZE-1)));
      info->end_of_file= ~(my_off_t) 0;
    }
  }
  info->type=type;
  info->error=0;
  init_functions(info);

  DBUG_RETURN(0);
} /* reinit_io_cache */



/*
  Read buffered.

  SYNOPSIS
    _my_b_read()
      info                      IO_CACHE pointer
      Buffer                    Buffer to retrieve count bytes from file
      Count                     Number of bytes to read into Buffer

  NOTE
    This function is only called from the my_b_read() macro when there
    isn't enough characters in the buffer to satisfy the request.

  WARNING

    When changing this function, be careful with handling file offsets
    (end-of_file, pos_in_file). Do not cast them to possibly smaller
    types than my_off_t unless you can be sure that their value fits.
    Same applies to differences of file offsets.

    When changing this function, check _my_b_read_r(). It might need the
    same change.

  RETURN
    0      we succeeded in reading all data
    1      Error: couldn't read requested characters. In this case:
             If info->error == -1, we got a read error.
             Otherwise info->error contains the number of bytes in Buffer.
*/

int _my_b_read(IO_CACHE *info, uchar *Buffer, size_t Count)
{
#ifdef ARIES_PROFILE
    aries::CPU_Timer t;
    t.begin();
#endif
  size_t length,diff_length,left_length, max_length;
  my_off_t pos_in_file;
  DBUG_ENTER("_my_b_read");

  /* If the buffer is not empty yet, copy what is available. */
  if ((left_length= (size_t) (info->read_end-info->read_pos)))
  {
    DBUG_ASSERT(Count >= left_length);	/* User is not using my_b_read() */
    memcpy(Buffer,info->read_pos, left_length);
    Buffer+=left_length;
    Count-=left_length;
  }

  /* pos_in_file always point on where info->buffer was read */
  pos_in_file=info->pos_in_file+ (size_t) (info->read_end - info->buffer);

  /*
    Whenever a function which operates on IO_CACHE flushes/writes
    some part of the IO_CACHE to disk it will set the property
    "seek_not_done" to indicate this to other functions operating
    on the IO_CACHE.
  */
  if (info->seek_not_done)
  {
    if ((inline_mysql_file_seek(info->file, pos_in_file, MY_SEEK_SET, MYF(0))
        != MY_FILEPOS_ERROR))
    {
      /* No error, reset seek_not_done flag. */
      info->seek_not_done= 0;
    }
    else
    {
      /*
        If the seek failed and the error number is ESPIPE, it is because
        info->file is a pipe or socket or FIFO.  We never should have tried
        to seek on that.  See Bugs#25807 and #22828 for more info.
      */
      DBUG_ASSERT(my_errno() != ESPIPE);
      info->error= -1;
      DBUG_RETURN(1);
    }
  }

  /*
    Calculate, how much we are within a IO_SIZE block. Ideally this
    should be zero.
  */
  diff_length= (size_t) (pos_in_file & (IO_SIZE-1));

  /*
    If more than a block plus the rest of the current block is wanted,
    we do read directly, without filling the buffer.
  */
  if (Count >= (size_t) (IO_SIZE+(IO_SIZE-diff_length)))
  {					/* Fill first intern buffer */
    size_t read_length;
    if (info->end_of_file <= pos_in_file)
    {
      /* End of file. Return, what we did copy from the buffer. */
      info->error= (int) left_length;
#ifdef ARIES_PROFILE
        info->read_time += t.end();
#endif
      DBUG_RETURN(1);
    }
    /*
      Crop the wanted count to a multiple of IO_SIZE and subtract,
      what we did already read from a block. That way, the read will
      end aligned with a block.
    */
    length=(Count & (size_t) ~(IO_SIZE-1))-diff_length;
    if ((read_length= inline_mysql_file_read(info->file,Buffer, length, info->myflags))
	!= length)
    {
      /*
        If we didn't get, what we wanted, we either return -1 for a read
        error, or (it's end of file), how much we got in total.
      */
      info->error= (read_length == (size_t) -1 ? -1 :
		    (int) (read_length+left_length));
#ifdef ARIES_PROFILE
        info->read_time += t.end();
#endif
      DBUG_RETURN(1);
    }
    Count-=length;
    Buffer+=length;
    pos_in_file+=length;
    left_length+=length;
    diff_length=0;
  }

  /*
    At this point, we want less than one and a partial block.
    We will read a full cache, minus the number of bytes, we are
    within a block already. So we will reach new alignment.
  */
  max_length= info->read_length-diff_length;
  /* We will not read past end of file. */
  if (info->type != READ_FIFO &&
      max_length > (info->end_of_file - pos_in_file))
    max_length= (size_t) (info->end_of_file - pos_in_file);
  /*
    If there is nothing left to read,
      we either are done, or we failed to fulfill the request.
    Otherwise, we read max_length into the cache.
  */
  if (!max_length)
  {
    if (Count)
    {
      /* We couldn't fulfil the request. Return, how much we got. */
      info->error= (int)left_length;
#ifdef ARIES_PROFILE
        info->read_time += t.end();
#endif
      DBUG_RETURN(1);
    }
    length=0;				/* Didn't read any chars */
  }
  else if ((length= inline_mysql_file_read(info->file,info->buffer, max_length,
                            info->myflags)) < Count ||
	   length == (size_t) -1)
  {
    /*
      We got an read error, or less than requested (end of file).
      If not a read error, copy, what we got.
    */
    if (length != (size_t) -1)
      memcpy(Buffer, info->buffer, length);
    info->pos_in_file= pos_in_file;
    /* For a read error, return -1, otherwise, what we got in total. */
    info->error= length == (size_t) -1 ? -1 : (int) (length+left_length);
    info->read_pos=info->read_end=info->buffer;
#ifdef ARIES_PROFILE
      info->read_time += t.end();
#endif
      DBUG_RETURN(1);
  }
  /*
    Count is the remaining number of bytes requested.
    length is the amount of data in the cache.
    Read Count bytes from the cache.
  */
  info->read_pos=info->buffer+Count;
  info->read_end=info->buffer+length;
  info->pos_in_file=pos_in_file;
  memcpy(Buffer, info->buffer, Count);
#ifdef ARIES_PROFILE
    info->read_time += t.end();
#endif
    DBUG_RETURN(0);
}

/* Read one byte when buffer is empty */

int _my_b_get(IO_CACHE *info)
{
  uchar buff;
  IO_CACHE_CALLBACK pre_read,post_read;
  if ((pre_read = info->pre_read))
    (*pre_read)(info);
  if ((*(info)->read_function)(info,&buff,1))
  {
      float s = ( info->read_time + 0.0 ) / ( 1000 * 1000 );
      LOG(INFO) << filename( info->file ) << " time ( read ): "
                <<  s << " s, " << s / 60 << "m";
      return my_b_EOF;
  }
  if ((post_read = info->post_read))
    (*post_read)(info);
  return (int) (uchar) buff;
}

/* Flush write cache */

#define LOCK_APPEND_BUFFER if (need_append_buffer_lock) \
  lock_append_buffer(info);
#define UNLOCK_APPEND_BUFFER if (need_append_buffer_lock) \
  unlock_append_buffer(info);

int my_b_flush_io_cache(IO_CACHE *info,
                        int need_append_buffer_lock MY_ATTRIBUTE((unused)))
{
    // size_t length;
    // my_off_t pos_in_file;
    my_bool append_cache= (info->type == SEQ_READ_APPEND);
    DBUG_ENTER("my_b_flush_io_cache");
    // DBUG_PRINT("enter", ("cache: 0x%lx", (long) info));

    if (!append_cache)
        need_append_buffer_lock= 0;

    if (info->type == WRITE_CACHE || append_cache)
    {
        // if (info->file == -1)
        // {
        //     if (real_open_cached_file(info))
        //         DBUG_RETURN((info->error= -1));
        // }
        // LOCK_APPEND_BUFFER;

        // if ((length=(size_t) (info->write_pos - info->write_buffer)))
        // {
        //     /*
        //       In case of a shared I/O cache with a writer we do direct write
        //       cache to read cache copy. Do it before the write here so that
        //       the readers can work in parallel with the write.
        //       copy_to_read_buffer() relies on info->pos_in_file.
        //     */
        //     if (info->share)
        //         copy_to_read_buffer(info, info->write_buffer, length);

        //     pos_in_file=info->pos_in_file;
        //     /*
        //   If we have append cache, we always open the file with
        //   O_APPEND which moves the pos to EOF automatically on every write
        //     */
        //     if (!append_cache && info->seek_not_done)
        //     {					/* File touched, do seek */
        //         if (mysql_file_seek(info->file, pos_in_file, MY_SEEK_SET, MYF(0)) ==
        //             MY_FILEPOS_ERROR)
        //         {
        //             UNLOCK_APPEND_BUFFER;
        //             DBUG_RETURN((info->error= -1));
        //         }
        //         if (!append_cache)
        //             info->seek_not_done=0;
        //     }
        //     if (!append_cache)
        //         info->pos_in_file+=length;
        //     info->write_end= (info->write_buffer+info->buffer_length-
        //                       ((pos_in_file+length) & (IO_SIZE-1)));

        //     if (mysql_file_write(info->file,info->write_buffer,length,
        //                          info->myflags | MY_NABP))
        //         info->error= -1;
        //     else
        //         info->error= 0;
        //     if (!append_cache)
        //     {
        //         set_if_bigger(info->end_of_file,(pos_in_file+length));
        //     }
        //     else
        //     {
        //         info->end_of_file+=(info->write_pos-info->append_read_pos);
        //         DBUG_ASSERT(info->end_of_file == mysql_file_tell(info->file, MYF(0)));
        //     }

        //     info->append_read_pos=info->write_pos=info->write_buffer;
        //     ++info->disk_writes;
        //     UNLOCK_APPEND_BUFFER;
        //     DBUG_RETURN(info->error);
        // }
    }
    UNLOCK_APPEND_BUFFER;
    DBUG_RETURN(0);
}
/*
  Free an IO_CACHE object

  SYNOPSOS
    end_io_cache()
    info		IO_CACHE Handle to free

  NOTES
    It's currently safe to call this if one has called init_io_cache()
    on the 'info' object, even if init_io_cache() failed.
    This function is also safe to call twice with the same handle.

  RETURN
   0  ok
   #  Error
*/

int end_io_cache(IO_CACHE *info)
{
  int error=0;
  IO_CACHE_CALLBACK pre_close;
  DBUG_ENTER("end_io_cache");
  // DBUG_PRINT("enter",("cache: 0x%lx", (ulong) info));

  /*
    Every thread must call remove_io_thread(). The last one destroys
    the share elements.
  */
  DBUG_ASSERT(!info->share || !info->share->total_threads);

  if ((pre_close=info->pre_close))
  {
    (*pre_close)(info);
    info->pre_close= 0;
  }
  if (info->alloced_buffer)
  {
    info->alloced_buffer=0;
    if (info->file != -1)			/* File doesn't exist */
      error= my_b_flush_io_cache(info,1);
    free(info->buffer);
    info->buffer=info->read_pos=(uchar*) 0;
  }
  if (info->type == SEQ_READ_APPEND)
  {
    /* Destroy allocated mutex */
    info->type= TYPE_NOT_SET;
    mysql_mutex_destroy(&info->append_buffer_lock);
  }
  DBUG_RETURN(error);
} /* end_io_cache */

// extern "C" {

/**
  Read buffered from the net.

  @retval
    1   if can't read requested characters
  @retval
    0   if record read
*/


int _my_b_net_read(IO_CACHE *info, uchar *Buffer,
                   size_t Count MY_ATTRIBUTE((unused)))
{
    ulong read_length;
    NET *net= current_thd->get_protocol_classic()->get_net();
    DBUG_ENTER("_my_b_net_read");

    if (!info->end_of_file)
        DBUG_RETURN(1);	/* because my_b_get (no _) takes 1 byte at a time */
    read_length=my_net_read(net);
    if (read_length == packet_error)
    {
        info->error= -1;
        DBUG_RETURN(1);
    }
    if (read_length == 0)
    {
        info->end_of_file= 0;			/* End of file from client */
        DBUG_RETURN(1);
    }
    /* to set up stuff for my_b_get (no _) */
    info->read_end = (info->read_pos = net->read_pos) + read_length;
    Buffer[0] = info->read_pos[0];		/* length is always 1 */

    /*
      info->request_pos is used by log_loaded_block() to know the size
      of the current block.
      info->pos_in_file is used by log_loaded_block() too.
    */
    info->pos_in_file+= read_length;
    info->request_pos=info->read_pos;

    info->read_pos++;

    DBUG_RETURN(0);
}

// } /* extern "C" */
