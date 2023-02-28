#include "utils.h"
#include <sys/stat.h>
#include <sys/statfs.h>
#include <dirent.h>
#include <server/mysql/include/my_sys.h>
size_t block_size( int fd )
{
    struct statfs st;
    if ( fstatfs(fd, &st) != -1 )
        return (size_t) st.f_bsize;
    else
        return MIN_BATCH_WRITE_BUFF_SIZE;
}
int64_t filesize(int fd)
{
    struct stat st;
    if ( fstat(fd, &st) == -1 )
    {
        return -1;
    }
    return (int64_t) st.st_size;
}
bool flushWriteBuff( int fd,
                     shared_ptr<WRITE_BUFF_INFO>& writeBuff )
{
    if ( writeBuff->getDataSize() > 0 )
    {
        size_t writtenSize = my_write( fd,
                                       writeBuff->get(),
                                       writeBuff->getDataSize(),
                                       MYF( MY_FNABP ) );
        if ( 0 != writtenSize )
        {
            return false;
        }
        writeBuff->clear();
    }
    return true;
}
bool batchWrite( int fd,
                 shared_ptr<WRITE_BUFF_INFO>& writeBuff,
                 uchar* data,
                 size_t dataSize,
                 bool flush )
{
    // batch write column
    size_t writtenSize;
    size_t appendedSize = writeBuff->append( data, dataSize );

    while ( appendedSize < dataSize )
    {
        if ( writeBuff->isFull() )
        {
            writtenSize = my_write(fd,
                                   writeBuff->get(),
                                   writeBuff->getDataSize(),
                                   MYF( MY_FNABP ));
            if ( 0 != writtenSize )
            {
                return false;
            }
            writeBuff->clear();
        }
        size_t leftSize = dataSize - appendedSize;
        if ( leftSize > 0 )
        {
            appendedSize += writeBuff->append( data + appendedSize, leftSize );
        }
    }
    if ( writeBuff->isFull() || flush )
    {
        writtenSize = my_write(fd,
                               writeBuff->get(),
                               writeBuff->getDataSize(),
                               MYF( MY_FNABP ));
        if ( 0 != writtenSize )
        {
            return false;
        }
        writeBuff->clear();
    }
    return true;
}

/**
 * list files or directories in a directory
*/
std::vector< std::string > listFiles( const string& dir, bool file )
{
    std::vector< std::string > files;
    string fileName;
    if ( auto dirPtr = opendir( dir.c_str() ) )
    {
        while ( auto f = readdir( dirPtr ) )
        {
            if ( '.' == f->d_name[0] )
            {
                continue;  // Skip everything that starts with a dot
            }
            if ( file )
            {
                if ( DT_REG != f->d_type )
                {
                    LOG(INFO) << "Skip: " << f->d_name;
                    continue;
                }
            }
            else
            {
                if ( DT_DIR != f->d_type )
                {
                    continue;
                }
            }
            fileName.assign( f->d_name );
            if (file)
            {
                LOG(INFO) << "Found file " << fileName;
            }
            else
            {
                LOG(INFO) << "Found dir " << fileName;
            }
            files.emplace_back(fileName);
        }
        closedir( dirPtr );
    }
    return files;
}

void partitionItems( const size_t itemCount,
                     const size_t partitionCount,
                     vector<size_t>& partitionItemCounts,
                     vector<size_t> *partitionItemStartIndex )
{
    size_t perPartitionItemCount = itemCount / partitionCount; // count of items for each partition
    size_t extraItemCount = itemCount % partitionCount;
    partitionItemCounts.assign( partitionCount, perPartitionItemCount );

    for ( size_t tmpJobIdx = 0; tmpJobIdx < extraItemCount; ++tmpJobIdx )
    {
        partitionItemCounts[ tmpJobIdx ] += 1;
    }
    if ( partitionItemStartIndex )
    {
        partitionItemStartIndex->assign( partitionCount, 0 );
        if ( partitionCount > 1 )
        {
            for ( size_t threadIdx = 1; threadIdx < partitionCount; ++threadIdx )
            {
                partitionItemStartIndex->operator[]( threadIdx ) =
                    partitionItemStartIndex->operator[]( threadIdx - 1 ) +
                        partitionItemCounts[ threadIdx -1 ];
            }
        }
    }
}