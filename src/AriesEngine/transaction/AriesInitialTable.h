/*
 * AriesInitialTable.h
 *
 *  Created on: Mar 12, 2020
 *      Author: lichi
 */

#ifndef ARIESINITIALTABLE_H_
#define ARIESINITIALTABLE_H_

#include <fstream>
#include <mutex>
#include <atomic>
#include "utils/utils.h"
#include "AriesTuple.h"
#include "../../AriesAssert.h"
#include "../../CudaAcc/AriesEngineDef.h"
#include "../AriesEngine/AriesDataDef.h"
#include "Compression/dict/AriesDict.h"
using namespace std;

/**
表按列存储结构示例：
/var/rateup/data/data/tpch_100
├── customer
│   ├── customer0_0
│   ├── customer1_0
│   ├── customer2_0
│   ├── customer6_dict_idx_0
| ... ...
│   └── metadata
├── lineitem
│   ├── lineitem0_0 // column 0, data block 0
│   ├── lineitem0_1 // column 0, data block 1
| ... ...
│   ├── lineitem1_0 // column 1, data block 0
│   ├── lineitem1_1 // column 1, data block 1
| ... ...
│   ├── lineitem13_dict_idx_0 // column 13, dict indice block 0
│   ├── lineitem13_dict_idx_1 // column 13, dict indice block 1
│   ├── lineitem13_dict_idx_2 // column 13, dict indice block 2
| ... ...
│   └── metadata
|.. ...

字典文件统一存储在/var/rateup/data/dict_data目录下

不同的数据库的数据文件存在不同的目录下，同一数据库中不同表的数据文件存在不同的子目录下。
每个表的数据目录下有一个metadata文件，记录该表的数据元信息，参见InitTableMetaInfo结构。
数据和字典索引文件按行数分块，每块最大行数为ARIES_BLOCK_FILE_ROW_COUNT。
列文件中的行定长存储。
每个列文件包含长度为ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE大小的头信息，参见BlockFileHeader结构。

*/

#ifdef ARIES_INIT_TABLE_TEST 
static constexpr uint32_t ARIES_BLOCK_FILE_ROW_COUNT = 8;
#elif !defined( NDEBUG )
static constexpr uint32_t ARIES_BLOCK_FILE_ROW_COUNT = 64 * 1024;
#else 
static constexpr uint32_t ARIES_BLOCK_FILE_ROW_COUNT = 20971520; // 20 * 1024 * 1024
#endif

static constexpr uint32_t ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE = ARIES_BLOCK_FILE_ROW_COUNT / 8; // ARIES_BLOCK_FILE_ROW_COUNT / 8;

#define ARIES_INIT_FILE_BITMAP_SIZE( x ) ( x ) * ARIES_BLOCK_FILE_SLOT_BITMAP_SIZE

const std::string ARIES_INIT_TABLE_META_FILE_NAME = "metadata";
const std::string ARIES_DICT_FILE_NAME_SUFFIX = "dict";
const std::string ARIES_DICT_INDEX_FILE_NAME_SUFFIX = "idx";
const std::string ARIES_DICT_COLUMN_FILE_NAME_SUFFIX = "_" + ARIES_DICT_FILE_NAME_SUFFIX + "_" + ARIES_DICT_INDEX_FILE_NAME_SUFFIX;

static constexpr uint32_t ARIES_INIT_TABLE_META_FILE_FIX_LEN = 4096;
struct ARIES_PACKED InitTableMetaInfo
{
    int32_t  m_version;
    uint64_t m_totalRowCount; // 表的总行数
    uint32_t m_blockMaxRowCount; // 一个block存储的数据最大行数: ARIES_BLOCK_FILE_ROW_COUNT
    uint32_t m_blockCount; // block个数，至少有一个block；对于空表，有一个只有block头信息的block文件
    uint64_t m_xlogSN = 0; // xlog序列号，预留
    // 28 bytes so far
    int8_t   m_reserved[ ARIES_INIT_TABLE_META_FILE_FIX_LEN - 28 ]; // 预留字段
    // 所有数据的bitmap，每一行对应一个bit, 表示数据文件中对应的行是否包含有效数据, 0: 无效，1：有效
    int8_t   m_slotBitmap[ 0 ];
};

struct ARIES_PACKED PartitionMetaInfoHeader
{
    int32_t Version;
    uint64_t RowCount;
    uint32_t BlockCount;
};

struct PartitionMetaInfo
{
    int32_t Version;
    uint64_t RowCount;
    uint32_t BlockCount;
    std::vector< int32_t > BlocksID;
};

// block file header actual size, with reserved spaces
static constexpr uint32_t ARIES_COLUMN_BLOCK_FILE_HEADER_SIZE = 4096;
/*
 * Header Info of Column file:
 * version: column file version
 * rows: row count in a block,  int32_t
 * containNull: contain null or not, int8_t
 * itemLen: actual length of one item, int16_t
 * slotBitmap: bit map of the state of slots of the block file
 *             1: occupied, 0: free
 * */
struct ARIES_PACKED BlockFileHeader {
    uint32_t rows; // actual rows in a block
    int8_t   containNull;
    uint16_t itemLen; // 包含nullable flag
    int8_t   checksum[ 8 ]; // 数据校验码, 预留
    int8_t   reserved[ 0 ];
};

BEGIN_ARIES_ENGINE_NAMESPACE

    int GetBlockFileHeaderInfo( IfstreamSPtr colFile,
                                BlockFileHeader& headerInfo );
    int GetBlockFileHeaderInfo( string filePath, BlockFileHeader& headerInfo );
    void ValidateBlockFile(
        IfstreamSPtr blockFile,
        const string& filePath,
        BlockFileHeader& headerInfo,
        AriesColumnType& dataType );

    //　目前只存储列存区数据的头信息，以后可将列数据直接交由此类管理
    //　对于列存区的RowId,均<0

    #define INITIAL_TABLE_ROWPOS_ADJUST(x) (-x - 1)
    #define BLOCK_IDX( rowIdx ) ( rowIdx ) / ARIES_BLOCK_FILE_ROW_COUNT
    #define BLOCK_SLOT_IDX( rowIdx ) ( rowIdx ) % ARIES_BLOCK_FILE_ROW_COUNT

    struct InitTableSlotPos
    {
        index_t m_slotIdx; // global slot index
        int m_blockIdx;
        index_t m_blockSlotIdx; // slot index in the block
    };

    struct UpdateRowData
    {
        index_t m_rowIdx; // row to update
        // parser->GetValidFlagLength() + valid column datas,
        // maintained by caller
        int8_t* m_colDataBuffs;
    };
    using UpdateRowDataPtr = shared_ptr< UpdateRowData >;

    using BlockDataBuff = vector< int8_t >;
    using BlockDataBuffSPtr = shared_ptr< BlockDataBuff >;
    class AriesInitialTable
    {
        struct ColumnFileInfo
        {
            AriesColumnType m_dataType;
            shared_ptr< mutex > m_lock;
            vector< string > m_blockFilePaths;
            vector< IfstreamSPtr > m_blockFileStreams;
            ColumnFileInfo( AriesColumnType argDataType,
                            const vector< string >& argBlkFilePaths,
                            const vector< IfstreamSPtr >& argBlkFileStreams ):
                m_dataType( argDataType ),
                m_blockFilePaths( argBlkFilePaths ),
                m_blockFileStreams( argBlkFileStreams )
            {
                m_lock = make_shared< mutex >();
            }
        };
        using ColumnFileInfoPtr = shared_ptr< ColumnFileInfo >;

    public:
        static const int32_t COLUMN_FILE_VERSION = 1;
        AriesInitialTable( const string& dbName,
                           const string& tableName );

        // used by import csv
        AriesInitialTable( const string& dbName,
                           const string& tableName,
                           const string& dataDir );
        ~AriesInitialTable();

        // static methods
        static string GetPartitionMetaFilePath( const string& dbName,
                                                const string& tableName,
                                                uint32_t partitionIndex );
        static string GetMetaFilePath( const string& dbName, const string& tableName );
        static void WriteMetaFile( const string& filePath,
                                   size_t totalRowCount,
                                   size_t blockCount = UINT64_MAX );
        static void WritePartitionMetaFile( const string& filePath,
                                            size_t totalRowCount,
                                            const std::vector< int32_t > &BlocksID );
        static void InitFiles( const string& dbName,
                               const string& tableName );

        static void WriteColumnBlockFileHeader( ColumnEntryPtr colEntry,
                                                int fd,
                                                const string& filePath,
                                                const uint32_t rowCnt,
                                                bool useMmap );
        static void WriteBlockFileHeader( int fd,
                                          const string& filePath,
                                          const uint32_t rowCnt,
                                          int8_t nullable,
                                          size_t itemStoreSize,
                                          bool useMmap );

        static int GetTotalRowCount( const string& metaInfoFile,
                                     uint64_t& totalRowCount );

        // file names and pathes
        static string GetColumnFileBaseName( const string& tableName, const int colIndex );
        static string GetBlockFileName( const string& tableName,
                                        const int colIndex,
                                        const uint32_t blockIndex,
                                        bool isDictEncoded );
        static string GetBlockFileName( const string& tableName,
                                        const int colIndex,
                                        const uint16_t parttitionIndex,
                                        const uint32_t blockIndex,
                                        bool isDictEncoded );
        static string GetDictIndiceFileBaseName( const string& tableName, const int colIndex );

        string GetMetaFilePath();
        string GetPartitionMetaFilePath( uint32_t partitionIndex );
        void WritePartitionMetaFile( uint32_t partitiontIndex, const PartitionMetaInfo &metaInfo );

    #ifdef ARIES_INIT_TABLE_CACHE
        map< int32_t, AriesColumnSPtr > GetAllColumns()
        {
            return m_table->GetAllColumns();
        }
    #else
        map< int32_t, AriesColumnSPtr > GetAllColumns()
        {
            return map< int32_t, AriesColumnSPtr >();
        }
    #endif

        void GetMetaInfo();

        std::vector< PartitionMetaInfo >& GetPartitionMetaInfo();

        void LoadPartitionMetaInfo();

        /**
         * @brief 更新init table文件中的多行数据
         */
        bool UpdateFileRows( const vector< UpdateRowDataPtr >& updateRowDatas );

        #ifdef BUILD_TEST
        bool IsSlotFree( const index_t slotIdx );
        #endif

        void SetTxMax( RowPos pos, TxId value )
        {
            assert( m_initialTupleHeader.size() == m_initTableMetaInfo.m_totalRowCount );
            CheckRowPos( pos );
            *( reinterpret_cast< atomic< TxId >* >( &m_initialTupleHeader[ -pos - 1 ] ) ) = value;
        }

        TxId GetTxMax( RowPos pos )
        {
            assert( m_initialTupleHeader.size() == m_initTableMetaInfo.m_totalRowCount );
            CheckRowPos( pos );
            return *( reinterpret_cast< atomic< TxId >* >( &m_initialTupleHeader[ -pos - 1 ] ) );
        }

        //对某行加锁
        void Lock( RowPos pos )
        {
            assert( m_pLocks );
            CheckRowPos( pos );
            int8_t expected = 0;
            while( !m_pLocks[ -pos - 1 ].compare_exchange_weak( expected, 1, std::memory_order_release, std::memory_order_relaxed ) )
                expected = 0;
        }

        bool TryLock( RowPos pos )
        {
            assert( m_pLocks );
            CheckRowPos( pos );
            int8_t expected = 0;
            return m_pLocks[ -pos - 1 ].compare_exchange_strong( expected, 1 );
        }

        //对某行解锁
        void Unlock( RowPos pos )
        {
            assert( m_pLocks );
            CheckRowPos( pos );
            int8_t expected = 1;
            m_pLocks[ -pos - 1 ].compare_exchange_strong( expected, 0 );
        }

        //返回block数量
        uint32_t GetBlockCount() const
        {
            return m_initTableMetaInfo.m_blockCount;
        }

        size_t GetBlockRowCount( uint32_t blockIndex );

        uint64_t GetTotalRowCount() const
        {
            return m_initTableMetaInfo.m_totalRowCount;
        }

        uint64_t GetCapacity() const
        {
            return m_capacity;
        }

        //创建一个包含columnIds所有数据的table，并返回。参见AriesTableBlock中的 AddColumn( int32_t columnId, AriesColumnSPtr column )
        //调用者会先调用CacheColumnData缓存数据，然后再调用此函数
        AriesTableBlockUPtr GetTable( const vector< int >& columnIds, const std::vector< AriesCommonExprUPtr >& partitionCondition = {} );

        // used for load csv
        AriesTableBlockUPtr GetTable( const vector< int >& columnIds,
                                      const int startBlockIndex,
                                      const uint32_t startBlockLineIndex );

        int8_t* GetTupleFieldContent( int32_t columnId, RowPos rowPos );

        // clear loaded data
        void Clear();

        std::string GetDbName() const;
        std::string GetTableName() const;

        // columnIndex: start from 0
        AriesDictSPtr GetColumnDict( int columnIndex );
        AriesDictSPtr GetColumnDict( const string& colName );

        string GetBlockFilePath( const int colIndex, const uint32_t blockIndex );

        // for partitioned table
        string GetBlockFilePath( const int colIndex,
                                 const uint16_t partitionIndex,
                                 const uint32_t blockIndex );

        // for test
        // read data of a specified partition
        void ReadColumn( int32_t colIndex,
                         AriesTableBlockUPtr& table,
                         uint32_t partitionIndex );
        void ReadDictColumn( int32_t colIndex,
                             AriesTableBlockUPtr& table,
                             uint32_t partitionIndex );
        AriesColumnSPtr ReadColumnBlockFiles(
            int32_t colIndex,
            uint32_t partitionIndex );

        AriesTableBlockUPtr GetPartitionData( const vector< int >& columnIds, uint32_t partitionIndex );

        /////////////////////////////////////////////////
        // apis for xlog recovery BEGIN

        /**
         * @brief 向 init table 文件中增加数据。
         * 优先使用block文件中的空洞，
         * 如果所有块都满了， 创建新块。
         *
         * @return 成功: 插入的slot位置
         *         失败： NULL_INDEX
        */
        vector< index_t > XLogRecoverInsertRows( const vector< int8_t* >& rowsData );

        void XLogRecoverInsertBatch( int8_t* columnsData,
                                     size_t columnCount,
                                     size_t* columnSizes,
                                     size_t rowCount,
                                     RowPos* rowposes );

        /**
         * @brief 向 dict 压缩的列的字典文件中增加数据。
         *
        */
        void XLogRecoverInsertDict( const vector< int8_t* >& rowsData );

        /**
         * @brief 删除init table文件中的多行数据
         */
        bool XLogRecoverDeleteRows( const vector< index_t >& rowIndice );

        /**
         * @brief 清理列文件中的空洞
         * 从前往后查找到空洞位置i，从后往前查找到有效数据行j，
         * 将j行数据填到i行位置。
         * 重复以上过程，直到 i > j
         */
        void Sweep();

        void XLogRecoverDone();
        static void InitXLogRecover();

        const vector< int8_t* >& GetBlockBitmaps() { return m_slotBitmaps; }

        // apis for xlog recovery END
        /////////////////////////////////////////////////

    private:
        void Open( const vector< int32_t >& columnIds );

        void SetCapacity()
        {
            m_capacity = m_initTableMetaInfo.m_blockCount * ARIES_BLOCK_FILE_ROW_COUNT;
        }

        void ValidateColumnFiles( const size_t colIndex );

        AriesColumnSPtr ReadColumnBlockFiles( int32_t colIndex,
                                              const std::vector< uint32_t >& filterBlockIndex = {} );

        /*
         * @brief 在所有的块中查找一个空闲的slot。
         *
         * @param startIdx 开始位置,
         *
         */
        InitTableSlotPos FindSlotHole( index_t startIdx );

        vector< InitTableSlotPos > FindAllSlotHoles();

        /*
         * @brief 在所有的块中反序查找一个有效数据的slot。
         *
         * @param endIdx 开始位置,
         */
        InitTableSlotPos ReverseFindDataSlot( index_t endIdx );

        vector< InitTableSlotPos > ReverseFindAllDataSlots( int64_t count );

        void UpdateBitmap( index_t& rowIdx, bool set );
        void UpdateBlockBitmap( InitTableSlotPos& pos, bool set );
        void UpdateBlockBitmaps( const int32_t blockIdx,
                                 const index_t startIdx,
                                 const size_t  count,
                                 const bool    set );
        // set all slots of a block, used in sweep
        void SetBlockBitmap( const int32_t blockIdx );
        void TruncateBlocks( const size_t emptyBlockCount );
        void BatchWriteBlockDatas( const vector< BlockDataBuffSPtr >& blockDataBuffs,
                                   uint64 startSlotIdx,
                                   const vector< int8_t* >& rowsData,
                                   size_t& rowIdx, // start row index
                                   uint32_t rowCount, // count of rows to write
                                   const int colCount,
                                   vector< index_t >& slotIndice );

        // replace an existing row: insert into a slot hole
        bool FillSlotHole( InitTableSlotPos& pos,
                           const int8_t* rowData );

        bool FillSlotHole( InitTableSlotPos& pos,
                           vector< int8_t* >& columnBuffers,
                           size_t rowIndex );
        /**
         * @brief 为所有的列创建一个新块
         */
        vector< pair< string, fd_helper_ptr > >
        NewBlockFiles( uint32_t reserveRowCount, uint32_t blockIndex );

        pair< string, fd_helper_ptr >
        NewBlockFile( uint32_t colIndex, uint32_t blockIndex, uint32_t reserveRowCount );

        vector< BlockDataBuffSPtr > XLogRecoverNewBlockBuffs( uint32_t reserveRowCount, uint32_t blockIndex );

        void InitColumnFiles();

        int32_t GetPartitionIDForBlock( const size_t index, const std::vector< AriesCommonExprUPtr >& partitionConditions );

        //将columnIds对应的数据文件分块缓存到内存
        //每个columnId在内存中对应多个内存块(block)
        //可以参考原来scan node的数据分块缓存逻辑
        //默认分块大小，使用ARIES_DATA_BLOCK_ROW_SIZE常量
        //返回总共数据行数
        //可以考虑将此函数包装在GetTable内部，简化外界使用
    #ifdef ARIES_INIT_TABLE_CACHE
        int64_t CacheColumnData( const vector< int32_t >& columnIds );
        void ReadColumn( int32_t colIndex );
        void ReadDictColumn( int32_t colIndex );
    #endif
        AriesTableBlockUPtr GetColumnData( const vector< int32_t >& columnIds, const std::vector< AriesCommonExprUPtr >& partitionCondition = {} );
        void ReadColumn( int32_t colIndex, AriesTableBlockUPtr& table, const std::vector< uint32_t >& filterBlockIndex = {} );
        void ReadDictColumn( int32_t colIndex,
                             AriesTableBlockUPtr& table,
                             const std::vector< uint32_t >& filterBlockIndex = {} );

        void ReadColumnMultiThreads( int32_t colIndex,
                                     uint64_t blockSize );
        vector< AriesDataBufferSPtr > ReadBlocks( const string& filePath,
                                                  AriesColumnType dataType,
                                                  uint64_t blockSize,
                                                  size_t startBlockIdx,
                                                  size_t blockCount );

        void CheckRowIndex( index_t rowIdx );

        BlockDataBuffSPtr GetColumnBlockDataBuffer( int32_t colIndex, int32_t blockIndex, bool createIfNotExists );
        void DeleteColumnBlockDataBuffer( int32_t colIndex, int32_t blockIndex );
        int8_t* GetDictBufferForXLogRecover( int32_t colIndex, bool createIfNotExists );
        void FlushXLogRecoverResult();
        void CheckRowPos( const RowPos pos ) const;

    private:
        string m_dbName;
        string m_tableName;
        string m_dataDir;
        string m_metaInfoFilePath;
        fd_helper_ptr m_metaInfoFd;
        // multiple of ARIES_BLOCK_FILE_ROW_COUNT
        uint64_t m_capacity;
        DatabaseEntrySPtr m_dbEntry;
        TableEntrySPtr m_tableEntry;
        size_t m_columnCount;
        int32_t m_bitmapLength;

    #ifdef ARIES_INIT_TABLE_CACHE
        AriesTableBlockUPtr m_table;
    #endif
        map< int32_t, ColumnFileInfoPtr > m_allColumnInfos; // used when reading column data
        mutex  m_columnInfoLock; // protect m_allColumnInfos

        // used for load csv
        int m_readStartBlockIndex = 0;
        uint32_t m_readStartBlockLineIndex = UINT32_MAX;

        //对应于列存数据的头信息，目前只需要t_xmax信息
        vector< TxId > m_initialTupleHeader;
        unique_ptr< int8_t[] > m_lockFlags; // lock 标志位
        atomic< int8_t >* m_pLocks;// 指向 m_lockFlags首地址
        vector< size_t > m_rowCountOfBlocks;
        InitTableMetaInfo m_initTableMetaInfo;
        vector< int8_t* > m_slotBitmaps;
        std::vector< PartitionMetaInfo > m_partitionMetaInfoArray;

        vector< size_t > m_itemStoreSizes;
        long m_pageSize;

        // use in xlog recovery
        // key: column_index + block_index, value: block data address
        static const size_t m_blockDeltaRowCount = 1024 * 1200;
        vector< bool > m_blockChanged;
        unordered_map< int64_t, BlockDataBuffSPtr > m_columnBlockDataBuffMap;

        map< int32_t, int32_t > m_blockPartitionID;
    };

    using AriesInitialTableSPtr = shared_ptr<AriesInitialTable>;

    bool WriteColumnDataIntoBlocks( AriesInitialTable &initTable,
                                    int colIndex,
                                    bool nullable,
                                    size_t itemSize,
                                    int8_t* buff,
                                    size_t rowCount );

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESINITIALTABLE_H_ */
