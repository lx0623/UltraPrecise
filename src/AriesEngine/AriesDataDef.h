/*
 * AriesDataDef.h
 *
 *  Created on: Oct 29, 2019
 *      Author: lichi
 */

#ifndef ARIESDATADEF_H_
#define ARIESDATADEF_H_
#include <map>
#include <set>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <boost/variant.hpp>
#include "../CudaAcc/AriesEngineDef.h"
using namespace aries_acc;

namespace aries {
    class AriesDict;
    using AriesDictSPtr = shared_ptr< AriesDict >;
}

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesInt64ArraySPtr GetPrefixSumOfBlockSize( const vector< int64_t >& blockSizePrefixSum );

//AriesColumn表示一个物化后的column，在SQL执行阶段，由于数据分块，可能被多个operator共享访问。
//内部用一个或多个AriesDataBufferSPtr 存储物化后的数据。
    class AriesIndices;
    using AriesIndicesSPtr = std::shared_ptr< AriesIndices >;

    class AriesColumn;
    using AriesColumnSPtr = std::shared_ptr< AriesColumn >;

    class AriesDictEncodedColumn;
    using AriesDictEncodedColumnSPtr = std::shared_ptr< AriesDictEncodedColumn >;

    using AriesRefferedColumn = boost::variant< AriesColumnSPtr, AriesDictEncodedColumnSPtr >;

    class AriesBaseColumn;
    using AriesBaseColumnSPtr = shared_ptr< AriesBaseColumn >;

    class AriesBaseColumn
    {
    public:
        EncodeType GetEncodeType() const
        {
            return m_encodeType;
        }
        AriesColumnType GetColumnType() const
        {
            return m_columnType;
        }
        void SetColumnType( AriesColumnType columnType )
        {
            m_columnType = columnType;
        }
        virtual int64_t GetRowCount() const = 0;
        virtual AriesBaseColumnSPtr Clone() const = 0;
        virtual AriesBaseColumnSPtr CloneWithNoContent() const;
        virtual AriesDataBufferSPtr GetDataBuffer() = 0;
        virtual AriesDataBufferSPtr GetDataBuffer( const AriesIndicesSPtr& indices, bool bRunOnGpu ) = 0;
        virtual AriesDataBufferSPtr GetDataBuffer( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu ) = 0;
        virtual int8_t* GetFieldContent( index_t index ) const = 0;
        virtual void MaterilizeSelfByIndices( const AriesIndicesSPtr& indices ) = 0;
        virtual bool IsGpuMaterilizeBetter( size_t resultCount ) = 0;

    protected:
        AriesColumnType m_columnType;
        EncodeType m_encodeType = EncodeType::NONE;
    };

    class AriesColumn : public AriesBaseColumn
    {
    public:
        AriesColumn();
        int64_t AddDataBuffer( const AriesDataBufferSPtr& dataBuffer );
        AriesDataBufferSPtr GetDataBuffer( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu );
        AriesDataBufferSPtr GetDataBuffer( const AriesIndicesSPtr& indices, bool bRunOnGpu );
        // AriesDataBufferSPtr GetDataBuffer( int64_t offset, int64_t count ) const;
        AriesDataBufferSPtr GetDataBuffer();

        vector< AriesDataBufferSPtr > GetDataBuffers( int64_t offset, int64_t& count, bool bStrict ) const;
        vector< AriesDataBufferSPtr > GetDataBuffers() const;
        size_t GetDataBlockCount() const;
        AriesColumnType GetColumnType() const;
        int64_t GetRowCount() const;
        AriesBaseColumnSPtr Clone() const;
        int8_t* GetFieldContent( index_t index ) const;
        void UpdateFieldContent( index_t index, int8_t* newData );
        const vector< int64_t >& GetBlockSizePsumArray() const;
        void PrefetchDataToCpu() const;
        void PrefetchDataToGpu() const;
        virtual void MaterilizeSelfByIndices( const AriesIndicesSPtr& indices );
        bool IsGpuMaterilizeBetter( size_t resultCount );

    private:
        AriesDataBufferSPtr GetDataBufferByCpu( const AriesIndicesArraySPtr& indices, bool hasNull ) const;
        AriesDataBufferSPtr GetDataBufferByCpu( int64_t offset, int64_t count ) const;
        AriesDataBufferSPtr GetDataBufferByCpu();
        AriesDataBufferSPtr GetDataBufferByGpu();

    private:
        int64_t m_totalCount;    //m_dataBlocks 中数据行总数
        vector< AriesDataBufferSPtr > m_dataBlocks;
        vector< int64_t > m_blockSizePrefixSum;    //m_dataBlocks 中block 行数的prefixsum，方便定位数据位置
    };

    class AriesIndices
    {
    public:
        AriesIndices();
        void AddIndices( const AriesIndicesArraySPtr& indices );
        const vector< AriesIndicesArraySPtr >& GetIndicesArray() const;
        //合并成一个返回，并清空m_indices，将新生成的合并后结果插入m_indices
        AriesIndicesArraySPtr GetIndices();
        //copy一份数据给外界
        AriesIndicesArraySPtr GetIndices( int64_t offset, int64_t count ) const;
        int64_t GetRowCount() const;
        void Clear();
        AriesIndicesSPtr Clone() const;
        void SetHasNull( bool bHasNull );
        bool HasNull() const;
        void PrefetchDataToCpu() const;
        void PrefetchDataToGpu() const;
        const vector< int64_t >& GetBlockSizePsumArray() const;
        void MoveToDevice( int deviceId );

    private:
        AriesIndicesArraySPtr GetIndicesByCpu();
        AriesIndicesArraySPtr GetIndicesByGpu();
        AriesIndicesArraySPtr GetIndicesByCpu( int64_t offset, int64_t count ) const;
        AriesIndicesArraySPtr GetIndicesByGpu( int64_t offset, int64_t count ) const;
        vector< AriesIndicesArraySPtr > GetIndicesByCpuAvoidCopy( int64_t offset, int64_t count ) const;

    private:
        bool m_bHasNull;
        vector< AriesIndicesArraySPtr > m_indices;
        int64_t m_rowCount;
        vector< int64_t > m_blockSizePrefixSum;    //方便定位数据位置
    };

    using AriesVariantIndicesArray = AriesDataBuffer;
    using AriesVariantIndicesArraySPtr = AriesDataBufferSPtr;
    using AriesVariantIndices = AriesColumn;
    using AriesVariantIndicesSPtr = AriesColumnSPtr;
    class AriesDictEncodedColumn : public AriesBaseColumn
    {
    public:
        AriesDictEncodedColumn( const aries::AriesDictSPtr& dict, const AriesVariantIndicesSPtr& indices );
        AriesBaseColumnSPtr Clone() const;
        AriesBaseColumnSPtr CloneWithNoContent() const;
        AriesDataBufferSPtr GetDictDataBuffer() const;
        aries::AriesDictSPtr GetDict() const;
        AriesVariantIndicesSPtr GetIndices() const;
        int64_t GetRowCount() const
        {
            return m_indices->GetRowCount();
        }
        AriesDataBufferSPtr GetDataBuffer( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu );
        AriesDataBufferSPtr GetDataBuffer( const AriesIndicesSPtr& indices, bool bRunOnGpu );
        AriesDataBufferSPtr GetDataBuffer();
        int8_t* GetFieldContent( index_t index ) const;

        virtual void MaterilizeSelfByIndices( const AriesIndicesSPtr& indices );
        bool IsGpuMaterilizeBetter( size_t resultCount );

    private:
        aries::AriesDictSPtr m_dict;
        AriesDataBufferSPtr m_mtrlzedBuff;
        // AriesIndicesSPtr m_indices;    //到字典的索引
        AriesVariantIndicesSPtr m_indices;    //到字典的索引
    };

    struct ColumnRefRowInfo
    {
        int64_t TotalRowNum; //为物化这个一个column的block，需要输入多少行数据
        int64_t MyRowNum; //这个column block物化后的数据有多少行
    };

    //表示一个未物化的column
    class AriesColumnReference;
    using AriesColumnReferenceSPtr = std::shared_ptr< AriesColumnReference >;
    class AriesColumnReference
    {
    public:
        AriesColumnReference( const AriesBaseColumnSPtr& refColumn );
        void SetIndices( const AriesIndicesSPtr& indices );
        AriesIndicesSPtr GetIndices() const;
        int64_t GetRowCount() const;
        AriesColumnReferenceSPtr CloneWithEmptyContent() const;
        AriesColumnReferenceSPtr Clone() const;
        AriesDataBufferSPtr GetDataBuffer( bool bRunOnGpu = false );
        AriesDataBufferSPtr GetDataBufferUsingIndices( const AriesIndicesArraySPtr& indices, bool hasNull, bool bRunOnGpu ) const;
        AriesBaseColumnSPtr GetReferredColumn() const;
        EncodeType GetReferredColumnEncodeType() const;
        ColumnRefRowInfo GetRowInfo() const;
        bool IsMaterializeNeeded() const;

    private:
        // AriesColumnSPtr m_refColumn;        //将引用的数据源看做一个连续数组
        AriesBaseColumnSPtr m_refColumn;
        AriesIndicesSPtr m_indices;    //表示数组下标
        // 对于字典压缩的列，保存物化后的临时结果，同时在table block中保留AriesColumnReference.
        // 适用于以下情况：
        // q12的group操作中，对字典压缩的列o_orderpriority进行了物化，
        // 但是在sum操作的case表达式中还需要对o_orderpriority进行等于比较，
        // 此等值比较已经被转换为索引值的比较。
        AriesDataBufferSPtr m_mtrlzedBuff;
    };

    struct AriesTableBlockStats
    {
        AriesTableBlockStats()
                : m_materializeTime( 0 ), m_updateIndiceTime( 0 ), m_getSubTableTime( 0 )
        {

        }

        AriesTableBlockStats( const AriesTableBlockStats& tableStats )
        {
            m_materializeTime = ( long )tableStats.m_materializeTime;
            m_updateIndiceTime = ( long )tableStats.m_updateIndiceTime;
            m_getSubTableTime = ( long )tableStats.m_getSubTableTime;
        }

        AriesTableBlockStats& operator=( const AriesTableBlockStats& tableStats )
        {
            m_materializeTime = ( long )tableStats.m_materializeTime;
            m_updateIndiceTime = ( long )tableStats.m_updateIndiceTime;
            m_getSubTableTime = ( long )tableStats.m_getSubTableTime;
            return *this;
        }

        void Print( const string& extraMsg ) const;
        string ToJson(int inputRows) const;
        AriesTableBlockStats& operator+=( const AriesTableBlockStats& tableStats )
        {
            m_materializeTime += tableStats.m_materializeTime;
            m_updateIndiceTime += tableStats.m_updateIndiceTime;
            m_getSubTableTime += tableStats.m_getSubTableTime;
            return *this;
        }

        bool operator <( const AriesTableBlockStats& tableStats ) const
        {
            return m_materializeTime + m_updateIndiceTime + m_getSubTableTime
                    < tableStats.m_materializeTime + tableStats.m_updateIndiceTime + tableStats.m_getSubTableTime;
        }
        atomic< long > m_materializeTime;
        atomic< long > m_updateIndiceTime;
        atomic< long > m_getSubTableTime;
    };

    // 表示一个表块，节点间传输数据的单位
    class AriesTableBlock;
    using AriesTableBlockUPtr = std::unique_ptr< AriesTableBlock >;
    class AriesTableBlock
    {
    public:
        AriesTableBlock();

        string GetColumnName( int32_t columnId ) const;
        void AddColumnName( int32_t columnId, const string& name );
        AriesLiteralValue GetLiteralValue();
        //添加属于这个table block的column。不能重复添加！
        //列如scan node会调用这个函数仅仅一次。后续block的添加由scan node通过自己保存的AriesColumnSPtr直接添加
        //该函数在进行数据输出时可能被调用，在收集数据时不应该被调用！
        void AddColumn( int32_t columnId, AriesColumnSPtr column );
        void AddColumn( int32_t columnId, AriesColumnReferenceSPtr columnRef );
        void AddColumn( int32_t columnId, const AriesDictEncodedColumnSPtr& dictColumn );

        //供scan node使用
        void AddColumn( const map< int32_t, AriesColumnSPtr >& columns, int64_t offset, int64_t count );

        //用一个column替换表中对应的column(只能替换掉原表中对应物化的列，非物化的列不能替换），目前只有sort节点可能会调用该函数(建议调用带多个参数的UpdateIndices一步到位）。其他场景支持，根据具体需求再进行设计
        void UpdateColumn( int32_t columnId, AriesColumnSPtr column );

        void UpdateColumnIds( const map< int32_t, int32_t >& idNewAndOld );

        //不在toKeep中的Column将被删除
        void KeepColumns( const set< int32_t >& toKeep );

        //在toRemove中的Column将被删除
        void RemoveColumns( const set< int32_t >& toRemove );

        //获取该table block包含的总数据行数
        int64_t GetRowCount() const;

        int64_t GetColumnCount() const;

        //仅仅在select常量时使用。需要在outputColumnIds为空时，将数据行数传递给下一个节点
        void SetRowCount( int64_t count );

        //当数据接收端需要累积数据进行处理时，调用此函数收集收到的table block。对调用者而言，所有的ｂｌｏｃｋ作为成为一个整体对外提供数据。
        void AddBlock( AriesTableBlockUPtr table );

        //数据接收端收集完数据后，根据自己的处理能力，进行分块读取。新生成的table block内部包含物化和非物化的列其中
        //1.对于物化的列直接是数据本身
        //2.对于非物化的列，需要保存自己的indices和完整的原始数据块引用(AriesColumnSPtr）
        AriesTableBlockUPtr GetSubTable( int64_t offset, int64_t count, bool bStrict = false ) const;

        AriesTableBlockUPtr GetSubTable2( int totalSliceCount, int sliceIdx ) const;

        AriesTableBlockUPtr GetOneBlock( int blockIndex ) const;
        size_t GetBlockCount() const;

        void SetBlockPartitionID( int blockIndex, int partition );

        //在join时，分别针对左表和右表调用MakeTableByColumns，生成两个新的table，然后调用MergeTable将两个新table横向合并成join的结果
        AriesTableBlockUPtr MakeTableByColumns( const vector< int32_t >& columnIds, bool bReturnCopy = true );
        void MergeTable( AriesTableBlockUPtr src );

        AriesTableBlockUPtr Clone( bool bDeepCopy = true );

        std::vector< int32_t > GetAllColumnsId() const;

        //隐式物化，获取某column完整数据。
        //需要遍历m_tables保证数据被完整获取。(sort是典型使用此函数的场景)
        AriesDataBufferSPtr GetColumnBuffer( int32_t columnId, bool bReturnCopy = false, bool bRunOnGpu = false );

        //生成临时物化给外界使用
        AriesDataBufferSPtr GetColumnBufferByIndices( int32_t columnId, const AriesIndicesArraySPtr& indices, bool bRunOnGpu = false ) const;

        //返回columnId对应的column是否没有被物化
        bool IsColumnUnMaterilized( int32_t columnId ) const;

        EncodeType GetColumnEncodeType( int32_t columnId ) const;

        //获取未物化的column
        AriesColumnReferenceSPtr GetUnMaterilizedColumn( int32_t columnId ) const;

        //获取物化的column,内部可能包含多个data buffer block!
        AriesColumnSPtr GetMaterilizedColumn( int32_t columnId ) const;

        AriesDataBufferSPtr GetDictEncodedColumnIndiceBuffer( const int32_t columnId );
        AriesDictEncodedColumnSPtr GetDictEncodedColumn( int32_t columnId ) const;

        //该函数在进行数据输出时调用，用以更新相关数据，内部操作流程如下：
        //1.遍历和更新m_indices。让所有未物化的列下标得到更新
        //2.将输入的indices添加到m_indices中
        //3.遍历m_columns，将除输入的columnId以外的物化列和输入的indices一起打包，生成新的m_columnReferences
        //4.清空m_columns
        //5.如果columnId != -1 且column != nullptr，则将输入的columnId，column插入到m_columns中，成为唯一的物化列
        void UpdateIndices( const AriesIndicesArraySPtr& indexArray, bool bHasNull = false, int32_t columnId = -1, AriesColumnSPtr column = nullptr );

        // TODO: is this necessary?
        bool VerifyContent() const;
        AriesTableBlockUPtr CloneWithNoContent() const;

        AriesColumnType GetColumnType( int32_t columnId ) const;
        vector< AriesColumnType > GetColumnTypes( const vector< int32_t >& columnIds ) const;
        static AriesTableBlockUPtr CreateTableWithNoRows( const vector< AriesColumnType >& types );

        void MaterilizeAll();
        ColumnRefRowInfo GetRowInfo() const;
        // void TryShrinkData( int marginSize = 100000, int ratio = 5 );
        map< int32_t, AriesColumnSPtr > GetAllColumns();
        bool ColumnExists( int32_t columnId ) const;
        void ResetTimeCostStats();
        void ResetAllStats();
        const AriesTableBlockStats& GetStats() const
        {
            return m_stats;
        }

        void SetStats( const AriesTableBlockStats& stats )
        {
            m_stats = stats;
        }

        int GetDeviceId() const
        {
            return m_deviceId;
        }

        void SetDeviceId( int deviceId )
        {
            m_deviceId = deviceId;
        }

        void MoveIndicesToDevice( int deviceId );

        vector< int > GetAllMaterilizedColumnIds() const;
        void MaterilizeColumns( const vector< int > columnIds, bool bRunOnGpu = false );
        void MaterilizeDictEncodedColumn( int columnId );
        AriesIndicesArraySPtr GetTheSharedIndiceForColumns( const vector< int > columnIds ) const;
        void ReplaceTheOnlyOneIndices( const AriesIndicesArraySPtr& indices, bool bHasNull = false );
        void MaterilizeAllDataBlocks();
        vector< int64_t > GetMaterilizedColumnDataBlockSizePsumArray() const;


        /**
         * @brief 获取该表的 partition ID
         * @return 如果没有 partition 则返回 -1
         */
        int32_t GetPartitionID() const;

        /**
         * @brief 获取该表用于 partition 的 column ID
         * @return 如果没有 partition 则返回 -1
         */
        int32_t GetPartitionedColumnID() const;

        void SetPartitionedColumnID( int32_t partitionedColumnID );
        void SetPartitionID( int32_t partitionID );

        void ClearPartitionInfo();

        const map< int32_t, int32_t >& GetPartitionInfo() const;
        void SetPartitionInfo( const map< int32_t, int32_t >& info );

    private:
        bool IsAllColumnsShareSameIndices() const;
        bool IsColumnsShareSameIndices( const vector< int > columnIds ) const;
        void CheckIfNeedMaterilize();
        map< int32_t, int32_t > GetColRefIndicePos() const;
        map< int32_t, int32_t > GetColRefIndicePos( const vector< int32_t >& columnIds ) const;
        set< AriesIndicesSPtr > FindReferredIndices( const map< int32_t, AriesColumnReferenceSPtr >& colRefs ) const;
        bool MergeOk( const AriesTableBlockUPtr& src ) const;
        void ShuffleIndices( const AriesIndicesArraySPtr& indices );
        bool IsMaterilizedColumnHasSameBlockSize() const;
        void CleanupIndices();

    private:
        map< int32_t, string > m_columnNames; //引擎不操作此数据
        map< int32_t, AriesColumnSPtr > m_columns;
        map< int32_t, AriesColumnReferenceSPtr > m_columnReferences;
        map< int32_t, AriesDictEncodedColumnSPtr > m_dictEncodedColumns;
        map< int32_t, int32_t > m_blockPartitionID;
        vector< AriesIndicesSPtr > m_indices;

        int32_t m_partitionID;
        int32_t m_partitionedColumnID;

        int64_t m_rowCount; //仅仅在主动调用了SetRowCount后才有效
        int m_deviceId;
        // for performance statistics
        mutable AriesTableBlockStats m_stats;
    };

END_ARIES_ENGINE_NAMESPACE

#endif /* ARIESDATADEF_H_ */
