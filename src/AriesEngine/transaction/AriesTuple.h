/*! 
  @file AriesTuple.h
  @brief Delta table中的tuple
 */

#pragma once
#include <stdio.h>
#include <cstdint>
#include <atomic>
#include "CudaAcc/AriesEngineDef.h"
#include "AriesTransaction.h"
#include "AriesColumnType.h"
#include "schema/TableEntry.h"

using namespace std;
using namespace aries_acc;
using namespace schema;

BEGIN_ARIES_ENGINE_NAMESPACE

/*!
 * RowPos值说明:
 * <0: 来至initialTable
 * >0: 来至deltaTable
 * =0: 无效值
 * */

#define INVALID_ROWPOS 0

    struct TupleHeader
    {
        TupleHeader() {
            m_xmin = INVALID_TX_ID;
            m_xmax = INVALID_TX_ID;
            m_initialRowPos = INVALID_ROWPOS;
            m_deadFlag = false;
            m_lockFlag = false;
        }

        void Initial(TxId xMin, TxId xMax, RowPos initialTableRowPos) {
            m_xmin = xMin;
            m_xmax = xMax;
            m_initialRowPos = initialTableRowPos;
            m_deadFlag = false;
            m_lockFlag = false;
        }

        TxId m_xmin; //!<插入本tuple的事务id
        atomic< TxId > m_xmax; //!<修改或删除本tuple的事务id
        RowPos m_initialRowPos; //!<列文件里的行号 ( -1, -2, ...)
        atomic< bool > m_deadFlag; //!<是否为dead tuple
        atomic< bool > m_lockFlag; //!<是否加了锁
    };

    typedef TupleHeader * pTupleHeader;

    /*
    需要更新或插入的数据结构
    map key: columnId, 从1开始计数
    map value: value buffer, 可包含多行数据
    */
    struct TupleData
    {
        map<int, AriesDataBufferSPtr> data;
    };

    using TupleDataSPtr = shared_ptr<TupleData>;


    bool TransferColumnData(const string& colName, const AriesDataBuffer& target, const AriesDataBuffer& source );

    class TupleParser
    {
    public:
        TupleParser(TableEntrySPtr &tableEntry);

        /*
        * 根据数据填写tuple的data区域
        * newBuf: 新数据存放的buffer
        * dataBuffer: 在newBuf里存放的数据Buffer
        * dataIndex: newBuf所存数据在Buffer里的位置
        * */
        void FillData( const std::vector< int8_t* >& columnBuffers, TupleDataSPtr dataBuffer, int dataIndex );

        AriesColumnType & GetColumnType( int colId );

        inline int GetTupleSize() const
        {
            return m_tupleSize;
        }

        inline size_t GetColumnsCount() const
        {
            return m_columnTypes.size();
        }

        inline const vector<AriesColumnType>& GetColumnTypes() const 
        {
            return m_columnTypes;
        }

    private:
        /*
        * ParsTableColumnInfo: 根据table的信息拆分出定长和变长列的id和对应的长度
        * 1, 拆分定长和定长列的id及其对于的长度
        * 2, 计算定长区域的数据长度
        * */
        void ParsTableColumnInfo( TableEntrySPtr &tableEntry );

    private:
        int m_tupleSize;                                //tuple的数据区长度
        vector<AriesColumnType> m_columnTypes;          //存放各列数据类型
        vector<string> m_columnName;
    };

    using TupleParserSPtr = shared_ptr<TupleParser>;

END_ARIES_ENGINE_NAMESPACE
