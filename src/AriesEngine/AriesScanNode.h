/*
 * AriesScanNode.h
 *
 *  Created on: Aug 31, 2018
 *      Author: lichi
 */

#pragma once

#include "AriesDataDef.h"
#include "AriesOpNode.h"
#include "../frontend/PhysicalTable.h"
using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE

#define USE_DATA_CACHE

    class AriesScanNode: public AriesOpNode
    {
        using IfstreamSPtr = shared_ptr<ifstream>;

        struct FileInfo
        {
            IfstreamSPtr FileStream;
            AriesColumnType DataType;
#ifdef USE_DATA_CACHE
            //added for data cache
            string colName;
            int blockId;
#endif
        };

    public:
        AriesScanNode(const string& dbName);
        ~AriesScanNode();

    public:
        void SetOutputColumnIds( const vector< int >& columnIds );
        void SetPhysicalTable( PhysicalTablePointer physicalTable );

        /*
         * @input parameter:
         *  filePath: full path of column file
         * @output parameters:
         *  rows: total rows in file
         *  containNull: rows contain Null or not
         *  itemLen: actual item length saved in column file
         *
         * @return:
         *   1 : success, but file format is older version, no output parameter saved in file, output is not changed
         *   0 : success, caller can use output parameters;
         *   -1: file is not exist
         *   -2: header size in file is not correct
         *   -3: header version in file is not correct
         * */
        static int getColumnFileHeaderInfo(string filePath, uint64_t &rows, int8_t &containNull, int16_t &itemLen);
        static int getHeaderInfo(IfstreamSPtr colFile, uint64_t &rows, int8_t &containNull, int16_t &itemLen);

    public:
        bool Open() override final;
        AriesOpResult GetNext() override final;
        void Close() override final;

    private:
        AriesDataBufferSPtr ReadNextBlock( FileInfo& fileInfo, int64_t blockSize, int64_t maxReadCount );

    private:
        string m_dbName;
        vector< int > m_outputColumnIds;
        std::map<int32_t, AriesColumnSPtr> m_columns;
        PhysicalTablePointer m_physicalTable;
        vector< FileInfo > m_files;
        int64_t m_readRowCount;
        int64_t m_totalRowNum;
#ifdef USE_DATA_CACHE
        //added for data cache
        string m_tableName;
#endif
    };

    using AriesScanNodeSPtr = shared_ptr< AriesScanNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace QueryEngine */
