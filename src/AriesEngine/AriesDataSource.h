/*
 * AriesDataSource.h
 *
 *  Created on: Feb 27, 2019
 *      Author: lichi
 */

#pragma once

#include "AriesDataBlock.h"
#include "CudaAcc/AriesSqlOperator.h"
#include "AriesAssert.h"

bool IsCurrentThdKilled();
void SendKillMessage();

BEGIN_ARIES_ENGINE_NAMESPACE

    class AriesDynamicKernel
    {
    public:
        AriesDynamicKernel()
        {
        }
        virtual ~AriesDynamicKernel()
        {
        }
        virtual void SetCuModule( const vector< CUmoduleSPtr >& modules )
        {
        }
        virtual string GetCudaKernelCode() const
        {
            return string();
        }
    };

    enum class DataSourceStatus
    {
        CONTINUE, END, ERROR
    };

    class AriesDataSource : public AriesDynamicKernel
    {
    public:
        AriesDataSource()
        {
        }
        virtual ~AriesDataSource()
        {
        }
        virtual bool Open() = 0;
        virtual pair< DataSourceStatus, AriesDataBlockUPtr > GetNext( int64_t blockSize = ARIES_DATA_BLOCK_ROW_SIZE ) = 0;
        virtual void Close() = 0;
        void SetLimit( int64_t offset, int64_t size )
        {
            m_limitOffset = offset;
            m_limitSize = size;
        }
        pair< int64_t, int64_t > GetOutputOffset( int64_t tupleNum )
        {
            int64_t startPos = -1;
            int64_t outputTupleNum = 0;
            if( tupleNum > 0 )
            {
                if( m_curPos + tupleNum > m_limitOffset )
                {
                    startPos = std::max( m_limitOffset - m_curPos, 0l );
                    outputTupleNum = std::min( tupleNum - startPos, m_limitSize - m_totalOutput );
                    ARIES_ASSERT( outputTupleNum > 0 , "outputTupleNum: " + to_string(outputTupleNum));
                }
            }
            return { startPos, outputTupleNum };
        }
    protected:
        int64_t m_limitOffset;
        int64_t m_limitSize;
        int64_t m_curPos;
        int64_t m_totalOutput;
    };

    using AriesDataSourceSPtr = shared_ptr< AriesDataSource >;

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */

