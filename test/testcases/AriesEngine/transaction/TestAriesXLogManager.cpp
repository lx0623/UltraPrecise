#include <gtest/gtest.h>

#include "AriesEngine/transaction/AriesXLogManager.h"
#include "../../../TestUtils.h"

using namespace aries_engine;

namespace aries_test
{

ARIES_UNIT_TEST_CLASS( TestAriesXLogManager )
{
protected:
    void SetUp() override
    {
        AriesXLogManager::GetInstance();
    }

    void TearDown() override
    {

    }
};

ARIES_UNIT_TEST_F( TestAriesXLogManager, constructor )
{

}

ARIES_UNIT_TEST_F( TestAriesXLogManager, reader )
{
    auto reader = AriesXLogManager::GetInstance().GetReader();

    do
    {
        auto log_pair = reader->Next();
        if ( log_pair.first == nullptr )
        {
            break;
        }

        // printf("operation code: %u, table_id = %ld, txid = %d\n", static_cast< uint16_t >( log_pair.first->operation ),
        //         log_pair.first->tableId, log_pair.first->txid );
    } while ( true );
}

}