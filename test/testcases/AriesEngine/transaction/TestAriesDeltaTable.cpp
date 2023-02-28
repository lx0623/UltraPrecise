/*
 * TestAriesDeltaTable.cpp
 *
 *  Created on: Mar 26, 2020
 *      Author: lichi
 */
#include <gtest/gtest.h>
#include "AriesEngine/transaction/AriesDeltaTable.h"
using namespace aries_engine;

TEST( AriesDeltaTable, delta )
{
    TxId currentTxId = 30;
    
    AriesDeltaTable deltaTable( 4 );

    Snapshot snapShot { 19, 19, vector< TxId >() };
    AriesTransManager& transManager = AriesTransManager::GetInstance();
    transManager.RemoveCommitLog( 100 );
    transManager.AddTx( 13, TransactionStatus::COMMITTED );
    transManager.AddTx( 14, TransactionStatus::ABORTED );
    transManager.AddTx( 15, TransactionStatus::COMMITTED );
    transManager.AddTx( 16, TransactionStatus::ABORTED );
    transManager.AddTx( 17, TransactionStatus::COMMITTED );
    transManager.AddTx( 18, TransactionStatus::ABORTED );
    transManager.AddTx( 19, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( 20, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( currentTxId, TransactionStatus::IN_PROGRESS );

    vector< RowPos > allVisibleIds;
    vector< RowPos > allInitialIds;
    vector< RowPos > visibleIds;
    vector< RowPos > initialIds;

    //tx13 insert 3 tuples, committed
    TxId txId = 13;
    bool unused;
    vector< RowPos > slots = deltaTable.ReserveSlot( 3, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId3 = slots[ 0 ];
    TupleHeader* header = deltaTable.GetTupleHeader( rowId3, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    RowPos rowId = slots[ 1 ];
    header = deltaTable.GetTupleHeader( rowId, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    RowPos rowId2 = slots[ 2 ];
    header = deltaTable.GetTupleHeader( rowId2, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    allVisibleIds.push_back( rowId3 );
    allVisibleIds.push_back( rowId );
    allVisibleIds.push_back( rowId2 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //tx14 insert a tuple, aborted
    txId = 14;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId4 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //tx15 update a tx3 tuple, committed
    txId = 15;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId5 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId5, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.SetTxMax( rowId3, txId );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId );
    allVisibleIds.push_back( rowId2 );
    allVisibleIds.push_back( rowId5 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //tx16 update a tx5 tuple, aborted
    txId = 16;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId6 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId6, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.SetTxMax( rowId5, txId );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //tx17 delete a tx3 tuple, committed
    txId = 17;
    deltaTable.SetTxMax( rowId, txId );
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId2 );
    allVisibleIds.push_back( rowId5 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );

    //tx18 delete a tx5 tuple, aborted
    txId = 18;
    deltaTable.SetTxMax( rowId5, txId );
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId2 );
    allVisibleIds.push_back( rowId5 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );

    //tx19 insert a tuple, inprogress
    txId = 19;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId9 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId9, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );

    //tx20 update a tx5 tuple and delete a tx3 tuple, inprogress
    txId = 20;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId10 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId10, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.SetTxMax( rowId5, txId );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    deltaTable.SetTxMax( rowId2, txId );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );

    //currentTxId insert a tuple, inprogress
    txId = currentTxId;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowIdcurrentTxId = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowIdcurrentTxId, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    allVisibleIds.push_back( rowIdcurrentTxId );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //currentTxId update a tuple, inprogress
    txId = currentTxId;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowIdcurrentTxId2 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowIdcurrentTxId2, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.SetTxMax( rowIdcurrentTxId, currentTxId );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId2 );
    allVisibleIds.push_back( rowId5 );
    allVisibleIds.push_back( rowIdcurrentTxId2 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 2 ], allVisibleIds[ 2 ] );

    //currentTxId delete a tuple, inprogress
    deltaTable.SetTxMax( rowIdcurrentTxId2, currentTxId );
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId2 );
    allVisibleIds.push_back( rowId5 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
}

TEST( AriesDeltaTable, initial )
{
    TxId currentTxId = 30;
    AriesDeltaTable deltaTable( 4 );
    Snapshot snapShot { 19, 19, vector< TxId >() };
    AriesTransManager& transManager = AriesTransManager::GetInstance();
    transManager.RemoveCommitLog( 100 );
    transManager.AddTx( 13, TransactionStatus::COMMITTED );
    transManager.AddTx( 14, TransactionStatus::ABORTED );
    transManager.AddTx( 15, TransactionStatus::COMMITTED );
    transManager.AddTx( 16, TransactionStatus::ABORTED );
    transManager.AddTx( 17, TransactionStatus::COMMITTED );
    transManager.AddTx( 18, TransactionStatus::ABORTED );
    transManager.AddTx( 19, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( 20, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( currentTxId, TransactionStatus::IN_PROGRESS );

    vector< RowPos > allVisibleIds;
    vector< RowPos > allInitialIds;
    vector< RowPos > visibleIds;
    vector< RowPos > initialIds;

    //tx13 delete a tuples, committed
    bool isContinuous;
    TxId txId = 13;
    vector< RowPos > slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    RowPos rowId3 = slots[ 0 ];
    TupleHeader* header = deltaTable.GetTupleHeader( rowId3, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -1 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    allInitialIds.push_back( -1 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );

    //tx14 delete a tuple, aborted
    txId = 14;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    RowPos rowId4 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -2 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );

    //tx15 update a tuple, committed
    txId = 15;
    bool unused;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId5 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId5, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    auto rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -3 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allVisibleIds.clear();
    allVisibleIds.push_back( rowId5 );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );

    //tx16 update a tuple, aborted
    txId = 16;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId6 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId6, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -4 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );

    //tx19 update, delete a tuple, inprogress
    txId = 19;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId9_1 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId9_1, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -5 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -6 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );

    //currentTxId update, delete a tuple, inprogress
    txId = currentTxId;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos currentTxId_1 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( currentTxId_1, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -7 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -8 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allVisibleIds.clear();
    allVisibleIds.push_back( rowId5 );
    allVisibleIds.push_back( currentTxId_1 );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -3 );
    allInitialIds.push_back( -7 );
    allInitialIds.push_back( -8 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );
    ASSERT_EQ( initialIds[ 2 ], allInitialIds[ 2 ] );
    ASSERT_EQ( initialIds[ 3 ], allInitialIds[ 3 ] );
}

TEST( AriesDeltaTable, complex )
{
    TxId currentTxId = 30;
    AriesDeltaTable deltaTable( 4 );
    Snapshot snapShot { 19, 19, vector< TxId >() };
    AriesTransManager& transManager = AriesTransManager::GetInstance();
    transManager.RemoveCommitLog( 100 );
    transManager.AddTx( 13, TransactionStatus::COMMITTED );
    transManager.AddTx( 14, TransactionStatus::ABORTED );
    transManager.AddTx( 15, TransactionStatus::COMMITTED );
    transManager.AddTx( 16, TransactionStatus::ABORTED );
    transManager.AddTx( 17, TransactionStatus::COMMITTED );
    transManager.AddTx( 18, TransactionStatus::ABORTED );
    transManager.AddTx( 19, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( 20, TransactionStatus::IN_PROGRESS );
    transManager.AddTx( currentTxId, TransactionStatus::IN_PROGRESS );

    vector< RowPos > allVisibleIds;
    vector< RowPos > allInitialIds;
    vector< RowPos > visibleIds;
    vector< RowPos > initialIds;

    //tx13 update a tuples, committed
    TxId txId = 13;
    bool unused;
    vector< RowPos > slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId3 = slots[ 0 ];
    TupleHeader* header = deltaTable.GetTupleHeader( rowId3, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    bool isContinuous;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    auto rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -1 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allInitialIds.push_back( -1 );
    allVisibleIds.push_back( rowId3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );

    //tx14 delete tx13 tuple, aborted
    txId = 14;
    header = deltaTable.GetTupleHeader( rowId3, AriesDeltaTableSlotType::AddedTuples );
    header->m_xmax = txId;
    //header->m_ctid = INVALID_ROWPOS;
    allInitialIds.clear();
    allVisibleIds.clear();
    allInitialIds.push_back( -1 );
    allVisibleIds.push_back( rowId3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );

    //tx15 delete tx13 tuple, committed
    txId = 15;
    header = deltaTable.GetTupleHeader( rowId3, AriesDeltaTableSlotType::AddedTuples );
    header->m_xmax = txId;
   // header->m_ctid = INVALID_ROWPOS;
    allInitialIds.clear();
    allVisibleIds.clear();
    allInitialIds.push_back( -1 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );

    //tx17 update a tuple, committed
    txId = 17;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos rowId7 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId7, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );
    
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -2 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allVisibleIds.clear();
    allVisibleIds.push_back( rowId7 );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -2 );

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );

    //tx19 delete tx17 tuple, inprogress
    txId = 19;
    header = deltaTable.GetTupleHeader( rowId7, AriesDeltaTableSlotType::AddedTuples );
    header->m_xmax = txId;
    //header->m_ctid = INVALID_ROWPOS;

    deltaTable.GetVisibleRowIdsInDeltaTable( currentTxId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );

    allVisibleIds.clear();
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -2 );
    deltaTable.GetVisibleRowIdsInDeltaTable( txId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );

    //currentTxId update, delete a tuple, inprogress
    txId = currentTxId;
    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused );
    RowPos currentTxId_1 = slots[ 0 ];
    header = deltaTable.GetTupleHeader( currentTxId_1, AriesDeltaTableSlotType::AddedTuples );
    header->Initial( txId, INVALID_TX_ID, INVALID_ROWPOS );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::AddedTuples );

    slots = deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::DeletedInitialTableTuples, isContinuous );
    rowId4delete = slots[ 0 ];
    header = deltaTable.GetTupleHeader( rowId4delete, AriesDeltaTableSlotType::DeletedInitialTableTuples );
    header->Initial( INVALID_TX_ID, txId, -3 );
    deltaTable.CompleteSlot( slots, AriesDeltaTableSlotType::DeletedInitialTableTuples );

    allVisibleIds.clear();
    allVisibleIds.push_back( rowId7 );
    allVisibleIds.push_back( currentTxId_1 );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -2 );
    allInitialIds.push_back( -3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( txId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( visibleIds[ 1 ], allVisibleIds[ 1 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );
    ASSERT_EQ( initialIds[ 2 ], allInitialIds[ 2 ] );

    header = deltaTable.GetTupleHeader( currentTxId_1, AriesDeltaTableSlotType::AddedTuples );
    header->m_xmax = txId;
    //header->m_ctid = INVALID_ROWPOS;
    allVisibleIds.clear();
    allVisibleIds.push_back( rowId7 );
    allInitialIds.clear();
    allInitialIds.push_back( -1 );
    allInitialIds.push_back( -2 );
    allInitialIds.push_back( -3 );

    deltaTable.GetVisibleRowIdsInDeltaTable( txId, snapShot, visibleIds, initialIds );
    ASSERT_EQ( visibleIds.size(), allVisibleIds.size() );
    ASSERT_EQ( initialIds.size(), allInitialIds.size() );
    ASSERT_EQ( visibleIds[ 0 ], allVisibleIds[ 0 ] );
    ASSERT_EQ( initialIds[ 0 ], allInitialIds[ 0 ] );
    ASSERT_EQ( initialIds[ 1 ], allInitialIds[ 1 ] );
    ASSERT_EQ( initialIds[ 2 ], allInitialIds[ 2 ] );
}

TEST( AriesDeltaTable, overReserveSlots )
{
    AriesDeltaTable deltaTable( 4 );
    bool unused;
    ASSERT_EQ( deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused ).size(), 1 );
    ASSERT_EQ( deltaTable.ReserveSlot( 4, AriesDeltaTableSlotType::AddedTuples, unused ).size(), 4 );
    deltaTable.FreeSlot( { 1 }, AriesDeltaTableSlotType::AddedTuples );
    ASSERT_EQ( deltaTable.ReserveSlot( 1, AriesDeltaTableSlotType::AddedTuples, unused ).size(), 1 );
}