//
// Created by david.shen on 2020/3/19.
//

#pragma once

#include "AriesEngine/transaction/AriesMvccTable.h"

BEGIN_ARIES_ENGINE_NAMESPACE

#define DELETE_INITIALTABLE_ROWID1 -1
#define DELETE_INITIALTABLE_ROWID2 -2
#define DELETE_INITIALTABLE_ROWID3 -3
#define UPDATE_INITIALTABLE_ROWID1 -4
#define UPDATE_INITIALTABLE_ROWID2 -5
#define UPDATE_INITIALTABLE_ROWID3 -6
#define INSERT_TXID 3
#define DELETE_TXID 4
#define UPDATE_TXID 5

class AriesMvccTestDataGenerator {
public:
    AriesMvccTestDataGenerator(AriesMvccTableSPtr &table);

    TupleDataSPtr InsertLineItemData();
    // TupleDataSPtr DeleteLineItemData();
    TupleDataSPtr UpdateLineItemData();

    /**
     * 该方法会产生 10 个 tuple，类型和状态对应的 tuple 个数为：
     * ----------------------------------------------------
     * |  类型    |  COMMITED  |  ABORTED  |  IN_PROGRESS  |
     * ----------------------------------------------------
     * | insert  |     2个     |    1个    |       1个     |
     * ----------------------------------------------------
     * | delete  |     1个     |    1个    |       1个     |
     * ----------------------------------------------------
     * | update  |     1个     |    1个    |       1个     |
     * ----------------------------------------------------
     * 其中 insert 的 L_ORDERKEY 值为 2147483647
     * update 的 L_ORDERKEY 的值为 2147483646
     */
    void GenerateTuples();

    /* for region table:
    CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));
    */
    static TupleDataSPtr GenerateRegionTupleData(int regionkey, string &name, string &comment);
    void InsertTuple(TupleDataSPtr newDataBuffer, int dataIndex, AriesTransactionPtr transaction, int cid=0, TransactionStatus status=TransactionStatus::COMMITTED);
    void ModifyTuple(RowPos rowId, TupleDataSPtr newDataBuffer, int dataIndex, AriesTransactionPtr transaction, int cid=0, TransactionStatus status=TransactionStatus::COMMITTED);
    void DeleteTuple(RowPos rowId, AriesTransactionPtr transaction, int cid=0, TransactionStatus status=TransactionStatus::COMMITTED);

private:
    AriesMvccTableSPtr m_table;
};

using AriesMvccTestDataGeneratorSPtr = shared_ptr<AriesMvccTestDataGenerator>;

END_ARIES_ENGINE_NAMESPACE
