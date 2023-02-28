//
// Created by david.shen on 2020/3/19.
//

#include "AriesMvccTestDataGenerator.h"
#include "datatypes/decimal.hxx"

BEGIN_ARIES_ENGINE_NAMESPACE

    AriesMvccTestDataGenerator::AriesMvccTestDataGenerator(AriesMvccTableSPtr &table)
    {
        m_table = table;
    }

    TupleDataSPtr AriesMvccTestDataGenerator::InsertLineItemData()
    {
        /*
        CREATE TABLE LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);
        */
        int columnId = 0;
        TupleDataSPtr newdata = make_shared<TupleData>();
        AriesColumnType intColumnType({AriesValueType::INT32}, false);
        int intValue = 2147483647;
        AriesDataBufferSPtr dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType decimalColumnType({aries::AriesValueType::COMPACT_DECIMAL, 15, 2}, false);
        aries_acc::Decimal decimalValue(15, 2, 0, "8888888888.88");
        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        AriesColumnType charColumnType({AriesValueType::CHAR, 1}, false);
        char charValue = 'Z';
        dataBuf = make_shared<AriesDataBuffer>(charColumnType, 1);
        memcpy(dataBuf->GetData(), &charValue, 1);
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(charColumnType, 1);
        memcpy(dataBuf->GetData(), &charValue, 1);
        newdata->data[++columnId] = dataBuf;

        AriesColumnType dateColumnType({AriesValueType::DATE}, false);
        aries_acc::AriesDate dateValue(2020, 3, 19);
        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string25ColumnType({AriesValueType::CHAR, 25}, false);
        char char25[25];
        memcpy(char25, "This is a rateup testcase", 25);
        dataBuf = make_shared<AriesDataBuffer>(string25ColumnType, 1);
        memcpy(dataBuf->GetData(), char25, sizeof(char25));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string10ColumnType({AriesValueType::CHAR, 10}, false);
        char char10[10];
        memcpy(char10, "a testcase", 10);
        dataBuf = make_shared<AriesDataBuffer>(string10ColumnType, 1);
        memcpy(dataBuf->GetData(), char10, sizeof(char10));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string44ColumnType({AriesValueType::CHAR, 44}, false);
        char char44[44];
        memcpy(char44, "This are more rateup testcases! yes!!!!!!!!!", sizeof(char44));
        dataBuf = make_shared<AriesDataBuffer>(string44ColumnType, 1);
        memcpy(dataBuf->GetData(), char44, sizeof(char44));
        newdata->data[++columnId] = dataBuf;

        return newdata;
    }

    // void AriesMvccTestDataGenerator::DeleteLineItemData()
    // {
    //     m_table->SetTxMax(DELETE_INITIALTABLE_ROWID, DELETE_TXID);
    //     m_table->ModifyTuple(DELETE_TXID, 0, DELETE_INITIALTABLE_ROWID, nullptr);
    // }

    TupleDataSPtr AriesMvccTestDataGenerator::UpdateLineItemData()
    {
        int columnId = 0;
        TupleDataSPtr newdata = make_shared<TupleData>();
        AriesColumnType intColumnType({AriesValueType::INT32}, false);
        int intValue = 2147483646;
        AriesDataBufferSPtr dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &intValue, sizeof(int));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType decimalColumnType({AriesValueType::COMPACT_DECIMAL, 15, 2}, false);
        aries_acc::Decimal decimalValue(15, 2, 0, "9999999999.99");
        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(decimalColumnType, 1);
        decimalValue.ToCompactDecimal((char *)(dataBuf->GetData()), decimalColumnType.GetDataTypeSize());
        newdata->data[++columnId] = dataBuf;

        AriesColumnType charColumnType({AriesValueType::CHAR, 1}, false);
        char charValue = 'Y';
        dataBuf = make_shared<AriesDataBuffer>(charColumnType, 1);
        memcpy(dataBuf->GetData(), &charValue, 1);
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(charColumnType, 1);
        memcpy(dataBuf->GetData(), &charValue, 1);
        newdata->data[++columnId] = dataBuf;

        AriesColumnType dateColumnType({AriesValueType::DATE}, false);
        aries_acc::AriesDate dateValue(2020, 3, 20);
        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        dataBuf = make_shared<AriesDataBuffer>(dateColumnType, 1);
        memcpy(dataBuf->GetData(), &dateValue, sizeof(aries_acc::AriesDate));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string25ColumnType({AriesValueType::CHAR, 25}, false);
        char char25[25];
        memcpy(char25, "This is a Rateup testcase", 25);
        dataBuf = make_shared<AriesDataBuffer>(string25ColumnType, 1);
        memcpy(dataBuf->GetData(), char25, sizeof(char25));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string10ColumnType({AriesValueType::CHAR, 10}, false);
        char char10[10];
        memcpy(char10, "A testcase", 10);
        dataBuf = make_shared<AriesDataBuffer>(string10ColumnType, 1);
        memcpy(dataBuf->GetData(), char10, sizeof(char10));
        newdata->data[++columnId] = dataBuf;

        AriesColumnType string44ColumnType({AriesValueType::CHAR, 44}, false);
        char char44[44];
        memcpy(char44, "This are more rateup testcases! yeah !!!!!!!!", sizeof(char44));
        dataBuf = make_shared<AriesDataBuffer>(string44ColumnType, 1);
        memcpy(dataBuf->GetData(), char44, sizeof(char44));
        newdata->data[++columnId] = dataBuf;

        // m_table->ModifyTuple(UPDATE_TXID, 0, UPDATE_INITIALTABLE_ROWID, newdata);
        return newdata;
    }

    void AriesMvccTestDataGenerator::GenerateTuples()
    {
        auto insert1 = AriesTransManager::GetInstance().NewTransaction();
        auto delete1 = AriesTransManager::GetInstance().NewTransaction();
        auto update1 = AriesTransManager::GetInstance().NewTransaction();
        auto insert2 = AriesTransManager::GetInstance().NewTransaction();
        auto delete2 = AriesTransManager::GetInstance().NewTransaction();
        auto update2 = AriesTransManager::GetInstance().NewTransaction();
        auto insert3 = AriesTransManager::GetInstance().NewTransaction();
        auto delete3 = AriesTransManager::GetInstance().NewTransaction();
        auto update3 = AriesTransManager::GetInstance().NewTransaction();
        auto insert4 = AriesTransManager::GetInstance().NewTransaction();


        // COMMITTED
        m_table->AddTuple( insert1, InsertLineItemData() );
        AriesTransManager::GetInstance().EndTransaction( insert1, TransactionStatus::COMMITTED );

        m_table->AddTuple( insert4, InsertLineItemData() );
        AriesTransManager::GetInstance().EndTransaction( insert4, TransactionStatus::COMMITTED );

        // ABORTED
        m_table->AddTuple( insert2, InsertLineItemData() );
        AriesTransManager::GetInstance().EndTransaction( insert2, TransactionStatus::ABORTED );

        // IN PROGRESS
        m_table->AddTuple( insert3, InsertLineItemData() );


        m_table->SetTxMax( DELETE_INITIALTABLE_ROWID1, delete1->GetTxId() );
        m_table->ModifyTuple( delete1, DELETE_INITIALTABLE_ROWID1, nullptr, 0 );
        AriesTransManager::GetInstance().EndTransaction( delete1, TransactionStatus::ABORTED );

        m_table->SetTxMax( DELETE_INITIALTABLE_ROWID2, delete2->GetTxId() );
        m_table->ModifyTuple( delete2, DELETE_INITIALTABLE_ROWID2, nullptr, 0 );
        AriesTransManager::GetInstance().EndTransaction( delete2, TransactionStatus::COMMITTED );

        m_table->SetTxMax( DELETE_INITIALTABLE_ROWID3, delete3->GetTxId() );
        m_table->ModifyTuple( delete3, DELETE_INITIALTABLE_ROWID3, nullptr, 0 );

        m_table->SetTxMax( UPDATE_INITIALTABLE_ROWID1, update1->GetTxId() );
        m_table->ModifyTuple( update1, UPDATE_INITIALTABLE_ROWID1, UpdateLineItemData(), 0 );
        AriesTransManager::GetInstance().EndTransaction( update1, TransactionStatus::ABORTED );

        m_table->SetTxMax( UPDATE_INITIALTABLE_ROWID1, update2->GetTxId() );
        m_table->ModifyTuple( update2, UPDATE_INITIALTABLE_ROWID1, UpdateLineItemData(), 0 );
        AriesTransManager::GetInstance().EndTransaction( update2, TransactionStatus::COMMITTED );

        m_table->SetTxMax( UPDATE_INITIALTABLE_ROWID3, update3->GetTxId() );
        m_table->ModifyTuple( update3, UPDATE_INITIALTABLE_ROWID3, UpdateLineItemData(), 0 );

    }

    /* region table:
    CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));
    */
    TupleDataSPtr AriesMvccTestDataGenerator::GenerateRegionTupleData(int regionkey, string &name, string &comment)
    {
        int columnId = 0;
        TupleDataSPtr newdata = make_shared<TupleData>();
        TupleDataSPtr data = make_shared<TupleData>();
        AriesColumnType intColumnType({AriesValueType::INT32}, false);
        AriesDataBufferSPtr dataBuf = make_shared<AriesDataBuffer>(intColumnType, 1);
        memcpy(dataBuf->GetData(), &regionkey, sizeof(int));
        newdata->data[++columnId] = dataBuf;
        AriesColumnType string25ColumnType({AriesValueType::CHAR, 25}, false);
        char char25[25] = {0};
        size_t size = name.size();
        if (size > 25) {
            size = 25;
        }
        memcpy(char25, name.data(), size);
        dataBuf = make_shared<AriesDataBuffer>(string25ColumnType, 1);
        memcpy(dataBuf->GetData(), char25, sizeof(char25));
        newdata->data[++columnId] = dataBuf;
        AriesColumnType string152ColumnType({AriesValueType::CHAR, 152}, true);
        char char152[152] = {0};
        char152[0] = 1;
        size = comment.size();
        if (size > 151) {
            size = 151;
        }
        memcpy(char152 + 1, comment.data(), size);
        dataBuf = make_shared<AriesDataBuffer>(string152ColumnType, 1);
        memcpy(dataBuf->GetData(), char152, sizeof(char152));
        newdata->data[++columnId] = dataBuf;
        return newdata;
    }

    void AriesMvccTestDataGenerator::InsertTuple(TupleDataSPtr newDataBuffer, int dataIndex, AriesTransactionPtr transaction, int cid, TransactionStatus status)
    {
        m_table->AddTuple(transaction, newDataBuffer );
        AriesTransManager::GetInstance().EndTransaction( transaction, status );
    }

    void AriesMvccTestDataGenerator::ModifyTuple(RowPos rowPos, TupleDataSPtr newDataBuffer, int dataIndex, AriesTransactionPtr transaction, int cid, TransactionStatus status)
    {
        m_table->SetTxMax(rowPos, transaction->GetTxId());
        m_table->ModifyTuple(transaction, rowPos, newDataBuffer, dataIndex);
        AriesTransManager::GetInstance().EndTransaction( transaction, status );
    }

    void AriesMvccTestDataGenerator::DeleteTuple(RowPos rowPos, AriesTransactionPtr transaction, int cid, TransactionStatus status)
    {
        m_table->SetTxMax(rowPos, transaction->GetTxId());
        m_table->ModifyTuple(transaction, rowPos, nullptr, 0);
        AriesTransManager::GetInstance().EndTransaction( transaction, status );
    }

END_ARIES_ENGINE_NAMESPACE
