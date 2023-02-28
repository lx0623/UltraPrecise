#pragma once

#include <cstdint>
#include <atomic>
#include <memory>

#include "AriesTuple.h"

BEGIN_ARIES_ENGINE_NAMESPACE

enum class OperationType : int16_t
{
    Insert,
    InsertDict, // 字典只增不删
    InsertBatch, // 批量插入数据
    Update,
    Delete,
    Commit,
    Abort,
    CheckPoint,
    Truncate
};

#define ARIES_XLOG_HEADER_MAGIC 0xFEEF

struct AriesXLogHeader
{
    uint16_t magic;
    OperationType operation;
    TableId tableId;
    TxId txid;
    RowPos sourcePos;
    RowPos targetPos;
    RowPos initialPos;
    int32_t dataLength;
} ARIES_PACKED;

struct AriesBatchInsertInfo
{
    size_t rowCount;
    size_t columnCount;
} ARIES_PACKED;

using AriesXLogHeaderSPtr = std::shared_ptr< AriesXLogHeader >;

END_ARIES_ENGINE_NAMESPACE
