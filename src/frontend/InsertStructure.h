#pragma once
#include <memory>
#include <vector>
#include "AbstractQuery.h"
#include "BasicRel.h"
#include "AbstractBiaodashi.h"
using namespace std;
namespace aries {
using VALUES = shared_ptr< vector< BiaodashiPointer > >;
using VALUES_LIST = vector< VALUES >;
class InsertStructure
{
public:
    InsertStructure( const BasicRelPointer& tableIdent );
    ~InsertStructure();

    string GetDbName() const { return m_dbName; }
    string GetTableName() const { return m_tableName; }

    void SetInsertColumns( const vector< BiaodashiPointer >& insertColumns );
    vector< BiaodashiPointer > GetInsertColumns() const
    {
        return m_insertColumns;
    }

    void SetInsertColumnValues( const VALUES_LIST& insertValuesList );
    void SetInsertColumnValues( VALUES_LIST&& insertValuesList );
    VALUES_LIST GetInsertColumnValues() const
    {
        return m_insertValuesList;
    }

    void SetOptUpdateColumnValues( const vector< BiaodashiPointer >& updateColumns,
                                   const vector< BiaodashiPointer >& updateValues );
    vector< BiaodashiPointer > GetOptUpdateColumns() const
    {
        return m_optUpdateColumns;
    }
    vector<BiaodashiPointer> GetOptUpdateValues() const
    {
        return m_optUpdateValues;
    }

    void SetSelectStructure( const AbstractQueryPointer& selectStructure )
    {
        m_selectStructure = selectStructure;
    }
    AbstractQueryPointer GetSelectStructure() const
    {
        return m_selectStructure;
    }
    void SetInsertRowsSelectStructure( const AbstractQueryPointer& selectStructure )
    {
        m_insertRowsSelectStructure = selectStructure;
    }
    AbstractQueryPointer GetInsertRowsSelectStructure() const
    {
        return m_insertRowsSelectStructure;
    }

private:
    string m_dbName;
    string m_tableName;
    vector< BiaodashiPointer > m_insertColumns;
    VALUES_LIST m_insertValuesList;

    vector< BiaodashiPointer > m_optUpdateColumns;
    vector< BiaodashiPointer > m_optUpdateValues;

    // insert ... select
    AbstractQueryPointer m_selectStructure;

    // used to check insert rows expressions
    AbstractQueryPointer m_insertRowsSelectStructure;
};
using InsertStructurePtr = shared_ptr< InsertStructure>;
}
