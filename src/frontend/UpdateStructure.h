#pragma once
#include <vector>
#include "SQLIdent.h"
#include "AbstractQuery.h"
#include "AbstractBiaodashi.h"
using namespace std;
namespace aries {
class UpdateStructure
{
public:
    UpdateStructure();
    ~UpdateStructure();

    void SetSelectStructure( const AbstractQueryPointer& selectStructure )
    {
        m_selectStructure = selectStructure;
    }
    AbstractQueryPointer GetSelectStructure() const
    {
        return m_selectStructure;
    }
    void SetTarget( const string& dbName, const string& tableName )
    {
        m_targetDbName = dbName;
        m_targetTableName = tableName;
    }
    string GetTargetDbName() const
    {
        return m_targetDbName;
    }
    string GetTargetTableName() const
    {
        return m_targetTableName;
    }
private:
    string m_targetDbName;
    string m_targetTableName;
    AbstractQueryPointer m_selectStructure;
};
using UpdateStructurePtr = shared_ptr< UpdateStructure >;
}