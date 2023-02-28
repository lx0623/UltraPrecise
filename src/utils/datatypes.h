#pragma once
#include <string>
using std::string;
struct TableMapKey {
    TableMapKey(const string &dbName, const string &tableName) {
        m_dbName = dbName;
        m_tableName = tableName;
    }
    bool operator <( const TableMapKey& src ) const
    {
        return m_dbName == src.m_dbName ? m_tableName < src.m_tableName : m_dbName < src.m_dbName;
    }
    string m_dbName;
    string m_tableName;
};