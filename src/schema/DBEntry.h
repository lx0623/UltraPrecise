#pragma once
#include <string>

using namespace std;

namespace aries {
namespace schema
{
class DBEntry {

private:
    string name;
public:
    const static string ROWID_COLUMN_NAME;

public:
    DBEntry() = default;
    DBEntry(string name);
    string GetName();
};
} // namespace schema
} // namespace areis
