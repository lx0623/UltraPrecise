#include "DBEntry.h"

#include <iostream>

namespace aries
{
namespace schema
{

const std::string DBEntry::ROWID_COLUMN_NAME = "__rateup_rowid__";
DBEntry::DBEntry(std::string name_): name(name_) {
}

string DBEntry::GetName() {
    return name;
}

} // namespace schema
} // namespace aries
