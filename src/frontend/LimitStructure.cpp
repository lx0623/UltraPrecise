//
// Created by 胡胜刚 on 2019-08-08.
//

#include <string>
#include "LimitStructure.h"

namespace aries {
std::string LimitStructure::ToString() {
    return "LIMIT " + std::to_string(Offset) + ", " + std::to_string(Limit);
}
} // namespace aries
