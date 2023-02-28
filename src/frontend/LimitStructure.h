//
// Created by 胡胜刚 on 2019-08-08.
//

#pragma once

#include <cstdint>
#include <memory>

namespace aries {
struct LimitStructure {
    std::int64_t Offset;
    std::int64_t Limit;

    LimitStructure(std::int64_t offset, std::int64_t limit) {
        Offset = offset;
        Limit = limit;
    }

    std::string ToString();
};

using LimitStructurePointer = std::shared_ptr<LimitStructure>;

} // namespace aries