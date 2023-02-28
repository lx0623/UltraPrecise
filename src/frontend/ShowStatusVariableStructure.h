//
// Created by tengjp on 19-8-19.
//

#ifndef AIRES_SHOWSTATUSVARIABLESTRUCTURE_H
#define AIRES_SHOWSTATUSVARIABLESTRUCTURE_H
#include "ShowStructure.h"

NAMESPACE_ARIES_START
class ShowStatusVariableStructure : public ShowStructure {
public:
    bool global;
};

using ShowStatusVariableStructurePtr = std::shared_ptr<ShowStatusVariableStructure>;
NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_SHOWSTATUSVARIABLESTRUCTURE_H
