//
// Created by tengjp on 19-8-16.
//

#ifndef AIRES_SHOWCOLUMNSSTRUCTURE_H
#define AIRES_SHOWCOLUMNSSTRUCTURE_H

#include "TableNameStructure.h"
#include "ShowStructure.h"
NAMESPACE_ARIES_START
class ShowColumnsStructure : public ShowStructure {
public:
    ShowColumnsStructure() {
        showCmd = SHOW_CMD::SHOW_COLUMNS;
        tableNameStructurePtr = std::make_shared<TableNameStructure>();
    }
    TableNameStructureSPtr tableNameStructurePtr;
};
using ShowColumnsStructurePtr = std::shared_ptr<ShowColumnsStructure>;
NAMESPACE_ARIES_END // namespace aries


#endif //AIRES_SHOWCOLUMNSSTRUCTURE_H
