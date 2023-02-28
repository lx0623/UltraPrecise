#ifndef ARIES_NODE_COLUMN_STRUCTURE
#define ARIES_NODE_COLUMN_STRUCTURE

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"

namespace aries {
//    typedef BiaodashiValueType ColumnValueType;

class NodeColumnStructure {

private:

    NodeColumnStructure(const NodeColumnStructure &arg);

    NodeColumnStructure &operator=(const NodeColumnStructure &arg);


    std::string name;
    ColumnValueType type;
    int length;
    bool nullable = false;


    ColumnStructurePointer possible_root;


    // /*where am I now*/
    // int relation_index = -1; /*0 means the first child of the node, ...*/
    // int column_index = -1; /*0 means the first column, ...*/

public:


    NodeColumnStructure( std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_nullable );

    std::string GetName();

    ColumnValueType GetValueType();

    bool IsNullable();

    std::string ToString();

    ColumnStructurePointer GetPossibleRoot();

    void SetPossibleRoot(ColumnStructurePointer arg_root);


    static std::shared_ptr<NodeColumnStructure>
    __createInstanceFromColumnStructure(ColumnStructurePointer arg_column_structure);

    /*we also need to clone myself*/
    std::shared_ptr<NodeColumnStructure> CloneAsNullable();

};


typedef std::shared_ptr<NodeColumnStructure> NodeColumnStructurePointer;

}

#endif
