#ifndef ARIES_NODE_RELATION_STRUCTURE
#define ARIES_NODE_RELATION_STRUCTURE

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "NodeColumnStructure.h"


namespace aries {

class NodeRelationStructure {

private:
    std::string name;
    size_t column_count = 0;

    std::vector<NodeColumnStructurePointer> columns;
    std::vector<std::string> tables_where_column_belong;

    NodeRelationStructure(const NodeRelationStructure &arg);

    NodeRelationStructure &operator=(const NodeRelationStructure &arg);


public:

    NodeRelationStructure();

    void AddColumn(NodeColumnStructurePointer arg_column, std::string arg_table_name);

    size_t GetColumnCount();

    NodeColumnStructurePointer GetColumnbyIndex(size_t arg_index);

    std::string GetColumnSourceTableNamebyIndex(size_t arg_index);

    std::string ToString();

};

typedef std::shared_ptr<NodeRelationStructure> NodeRelationStructurePointer;
}//namespace

#endif
