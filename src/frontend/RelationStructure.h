#ifndef ARIES_RELATION_STRUCTURE
#define ARIES_RELATION_STRUCTURE


#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "ColumnStructure.h"


namespace aries {
class RelationStructure {

private:

    std::string name;

    ColumnStructurePointer primary_column = nullptr;

    int column_count = 0;


    std::vector<ColumnStructurePointer> columns;
    std::map<std::string, ColumnStructurePointer> name_column_map;
    std::map<std::string, int> name_location_map;
    std::map< std::string, schema::TableConstraintSPtr > constraints;


    RelationStructure(const RelationStructure &arg);

    RelationStructure &operator=(const RelationStructure &arg);

public:

    RelationStructure();

    /*simple clone just re-use existing columns!*/
    std::shared_ptr<RelationStructure> SimpleClone();

    void SetName(std::string arg_name);

    std::string GetName();

    size_t GetColumnCount();

    bool AddColumn(ColumnStructurePointer arg_column_p);

    ColumnStructurePointer GetColumn(size_t arg_index);

    /*we assume this function is called after FindColumn return true!!!*/
    int LocateColumn(std::string arg_column_name);

    ColumnStructurePointer FindColumn(std::string arg_column_name);

    std::string ToString();

    void ResetNameForTheOnlyColumn(std::string agg_column_name);

    void SetConstraints( const std::map< std::string, schema::TableConstraintSPtr >& constraints );

    const std::map< std::string, schema::TableConstraintSPtr >& GetConstraints() const;
};

typedef std::shared_ptr<RelationStructure> RelationStructurePointer;

}

#endif
