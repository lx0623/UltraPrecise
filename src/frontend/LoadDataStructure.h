//
// Created by tengjp on 19-12-9.
//

#ifndef AIRES_LOADDATASTRUCTURE_H
#define AIRES_LOADDATASTRUCTURE_H

#include <string>
#include <memory>
#include <AriesDefinition.h>
#include "frontend/BasicRel.h"

NAMESPACE_ARIES_START

enum enum_filetype
{
    FILETYPE_CSV,
    FILETYPE_XML
};
enum thr_lock_type
{
    TL_WRITE_DEFAULT,
    TL_WRITE_CONCURRENT_INSERT,
    TL_WRITE_LOW_PRIORITY
};
enum class On_duplicate { ERROR, IGNORE_DUP, REPLACE_DUP };

struct Line_separators {
    std::shared_ptr<std::string> line_term;
    std::shared_ptr<std::string> line_start;

    void cleanup() { line_term = line_start = nullptr; }
    void merge_line_separators(const Line_separators &s) {
        if (s.line_term != nullptr) line_term = s.line_term;
        if (s.line_start != nullptr) line_start = s.line_start;
    }
};

/**
  Helper for the sql_exchange class
*/

struct Field_separators {
    std::shared_ptr<std::string> field_term;
    std::shared_ptr<std::string> escaped;
    std::shared_ptr<std::string> enclosed;
    bool opt_enclosed;

    void cleanup() {
        field_term = escaped = enclosed = nullptr;
        opt_enclosed = false;
    }
    void merge_field_separators(const Field_separators &s) {
        if (s.field_term != nullptr) field_term = s.field_term;
        if (s.escaped != nullptr) escaped = s.escaped;
        if (s.enclosed != nullptr) enclosed = s.enclosed;
        // TODO: a bug?
        // OPTIONALLY ENCLOSED BY x ENCLOSED BY y == OPTIONALLY ENCLOSED BY y
        if (s.opt_enclosed) opt_enclosed = s.opt_enclosed;
    }
};

/**
  Used to hold information about file and file structure in exchange
  via non-DB file (...INTO OUTFILE..., ...LOAD DATA...)
  XXX: We never call destructor for objects of this class.
*/

class sql_exchange final {
public:
    const string file_name;
    const string cs_name;
    bool dumpfile;
    unsigned long skip_lines;
    enum enum_filetype filetype; /* load XML, Added by Arnold & Erik */
    Field_separators field;
    Line_separators line;
    // const CHARSET_INFO *cs;
    sql_exchange(const string& name, const string& csName, bool dumpfile_flag,
                 enum_filetype filetype_arg = FILETYPE_CSV);
    bool escaped_given(void);
};

class LoadDataStructure {
public:
    LoadDataStructure(enum_filetype fileType,
                      thr_lock_type lockType,
                      bool optLocal,
                      const string& file,
                      On_duplicate optOnDuplicate,
                      const std::shared_ptr<BasicRel>& argTableIdent,
                      const string& charset,
                      const Field_separators& fieldSeparators,
                      const Line_separators& lineSeparators,
                      ulong skipLines)
                      : tableIdent(argTableIdent),
                        isLocalFile(optLocal),
                        thrLockType(lockType),
                        onDuplicate(optOnDuplicate),
                        sqlExchange( file, charset, false, fileType)
    {
        sqlExchange.field.merge_field_separators( fieldSeparators );
        sqlExchange.line.merge_line_separators( lineSeparators );
        sqlExchange.skip_lines = skipLines;

    }
    std::shared_ptr<BasicRel> tableIdent;
    bool isLocalFile;
    thr_lock_type thrLockType;
    On_duplicate onDuplicate;

    sql_exchange sqlExchange;
};
using LoadDataStructurePtr = std::shared_ptr<LoadDataStructure>;

NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_LOADDATASTRUCTURE_H
