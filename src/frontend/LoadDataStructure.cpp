//
// Created by tengjp on 19-12-9.
//

#include "LoadDataStructure.h"

NAMESPACE_ARIES_START

static const auto default_line_term = std::make_shared<string>("\n");
static const auto default_escaped = std::make_shared<string>("\\");
static const auto default_field_term = std::make_shared<string>("\t");
static const auto default_xml_row_term = std::make_shared<string>("<row>");
static const auto my_empty_string = std::make_shared<string>("");

sql_exchange::sql_exchange(const string& name, const string& csName, bool flag,
                           enum enum_filetype filetype_arg)
        : file_name(name), cs_name(csName), dumpfile(flag), skip_lines(0) {
    filetype = filetype_arg;
    field.field_term = default_field_term;
    field.escaped = default_escaped;
    field.opt_enclosed = 0;
    field.enclosed = line.line_start = my_empty_string;

    line.line_term =
            filetype == FILETYPE_CSV ? default_line_term : default_xml_row_term;
    // cs = NULL;
}

bool sql_exchange::escaped_given(void) {
    return field.escaped != default_escaped;
}
NAMESPACE_ARIES_END // namespace aries

