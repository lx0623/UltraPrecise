//
// Created by 胡胜刚 on 2019/11/8.
//
#include <gtest/gtest.h>

#include <AriesEngine/AriesScanNode.h>
#include <AriesEngineWrapper/AriesEngineShell.h>
#include <schema/SchemaManager.h>
#include <frontend/SchemaBuilder.h>

using namespace aries;

TEST(scannode, t1)
{
    /*
    aries_engine::AriesEngineShell shell;
    std::string table_name = "lineitem";
    ASSERT_TRUE(schema::SchemaManager::Load("/var/rateup/data"));

    auto database = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName("scale_1");
    auto db = SchemaBuilder::BuildFromDatabase(database.get());
    auto table =  db->FindPhysicalTable("lineitem");

    vector< int > columns_id;

    for (int i = 1; i <= table->GetRelationStructure()->GetColumnCount(); i++)
    {
        columns_id.emplace_back(i);
    }

    auto node = aries_engine::AriesEngineShell::MakeScanNode("scale_1", table, columns_id);

    auto column_file_path = database->GetColumnLocationString_ByIndex(table_name, 0);

    uint64_t rows;
    int8_t has_null;
    int16_t item_len;
    ASSERT_EQ(aries_engine::AriesScanNode::getColumnFileHeaderInfo(column_file_path + "_0", rows, has_null, item_len), 0);

    ASSERT_TRUE( node->Open() );

    uint64_t read_count = 0;
    AriesOpResult result;
    do {
        result = node->GetNext();
        if (result.Status == AriesOpNodeStatus::ERROR)
        {
            std::cout << "read error" << std::endl;
        }
        else if (result.Status == AriesOpNodeStatus::CONTINUE)
        {
            std::cout << "have more data to read" << std::endl;
            read_count += result.TableBlock->GetRowCount();
        }
        else
        {
            std::cout << "read all data!" << std::endl;
            read_count += result.TableBlock->GetRowCount();
        }

        ASSERT_NE(result.Status, AriesOpNodeStatus::ERROR);
        ASSERT_EQ( result.TableBlock->GetColumnCount(), columns_id.size());
    } while (result.Status == AriesOpNodeStatus::CONTINUE);

    ASSERT_EQ(read_count, rows);
    */
}