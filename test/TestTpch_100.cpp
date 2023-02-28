#include <gtest/gtest.h>
#include <string>

#include "TestCommonBase.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace std;
using std::string;


TEST(tpch_100, q1)
{
    TestTpch(100, 1);
}

TEST(tpch_100, q2)
{
    TestTpch(100, 2);
}

TEST(tpch_100, q3)
{
    TestTpch(100, 3);
}

TEST(tpch_100, q4)
{
    TestTpch(100, 4);
}

TEST(tpch_100, q5)
{
    TestTpch(100, 5);
}

TEST(tpch_100, q6)
{
    TestTpch(100, 6);
}

TEST(tpch_100, q7)
{
    TestTpch(100, 7);
}

TEST(tpch_100, q8)
{
    TestTpch(100, 8);
}

TEST(tpch_100, q9)
{
    TestTpch(100, 9);
}

TEST(tpch_100, q10)
{
    TestTpch(100, 10);
}

TEST(tpch_100, q11)
{
    TestTpch(100, 11);
}

TEST(tpch_100, q12)
{
    TestTpch(100, 12);
}

TEST(tpch_100, q13)
{
    TestTpch(100, 13);
}

TEST(tpch_100, q14)
{
    TestTpch(100, 14);
}

TEST(tpch_100, q15)
{
    TestTpch(100, 15);
}

TEST(tpch_100, q16)
{
    TestTpch(100, 16);
}

TEST(tpch_100, q17)
{
   TestTpch(100, 17);
}

TEST(tpch_100, q18)
{
    TestTpch(100, 18);
}

TEST(tpch_100, q19)
{
    TestTpch(100, 19);
}

TEST(tpch_100, q20)
{
    TestTpch(100, 20);
}

TEST(tpch_100, q21)
{
    TestTpch(100, 21);
}

TEST(tpch_100, q22)
{
    TestTpch(100, 22);
}


//power test order {14, 2, 9, 20, 6, 17, 18, 8, 21, 13, 3, 22, 16, 4, 11, 15, 1, 10, 19, 5, 7, 12}
TEST(power, p1)
{
    TestTpch(100, 14);
    std::cout << "p1 means q14" << std::endl;
}

TEST(power, p2)
{
    TestTpch(100, 2);
    std::cout << "p2 means q2" << std::endl;
}

TEST(power, p3)
{
    TestTpch(100, 9);
    std::cout << "p3 means q9" << std::endl;
}

TEST(power, p4)
{
    TestTpch(100, 20);
    std::cout << "p4 means q20" << std::endl;
}

TEST(power, p5)
{
    TestTpch(100, 6);
    std::cout << "p5 means q6" << std::endl;
}

TEST(power, p6)
{
    TestTpch(100, 17);
    std::cout << "p6 means q17" << std::endl;
}


TEST(power, p7)
{
    TestTpch(100, 18);
    std::cout << "p7 means q18" << std::endl;
}

TEST(power, p8)
{
    TestTpch(100, 8);
    std::cout << "p8 means q8" << std::endl;
}

TEST(power, p9)
{
    TestTpch(100, 21);
    std::cout << "p9 means q21" << std::endl;
}

TEST(power, p10)
{
    TestTpch(100, 13);
    std::cout << "p10 means q13" << std::endl;
}

TEST(power, p11)
{
    TestTpch(100, 3);
    std::cout << "p11 means q3" << std::endl;
}

TEST(power, p12)
{
    TestTpch(100, 22);
    std::cout << "p12 means q22" << std::endl;
}

TEST(power, p13)
{
    TestTpch(100, 16);
    std::cout << "p13 means q16" << std::endl;
}

TEST(power, p14)
{
    TestTpch(100, 4);
    std::cout << "p14 means q4" << std::endl;
}

TEST(power, p15)
{
    TestTpch(100, 11);
    std::cout << "p15 means q11" << std::endl;
}

TEST(power, p16)
{
    TestTpch(100, 15);
    std::cout << "p16 means q15" << std::endl;
}

TEST(power, p17)
{
    TestTpch(100, 1);
    std::cout << "p17 means q1" << std::endl;
}

TEST(power, p18)
{
    TestTpch(100, 10);
    std::cout << "p18 means q10" << std::endl;
}

TEST(power, p19)
{
    TestTpch(100, 19);
    std::cout << "p19 means q19" << std::endl;
}

TEST(power, p20)
{
    TestTpch(100, 5);
    std::cout << "p20 means q5" << std::endl;
}

TEST(power, p21)
{
    TestTpch(100, 7);
    std::cout << "p21 means q7" << std::endl;
}

TEST(power, p22)
{
    TestTpch(100, 12);
    std::cout << "p22 means q12" << std::endl;
}


//power test order {14, 2, 9, 20, 6, 17, 18, 8, 21, 13, 3, 22, 16, 4, 11, 15, 1, 10, 19, 5, 7, 12}
TEST(power_clean, p1)
{
    TestTpch(100, 14);
    std::cout << "p1 means q14" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p2)
{
    TestTpch(100, 2);
    std::cout << "p2 means q2" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p3)
{
    TestTpch(100, 9);
    std::cout << "p3 means q9" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p4)
{
    TestTpch(100, 20);
    std::cout << "p4 means q20" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p5)
{
    TestTpch(100, 6);
    std::cout << "p5 means q6" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p6)
{
    TestTpch(100, 17);
    std::cout << "p6 means q17" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p7)
{
    TestTpch(100, 18);
    std::cout << "p7 means q18" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p8)
{
    TestTpch(100, 8);
    std::cout << "p8 means q8" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p9)
{
    TestTpch(100, 21);
    std::cout << "p9 means q21" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p10)
{
    TestTpch(100, 13);
    std::cout << "p10 means q13" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p11)
{
    TestTpch(100, 3);
    std::cout << "p11 means q3" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p12)
{
    TestTpch(100, 22);
    std::cout << "p12 means q22" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p13)
{
    TestTpch(100, 16);
    std::cout << "p13 means q16" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p14)
{
    TestTpch(100, 4);
    std::cout << "p14 means q4" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p15)
{
    TestTpch(100, 11);
    std::cout << "p15 means q11" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p16)
{
    TestTpch(100, 15);
    std::cout << "p16 means q15" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p17)
{
    TestTpch(100, 1);
    std::cout << "p17 means q1" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p18)
{
    TestTpch(100, 10);
    std::cout << "p18 means q10" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p19)
{
    TestTpch(100, 19);
    std::cout << "p19 means q19" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p20)
{
    TestTpch(100, 5);
    std::cout << "p20 means q5" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p21)
{
    TestTpch(100, 7);
    std::cout << "p21 means q7" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(power_clean, p22)
{
    TestTpch(100, 12);
    std::cout << "p22 means q12" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}