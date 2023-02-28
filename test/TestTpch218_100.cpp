#include <gtest/gtest.h>
#include <string>

#include "TestCommonBase.h"
#include "AriesEngine/transaction/AriesInitialTableManager.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace std;
using std::string;


TEST(tpch218_100, q1)
{
    TestTpch218(100, 1);
}

TEST(tpch218_100, q2)
{
    TestTpch218(100, 2);
}

TEST(tpch218_100, q3)
{
    TestTpch218(100, 3);
}

TEST(tpch218_100, q4)
{
    TestTpch218(100, 4);
}

TEST(tpch218_100, q5)
{
    TestTpch218(100, 5);
}

TEST(tpch218_100, q6)
{
    TestTpch218(100, 6);
}

TEST(tpch218_100, q7)
{
    TestTpch218(100, 7);
}

TEST(tpch218_100, q8)
{
    TestTpch218(100, 8);
}

TEST(tpch218_100, q9)
{
    TestTpch218(100, 9);
}

TEST(tpch218_100, q10)
{
    TestTpch218(100, 10);
}

TEST(tpch218_100, q11)
{
    TestTpch218(100, 11);
}

TEST(tpch218_100, q12)
{
    TestTpch218(100, 12);
}

TEST(tpch218_100, q13)
{
    TestTpch218(100, 13);
}

TEST(tpch218_100, q14)
{
    TestTpch218(100, 14);
}

TEST(tpch218_100, q15)
{
    TestTpch218(100, 150, false);
    TestTpch218(100, 15);
    TestTpch218(100, 151, false);
}

TEST(tpch218_100, q16)
{
    TestTpch218(100, 16);
}

TEST(tpch218_100, q17)
{
   TestTpch218(100, 17);
}

TEST(tpch218_100, q18)
{
    TestTpch218(100, 18);
}

TEST(tpch218_100, q19)
{
    TestTpch218(100, 19);
}

TEST(tpch218_100, q20)
{
    TestTpch218(100, 20);
}

TEST(tpch218_100, q21)
{
    TestTpch218(100, 21);
}

TEST(tpch218_100, q22)
{
    TestTpch218(100, 22);
}


//power test order {14, 2, 9, 20, 6, 17, 18, 8, 21, 13, 3, 22, 16, 4, 11, 15, 1, 10, 19, 5, 7, 12}
TEST(power218_100, p1)
{
    TestTpch218(100, 14);
    std::cout << "p1 means q14" << std::endl;
}

TEST(power218_100, p2)
{
    TestTpch218(100, 2);
    std::cout << "p2 means q2" << std::endl;
}

TEST(power218_100, p3)
{
    TestTpch218(100, 9);
    std::cout << "p3 means q9" << std::endl;
}

TEST(power218_100, p4)
{
    TestTpch218(100, 20);
    std::cout << "p4 means q20" << std::endl;
}

TEST(power218_100, p5)
{
    TestTpch218(100, 6);
    std::cout << "p5 means q6" << std::endl;
}

TEST(power218_100, p6)
{
    TestTpch218(100, 17);
    std::cout << "p6 means q17" << std::endl;
}


TEST(power218_100, p7)
{
    TestTpch218(100, 18);
    std::cout << "p7 means q18" << std::endl;
}

TEST(power218_100, p8)
{
    TestTpch218(100, 8);
    std::cout << "p8 means q8" << std::endl;
}

TEST(power218_100, p9)
{
    TestTpch218(100, 21);
    std::cout << "p9 means q21" << std::endl;
}

TEST(power218_100, p10)
{
    TestTpch218(100, 13);
    std::cout << "p10 means q13" << std::endl;
}

TEST(power218_100, p11)
{
    TestTpch218(100, 3);
    std::cout << "p11 means q3" << std::endl;
}

TEST(power218_100, p12)
{
    TestTpch218(100, 22);
    std::cout << "p12 means q22" << std::endl;
}

TEST(power218_100, p13)
{
    TestTpch218(100, 16);
    std::cout << "p13 means q16" << std::endl;
}

TEST(power218_100, p14)
{
    TestTpch218(100, 4);
    std::cout << "p14 means q4" << std::endl;
}

TEST(power218_100, p15)
{
    TestTpch218(100, 11);
    std::cout << "p15 means q11" << std::endl;
}

TEST(power218_100, p16)
{
    TestTpch218(100, 150, false);
    TestTpch218(100, 15);
    TestTpch218(100, 151, false);
    std::cout << "p16 means q15" << std::endl;
}

TEST(power218_100, p17)
{
    TestTpch218(100, 1);
    std::cout << "p17 means q1" << std::endl;
}

TEST(power218_100, p18)
{
    TestTpch218(100, 10);
    std::cout << "p18 means q10" << std::endl;
}

TEST(power218_100, p19)
{
    TestTpch218(100, 19);
    std::cout << "p19 means q19" << std::endl;
}

TEST(power218_100, p20)
{
    TestTpch218(100, 5);
    std::cout << "p20 means q5" << std::endl;
}

TEST(power218_100, p21)
{
    TestTpch218(100, 7);
    std::cout << "p21 means q7" << std::endl;
}

TEST(power218_100, p22)
{
    TestTpch218(100, 12);
    std::cout << "p22 means q12" << std::endl;
}


//power test order {14, 2, 9, 20, 6, 17, 18, 8, 21, 13, 3, 22, 16, 4, 11, 15, 1, 10, 19, 5, 7, 12}
TEST(powerClean218_100, p1)
{
    TestTpch218(100, 14);
    std::cout << "p1 means q14" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p2)
{
    TestTpch218(100, 2);
    std::cout << "p2 means q2" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p3)
{
    TestTpch218(100, 9);
    std::cout << "p3 means q9" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p4)
{
    TestTpch218(100, 20);
    std::cout << "p4 means q20" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p5)
{
    TestTpch218(100, 6);
    std::cout << "p5 means q6" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p6)
{
    TestTpch218(100, 17);
    std::cout << "p6 means q17" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p7)
{
    TestTpch218(100, 18);
    std::cout << "p7 means q18" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p8)
{
    TestTpch218(100, 8);
    std::cout << "p8 means q8" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p9)
{
    TestTpch218(100, 21);
    std::cout << "p9 means q21" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p10)
{
    TestTpch218(100, 13);
    std::cout << "p10 means q13" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p11)
{
    TestTpch218(100, 3);
    std::cout << "p11 means q3" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p12)
{
    TestTpch218(100, 22);
    std::cout << "p12 means q22" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p13)
{
    TestTpch218(100, 16);
    std::cout << "p13 means q16" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p14)
{
    TestTpch218(100, 4);
    std::cout << "p14 means q4" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p15)
{
    TestTpch218(100, 11);
    std::cout << "p15 means q11" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p16)
{
    TestTpch218(100, 150, false);
    TestTpch218(100, 15);
    TestTpch218(100, 151, false);
    std::cout << "p16 means q15" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p17)
{
    TestTpch218(100, 1);
    std::cout << "p17 means q1" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p18)
{
    TestTpch218(100, 10);
    std::cout << "p18 means q10" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p19)
{
    TestTpch218(100, 19);
    std::cout << "p19 means q19" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p20)
{
    TestTpch218(100, 5);
    std::cout << "p20 means q5" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p21)
{
    TestTpch218(100, 7);
    std::cout << "p21 means q7" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}

TEST(powerClean218_100, p22)
{
    TestTpch218(100, 12);
    std::cout << "p22 means q12" << std::endl;
    AriesInitialTableManager::GetInstance().allPrefetchToCpu();
}