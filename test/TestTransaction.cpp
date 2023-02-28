// #include <gtest/gtest.h>
// #include <string>
// #include <thread>

// #include "AriesEngine/transaction/AriesTransManager.h"
// #include "CudaAcc/AriesSqlOperator.h"
// #include "testcases/AriesEngine/transaction/AriesMvccTestDataGenerator.h"
// #include "AriesEngine/transaction/AriesMvccTableManager.h"
// #include "frontend/SQLExecutor.h"
// #include "AriesEngineWrapper/AriesMemTable.h"

// using namespace aries_engine;
// using namespace aries_acc;
// using namespace std;
// using std::string;


// /**
//  * 第一种情况: A->B->C + C->A => A->B->C->A
//  */
// TEST( trans, deadlock1 )
// {
//     auto trans_a = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_b = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_c = AriesTransManager::GetInstance().NewTransaction();

//     thread t_a( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_a, trans_b->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::ABORTED );
//         }
        
//     });

//     thread t_b( [ = ]
//     {
//         ::sleep( 1 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_b, trans_c->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::ABORTED );
//         }
        
//     });

//     thread t_c( [ = ]
//     {
//         ::sleep( 3 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_c, trans_a->GetTxId() );
//             ASSERT_TRUE( false );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::ABORTED );
//             ASSERT_TRUE( true );
//         }

//     });

//     t_a.join();
//     t_b.join();
//     t_c.join();
// }


// /**
//  * 第二种情况： A->B->C + C->B => A->B->C->B
//  */
// TEST( trans, deadlock2 )
// {
//     auto trans_a = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_b = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_c = AriesTransManager::GetInstance().NewTransaction();

//     thread t_a( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_a, trans_b->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::ABORTED );
//         }
        
//     });

//     thread t_b( [ = ]
//     {
//         ::sleep( 1 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_b, trans_c->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::ABORTED );
//         }
        
//     });

//     thread t_c( [ = ]
//     {
//         ::sleep( 3 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_c, trans_b->GetTxId() );
//             ASSERT_TRUE( false );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::ABORTED );
//             ASSERT_TRUE( true );
//         }

//     });

//     t_a.join();
//     t_b.join();
//     t_c.join();
// }

// /**
//  * 第三种情况： F->D + D->A => F->D->A
//  */
// TEST( trans, deadlock3 )
// {
//     auto trans_f = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_d = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_a = AriesTransManager::GetInstance().NewTransaction();

//     thread t_f( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_f, trans_d->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_d( [ = ]
//     {
//         ::sleep( 1 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_d, trans_a->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_a( [ = ]
//     {
//         ::sleep( 3 );
//         AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::COMMITTED );
//     });

//     t_f.join();
//     t_d.join();
//     t_a.join();
// }

// /**
//  * 第六种情况：
//  *      链表1: A->B->C->D
//  *      链表2: C->D->F->E->B
//  */
// TEST( trans, deadlock6 )
// {
//     auto trans_a = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_b = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_c = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_d = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_e = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_f = AriesTransManager::GetInstance().NewTransaction();

//     thread t_a( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_a, trans_b->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_b( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_b, trans_c->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_c( [ = ]
//     {
//         ::sleep( 3 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_c, trans_d->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::COMMITTED );
//             ASSERT_TRUE( false );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::ABORTED );
//             ASSERT_TRUE( true );
//         }
//     });

//     thread t_d( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_d, trans_f->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
//     });

//     thread t_f( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_f, trans_e->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
//     });

//     thread t_e( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_e, trans_b->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_e, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_e, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
//     });

//     t_a.join();
//     t_b.join();
//     t_c.join();
//     t_d.join();
//     t_e.join();
//     t_f.join();
// }

// /**
//  * 第七种情况：
//  *      链表1: A->B->C->D
//  *      链表2: D->F->A
//  */
// TEST( trans, deadlock7 )
// {
//     auto trans_a = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_b = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_c = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_d = AriesTransManager::GetInstance().NewTransaction();
//     auto trans_f = AriesTransManager::GetInstance().NewTransaction();

//     thread t_a( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_a, trans_b->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_a, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_b( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_b, trans_c->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_b, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
        
//     });

//     thread t_c( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_c, trans_d->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_c, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
//     });

//     thread t_d( [ = ]
//     {
//         ::sleep( 3 );
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_d, trans_f->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::COMMITTED );
//             ASSERT_TRUE( false );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_d, TransactionStatus::ABORTED );
//         }
//     });

//     thread t_f( [ = ]
//     {
//         try
//         {
//             AriesTransManager::GetInstance().WaitForTxEnd( trans_f, trans_a->GetTxId() );
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::COMMITTED );
//         }
//         catch ( AriesTransManager::DeadLockDetected& e )
//         {
//             AriesTransManager::GetInstance().EndTransaction( trans_f, TransactionStatus::ABORTED );
//             ASSERT_TRUE( false );
//         }
//     });

//     t_a.join();
//     t_b.join();
//     t_c.join();
//     t_d.join();
//     t_f.join();
// }

// TEST( trans, create_visible_ids )
// {
//     auto invisible_ids = std::make_shared< AriesInt32Array >( 10 );
    
//     for ( int i = 0; i < 10; i++ )
//     {
//         invisible_ids->GetData()[ i ] = 0 - i * 10 - 1;
//     }

//     auto r = CreateVisibleRowIds( 100, invisible_ids );
//     ASSERT_EQ( r->GetItemCount(), 90 );

//     for ( int i = 0; i < r->GetItemCount(); i++ )
//     {
//         for ( int j = 0; j < invisible_ids->GetItemCount(); j++ )
//         {
//             ASSERT_NE( r->GetInt32( i ), invisible_ids->GetData()[ j ] );
//         }
//         ASSERT_EQ( r->GetInt32( i ), 0 - i - i / 9 - 2);
//     }

//     auto indices = ConvertRowIdToIndices( r );

//     ASSERT_EQ( indices->GetItemCount(), r->GetItemCount() );

//     for ( int i = 0; i < indices->GetItemCount(); i++ )
//     {
//         ASSERT_EQ( indices->GetData()[ i ], i + i / 9 + 1);
//     }
// }
