#ifndef MY_DBUG_INCLUDED
#define MY_DBUG_INCLUDED
#define DBUG_ENTER(a)
#define DBUG_RETURN(a1) do {return(a1);} while(0)
#define DBUG_VOID_RETURN do {return;} while(0)
#define DBUG_ASSERT(A) assert(A)
#endif