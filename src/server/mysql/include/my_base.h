//
// Created by tengjp on 19-7-25.
//

#ifndef AIRES_MY_BASE_H
#define AIRES_MY_BASE_H
typedef ulong key_part_map;
#define HA_WHOLE_KEY  (~(key_part_map)0)
/*
  Errorcodes given by handler functions

  opt_sum_query() assumes these codes are > 1
  Do not add error numbers before HA_ERR_FIRST.
  If necessary to add lower numbers, change HA_ERR_FIRST accordingly.
*/
#define HA_ERR_FIRST            120	/* Copy of first error nr.*/

#define HA_ERR_KEY_NOT_FOUND	120	/* Didn't find key on read or update */
#define HA_ERR_FOUND_DUPP_KEY	121	/* Dupplicate key on write */
#define HA_ERR_INTERNAL_ERROR   122	/* Internal error */
#define HA_ERR_RECORD_CHANGED	123	/* Uppdate with is recoverable */
#define HA_ERR_WRONG_INDEX	124	/* Wrong index given to function */
#define HA_ERR_CRASHED		126	/* Indexfile is crashed */
#define HA_ERR_WRONG_IN_RECORD	127	/* Record-file is crashed */
#define HA_ERR_OUT_OF_MEM	128	/* Record-file is crashed */
#define HA_ERR_NOT_A_TABLE      130     /* not a MYI file - no signature */
#define HA_ERR_WRONG_COMMAND	131	/* Command not supported */
#define HA_ERR_OLD_FILE		132	/* old databasfile */
#define HA_ERR_NO_ACTIVE_RECORD 133	/* No record read in update() */
#define HA_ERR_RECORD_DELETED	134	/* A record is not there */
#define HA_ERR_RECORD_FILE_FULL 135	/* No more room in file */
#define HA_ERR_INDEX_FILE_FULL	136	/* No more room in file */
#define HA_ERR_END_OF_FILE	137	/* end in next/prev/first/last */
#define HA_ERR_UNSUPPORTED	138	/* unsupported extension used */
#define HA_ERR_TO_BIG_ROW	139	/* Too big row */
#define HA_WRONG_CREATE_OPTION	140	/* Wrong create option */
#define HA_ERR_FOUND_DUPP_UNIQUE 141	/* Dupplicate unique on write */
#define HA_ERR_UNKNOWN_CHARSET	 142	/* Can't open charset */
#define HA_ERR_WRONG_MRG_TABLE_DEF 143	/* conflicting tables in MERGE */
#define HA_ERR_CRASHED_ON_REPAIR 144	/* Last (automatic?) repair failed */
#define HA_ERR_CRASHED_ON_USAGE  145	/* Table must be repaired */
#define HA_ERR_LOCK_WAIT_TIMEOUT 146
#define HA_ERR_LOCK_TABLE_FULL   147
#define HA_ERR_READ_ONLY_TRANSACTION 148 /* Updates not allowed */
#define HA_ERR_LOCK_DEADLOCK	 149
#define HA_ERR_CANNOT_ADD_FOREIGN 150    /* Cannot add a foreign key constr. */
#define HA_ERR_NO_REFERENCED_ROW 151     /* Cannot add a child row */
#define HA_ERR_ROW_IS_REFERENCED 152     /* Cannot delete a parent row */
#define HA_ERR_NO_SAVEPOINT	 153     /* No savepoint with that name */
#define HA_ERR_NON_UNIQUE_BLOCK_SIZE 154 /* Non unique key block size */
#define HA_ERR_NO_SUCH_TABLE     155     /* The table does not exist in engine */
#define HA_ERR_TABLE_EXIST       156     /* The table existed in storage engine */
#define HA_ERR_NO_CONNECTION     157     /* Could not connect to storage engine */
/* NULLs are not supported in spatial index */
#define HA_ERR_NULL_IN_SPATIAL   158
#define HA_ERR_TABLE_DEF_CHANGED 159     /* The table changed in storage engine */
/* There's no partition in table for given value */
#define HA_ERR_NO_PARTITION_FOUND 160
#define HA_ERR_RBR_LOGGING_FAILED 161    /* Row-based binlogging of row failed */
#define HA_ERR_DROP_INDEX_FK      162    /* Index needed in foreign key constr */
/*
  Upholding foreign key constraints would lead to a duplicate key error
  in some other table.
*/
#define HA_ERR_FOREIGN_DUPLICATE_KEY 163
/* The table changed in storage engine */
#define HA_ERR_TABLE_NEEDS_UPGRADE 164
#define HA_ERR_TABLE_READONLY      165   /* The table is not writable */

#define HA_ERR_AUTOINC_READ_FAILED 166   /* Failed to get next autoinc value */
#define HA_ERR_AUTOINC_ERANGE    167     /* Failed to set row autoinc value */
#define HA_ERR_GENERIC           168     /* Generic error */
/* row not actually updated: new values same as the old values */
#define HA_ERR_RECORD_IS_THE_SAME 169
/* It is not possible to log this statement */
#define HA_ERR_LOGGING_IMPOSSIBLE 170    /* It is not possible to log this
                                            statement */
#define HA_ERR_CORRUPT_EVENT      171    /* The event was corrupt, leading to
                                            illegal data being read */
#define HA_ERR_NEW_FILE	          172    /* New file format */
#define HA_ERR_ROWS_EVENT_APPLY   173    /* The event could not be processed
                                            no other hanlder error happened */
#define HA_ERR_INITIALIZATION     174    /* Error during initialization */
#define HA_ERR_FILE_TOO_SHORT	  175    /* File too short */
#define HA_ERR_WRONG_CRC	  176    /* Wrong CRC on page */
#define HA_ERR_TOO_MANY_CONCURRENT_TRXS 177 /*Too many active concurrent transactions */
/* There's no explicitly listed partition in table for the given value */
#define HA_ERR_NOT_IN_LOCK_PARTITIONS 178
#define HA_ERR_INDEX_COL_TOO_LONG 179    /* Index column length exceeds limit */
#define HA_ERR_INDEX_CORRUPT      180    /* InnoDB index corrupted */
#define HA_ERR_UNDO_REC_TOO_BIG   181    /* Undo log record too big */
#define HA_FTS_INVALID_DOCID      182    /* Invalid InnoDB Doc ID */
#define HA_ERR_TABLE_IN_FK_CHECK  183    /* Table being used in foreign key check */
#define HA_ERR_TABLESPACE_EXISTS  184    /* The tablespace existed in storage engine */
#define HA_ERR_TOO_MANY_FIELDS    185    /* Table has too many columns */
#define HA_ERR_ROW_IN_WRONG_PARTITION 186 /* Row in wrong partition */
#define HA_ERR_INNODB_READ_ONLY   187    /* InnoDB is in read only mode. */
#define HA_ERR_FTS_EXCEED_RESULT_CACHE_LIMIT  188 /* FTS query exceeds result cache limit */
#define HA_ERR_TEMP_FILE_WRITE_FAILURE	189	/* Temporary file write failure */
#define HA_ERR_INNODB_FORCED_RECOVERY 190	/* Innodb is in force recovery mode */
#define HA_ERR_FTS_TOO_MANY_WORDS_IN_PHRASE	191 /* Too many words in a phrase */
#define HA_ERR_TMP_TABLE_MAX_FILE_SIZE_EXCEEDED 192 /* on-disk temp table too large */
#define HA_ERR_FAILED_TO_LOCK_REC_NOWAIT 193 /* Failed to lock a record and didn't wait */
#define HA_ERR_QUERY_INTERRUPTED 194  /* The query was interrupted */
#define HA_ERR_LAST               194    /* Copy of last error nr */

/* Number of different errors */
#define HA_ERR_ERRORS            (HA_ERR_LAST - HA_ERR_FIRST + 1)

#define HA_POS_ERROR	(~ (ulonglong) 0)

#endif //AIRES_MY_BASE_H
