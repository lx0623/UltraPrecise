//
// Created by tengjp on 19-7-29.
//
#include "./include/my_bitmap.h"
#include "./include/mysql_thread.h"
void create_last_word_mask(MY_BITMAP *map)
{
    /* Get the number of used bits (1..8) in the last byte */
    unsigned int const used= 1U + ((map->n_bits-1U) & 0x7U);

    /*
      Create a mask with the upper 'unused' bits set and the lower 'used'
      bits clear. The bits within each byte is stored in big-endian order.
     */
    unsigned char const mask= (~((1 << used) - 1)) & 255;

    /*
      The first bytes are to be set to zero since they represent real  bits
      in the bitvector. The last bytes are set to 0xFF since they  represent
      bytes not used by the bitvector. Finally the last byte contains  bits
      as set by the mask above.
    */
    unsigned char *ptr= (unsigned char*)&map->last_word_mask;

    map->last_word_ptr= map->bitmap + no_words_in_map(map)-1;
    switch (no_bytes_in_map(map) & 3) {
        case 1:
            map->last_word_mask= ~0U;
            ptr[0]= mask;
            return;
        case 2:
            map->last_word_mask= ~0U;
            ptr[0]= 0;
            ptr[1]= mask;
            return;
        case 3:
            map->last_word_mask= 0U;
            ptr[2]= mask;
            ptr[3]= 0xFFU;
            return;
        case 0:
            map->last_word_mask= 0U;
            ptr[3]= mask;
            return;
    }
}

my_bool bitmap_init(MY_BITMAP *map, my_bitmap_map *buf, uint n_bits,
                    my_bool thread_safe MY_ATTRIBUTE((unused)))
{
    // DBUG_ENTER("bitmap_init");
    if (!buf)
    {
        uint size_in_bytes= bitmap_buffer_size(n_bits);
        uint extra= 0;

        if (thread_safe)
        {
            size_in_bytes= ALIGN_SIZE(size_in_bytes);
            extra= sizeof(mysql_mutex_t);
        }
        map->mutex= 0;

        if (!(buf= (my_bitmap_map*) malloc(size_in_bytes+extra)))
            DBUG_RETURN(1);

        if (thread_safe)
        {
            map->mutex= (mysql_mutex_t *) ((char*) buf + size_in_bytes);
            mysql_mutex_init(map->mutex, MY_MUTEX_INIT_FAST);
        }

    }

    else
    {
        DBUG_ASSERT(thread_safe == 0);
        map->mutex= NULL;
    }


    map->bitmap= buf;
    map->n_bits= n_bits;
    create_last_word_mask(map);
    bitmap_clear_all(map);
    DBUG_RETURN(0);
}


void bitmap_free(MY_BITMAP *map)
{
    // DBUG_ENTER("bitmap_free");
    if (map->bitmap)
    {
        if (map->mutex)
            mysql_mutex_destroy(map->mutex);

        free(map->bitmap);
        map->bitmap=0;
    }
    DBUG_VOID_RETURN;
}

