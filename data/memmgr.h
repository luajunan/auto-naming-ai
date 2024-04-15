#ifndef MEMMGR_H
#define MEMMGR_H

#define DEBUG_MEMMGR_SUPPORT_STATS 1

#define POOL_SIZE 8 * 1024
#define MIN_POOL_ALLOC_QUANTAS 16


typedef unsigned char byte;
typedef unsigned long ulong;


void memmgr_init();

void* memmgr_alloc(ulong nbytes);

void memmgr_free(void* ap);

void memmgr_print_stats();


#endif
