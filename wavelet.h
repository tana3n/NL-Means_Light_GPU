typedef struct {
    BYTE* src;
    BYTE* dest;
    LONG atomic_counter;
    int srcwidth;
    int srcheight;
    int destwidth;
    int destheight;
    int src_bpl;
    int dest_bpl;
} WAVELET_PARAM, *LPWAVELET_PARAM;

typedef struct {
    BYTE* src;
    BYTE* dest;
    int width;
    int height;
    int src_bpl;
    int dest_bpl;
    float str;
} BLEND_PARAM;

extern BYTE* work;
extern int worksize;

void fwt53(int thread_id,int thread_num,void *param1, void* param2);
void fwt53_LL(int thread_id,int thread_num,void *param1, void* param2);
void iwt53(int thread_id,int thread_num,void *param1, void* param2);
void blend_lo(int thread_id,int thread_num,void *param1, void* param2);