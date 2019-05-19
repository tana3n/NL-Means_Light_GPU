#include <windows.h>
#include "wavelet.h"
#include <emmintrin.h>

static const __m128 predict = { -0.5f, -0.5f, -0.5f, -0.5f};
static const __m128 update =  { 0.25f, 0.25f, 0.25f, 0.25f};
static const __m128 i_predict = { 0.5f, 0.5f, 0.5f, 0.5f};
static const __m128 i_update = {-0.25f, -0.25f, -0.25f, -0.25f};

void fwt53(int thread_id,int thread_num,void *param1, void* param2)
{
    WAVELET_PARAM* prm = (WAVELET_PARAM*)param1;
    int width = prm->srcwidth;
    int height = prm->srcheight;
    BYTE* src = prm->src;
    BYTE* dest = prm->dest;
    BYTE* ywork = work + worksize * thread_id;
    BYTE* xwork = ywork + 64 * (4 + 2 + 2);
    LONG* atomic_counter = (LONG*)(&prm->atomic_counter);
    int src_bpl = prm->src_bpl;
    int dest_bpl = prm->dest_bpl;
    
    int maxblock = (height + 3) / 4;
    int offsety = (prm->destheight / 2) * dest_bpl;
    int offsetx = (prm->destwidth / 2) * 12;
    while (TRUE)
    {
        int block = InterlockedIncrement(atomic_counter) - 1;
        if (block >= maxblock) break;
        int y0, y1, y2;
        y1 = block * 4;
        y0 = y1 - 2;
        y2 = y1 + 5;
        int yloop1, yloop2, yloop3;
        yloop2 = y2 - y0;
        if (y0 < 0)
        {
            yloop1 = 2;
            yloop2 -= 2;
        }
        else
        {
            yloop1 = 0;
        }
        if (y2 > height)
        {
            yloop3 = y2 - height;
            yloop2 -= yloop3;
        }
        else
        {
            yloop3 = 0;
        }

        int xloopcount = (width * 3 + 15) / 16;
        BYTE* src2 = src + (y0 + yloop1) * src_bpl;
        BYTE* dest2 = dest + y1 * dest_bpl / 2;
        BYTE* ywork2 = ywork + yloop1 * 64;
        BYTE* xwork2;
        __asm
        {
            mov             ecx, xloopcount
            mov             edi, xwork
            mov             eax, src_bpl
            mov             esi, src2
        align 16
        xloop_1:
            mov             xloopcount, ecx
            mov             xwork2, edi
            mov             edi, ywork2
            mov             ecx, yloop2
            
        align 16
        loop_1:
            movdqa          xmm1, [esi]
            movdqa          xmm3, [esi+16]
            punpcklwd       xmm0, xmm1
            punpckhwd       xmm1, xmm1
            punpcklwd       xmm2, xmm3
            punpckhwd       xmm3, xmm3
            psrad           xmm0, 16
            psrad           xmm1, 16
            psrad           xmm2, 16
            psrad           xmm3, 16
            cvtdq2ps        xmm0, xmm0
            cvtdq2ps        xmm1, xmm1
            cvtdq2ps        xmm2, xmm2
            cvtdq2ps        xmm3, xmm3
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            add             esi, eax
            add             edi, 64
            sub             ecx, 1
            jnz             loop_1
            
            mov             ecx, yloop3
            test            ecx, ecx
            jz              loop_2_end
            sub             esi, eax
            sub             esi, eax
        align 16
        loop_2:
            movdqa          xmm1, [esi]
            movdqa          xmm3, [esi+16]
            punpcklwd       xmm0, xmm1
            punpckhwd       xmm1, xmm1
            punpcklwd       xmm2, xmm3
            punpckhwd       xmm3, xmm3
            psrad           xmm0, 16
            psrad           xmm1, 16
            psrad           xmm2, 16
            psrad           xmm3, 16
            cvtdq2ps        xmm0, xmm0
            cvtdq2ps        xmm1, xmm1
            cvtdq2ps        xmm2, xmm2
            cvtdq2ps        xmm3, xmm3
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            sub             esi, eax
            add             edi, 64
            sub             ecx, 1
            jnz             loop_2
        loop_2_end:
            
            mov             ecx, yloop1
            test            ecx, ecx
            jz              padding_end
            mov             edi, ywork2
            movaps          xmm0, [edi+64]
            movaps          xmm1, [edi+80]
            movaps          xmm2, [edi+96]
            movaps          xmm3, [edi+112]
            movaps          [edi-64 ], xmm0
            movaps          [edi-48 ], xmm1
            movaps          [edi-32 ], xmm2
            movaps          [edi-16 ], xmm3
            movaps          xmm0, [edi+128]
            movaps          xmm1, [edi+144]
            movaps          xmm2, [edi+160]
            movaps          xmm3, [edi+176]
            movaps          [edi-128 ], xmm0
            movaps          [edi-112 ], xmm1
            movaps          [edi-96  ], xmm2
            movaps          [edi-80  ], xmm3
        padding_end:
            
            movaps          xmm7, predict
            movaps          xmm6, update
            mov             esi, ywork
            mov             edi, xwork2
            mov             ecx, 2
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            addps           xmm2, [esi+160]
            addps           xmm3, [esi+176]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            mulps           xmm2, xmm7
            mulps           xmm3, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            addps           xmm2, [esi+96]
            addps           xmm3, [esi+112]
            movaps          [esi+64], xmm0
            movaps          [esi+80], xmm1
            movaps          [esi+96], xmm2
            movaps          [esi+112], xmm3
            add             esi, 128
        align 16
        loop_3:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            addps           xmm2, [esi+160]
            addps           xmm3, [esi+176]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            mulps           xmm2, xmm7
            mulps           xmm3, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            addps           xmm2, [esi+96]
            addps           xmm3, [esi+112]
            movaps          [esi+64], xmm0
            movaps          [esi+80], xmm1
            movaps          [esi+96], xmm2
            movaps          [esi+112], xmm3
            addps           xmm0, [esi-64]
            addps           xmm1, [esi-48]
            addps           xmm2, [esi-32]
            addps           xmm3, [esi-16]
            mulps           xmm0, xmm6
            mulps           xmm1, xmm6
            mulps           xmm2, xmm6
            mulps           xmm3, xmm6
            addps           xmm0, [esi]
            addps           xmm1, [esi+16]
            addps           xmm2, [esi+32]
            addps           xmm3, [esi+48]
            movaps          [esi], xmm0
            movaps          [esi+16], xmm1
            movaps          [esi+32], xmm2
            movaps          [esi+48], xmm3
            add             esi, 128
            sub             ecx, 1
            jnz             loop_3
            
            sub             esi, 64*4
            mov             ecx, 4
        align 16
        loop_4:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+64]
            movaps          xmm2, [esi+128]
            movaps          xmm3, [esi+192]
            movaps          xmm4, xmm0
            movaps          xmm5, xmm1
            unpcklps        xmm4, xmm2
            unpckhps        xmm0, xmm2
            unpcklps        xmm5, xmm3
            unpckhps        xmm1, xmm3
            movaps          xmm2, xmm4
            movaps          xmm3, xmm0
            unpcklps        xmm2, xmm5
            unpckhps        xmm4, xmm5
            unpcklps        xmm3, xmm1
            unpckhps        xmm0, xmm1
            movaps          [edi], xmm2
            movaps          [edi+16], xmm4
            movaps          [edi+32], xmm3
            movaps          [edi+48], xmm0
            add             esi, 16
            add             edi, 64
            sub             ecx, 1
            jnz             loop_4
            
            mov             esi, src2
            mov             ecx, xloopcount
            add             esi, 32
            mov             src2, esi
            sub             ecx, 1
            jnz             xloop_1
        }
        xwork2 = xwork + 48 * width;
        __asm
        {
            mov             edx, xwork2
            movaps          xmm0, [edx-96]
            movaps          xmm1, [edx-80]
            movaps          xmm2, [edx-64]
            movaps          [edx], xmm0
            movaps          [edx+16], xmm1
            movaps          [edx+32], xmm2
        }
        xwork2 = xwork;
        for(int x=0; x<width; x+=160)
        {
            int xloop1;
            if (width - x > 160) xloop1 = 80;
            else xloop1 = (width - x) / 2;
            
            if (x!=0)
            {
                __asm
                {
                    movaps          xmm7, predict
                    movaps          xmm6, update
                    mov             esi, xwork2
                    mov             ecx, xloop1
                align 16
                loop_5:
                    movaps          xmm0, [esi]
                    movaps          xmm1, [esi+16]
                    movaps          xmm2, [esi+32]
                    movaps          xmm3, [esi+96]
                    movaps          xmm4, [esi+112]
                    movaps          xmm5, [esi+128]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm7
                    mulps           xmm4, xmm7
                    mulps           xmm5, xmm7
                    addps           xmm3, [esi+48]
                    addps           xmm4, [esi+64]
                    addps           xmm5, [esi+80]
                    movaps          [esi+48], xmm3
                    movaps          [esi+64], xmm4
                    movaps          [esi+80], xmm5
                    addps           xmm3, [esi-48]
                    addps           xmm4, [esi-32]
                    addps           xmm5, [esi-16]
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    movaps          [esi], xmm3
                    movaps          [esi+16], xmm4
                    movaps          [esi+32], xmm5
                    add             esi, 96
                    sub             ecx, 1
                    jnz             loop_5
                }
            }
            else
            {
                __asm
                {
                    movaps          xmm7, predict
                    movaps          xmm6, update
                    mov             esi, xwork2
                    mov             ecx, xloop1
                    movaps          xmm0, [esi]
                    movaps          xmm1, [esi+16]
                    movaps          xmm2, [esi+32]
                    addps           xmm0, [esi+96]
                    addps           xmm1, [esi+112]
                    addps           xmm2, [esi+128]
                    mulps           xmm0, xmm7
                    mulps           xmm1, xmm7
                    mulps           xmm2, xmm7
                    addps           xmm0, [esi+48]
                    addps           xmm1, [esi+64]
                    addps           xmm2, [esi+80]
                    movaps          [esi-48], xmm0
                    movaps          [esi-32], xmm1
                    movaps          [esi-16], xmm2
                align 16
                loop_6:
                    movaps          xmm0, [esi]
                    movaps          xmm1, [esi+16]
                    movaps          xmm2, [esi+32]
                    movaps          xmm3, [esi+96]
                    movaps          xmm4, [esi+112]
                    movaps          xmm5, [esi+128]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm7
                    mulps           xmm4, xmm7
                    mulps           xmm5, xmm7
                    addps           xmm3, [esi+48]
                    addps           xmm4, [esi+64]
                    addps           xmm5, [esi+80]
                    movaps          [esi+48], xmm3
                    movaps          [esi+64], xmm4
                    movaps          [esi+80], xmm5
                    addps           xmm3, [esi-48]
                    addps           xmm4, [esi-32]
                    addps           xmm5, [esi-16]
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    movaps          [esi], xmm3
                    movaps          [esi+16], xmm4
                    movaps          [esi+32], xmm5
                    add             esi, 96
                    sub             ecx, 1
                    jnz             loop_6
                }
            }
            
            BYTE* xwork3 = xwork2;
            BYTE* dest3 = dest2;
            __asm
            {
                mov             ebx, 2
                mov             esi, xwork3
                mov             edi, dest3
            align 16
            loop_7:
                mov             edx, offsety
                mov             ecx, xloop1
                add             edx, edi
            align 16
            loop_8:
                movlps          xmm0, [esi]
                movlps          xmm1, [esi+16]
                movlps          xmm2, [esi+32]
                movlps          xmm3, [esi+96]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi], xmm0
                movntps         [edx], xmm2
                movlps          xmm0, [esi+112]
                movlps          xmm1, [esi+128]
                movlps          xmm2, [esi+192]
                movlps          xmm3, [esi+208]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+16], xmm0
                movntps         [edx+16], xmm2
                movlps          xmm0, [esi+224]
                movlps          xmm1, [esi+288]
                movlps          xmm2, [esi+304]
                movlps          xmm3, [esi+320]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+32], xmm0
                movntps         [edx+32], xmm2
                add             esi, 384
                add             edi, 48
                add             edx, 48
                sub             ecx, 4
                jnle            loop_8
                
                mov             esi, xwork3
                mov             edi, dest3
                mov             edx, offsety
                add             esi, 48
                add             edi, offsetx
                mov             ecx, xloop1
                add             edx, edi
            align 16
            loop_9:
                movlps          xmm0, [esi]
                movlps          xmm1, [esi+16]
                movlps          xmm2, [esi+32]
                movlps          xmm3, [esi+96]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi], xmm0
                movntps         [edx], xmm2
                movlps          xmm0, [esi+112]
                movlps          xmm1, [esi+128]
                movlps          xmm2, [esi+192]
                movlps          xmm3, [esi+208]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+16], xmm0
                movntps         [edx+16], xmm2
                movlps          xmm0, [esi+224]
                movlps          xmm1, [esi+288]
                movlps          xmm2, [esi+304]
                movlps          xmm3, [esi+320]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+32], xmm0
                movntps         [edx+32], xmm2
                add             esi, 384
                add             edi, 48
                add             edx, 48
                sub             ecx, 4
                jnle            loop_9
                
                mov             edi, dest3
                mov             esi, xwork3
                add             edi, dest_bpl
                add             esi, 8
                mov             dest3, edi
                mov             xwork3, esi
                sub             ebx, 1
                jnz             loop_7
            }
            xwork2 += 160 * 48;
            dest2 += 80 * 12;
        }
    }
}

void fwt53_LL(int thread_id,int thread_num,void *param1, void* param2)
{
    WAVELET_PARAM* prm = (WAVELET_PARAM*)param1;
    int width = prm->srcwidth;
    int height = prm->srcheight;
    BYTE* src = prm->src;
    BYTE* dest = prm->dest;
    BYTE* ywork = work + worksize * thread_id;
    BYTE* xwork = ywork + 64 * (8 + 2 + 2);
    LONG* atomic_counter = (LONG*)(&prm->atomic_counter);
    int src_bpl = prm->src_bpl;
    int dest_bpl = prm->dest_bpl;
    int maxblock = (height + 7) / 8;
    while (TRUE)
    {
        int block = InterlockedIncrement(atomic_counter) - 1;
        if (block >= maxblock) break;
        int y0, y1, y2;
        y1 = block * 8;
        y0 = y1 - 2;
        y2 = y1 + 9;
        int yloop1, yloop2, yloop3;
        yloop2 = y2 - y0;
        if (y0 < 0)
        {
            yloop1 = 2;
            yloop2 -= 2;
        }
        else
        {
            yloop1 = 0;
        }
        if (y2 > height)
        {
            yloop3 = y2 - height;
            yloop2 -= yloop3;
        }
        else
        {
            yloop3 = 0;
        }

        int xloopcount = (width * 3 + 15) / 16;
        BYTE* src2 = src + (y0 + yloop1) * src_bpl;
        BYTE* dest2 = dest + y1 * dest_bpl / 2;
        BYTE* ywork2 = ywork + yloop1 * 64;
        BYTE* xwork2;
        __asm
        {
            mov             ecx, xloopcount
            mov             edi, xwork
            mov             eax, src_bpl
            mov             esi, src2
        align 16
        xloop_1:
            mov             xloopcount, ecx
            mov             xwork2, edi
            mov             edi, ywork2
            mov             ecx, yloop2
            
        align 16
        loop_1:
            movdqa          xmm1, [esi]
            movdqa          xmm3, [esi+16]
            punpcklwd       xmm0, xmm1
            punpckhwd       xmm1, xmm1
            punpcklwd       xmm2, xmm3
            punpckhwd       xmm3, xmm3
            psrad           xmm0, 16
            psrad           xmm1, 16
            psrad           xmm2, 16
            psrad           xmm3, 16
            cvtdq2ps        xmm0, xmm0
            cvtdq2ps        xmm1, xmm1
            cvtdq2ps        xmm2, xmm2
            cvtdq2ps        xmm3, xmm3
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            add             esi, eax
            add             edi, 64
            sub             ecx, 1
            jnz             loop_1
            
            mov             ecx, yloop3
            test            ecx, ecx
            jz              loop_2_end
            sub             esi, eax
            sub             esi, eax
        align 16
        loop_2:
            movdqa          xmm1, [esi]
            movdqa          xmm3, [esi+16]
            punpcklwd       xmm0, xmm1
            punpckhwd       xmm1, xmm1
            punpcklwd       xmm2, xmm3
            punpckhwd       xmm3, xmm3
            psrad           xmm0, 16
            psrad           xmm1, 16
            psrad           xmm2, 16
            psrad           xmm3, 16
            cvtdq2ps        xmm0, xmm0
            cvtdq2ps        xmm1, xmm1
            cvtdq2ps        xmm2, xmm2
            cvtdq2ps        xmm3, xmm3
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            sub             esi, eax
            add             edi, 64
            sub             ecx, 1
            jnz             loop_2
        loop_2_end:
            
            mov             ecx, yloop1
            test            ecx, ecx
            jz              padding_end
            mov             edi, ywork2
            movaps          xmm0, [edi+64]
            movaps          xmm1, [edi+80]
            movaps          xmm2, [edi+96]
            movaps          xmm3, [edi+112]
            movaps          [edi-64 ], xmm0
            movaps          [edi-48 ], xmm1
            movaps          [edi-32 ], xmm2
            movaps          [edi-16 ], xmm3
            movaps          xmm0, [edi+128]
            movaps          xmm1, [edi+144]
            movaps          xmm2, [edi+160]
            movaps          xmm3, [edi+176]
            movaps          [edi-128 ], xmm0
            movaps          [edi-112 ], xmm1
            movaps          [edi-96  ], xmm2
            movaps          [edi-80  ], xmm3
        padding_end:
            
            movaps          xmm7, predict
            movaps          xmm6, update
            mov             esi, ywork
            mov             edi, xwork2
            mov             ecx, 4
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            add             esi, 128
        align 16
        loop_3:
            movaps          xmm2, [esi]
            movaps          xmm3, [esi+16]
            movaps          xmm4, [esi+128]
            movaps          xmm5, [esi+144]
            addps           xmm4, xmm2
            addps           xmm5, xmm3
            mulps           xmm4, xmm7
            mulps           xmm5, xmm7
            addps           xmm4, [esi+64]
            addps           xmm5, [esi+80]
            addps           xmm0, xmm4
            addps           xmm1, xmm5
            mulps           xmm0, xmm6
            mulps           xmm1, xmm6
            addps           xmm2, xmm0
            addps           xmm3, xmm1
            movaps          xmm0, xmm4
            movaps          xmm1, xmm5
            movaps          [esi], xmm2
            movaps          [esi+16], xmm3
            add             esi, 128
            sub             ecx, 1
            jnz             loop_3
            
            sub             esi, 128*5-32
            mov             ecx, 4
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            add             esi, 128
        align 16
        loop_4:
            movaps          xmm2, [esi]
            movaps          xmm3, [esi+16]
            movaps          xmm4, [esi+128]
            movaps          xmm5, [esi+144]
            addps           xmm4, xmm2
            addps           xmm5, xmm3
            mulps           xmm4, xmm7
            mulps           xmm5, xmm7
            addps           xmm4, [esi+64]
            addps           xmm5, [esi+80]
            addps           xmm0, xmm4
            addps           xmm1, xmm5
            mulps           xmm0, xmm6
            mulps           xmm1, xmm6
            addps           xmm2, xmm0
            addps           xmm3, xmm1
            movaps          xmm0, xmm4
            movaps          xmm1, xmm5
            movaps          [esi], xmm2
            movaps          [esi+16], xmm3
            add             esi, 128
            sub             ecx, 1
            jnz             loop_4
            
            sub             esi, 128*4+32
            mov             ecx, 4
        align 16
        loop_5:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+128]
            movaps          xmm2, [esi+256]
            movaps          xmm3, [esi+384]
            movaps          xmm4, xmm0
            movaps          xmm5, xmm1
            unpcklps        xmm4, xmm2
            unpckhps        xmm0, xmm2
            unpcklps        xmm5, xmm3
            unpckhps        xmm1, xmm3
            movaps          xmm2, xmm4
            movaps          xmm3, xmm0
            unpcklps        xmm2, xmm5
            unpckhps        xmm4, xmm5
            unpcklps        xmm3, xmm1
            unpckhps        xmm0, xmm1
            movaps          [edi], xmm2
            movaps          [edi+16], xmm4
            movaps          [edi+32], xmm3
            movaps          [edi+48], xmm0
            add             esi, 16
            add             edi, 64
            sub             ecx, 1
            jnz             loop_5
            
            mov             esi, src2
            mov             ecx, xloopcount
            add             esi, 32
            mov             src2, esi
            sub             ecx, 1
            jnz             xloop_1
        }
        xwork2 = xwork + 48 * width;
        __asm
        {
            mov             edx, xwork2
            movaps          xmm0, [edx-96]
            movaps          xmm1, [edx-96+16]
            movaps          xmm2, [edx-96+32]
            movaps          [edx], xmm0
            movaps          [edx+16], xmm1
            movaps          [edx+32], xmm2
        }
        xwork2 = xwork;
        
        for(int x=0; x<width; x+=160)
        {
            int xloop1;
            if (width - x > 160) xloop1 = 80;
            else xloop1 = (width - x) / 2;
            
            if (x!=0)
            {
                __asm
                {
                    movaps          xmm7, predict
                    movaps          xmm6, update
                    mov             esi, xwork2
                    mov             ecx, xloop1
                    movaps          xmm3, [esi-48]
                    movaps          xmm4, [esi-32]
                    movaps          xmm5, [esi-16]
                align 16
                loop_6:
                    movaps          xmm0, [esi]
                    movaps          xmm1, [esi+16]
                    movaps          xmm2, [esi+32]
                    addps           xmm0, [esi+96]
                    addps           xmm1, [esi+112]
                    addps           xmm2, [esi+128]
                    mulps           xmm0, xmm7
                    mulps           xmm1, xmm7
                    mulps           xmm2, xmm7
                    addps           xmm0, [esi+48]
                    addps           xmm1, [esi+64]
                    addps           xmm2, [esi+80]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, [esi]
                    addps           xmm4, [esi+16]
                    addps           xmm5, [esi+32]
                    movaps          [esi], xmm3
                    movaps          [esi+16], xmm4
                    movaps          [esi+32], xmm5
                    movaps          xmm3, xmm0
                    movaps          xmm4, xmm1
                    movaps          xmm5, xmm2
                    add             esi, 96
                    sub             ecx, 1
                    jnz             loop_6
                    movaps          [esi-48], xmm3
                    movaps          [esi-32], xmm4
                    movaps          [esi-16], xmm5
                }
            }
            else
            {
                __asm
                {
                    movaps          xmm7, predict
                    movaps          xmm6, update
                    mov             esi, xwork2
                    mov             ecx, xloop1
                    movaps          xmm3, [esi]
                    movaps          xmm4, [esi+16]
                    movaps          xmm5, [esi+32]
                    addps           xmm3, [esi+96]
                    addps           xmm4, [esi+112]
                    addps           xmm5, [esi+128]
                    mulps           xmm3, xmm7
                    mulps           xmm4, xmm7
                    mulps           xmm5, xmm7
                    addps           xmm3, [esi+48]
                    addps           xmm4, [esi+64]
                    addps           xmm5, [esi+80]
                align 16
                loop_7:
                    movaps          xmm0, [esi]
                    movaps          xmm1, [esi+16]
                    movaps          xmm2, [esi+32]
                    addps           xmm0, [esi+96]
                    addps           xmm1, [esi+112]
                    addps           xmm2, [esi+128]
                    mulps           xmm0, xmm7
                    mulps           xmm1, xmm7
                    mulps           xmm2, xmm7
                    addps           xmm0, [esi+48]
                    addps           xmm1, [esi+64]
                    addps           xmm2, [esi+80]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, [esi]
                    addps           xmm4, [esi+16]
                    addps           xmm5, [esi+32]
                    movaps          [esi], xmm3
                    movaps          [esi+16], xmm4
                    movaps          [esi+32], xmm5
                    movaps          xmm3, xmm0
                    movaps          xmm4, xmm1
                    movaps          xmm5, xmm2
                    add             esi, 96
                    sub             ecx, 1
                    jnz             loop_7
                    movaps          [esi-48], xmm3
                    movaps          [esi-32], xmm4
                    movaps          [esi-16], xmm5
                }
            }
            
            BYTE* xwork3 = xwork2;
            BYTE* dest3 = dest2;
            __asm
            {
                mov             ebx, 2
                mov             esi, xwork3
                mov             edi, dest3
                mov             eax, dest_bpl
            loop_8:
                mov             ecx, xloop1
            align 16
            loop_9:
                movlps          xmm0, [esi]
                movlps          xmm1, [esi+16]
                movlps          xmm2, [esi+32]
                movlps          xmm3, [esi+96]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi], xmm0
                movntps         [edi+eax], xmm2
                movlps          xmm0, [esi+112]
                movlps          xmm1, [esi+128]
                movlps          xmm2, [esi+192]
                movlps          xmm3, [esi+208]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+16], xmm0
                movntps         [edi+eax+16], xmm2
                movlps          xmm0, [esi+224]
                movlps          xmm1, [esi+288]
                movlps          xmm2, [esi+304]
                movlps          xmm3, [esi+320]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movntps         [edi+32], xmm0
                movntps         [edi+eax+32], xmm2
                add             esi, 384
                add             edi, 48
                sub             ecx, 4
                jnle            loop_9
                
                mov             edi, dest3
                mov             esi, xwork3
                lea             edi, [edi+eax*2]
                add             esi, 8
                mov             dest3, edi
                mov             xwork3, esi
                sub             ebx, 1
                jnz             loop_8
            }
            xwork2 += 160 * 48;
            dest2 += 80 * 12;
        }
    }
}

void iwt53(int thread_id,int thread_num,void *param1, void* param2)
{
    WAVELET_PARAM* prm = (WAVELET_PARAM*)param1;
    int width = prm->destwidth;
    int height = (prm->destheight+7)&~7;
    int halfw = prm->destwidth / 2;
    int halfh = height / 2;
    BYTE* src = prm->src;
    BYTE* dest = prm->dest;
    BYTE* ywork = work + worksize * thread_id;
    BYTE* xwork = ywork + 64 * (4 + 2 + 2);
    LONG* atomic_counter = (LONG*)(&prm->atomic_counter);
    int src_bpl = prm->src_bpl;
    int dest_bpl = prm->dest_bpl;
    int maxblock = (height+3) / 4;
    int v_hi = (prm->srcheight / 2) * src_bpl;
    int h_hi = (prm->srcwidth / 2) * 12;
    int xworkhalf = ((((halfw + 15) &~15) + 1) * 48 +127) &~127;
    int xloop1 = (halfw * 3 + 15) / 16;
    while (TRUE)
    {
        int block = InterlockedIncrement(atomic_counter) - 1;
        if (block >= maxblock) break;
        int y0, y1, y2;
        y1 = block * 2;
        y0 = y1 - 1;
        y2 = y1 + 3;
        int yloop1, yloop2, yloop3;
        yloop2 = y2 - y0;
        if (y0 < 0)
        {
            yloop1 = 1;
            yloop2 -= 1;
        }
        else
        {
            yloop1 = 0;
        }
        if (y2 > halfh)
        {
            yloop3 = y2 - halfh;
            yloop2 -= yloop3;
        }
        else
        {
            yloop3 = 0;
        }

        BYTE* src2 = src + (y0 + yloop1) * src_bpl;
        BYTE* dest2 = dest + y1 * dest_bpl * 2;
        BYTE* ywork2 = ywork + yloop1 * 128;
        BYTE* xwork2 = xwork;
        BYTE* src3 = src2;
        int xloop2;
        __asm
        {
            mov             ecx, xloop1
            mov             esi, src3
            mov             eax, src_bpl
            mov             edx, v_hi
            mov             ebx, 2
        align 16
        xloop_1:
            mov             xloop2, ecx
            mov             edi, ywork2
            mov             ecx, yloop2
        align 16
        loop_1:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            movaps          xmm0, [esi+edx]
            movaps          xmm1, [esi+edx+16]
            movaps          xmm2, [esi+edx+32]
            movaps          xmm3, [esi+edx+48]
            movaps          [edi+64], xmm0
            movaps          [edi+80], xmm1
            movaps          [edi+96], xmm2
            movaps          [edi+112], xmm3
            add             esi, eax
            add             edi, 128
            sub             ecx, 1
            jnz             loop_1
            
            mov             ecx, yloop3
            test            ecx, ecx
            jz              loop_2_end
            lea             esi, [edi-192]
        align 16
        loop_2:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            movaps          [edi+64], xmm0
            movaps          [edi+80], xmm1
            movaps          [edi+96], xmm2
            movaps          [edi+112], xmm3
            movaps          xmm0, [esi+64]
            movaps          xmm1, [esi+80]
            movaps          xmm2, [esi+96]
            movaps          xmm3, [esi+112]
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
            movaps          [edi+48], xmm3
            sub             esi, 128
            add             edi, 128
            sub             ecx, 1
            jnz             loop_2
            loop_2_end:
            
            mov             ecx, yloop1
            test            ecx, ecx
            jz              padding_end
            mov             edi, ywork2
            mov             esi, edi
            movaps          xmm0, [edi+64]
            movaps          xmm1, [edi+80]
            movaps          xmm2, [edi+96]
            movaps          xmm3, [esi+112]
            movaps          [edi-64], xmm0
            movaps          [edi-48], xmm1
            movaps          [edi-32], xmm2
            movaps          [edi-16], xmm3
            movaps          xmm0, [esi+128]
            movaps          xmm1, [esi+144]
            movaps          xmm2, [esi+160]
            movaps          xmm3, [esi+176]
            movaps          [edi-128], xmm0
            movaps          [edi-112], xmm1
            movaps          [edi-96], xmm2
            movaps          [edi-80], xmm3
        padding_end:
            
            movaps          xmm7, i_update
            movaps          xmm6, i_predict
            mov             esi, ywork
            mov             edi, xwork2
            mov             ecx, 2
            add             esi, 64
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            addps           xmm2, [esi+160]
            addps           xmm3, [esi+176]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            mulps           xmm2, xmm7
            mulps           xmm3, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            addps           xmm2, [esi+96]
            addps           xmm3, [esi+112]
            movaps          [esi+64], xmm0
            movaps          [esi+80], xmm1
            movaps          [esi+96], xmm2
            movaps          [esi+112], xmm3
            add             esi, 128
        align 16
        loop_3:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+16]
            movaps          xmm2, [esi+32]
            movaps          xmm3, [esi+48]
            addps           xmm0, [esi+128]
            addps           xmm1, [esi+144]
            addps           xmm2, [esi+160]
            addps           xmm3, [esi+176]
            mulps           xmm0, xmm7
            mulps           xmm1, xmm7
            mulps           xmm2, xmm7
            mulps           xmm3, xmm7
            addps           xmm0, [esi+64]
            addps           xmm1, [esi+80]
            addps           xmm2, [esi+96]
            addps           xmm3, [esi+112]
            movaps          [esi+64], xmm0
            movaps          [esi+80], xmm1
            movaps          [esi+96], xmm2
            movaps          [esi+112], xmm3
            addps           xmm0, [esi-64]
            addps           xmm1, [esi-48]
            addps           xmm2, [esi-32]
            addps           xmm3, [esi-16]
            mulps           xmm0, xmm6
            mulps           xmm1, xmm6
            mulps           xmm2, xmm6
            mulps           xmm3, xmm6
            addps           xmm0, [esi]
            addps           xmm1, [esi+16]
            addps           xmm2, [esi+32]
            addps           xmm3, [esi+48]
            movaps          [esi], xmm0
            movaps          [esi+16], xmm1
            movaps          [esi+32], xmm2
            movaps          [esi+48], xmm3
            add             esi, 128
            sub             ecx, 1
            jnz             loop_3
            
            sub             esi, 64*5
            mov             ecx, 4
        align 16
        loop_4:
            movaps          xmm0, [esi]
            movaps          xmm1, [esi+64]
            movaps          xmm2, [esi+128]
            movaps          xmm3, [esi+192]
            movaps          xmm4, xmm0
            movaps          xmm5, xmm1
            unpcklps        xmm0, xmm2
            unpckhps        xmm4, xmm2
            unpcklps        xmm1, xmm3
            unpckhps        xmm5, xmm3
            movaps          xmm2, xmm0
            movaps          xmm3, xmm4
            unpcklps        xmm0, xmm1
            unpckhps        xmm2, xmm1
            unpcklps        xmm4, xmm5
            unpckhps        xmm3, xmm5
            movaps          [edi], xmm0
            movaps          [edi+16], xmm2
            movaps          [edi+32], xmm4
            movaps          [edi+48], xmm3
            add             esi, 16
            add             edi, 64
            sub             ecx, 1
            jnz             loop_4
            
            mov             esi, src3
            mov             ecx, xloop2
            add             esi, 64
            mov             xwork2, edi
            mov             src3, esi
            sub             ecx, 1
            jnz             xloop_1
            
            sub             ebx, 1
            jz              xloop_1_end
            
            mov             esi, src2
            mov             edi, xwork
            add             esi, h_hi
            add             edi, xworkhalf
            mov             src3, esi
            mov             xwork2, edi
            mov             ecx, xloop1
            jmp             xloop_1
        xloop_1_end:
        }

        xwork2 = xwork + 48 * halfw;
        BYTE* xwork3 = xwork + 48 * halfw + xworkhalf;
        __asm
        {
            mov             esi, xwork2
            mov             edi, xwork3
            movaps          xmm0, [esi-48]
            movaps          xmm1, [esi-32]
            movaps          xmm2, [esi-16]
            movaps          [esi], xmm0
            movaps          [esi+16], xmm1
            movaps          [esi+32], xmm2
            movaps          xmm0, [edi-96]
            movaps          xmm1, [edi-80]
            movaps          xmm2, [edi-64]
            movaps          [edi], xmm0
            movaps          [edi+16], xmm1
            movaps          [edi+32], xmm2
        }
        
        xwork2 = xwork;
        for(int x=0; x<halfw; x+=80)
        {
            int xloop3;
            if (halfw - x > 80) xloop3 = 80;
            else xloop3 = halfw - x;
            if(x!=0)
            {
                __asm
                {
                    movaps          xmm7, i_update
                    movaps          xmm6, i_predict
                    mov             esi, xwork2
                    mov             edx, xworkhalf
                    mov             ecx, xloop3
                align 16
                loop_5:
                    movaps          xmm0, [esi+edx]
                    movaps          xmm1, [esi+edx+16]
                    movaps          xmm2, [esi+edx+32]
                    movaps          xmm3, [esi+edx+48]
                    movaps          xmm4, [esi+edx+64]
                    movaps          xmm5, [esi+edx+80]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm7
                    mulps           xmm4, xmm7
                    mulps           xmm5, xmm7
                    addps           xmm3, [esi+48]
                    addps           xmm4, [esi+64]
                    addps           xmm5, [esi+80]
                    movaps          [esi+48], xmm3
                    movaps          [esi+64], xmm4
                    movaps          [esi+80], xmm5
                    addps           xmm3, [esi]
                    addps           xmm4, [esi+16]
                    addps           xmm5, [esi+32]
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    movaps          [esi+edx], xmm3
                    movaps          [esi+edx+16], xmm4
                    movaps          [esi+edx+32], xmm5
                    add             esi, 48
                    sub             ecx, 1
                    jnz             loop_5
                }
            }
            else
            {
                __asm
                {
                    movaps          xmm7, i_update
                    movaps          xmm6, i_predict
                    mov             esi, xwork2
                    mov             edx, xworkhalf
                    mov             ecx, xloop3
                    movaps          xmm0, [esi+edx]
                    movaps          xmm1, [esi+edx+16]
                    movaps          xmm2, [esi+edx+32]
                    addps           xmm0, xmm0
                    addps           xmm1, xmm1
                    addps           xmm2, xmm2
                    mulps           xmm0, xmm7
                    mulps           xmm1, xmm7
                    mulps           xmm2, xmm7
                    addps           xmm0, [esi]
                    addps           xmm1, [esi+16]
                    addps           xmm2, [esi+32]
                    movaps          [esi], xmm0
                    movaps          [esi+16], xmm1
                    movaps          [esi+32], xmm2
                align 16
                loop_6:
                    movaps          xmm0, [esi+edx]
                    movaps          xmm1, [esi+edx+16]
                    movaps          xmm2, [esi+edx+32]
                    movaps          xmm3, [esi+edx+48]
                    movaps          xmm4, [esi+edx+64]
                    movaps          xmm5, [esi+edx+80]
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    mulps           xmm3, xmm7
                    mulps           xmm4, xmm7
                    mulps           xmm5, xmm7
                    addps           xmm3, [esi+48]
                    addps           xmm4, [esi+64]
                    addps           xmm5, [esi+80]
                    movaps          [esi+48], xmm3
                    movaps          [esi+64], xmm4
                    movaps          [esi+80], xmm5
                    addps           xmm3, [esi]
                    addps           xmm4, [esi+16]
                    addps           xmm5, [esi+32]
                    mulps           xmm3, xmm6
                    mulps           xmm4, xmm6
                    mulps           xmm5, xmm6
                    addps           xmm3, xmm0
                    addps           xmm4, xmm1
                    addps           xmm5, xmm2
                    movaps          [esi+edx], xmm3
                    movaps          [esi+edx+16], xmm4
                    movaps          [esi+edx+32], xmm5
                    add             esi, 48
                    sub             ecx, 1
                    jnz             loop_6
                }
            }
            
            BYTE* xwork3 = xwork2;
            BYTE* dest3 = dest2;
            __asm
            {
                mov             ebx, 2
                mov             esi, xwork3
                mov             edi, dest3
                mov             edx, xworkhalf
                mov             eax, dest_bpl
            align 16
            loop_7:
                mov             ecx, xloop3
                
            align 16
            loop_8:
                movlps          xmm0, [esi]
                movlps          xmm1, [esi+16]
                movlps          xmm2, [esi+32]
                movlps          xmm3, [esi+edx]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movlps          xmm1, [esi+edx+16]
                movlps          xmm3, [esi+edx+32]
                movlps          xmm4, [esi+48]
                movlps          xmm5, [esi+64]
                unpcklps        xmm1, xmm4
                unpcklps        xmm3, xmm5
                movaps          xmm4, xmm1
                unpcklps        xmm1, xmm3
                unpckhps        xmm4, xmm3
                cvtps2dq        xmm0, xmm0
                cvtps2dq        xmm2, xmm2
                cvtps2dq        xmm1, xmm1
                cvtps2dq        xmm4, xmm4
                packssdw        xmm0, xmm1
                packssdw        xmm2, xmm4
                movntdq         [edi], xmm0
                movntdq         [edi+eax], xmm2
                
                movlps          xmm0, [esi+80]
                movlps          xmm1, [esi+edx+48]
                movlps          xmm2, [esi+edx+64]
                movlps          xmm3, [esi+edx+80]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movlps          xmm1, [esi+96]
                movlps          xmm3, [esi+112]
                movlps          xmm4, [esi+128]
                movlps          xmm5, [esi+edx+96]
                unpcklps        xmm1, xmm4
                unpcklps        xmm3, xmm5
                movaps          xmm4, xmm1
                unpcklps        xmm1, xmm3
                unpckhps        xmm4, xmm3
                cvtps2dq        xmm0, xmm0
                cvtps2dq        xmm2, xmm2
                cvtps2dq        xmm1, xmm1
                cvtps2dq        xmm4, xmm4
                packssdw        xmm0, xmm1
                packssdw        xmm2, xmm4
                movntdq         [edi+16], xmm0
                movntdq         [edi+eax+16], xmm2
                
                movlps          xmm0, [esi+edx+112]
                movlps          xmm1, [esi+edx+128]
                movlps          xmm2, [esi+144]
                movlps          xmm3, [esi+160]
                unpcklps        xmm0, xmm2
                unpcklps        xmm1, xmm3
                movaps          xmm2, xmm0
                unpcklps        xmm0, xmm1
                unpckhps        xmm2, xmm1
                movlps          xmm1, [esi+176]
                movlps          xmm3, [esi+edx+144]
                movlps          xmm4, [esi+edx+160]
                movlps          xmm5, [esi+edx+176]
                unpcklps        xmm1, xmm4
                unpcklps        xmm3, xmm5
                movaps          xmm4, xmm1
                unpcklps        xmm1, xmm3
                unpckhps        xmm4, xmm3
                cvtps2dq        xmm0, xmm0
                cvtps2dq        xmm2, xmm2
                cvtps2dq        xmm1, xmm1
                cvtps2dq        xmm4, xmm4
                packssdw        xmm0, xmm1
                packssdw        xmm2, xmm4
                movntdq         [edi+32], xmm0
                movntdq         [edi+eax+32], xmm2
                add             esi, 192
                add             edi, 48
                sub             ecx, 4
                jnle            loop_8
                
                mov             edi, dest3
                mov             esi, xwork3
                lea             edi, [edi+eax*2]
                add             esi, 8
                mov             dest3, edi
                mov             xwork3, esi
                sub             ebx, 1
                jnz             loop_7
            }
            xwork2 += 80 * 48;
            dest2 += 160 * 6;
        }
    }
}

void blend_lo(int thread_id,int thread_num,void *param1, void* param2)
{
    BLEND_PARAM* prm = (BLEND_PARAM*)param1;
    int width = prm->width;
    int height = prm->height;
    int y_start = thread_id * height / thread_num;
    int y_end = (thread_id+1) * height / thread_num;
    int xloopcount = width / 4;
    int yloopcount = y_end - y_start;
    BYTE* src = prm->src + y_start * prm->src_bpl;
    BYTE* dest = prm->dest + y_start * prm->dest_bpl;
    int deststep = prm->dest_bpl - xloopcount * 48;
    int srcstep = prm->src_bpl - xloopcount * 48;
    float w1 = prm->str / 100;
    float w2 = 1.0f - w1;
    __asm
    {
        movss           xmm6, w2
        movss           xmm7, w1
        shufps          xmm6, xmm6, 0
        shufps          xmm7, xmm7, 0
        mov             edx, dest
        mov             eax, src
        mov             ebx, yloopcount
    yloop:
        mov             ecx, xloopcount
    align 16
    xloop:
        movaps          xmm0, [edx]
        movaps          xmm1, [edx+16]
        movaps          xmm2, [edx+32]
        movaps          xmm3, [eax]
        movaps          xmm4, [eax+16]
        movaps          xmm5, [eax+32]
        mulps           xmm0, xmm6
        mulps           xmm1, xmm6
        mulps           xmm2, xmm6
        mulps           xmm3, xmm7
        mulps           xmm4, xmm7
        mulps           xmm5, xmm7
        addps           xmm0, xmm3
        addps           xmm1, xmm4
        addps           xmm2, xmm5
        movaps          [edx], xmm0
        movaps          [edx+16], xmm1
        movaps          [edx+32], xmm2
        add             edx, 48
        add             eax, 48
        sub             ecx, 1
        jnz             xloop
        add             edx, deststep
        add             eax, srcstep
        sub             ebx, 1
        jnz             yloop
    }
}
