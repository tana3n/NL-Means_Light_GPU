#include <windows.h>
//AviUtlのFiler.h
#include "filter.h"
#include <stdio.h>
/*一時的に作業フォルダに配置*/
#include <d3d9.h>
//#include <d3dx9.h>
#include "resource/d3dx9.h"
#include <d3d9types.h>
#pragma comment( lib, "d3d9.lib" )
#pragma comment( lib, "resource/d3dx9.lib" )
#include <emmintrin.h>
#include "wavelet.h"

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }
#endif

#define TRACK_N    8                                    //    トラックバーの数
TCHAR* track_name[] = { (TCHAR*)"輝度空間",(TCHAR*)"輝度時間",(TCHAR*)"輝度分散",(TCHAR*)"色差空間",(TCHAR*)"色差時間",(TCHAR*)"色差分散",(TCHAR*)"アダプタ",(TCHAR*)"保護" };//    トラックバーの名前
int track_default[] = { 1,   0,  40,  1,  0,  40,  0, 100 };//    トラックバーの初期値
int track_s[] = { 0,   0,   1,  0,  0,   1,  0, 0 };      //    トラックバーの下限値
int track_e[] = { 5,   3, 100,  5,  3, 100,  0, 100 };    //    トラックバーの上限値
#define CHECK_N 1
TCHAR* check_name[] = { (TCHAR*)"Use UpdateSurface" };
int        check_default[] = { 1 };

FILTER_DLL filter = {
    FILTER_FLAG_EX_INFORMATION | FILTER_FLAG_NO_INIT_DATA,    //    フィルタのフラグ
    0,0,                            //    設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
	(TCHAR*)"NL-Means-Light for GPU TypeC", //    フィルタの名前
    TRACK_N,                        //    トラックバーの数 (0なら名前初期値等もNULLでよい)
    track_name,                     //    トラックバーの名前郡へのポインタ
    track_default,                  //    トラックバーの初期値郡へのポインタ
    track_s,track_e,                //    トラックバーの数値の下限上限 (NULLなら全て0～256)
    CHECK_N,                        //    チェックボックスの数 (0なら名前初期値等もNULLでよい)
    check_name,                     //    チェックボックスの名前郡へのポインタ
    check_default,                  //    チェックボックスの初期値郡へのポインタ
    func_proc,                      //    フィルタ処理関数へのポインタ (NULLなら呼ばれません)
    NULL,                           //    開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_exit,                      //    終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_update,                    //    設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_WndProc,                   //    設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,NULL,                      //    システムで使いますので使用しないでください
    NULL,                           //  拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
    NULL,                           //  拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
	(TCHAR*)"NL-Means-Light for GPU TypeC Ver.111125 mod.190519",
    //  フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
    NULL,                           //    セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                           //    セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};

typedef struct
{
    IDirect3DTexture9* Tex;
    int index;
} TextureCache;

typedef struct
{
    float x, y, z, w;
} VERTEX;

typedef struct
{
    int x, y, z, w;
} int4;
typedef struct
{
    short* src;
    short* src2;
    short* dest;
    int srcpitch;
    int srcpitch2;
    int destpitch;
    int width;
    int height;
    int radius;
    int radius2;
} TexReadWriteParam;

IDirect3D9* direct3D;
IDirect3DDevice9* device;
IDirect3DTexture9* SrcSysTexture;
IDirect3DTexture9* DestTexture;
IDirect3DTexture9* DestSysTexture;
TextureCache SrcTexture[7];

IDirect3DVertexBuffer9* vertexBuffer;
IDirect3DPixelShader9* ps[2][4];
IDirect3DQuery9* query;

UINT adapter;
int currentCacheIndex;
int maxCacheCount;
int mode = 0;

BOOL initialized = FALSE;
BOOL needCacheClear;
BOOL clearAviUtlCache;
int prerenderingFrameIndex = -1;
int frameCount;

int srcWidth;
int srcHeight;
int lumaRadius;
int chromaRadius;
int halfw;
int halfh;

float lumaH2;
float chromaH2;
D3DXVECTOR4 pscf[16];
int4        psci[2];

int maxthread;
WAVELET_PARAM before, after;
BYTE* work;
int worksize;
float str;

static const __m128i i128_pw_2048 = _mm_set1_epi16(2048);
static const __m128i i128_pw_4096 = _mm_set1_epi16(4096);
static const __m128i i128_CbCr_rev1 = _mm_set_epi16(0,4096,0,0,4096,0,0,4096);
static const __m128i i128_CbCr_rev2 = _mm_set_epi16(4096, 0,0,4096,0,0,4096,0);
static const __m128i i128_CbCr_rev3 = _mm_set_epi16(0,0,4096,0,0,4096,0,0);
static const __m128i i128_pw_16384 = _mm_set1_epi16(16384);
static const __m128i i128_pw_4 = _mm_set1_epi16(4);

//---------------------------------------------------------------------
//        フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable( void )
{
    direct3D = Direct3DCreate9(D3D_SDK_VERSION);
    if (direct3D != NULL)
    {
        track_e[6] = direct3D->GetAdapterCount() - 1;
        
    }
    return &filter;
}

void ShowErrorMsg(char* message)
{
    MessageBox(NULL, message,"warning", MB_ICONWARNING);
}

void ReleaseTexture()
{
    SAFE_RELEASE(SrcSysTexture);
    for (int i=0; i<7; i++)
    {
        SAFE_RELEASE(SrcTexture[i].Tex);
        SrcTexture[i].index = -1;
    }
    SAFE_RELEASE(DestTexture);
    SAFE_RELEASE(DestSysTexture);
    currentCacheIndex = 0;
    srcWidth = 0;
    srcHeight = 0;
}

void FinalizeD3D()
{
    ReleaseTexture();
    SAFE_RELEASE(query);
    SAFE_RELEASE(vertexBuffer);
    for (int i=0; i<4; i++)
    {
        SAFE_RELEASE(ps[0][i]);
        SAFE_RELEASE(ps[1][i]);
    }
    SAFE_RELEASE(device);
    SAFE_RELEASE(direct3D);
    initialized = FALSE;
}

BOOL skipfilter = FALSE;

bool InitD3D(HWND hwnd, HMODULE hModule, UINT _adapter)
{
    while (TRUE)
    {
        if (direct3D == NULL){
            direct3D = Direct3DCreate9(D3D_SDK_VERSION);
            if (direct3D == NULL)
            {
                ShowErrorMsg((TCHAR*)"Direct3DCreate9に失敗");
                skipfilter = TRUE;
                return false;
            }
        }
        adapter = _adapter;
        D3DPRESENT_PARAMETERS presentParameters = {0};
        presentParameters.SwapEffect = D3DSWAPEFFECT_DISCARD;
        presentParameters.Windowed = TRUE;
        presentParameters.MultiSampleType = D3DMULTISAMPLE_NONE;
        
        if (FAILED(direct3D->CreateDevice(adapter, D3DDEVTYPE_HAL, hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED | D3DCREATE_FPU_PRESERVE, &presentParameters, &device)))
        {
            if (MessageBox(NULL, "CreateDeviceに失敗", "NL-Means Light for GPU", MB_RETRYCANCEL | MB_ICONWARNING) == IDCANCEL)
            {
                skipfilter = TRUE;
                return false;
            }
            FinalizeD3D();
        }
        else
        {
            break;
        }
    }

    IDirect3DVertexDeclaration9* vertexDeclaration;
    const D3DVERTEXELEMENT9 VERTEX_ELEMENTS[] = {
        {0,  0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITIONT, 0},
        D3DDECL_END()
    };
    
    
    device->CreateVertexDeclaration(VERTEX_ELEMENTS, &vertexDeclaration);
    device->SetVertexDeclaration(vertexDeclaration);
    SAFE_RELEASE(vertexDeclaration);
    for(int i=0; i<7; i++)
    {
        device->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        device->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_POINT);
    }
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->CreateQuery(D3DQUERYTYPE_EVENT, &query);
    initialized = TRUE;
    return true;
}

bool CreateSystemTexture(int width, int height, D3DFORMAT Format, LPDIRECT3DTEXTURE9* ppTexture)
{
    if (FAILED(D3DXCreateTexture(device, width, height, 1, 0, Format, D3DPOOL_SYSTEMMEM, ppTexture)))
    {
        ShowErrorMsg((TCHAR*)"D3DXCreateTextureに失敗");
        return false;
    }
    return true;
}

bool CreateTexture(int width, int height, DWORD Usage, D3DFORMAT Format, LPDIRECT3DTEXTURE9* ppTexture)
{
    if (FAILED(D3DXCreateTexture(device, width, height, 1, Usage, Format, D3DPOOL_DEFAULT, ppTexture)))
    {
        ShowErrorMsg((TCHAR*)"D3DXCreateTextureに失敗");
        return false;
    }
    return true;
}

bool InitTexture(int width, int height, int radius, int radius2)
{
    ReleaseTexture();
    SAFE_RELEASE(vertexBuffer);
    
    srcWidth = width;
    srcHeight = height;
    lumaRadius = radius;
    chromaRadius = radius2;
    int w8 = (width+7)&~7;
    halfw = w8 / 2;
    halfh = (height+1) / 2;
    
    int DestTexWidth, SrcTexWidth;
    switch(mode)
    {
    case 1:
        DestTexWidth = halfw;
        SrcTexWidth = (w8 + 16 + 16) / 2;
        break;
    case 2:
        DestTexWidth = halfw * 2;
        SrcTexWidth = w8 + 16 + 16;
        break;
    case 3:
        DestTexWidth = halfw * 3;
        SrcTexWidth = ((w8+16+16)/2)*3;
        break;
    }
    if (CreateSystemTexture(SrcTexWidth, height, D3DFMT_G16R16, &SrcSysTexture) == FALSE)
    {
        return false;
    }
    for (int i=0; i<7; i++)
    {
        if (CreateTexture(SrcTexWidth, height, D3DUSAGE_DYNAMIC, D3DFMT_G16R16, &SrcTexture[i].Tex) == FALSE)
        {
            return false;
        }
    }
    if (CreateSystemTexture(DestTexWidth, halfh, D3DFMT_A16B16G16R16, &DestSysTexture) == FALSE)
    {
        return false;
    }
    if (CreateTexture(DestTexWidth, halfh, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16, &DestTexture) == FALSE)
    {
        return false;
    }
    
    VERTEX v[12] =
    {
        {          0.0f,         0.0f, 0.0f, 1.0f},
        {  (float)halfw,         0.0f, 0.0f, 1.0f},
        {          0.0f, (float)halfh, 0.0f, 1.0f},
        {  (float)halfw, (float)halfh, 0.0f, 1.0f},
        
        {  (float)halfw,         0.0f, 0.0f, 1.0f},
        {(float)halfw*2,         0.0f, 0.0f, 1.0f},
        {  (float)halfw, (float)halfh, 0.0f, 1.0f},
        {(float)halfw*2, (float)halfh, 0.0f, 1.0f},
        
        {(float)halfw*2,         0.0f, 0.0f, 1.0f},
        {(float)halfw*3,         0.0f, 0.0f, 1.0f},
        {(float)halfw*2, (float)halfh, 0.0f, 1.0f},
        {(float)halfw*3, (float)halfh, 0.0f, 1.0f},
    };
    device->CreateVertexBuffer(sizeof(VERTEX)*12, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vertexBuffer, NULL);
    void* p;
    vertexBuffer->Lock( 0, 0, &p, 0);
    CopyMemory(p, v, sizeof(v));
    vertexBuffer->Unlock();
    
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<3; j++)
        {
            pscf[i*3+j].x = (float)(j - 1) / SrcTexWidth;
            pscf[i*3+j].y = (float)(i - 1) / height;
        }
    }

    pscf[12].x = 1.0f / SrcTexWidth;
    pscf[12].y = 1.0f / height;
    return true;
}

void ClearCache(int count)
{
    for (int i=0; i<7; i++)
    {
        SrcTexture[i].index = -1;
    }
    maxCacheCount = count;
    currentCacheIndex = 0;
}

void WriteTextureMode1(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = prm->height * thread_id / thread_num;
    int y_end = prm->height * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = prm->width - 15;
    int xloopcount2 = (16 - (prm->width & 15)) & 7;

    int srcstep = prm->srcpitch - prm->width * 6;
    int deststep = prm->destpitch - ((prm->width + 7) &~7) * 2 - 16;
    short* src = prm->src + (prm->srcpitch * y_start) / 2;
    short* dest = prm->dest + (prm->destpitch * y_start) / 2;
    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             eax, yloopcount
        mov             ebx, xloopcount
        movdqa          xmm7, i128_pw_16384
    
    yloop:
        pinsrw          xmm0, [esi], 0
        punpcklwd       xmm0, xmm0
        pshufd          xmm0, xmm0, 0
        paddw           xmm0, xmm7
        movntdq         [edi], xmm0
        add             edi, 16
        mov             ecx, ebx
    align 16
    xloop:
        pinsrw          xmm0, [esi   ], 0
        pinsrw          xmm0, [esi+6 ], 1
        pinsrw          xmm0, [esi+12], 2
        pinsrw          xmm0, [esi+18], 3
        pinsrw          xmm0, [esi+24], 4
        pinsrw          xmm0, [esi+30], 5
        pinsrw          xmm0, [esi+36], 6
        pinsrw          xmm0, [esi+42], 7
        pinsrw          xmm1, [esi+48], 0
        pinsrw          xmm1, [esi+54], 1
        pinsrw          xmm1, [esi+60], 2
        pinsrw          xmm1, [esi+66], 3
        pinsrw          xmm1, [esi+72], 4
        pinsrw          xmm1, [esi+78], 5
        pinsrw          xmm1, [esi+84], 6
        pinsrw          xmm1, [esi+90], 7
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        movntdq         [edi   ], xmm0
        movntdq         [edi+16], xmm1
        add             esi, 96
        add             edi, 32
        sub             ecx, 16
        jnle            xloop
        add             ecx, 15
        jz              xloop3_end
        
    align 16
    xloop2:
        mov             dx, [esi]
        add             dx, 16384
        mov             [edi], dx
        add             esi, 6
        add             edi, 2
        sub             ecx, 1
        jnz             xloop2
        
        mov             ecx, xloopcount2
        test            ecx, ecx
        jz              xloop3_end
        mov             dx, [esi-6]
        add             dx, 16384
    align 16
    xloop3:
        mov             [edi], dx
        add             edi, 2
        sub             ecx, 1
        jnz             xloop3
    xloop3_end:
        pinsrw          xmm0, [esi-6], 0
        punpcklwd       xmm0, xmm0
        pshufd          xmm0, xmm0, 0
        paddw           xmm0, xmm7
        movntdq         [edi], xmm0
        add             esi, srcstep
        add             edi, deststep
        sub             eax, 1
        jnz             yloop

        pop             ebx
    }
}

void WriteTextureMode2(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = prm->height * thread_id / thread_num;
    int y_end = prm->height * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = prm->width - 7;
    int xloopcount2 = (8 - (prm->width & 7)) & 7;
    int srcstep = prm->srcpitch - prm->width * 6;
    int deststep = prm->destpitch - ((prm->width + 7) & ~7) * 2 - 16;
    int span = ((prm->width + 7) &~7) * 2 + 16 + 16;
    short* src = prm->src + (prm->srcpitch * y_start) / 2;
    short* dest = prm->dest + (prm->destpitch * y_start) / 2 ;
    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             edx, span
        movdqa          xmm7, i128_pw_16384
        
    yloop:
        pinsrw          xmm0, [esi+2], 0
        pinsrw          xmm1, [esi+4], 0
        punpcklwd       xmm0, xmm0
        punpcklwd       xmm1, xmm1
        pshufd          xmm0, xmm0, 0
        pshufd          xmm1, xmm1, 0
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        movntdq         [edi    ], xmm0
        movntdq         [edi+edx], xmm1
        add             edi, 16
        mov             ecx, xloopcount
    align 16
    xloop:
        movdqa          xmm0, [esi   ]
        movdqa          xmm1, [esi+16]
        movdqa          xmm2, [esi+32]
// 00 10 20 01 11 21 02 12
// 22 03 13 23 04 14 24 05
// 15 25 06 16 26 07 17 27
        punpcklqdq      xmm3, xmm0
        psrldq          xmm0, 8
        punpcklqdq      xmm4, xmm1
        punpckhwd       xmm3, xmm1
        punpcklwd       xmm0, xmm2
        punpckhwd       xmm4, xmm2
// 00 04 10 14 20 24 01 05
// 11 15 21 25 02 06 12 26
// 22 26 03 07 13 17 23 27
        punpcklqdq      xmm1, xmm3
        psrldq          xmm3, 8
        punpcklqdq      xmm2, xmm0
        punpckhwd       xmm1, xmm0
        punpcklwd       xmm3, xmm4
        punpckhwd       xmm2, xmm4
// 00 02 04 06 10 12 14 16
// 20 22 24 26 01 03 05 07
// 11 13 15 17 21 23 25 27
        psrldq          xmm1, 8
        pslldq          xmm3, 8
        punpcklwd       xmm1, xmm2
        punpckhwd       xmm3, xmm2
// 10 11 12 13 14 15 16 17
// 20 21 22 23 24 25 26 27
        paddw           xmm1, xmm7
        paddw           xmm3, xmm7
        movntdq         [edi    ], xmm1
        movntdq         [edi+edx], xmm3
        add             esi, 48
        add             edi, 16
        sub             ecx, 8
        jnle            xloop
        
        add             ecx, 7
        jnz             xloop2
        pinsrw          xmm0, [esi-6+2], 0
        pinsrw          xmm1, [esi-6+4], 0
        punpcklwd       xmm0, xmm0
        punpcklwd       xmm1, xmm1
        pshufd          xmm0, xmm0, 0
        pshufd          xmm1, xmm1, 0
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        movntdq         [edi    ], xmm0
        movntdq         [edi+edx], xmm1
        add             esi, srcstep
        add             edi, deststep
        sub             yloopcount, 1
        jnz             yloop
        jmp             yloop_end
    align 16
    xloop2:
        mov             ax, [esi+2]
        mov             bx, [esi+4]
        add             ax, 16384
        add             bx, 16384
        mov             [edi    ], ax
        mov             [edi+edx], bx
        add             esi, 6
        add             edi, 2
        sub             ecx, 1
        jnz             xloop2
        
        mov             ecx, xloopcount2
        mov             ax, [esi-6+2]
        mov             bx, [esi-6+4]
        add             ax, 16384
        add             bx, 16384
    align 16
    xloop3:
        mov             [edi    ], ax
        mov             [edi+edx], bx
        add             edi, 2
        sub             ecx, 1
        jnz             xloop3
        
        pinsrw          xmm0, eax, 0
        pinsrw          xmm1, ebx, 0
        punpcklwd       xmm0, xmm0
        punpcklwd       xmm1, xmm1
        pshufd          xmm0, xmm0, 0
        pshufd          xmm1, xmm1, 0
        movntdq         [edi    ], xmm0
        movntdq         [edi+edx], xmm1
        add             esi, srcstep
        add             edi, deststep
        sub             yloopcount, 1
        jnz             yloop
    yloop_end:
        pop             ebx
    }
}

void WriteTextureMode3(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = prm->height * thread_id / thread_num;
    int y_end = prm->height * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = prm->width - 7;
    int span = ((prm->width + 7) &~7) * 2 + 16 + 16;
    int srcstep = prm->srcpitch - (prm->width &~7) * 6;
    int deststep = prm->destpitch - (prm->width &~7) * 2 - 16;
    short* src = prm->src + (prm->srcpitch * y_start) / 2;
    short* dest = prm->dest + (prm->destpitch * y_start) / 2;
    __m128i i128_mask, i128_mask2;
    
    WORD* p1 = (WORD*)&i128_mask;
    WORD* p2 = (WORD*)&i128_mask2;
    for (int i=0; i< prm->width % 8; i++)
    {
        *p1 = 0xFFFF;
        *p2 = 0;
        p1++;
        p2++;
    }
    for (int i= prm->width % 8; i < 8; i++)
    {
        *p1 = 0;
        *p2 = 0xFFFF;
        p1++;
        p2++;
    }

    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             eax, xloopcount
        mov             ebx, yloopcount
        mov             edx, span
        movdqa          xmm6, i128_mask2
        movdqa          xmm7, i128_pw_16384
        
    yloop:
        pinsrw          xmm0, [esi  ], 0
        pinsrw          xmm1, [esi+2], 0
        pinsrw          xmm2, [esi+4], 0
        punpcklwd       xmm0, xmm0
        punpcklwd       xmm1, xmm1
        punpcklwd       xmm2, xmm2
        pshufd          xmm0, xmm0, 0
        pshufd          xmm1, xmm1, 0
        pshufd          xmm2, xmm2, 0
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        paddw           xmm2, xmm7
        movntdq         [edi      ], xmm0
        movntdq         [edi+edx  ], xmm1
        movntdq         [edi+edx*2], xmm2
        add             edi, 16
        mov             ecx, eax
    align 16
    xloop:
        movdqa          xmm0, [esi   ]
        movdqa          xmm1, [esi+16]
        movdqa          xmm2, [esi+32]
// 00 10 20 01 11 21 02 12
// 22 03 13 23 04 14 24 05
// 15 25 06 16 26 07 17 27
        punpcklqdq      xmm3, xmm0
        psrldq          xmm0, 8
        punpcklqdq      xmm4, xmm1
        punpckhwd       xmm3, xmm1
        punpcklwd       xmm0, xmm2
        punpckhwd       xmm4, xmm2
// 00 04 10 14 20 24 01 05
// 11 15 21 25 02 06 12 26
// 22 26 03 07 13 17 23 27
        punpcklqdq      xmm1, xmm3
        psrldq          xmm3, 8
        punpcklqdq      xmm2, xmm0
        punpckhwd       xmm1, xmm0
        punpcklwd       xmm3, xmm4
        punpckhwd       xmm2, xmm4
// 00 02 04 06 10 12 14 16
// 20 22 24 26 01 03 05 07
// 11 13 15 17 21 23 25 27
        punpcklqdq      xmm0, xmm1
        psrldq          xmm1, 8
        punpcklqdq      xmm4, xmm3
        punpckhwd       xmm0, xmm3
        punpcklwd       xmm1, xmm2
        punpckhwd       xmm4, xmm2
// 00 01 02 03 04 05 06 07
// 10 11 12 13 14 15 16 17
// 20 21 22 23 24 25 26 27
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        paddw           xmm4, xmm7
        movntdq         [edi      ], xmm0
        movntdq         [edi+edx  ], xmm1
        movntdq         [edi+edx*2], xmm4
        add             esi, 48
        add             edi, 16
        sub             ecx, 8
        jnle            xloop
        
        add             ecx, 7
        jnz             last16pixel
        pinsrw          xmm0, [esi-6], 0
        pinsrw          xmm1, [esi-4], 0
        pinsrw          xmm2, [esi-2], 0
        punpcklwd       xmm0, xmm0
        punpcklwd       xmm1, xmm1
        punpcklwd       xmm2, xmm2
        pshufd          xmm0, xmm0, 0
        pshufd          xmm1, xmm1, 0
        pshufd          xmm2, xmm2, 0
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        paddw           xmm2, xmm7
        movntdq         [edi      ], xmm0
        movntdq         [edi+edx  ], xmm1
        movntdq         [edi+edx*2], xmm2
        add             esi, srcstep
        add             edi, deststep
        sub             ebx, 1
        jnz             yloop
        jmp             yloop_end
        
    last16pixel:
        lea             ecx, [ecx+ecx*2]
        movdqa          xmm0, [esi]
        movdqa          xmm1, [esi+16]
        movdqa          xmm2, [esi+32]
        movdqa          xmm5, i128_mask
        punpcklqdq      xmm3, xmm0
        psrldq          xmm0, 8
        punpcklqdq      xmm4, xmm1
        punpckhwd       xmm3, xmm1
        punpcklwd       xmm0, xmm2
        punpckhwd       xmm4, xmm2
        punpcklqdq      xmm1, xmm3
        psrldq          xmm3, 8
        punpcklqdq      xmm2, xmm0
        punpckhwd       xmm1, xmm0
        punpcklwd       xmm3, xmm4
        punpckhwd       xmm2, xmm4
        punpcklqdq      xmm0, xmm1
        psrldq          xmm1, 8
        punpcklqdq      xmm4, xmm3
        punpckhwd       xmm0, xmm3
        punpcklwd       xmm1, xmm2
        punpckhwd       xmm4, xmm2
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        paddw           xmm4, xmm7
        pand            xmm0, xmm5
        pand            xmm1, xmm5
        pand            xmm4, xmm5
        pinsrw          xmm2, [esi+ecx*2-6], 0
        pinsrw          xmm3, [esi+ecx*2-4], 0
        pinsrw          xmm5, [esi+ecx*2-2], 0
        punpcklwd       xmm2, xmm2
        punpcklwd       xmm3, xmm3
        punpcklwd       xmm5, xmm5
        pshufd          xmm2, xmm2, 0
        pshufd          xmm3, xmm3, 0
        pshufd          xmm5, xmm5, 0
        paddw           xmm2, xmm7
        paddw           xmm3, xmm7
        paddw           xmm5, xmm7
        movntdq         [edi+16      ], xmm2
        movntdq         [edi+edx+16  ], xmm3
        movntdq         [edi+edx*2+16], xmm5
        pand            xmm2, xmm6
        pand            xmm3, xmm6
        pand            xmm5, xmm6
        por             xmm0, xmm2
        por             xmm1, xmm3
        por             xmm4, xmm5
        movntdq         [edi      ], xmm0
        movntdq         [edi+edx  ], xmm1
        movntdq         [edi+edx*2], xmm4
        add             esi, srcstep
        add             edi, deststep
        sub             ebx, 1
        jnz             yloop
    yloop_end:
        pop             ebx
    }
}

void ReadTextureMode1(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = halfh * thread_id / thread_num;
    int y_end = halfh * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = halfw / 4;
    int srcstep = prm->srcpitch - xloopcount * 32;
    int srcstep2 = prm->srcpitch2 * 2 - xloopcount * 48;
    int deststep = prm->destpitch * 2 - xloopcount * 48;
    short* src = prm->src + prm->srcpitch * y_start / 2;
    short* src2 = prm->src2 + prm->srcpitch2 * y_start;
    short* dest = prm->dest + prm->destpitch  * y_start;
    int destpitch = prm->destpitch;
    int srcpitch2 = prm->srcpitch2;
    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             edx, src2
        mov             eax, destpitch
        mov             ebx, srcpitch2
        movdqa          xmm7, i128_pw_16384
    yloop:
        mov             ecx, xloopcount
        
    align 16
    xloop1:
        movdqa          xmm0, [edx   ]
        movdqa          xmm1, [edx+16]
        movdqa          xmm2, [edx+32]
        movdqa          xmm3, [edx+ebx]
        movdqa          xmm4, [edx+ebx+16]
        movdqa          xmm5, [edx+ebx+32]
        paddw           xmm0, xmm7
        paddw           xmm1, xmm7
        paddw           xmm2, xmm7
        paddw           xmm3, xmm7
        paddw           xmm4, xmm7
        paddw           xmm5, xmm7
        pinsrw          xmm0, [esi   ], 0
        pinsrw          xmm3, [esi+2 ], 0
        pinsrw          xmm0, [esi+4 ], 3
        pinsrw          xmm3, [esi+6 ], 3
        pinsrw          xmm0, [esi+8 ], 6
        pinsrw          xmm3, [esi+10] ,6
        pinsrw          xmm1, [esi+12], 1
        pinsrw          xmm4, [esi+14], 1
        pinsrw          xmm1, [esi+16], 4
        pinsrw          xmm4, [esi+18], 4
        pinsrw          xmm1, [esi+20], 7
        pinsrw          xmm4, [esi+22], 7
        pinsrw          xmm2, [esi+24], 2
        pinsrw          xmm5, [esi+26], 2
        pinsrw          xmm2, [esi+28], 5
        pinsrw          xmm5, [esi+30], 5
        psubw           xmm0, xmm7
        psubw           xmm1, xmm7
        psubw           xmm2, xmm7
        psubw           xmm3, xmm7
        psubw           xmm4, xmm7
        psubw           xmm5, xmm7
        movntdq         [edi   ], xmm0
        movntdq         [edi+16], xmm1
        movntdq         [edi+32], xmm2
        movntdq         [edi+eax   ], xmm3
        movntdq         [edi+eax+16], xmm4
        movntdq         [edi+eax+32], xmm5
        add             esi, 32
        add             edx, 48
        add             edi, 48
        sub             ecx, 1
        jnz             xloop1
        
        add             esi, srcstep
        add             edx, srcstep2
        add             edi, deststep
        sub             yloopcount, 1
        jnz             yloop
        pop             ebx
    }
}

void ReadTextureMode2(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = halfh * thread_id / thread_num;
    int y_end = halfh * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = halfw / 4;
    int srcstep = prm->srcpitch - xloopcount * 32;
    int srcstep2 = prm->srcpitch2 - xloopcount * 48;
    int deststep = prm->destpitch - xloopcount * 48;
    short* src = prm->src + prm->srcpitch * y_start / 2;
    short* src2 = prm->src2 + prm->srcpitch2 * y_start;
    short* dest = prm->dest + prm->destpitch  * y_start;
    int span = halfw * 8;

    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             edx, src2
        mov             eax, span
        mov             ebx, yloopcount
        movdqa          xmm7, i128_pw_16384
    yloop:
        mov             ecx, xloopcount
    align 16
    xloop1:
        pinsrw          xmm0, [esi   ], 1
        pinsrw          xmm0, [esi+4 ], 4
        pinsrw          xmm0, [esi+8 ], 7
        pinsrw          xmm1, [esi+12], 2
        pinsrw          xmm1, [esi+16], 5
        pinsrw          xmm2, [esi+20], 0
        pinsrw          xmm2, [esi+24], 3
        pinsrw          xmm2, [esi+28], 6
        pinsrw          xmm0, [esi+eax   ], 2
        pinsrw          xmm0, [esi+eax+4 ], 5
        pinsrw          xmm1, [esi+eax+8 ], 0
        pinsrw          xmm1, [esi+eax+12], 3
        pinsrw          xmm1, [esi+eax+16], 6
        pinsrw          xmm2, [esi+eax+20], 1
        pinsrw          xmm2, [esi+eax+24], 4
        pinsrw          xmm2, [esi+eax+28], 7
        psubw           xmm0, xmm7
        psubw           xmm1, xmm7
        psubw           xmm2, xmm7
        pinsrw          xmm0, [edx   ], 0
        pinsrw          xmm0, [edx+6 ], 3
        pinsrw          xmm0, [edx+12], 6
        pinsrw          xmm1, [edx+18], 1
        pinsrw          xmm1, [edx+24], 4
        pinsrw          xmm1, [edx+30], 7
        pinsrw          xmm2, [edx+36], 2
        pinsrw          xmm2, [edx+42], 5
        movntdq         [edi   ], xmm0
        movntdq         [edi+16], xmm1
        movntdq         [edi+32], xmm2
        add             esi, 32
        add             edx, 48
        add             edi, 48
        sub             ecx, 1
        jnz             xloop1
        
        mov             esi, src
        add             edx, srcstep2
        add             edi, deststep
        mov             ecx, xloopcount
    align 16
    xloop2:
        pinsrw          xmm0, [esi+2 ], 1
        pinsrw          xmm0, [esi+6 ], 4
        pinsrw          xmm0, [esi+10], 7
        pinsrw          xmm1, [esi+14], 2
        pinsrw          xmm1, [esi+18], 5
        pinsrw          xmm2, [esi+22], 0
        pinsrw          xmm2, [esi+26], 3
        pinsrw          xmm2, [esi+30], 6
        pinsrw          xmm0, [esi+eax+2 ], 2
        pinsrw          xmm0, [esi+eax+6 ], 5
        pinsrw          xmm1, [esi+eax+10], 0
        pinsrw          xmm1, [esi+eax+14], 3
        pinsrw          xmm1, [esi+eax+18], 6
        pinsrw          xmm2, [esi+eax+22], 1
        pinsrw          xmm2, [esi+eax+26], 4
        pinsrw          xmm2, [esi+eax+30], 7
        psubw           xmm0, xmm7
        psubw           xmm1, xmm7
        psubw           xmm2, xmm7
        pinsrw          xmm0, [edx   ], 0
        pinsrw          xmm0, [edx+6 ], 3
        pinsrw          xmm0, [edx+12], 6
        pinsrw          xmm1, [edx+18], 1
        pinsrw          xmm1, [edx+24], 4
        pinsrw          xmm1, [edx+30], 7
        pinsrw          xmm2, [edx+36], 2
        pinsrw          xmm2, [edx+42], 5
        movntdq         [edi   ], xmm0
        movntdq         [edi+16], xmm1
        movntdq         [edi+32], xmm2
        add             esi, 32
        add             edx, 48
        add             edi, 48
        sub             ecx, 1
        jnz             xloop2
        
        add             esi, srcstep
        add             edx, srcstep2
        add             edi, deststep
        mov             src, esi
        sub             ebx, 1
        jnz             yloop
        pop             ebx
    }
}

void ReadTextureMode3(int thread_id,int thread_num,void *param1,void *param2)
{
    TexReadWriteParam* prm = (TexReadWriteParam*)param1;
    int y_start = halfh * thread_id / thread_num;
    int y_end = halfh * (thread_id + 1) / thread_num;
    int yloopcount = y_end - y_start;
    int xloopcount = halfw / 4;
    int srcstep = prm->srcpitch - xloopcount * 32;
    int deststep = prm->destpitch * 2 - xloopcount * 48;
    short* src = prm->src + prm->srcpitch * y_start / 2;
    short* dest = prm->dest + prm->destpitch  * y_start;
    int span = halfw * 8;
    int span2 = prm->destpitch;
    __asm
    {
        push            ebx
        mov             esi, src
        mov             edi, dest
        mov             eax, span
        mov             edx, span2
        mov             ebx, yloopcount
        movdqa          xmm7, i128_pw_16384
    yloop:
        mov             ecx, xloopcount
    align 16
    xloop1:
        movdqa          xmm0, [esi         ]
        movdqa          xmm1, [esi+16      ]
        movdqa          xmm2, [esi+eax     ]
        movdqa          xmm3, [esi+eax+16  ]
        movdqa          xmm4, [esi+eax*2   ]
        movdqa          xmm5, [esi+eax*2+16]
// 00 08 01 09 02 0A 03 0B
// 04 0C 05 0D 06 0E 07 0F
// 10 18 11 19 12 1A 13 1B
// 14 1C 15 1D 16 1E 17 1F
// 20 28 21 29 22 2A 23 2B
// 24 2C 25 2D 26 2E 27 2F
        pshufd          xmm0, xmm0, 0x8D //2, 0, 3, 1
        pshufd          xmm1, xmm1, 0x8D
        pshufd          xmm2, xmm2, 0x8D
        pshufd          xmm3, xmm3, 0x8D
        pshufd          xmm4, xmm4, 0x8D
        pshufd          xmm5, xmm5, 0x8D
// 01 09 03 0B 00 08 02 0A
// 05 0D 07 0F 04 0C 06 0E
// 11 19 13 1B 10 18 12 1A
// 15 1D 17 1F 14 1C 16 1E
// 21 29 23 2B 20 28 22 2A
// 25 2D 27 2F 24 2C 26 2E
        punpcklqdq      xmm6, xmm0
        punpckhwd       xmm0, xmm2
        punpcklwd       xmm2, xmm4
        punpckhwd       xmm4, xmm6
        punpcklqdq      xmm6, xmm1
        punpckhwd       xmm1, xmm3
        punpcklwd       xmm3, xmm5
        punpckhwd       xmm5, xmm6
// 00 10 08 18 02 12 0A 1A
// 11 21 19 29 13 23 1B 2B
// 20 01 28 09 22 03 2A 0B
// 04 14 0C 1C 06 16 0E 1E
// 15 25 1D 2D 17 27 1F 2F
// 24 05 2C 0D 26 07 2E 0F
        pshufd          xmm6, xmm0, 0x0E
        punpckldq       xmm0, xmm4
        punpckhdq       xmm4, xmm2
        punpckldq       xmm2, xmm6
        pshufd          xmm6, xmm1, 0x0E
        punpckldq       xmm1, xmm5
        punpckhdq       xmm5, xmm3
        punpckldq       xmm3, xmm6
// 00 10 20 01 08 18 28 09
// 22 03 13 23 2A 0B 1B 2B
// 11 21 02 12 19 29 0A 1A
// 04 14 24 05 0C 1C 2C 0D
// 26 07 17 27 2E 0F 1F 2F
// 15 25 06 16 1D 2D 0E 1E
        movdqa          xmm6, xmm0
        punpcklqdq      xmm0, xmm2
        punpckhqdq      xmm6, xmm2
        movdqa          xmm2, xmm4
        punpcklqdq      xmm4, xmm1
        punpckhqdq      xmm2, xmm1
        movdqa          xmm1, xmm3
        punpcklqdq      xmm3, xmm5
        punpckhqdq      xmm1, xmm5
// 00 10 20 01 11 21 02 12
// 08 18 28 09 19 29 0A 1A
// 22 03 13 23 04 14 24 05
// 2A 0B 1B 2B 0C 1C 2C 0D
// 15 25 06 16 26 07 17 27
// 1D 2D 0E 1E 2E 0F 1F 2F
        psubw           xmm0, xmm7
        psubw           xmm6, xmm7
        psubw           xmm4, xmm7
        psubw           xmm2, xmm7
        psubw           xmm3, xmm7
        psubw           xmm1, xmm7
        movntdq         [edi       ], xmm0
        movntdq         [edi+16    ], xmm4
        movntdq         [edi+32    ], xmm3
        movntdq         [edi+edx   ], xmm6
        movntdq         [edi+edx+16], xmm2
        movntdq         [edi+edx+32], xmm1
        add             esi, 32
        add             edi, 48
        sub             ecx, 1
        jnz             xloop1
        
        add             esi, srcstep
        add             edi, deststep
        sub             ebx, 1
        jnz             yloop
        pop             ebx
    }
}

IDirect3DTexture9* GetSrcTexture(FILTER* fp, FILTER_PROC_INFO* fpip, int frameIndex)
{
    frameIndex = max(0, min(fpip->frame_n - 1, frameIndex));
    for (int i=0; i<maxCacheCount; i++)
    {
        if (SrcTexture[i].index == frameIndex)
        {
            return SrcTexture[i].Tex;
        }
    }
    IDirect3DTexture9* tex;
    if (fp->check[0] == 0)
    {
        tex = SrcTexture[currentCacheIndex].Tex;
    }
    else
    {
        tex = SrcSysTexture;
    }
    D3DLOCKED_RECT lockedRect = {0};
    if (FAILED(tex->LockRect(0, &lockedRect, NULL, D3DLOCK_DISCARD | D3DLOCK_NOSYSLOCK)))
    {
        ShowErrorMsg((TCHAR*)"LockRectに失敗");
        return FALSE;
    }
    TexReadWriteParam param;
    param.src = (short*)(fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, frameIndex, NULL, NULL));
    param.dest = (short*)lockedRect.pBits;
    param.srcpitch = ((srcWidth+7)&~7) * 6;
    param.destpitch = lockedRect.Pitch;
    param.width = srcWidth;
    param.height = srcHeight;
    param.radius = lumaRadius;
    param.radius2 = chromaRadius;
    
    switch(mode)
    {
    case 1:
        fp->exfunc->exec_multi_thread_func(WriteTextureMode1, (void*)&param, NULL);
        break;
    case 2:
        fp->exfunc->exec_multi_thread_func(WriteTextureMode2, (void*)&param, NULL);
        break;
    case 3:
        fp->exfunc->exec_multi_thread_func(WriteTextureMode3, (void*)&param, NULL);
        break;
    }

    if (FAILED(tex->UnlockRect(0)))
    {
        ShowErrorMsg((TCHAR*)"UnlockRectに失敗");
        return FALSE;
    }
    if (fp->check[0] != 0)
    {
        IDirect3DSurface9* src;
        IDirect3DSurface9* dest;
        SrcSysTexture->GetSurfaceLevel(0, &src);
        SrcTexture[currentCacheIndex].Tex->GetSurfaceLevel(0, &dest);
        device->UpdateSurface(src, NULL, dest, NULL);
        SAFE_RELEASE(src);
        SAFE_RELEASE(dest);
        tex = SrcTexture[currentCacheIndex].Tex;
    }
    SrcTexture[currentCacheIndex].index = frameIndex;
    currentCacheIndex++;
    if (currentCacheIndex >= maxCacheCount) currentCacheIndex = 0;
    
    return tex;
}

IDirect3DPixelShader9* GetPixelShader(int spaceRadius, int timeRadius, HMODULE hModule)
{
    if (ps[spaceRadius & 1][timeRadius] != NULL)
    {
        return ps[spaceRadius & 1][timeRadius];
    }
    IDirect3DPixelShader9* pixelShader;
    char resourcename[10];
    if (spaceRadius & 1)
        sprintf_s(resourcename, sizeof(resourcename), "ODD_T%d", timeRadius);
    else
        sprintf_s(resourcename, sizeof(resourcename), "EVEN_T%d", timeRadius);
    HRSRC hRsrc;
    HANDLE hRes;
    void* lpPS;
    hRsrc = FindResource(hModule, resourcename, "PIXELSHADER");
    if (hRsrc != NULL)
    {
        hRes = LoadResource(hModule, hRsrc);
        if (hRes != NULL)
        {
            lpPS = LockResource(hRes);
            if (lpPS != NULL)
            {
                if (device->CreatePixelShader((DWORD*)lpPS, &pixelShader) == D3D_OK)
                {
                    ps[spaceRadius & 1][timeRadius] = pixelShader;
                    return pixelShader;
                }
            }
        }
    }
    ShowErrorMsg((TCHAR*)"ピクセルシェーダの作成に失敗");
    return NULL;
}

bool Render(FILTER *fp,FILTER_PROC_INFO *fpip, int frameIndex)
{
    int count = max(fp->track[1], fp->track[4]) * 2 + 1;
    if (count != maxCacheCount)
    {
        ClearCache(count);
    }
    if (srcWidth != fpip->w || srcHeight != fpip->h || lumaRadius != fp->track[0] || chromaRadius != fp->track[3])
    {
        if (InitTexture(fpip->w, fpip->h, fp->track[0], fp->track[3]) == FALSE)
        {
            ShowErrorMsg((TCHAR*)"initTextureに失敗");
            FinalizeD3D();
            return FALSE;
        }
    }
    

    IDirect3DSurface9* Surface;
    device->BeginScene();
    DestTexture->GetSurfaceLevel(0, &Surface);
    device->SetRenderTarget(0, Surface);
    device->SetStreamSource(0, vertexBuffer, 0, sizeof(VERTEX));
    int timeRadius;
    int range, range2;
    switch(mode)
    {
    case 1:
        timeRadius = fp->track[1];
        range = lumaRadius*2+1;
        range2 = lumaRadius;
        device->SetPixelShader(GetPixelShader(lumaRadius, timeRadius, fp->dll_hinst));
        pscf[13].x = (float)(lumaRadius/2);
        pscf[13].y = (float)lumaRadius;
        pscf[14].x = (float)lumaH2;
        pscf[15].x = 4;
        psci[0].x = range;
        psci[1].x = range2;

        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->SetPixelShaderConstantI(0, (int*)psci, 2);

        for (int i=-timeRadius; i<=timeRadius; i++)
        {
            device->SetTexture(timeRadius+i, GetSrcTexture(fp, fpip, frameIndex+i));
        }
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        break;
    case 2:
        timeRadius = fp->track[4];
        range = chromaRadius*2+1;
        range2 = chromaRadius;
        device->SetPixelShader(GetPixelShader(chromaRadius, timeRadius, fp->dll_hinst));
        pscf[13].x = (float)(chromaRadius/2);
        pscf[13].y = (float)chromaRadius;
        pscf[14].x = (float)chromaH2;
        pscf[15].x = 4;
        psci[0].x = range;
        psci[1].x = range2;
        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->SetPixelShaderConstantI(0, (int*)psci, 2);

        for (int i=-timeRadius; i<=timeRadius; i++)
        {
            device->SetTexture(timeRadius+i, GetSrcTexture(fp, fpip, frameIndex+i));
        }
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        pscf[15].x = 12;
        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 4, 2);
        break;
    case 3:
        timeRadius = fp->track[1];
        range = lumaRadius*2+1;
        range2 = lumaRadius;
        device->SetPixelShader(GetPixelShader(lumaRadius, timeRadius, fp->dll_hinst));
        pscf[13].x = (float)(lumaRadius/2);
        pscf[13].y = (float)lumaRadius;
        pscf[14].x = (float)lumaH2;
        pscf[15].x = 4;
        psci[0].x = range;
        psci[1].x = range2;

        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->SetPixelShaderConstantI(0, (int*)psci, 2);

        for (int i=-timeRadius; i<=timeRadius; i++)
        {
            device->SetTexture(timeRadius+i, GetSrcTexture(fp, fpip, frameIndex+i));
        }
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        timeRadius = fp->track[4];
        range = chromaRadius*2+1;
        range2 = chromaRadius;
        device->SetPixelShader(GetPixelShader(chromaRadius, timeRadius, fp->dll_hinst));
        pscf[13].x = (float)(chromaRadius/2);
        pscf[13].y = (float)chromaRadius;
        pscf[14].x = (float)chromaH2;
        pscf[15].x = 12;
        psci[0].x = range;
        psci[1].x = range2;
        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->SetPixelShaderConstantI(0, (int*)psci, 2);

        for (int i=-timeRadius; i<=timeRadius; i++)
        {
            device->SetTexture(timeRadius+i, GetSrcTexture(fp, fpip, frameIndex+i));
        }
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 4, 2);
        pscf[15].x = 20;
        device->SetPixelShaderConstantF(0, (float*)pscf, 16);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 8, 2);
        
        break;
    }
    device->EndScene();
    SAFE_RELEASE(Surface);
    return true;
}

void copyframe(FILTER *fp, FILTER_PROC_INFO *fpip)
{
    BYTE* dest = (BYTE*)fpip->ycp_edit;
    BYTE* src = (BYTE*)(fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, fpip->frame, NULL, NULL));
    int w = fpip->w;
    int h = fpip->h;
    int destbpl = fpip->max_w * 6;
    int srcbpl = ((w+7)&~7) * 6;
    for (int y=0; y<h; y++)
    {
        memcpy(dest, src, w * 6);
        dest += destbpl;
        src += srcbpl;
    }
}

void releaseWaveletWork()
{
    if(work != NULL) _mm_free(work);
    work = NULL;
    if(before.dest != NULL) _mm_free(before.dest);
    if(after.dest != NULL) _mm_free(after.dest);
    before.dest = NULL;
    after.dest = NULL;
}

void check_threadcount( int thread_id,int thread_num,void *param1,void *param2 )
{
    if (thread_id == 0) *(int*)param1 = thread_num;
}

BOOL func_proc( FILTER *fp,FILTER_PROC_INFO *fpip )
{
    int width;
    int height;
    const int frameIndex = fpip->frame;
    
    int newmode = 0;
    if (fp->track[0] != 0 || fp->track[1] != 0) newmode |= 1;
    if (fp->track[3] != 0 || fp->track[4] != 0) newmode |= 2;
    if (newmode != mode)
    {
        mode = newmode;
        srcWidth = 0;
        srcHeight = 0;
        if (prerenderingFrameIndex != -1)
        {
            needCacheClear = TRUE;
        }
        ClearCache(max(fp->track[1], fp->track[4]) * 2 + 1);
    }
    
    fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, frameIndex, &width, &height);
    fpip->w = width;
    fpip->h = height;
    
    int timeRadius = max(fp->track[1], fp->track[4]);
    int count = timeRadius * 2 + 1;
    if (clearAviUtlCache || fpip->frame_n != frameCount)
    {
        fp->exfunc->set_ycp_filtering_cache_size(fp, (width+7)&~7, (height+1)&~1, 0, NULL);
        clearAviUtlCache = FALSE;
        frameCount = fpip->frame_n;
    }
    fp->exfunc->set_ycp_filtering_cache_size(fp, (width+7)&~7, (height+1)&~1, count + 1, NULL);
    
    if (mode == 0 || skipfilter == TRUE)
    {
        copyframe(fp, fpip);
        return TRUE;
    }
    if (initialized == FALSE)
    {
        if (InitD3D(fp->hwnd, fp->dll_hinst, fp->track[6]) == FALSE)
        {
            FinalizeD3D();
            copyframe(fp, fpip);
            return FALSE;
        }
    }
    
    int resetflag = 0;
    if (device->TestCooperativeLevel() != D3D_OK)
    {
        FinalizeD3D();
        prerenderingFrameIndex = -1;
        if (InitD3D(fp->hwnd, fp->dll_hinst, fp->track[6]) == false)
        {
            FinalizeD3D();
            copyframe(fp, fpip);
            return FALSE;
        }
        resetflag = 1;
    }

    if (needCacheClear != FALSE || (prerenderingFrameIndex != frameIndex && prerenderingFrameIndex != -1))
    {
        needCacheClear = FALSE;
        prerenderingFrameIndex = -1;
        query->Issue(D3DISSUE_END);
        while(query->GetData(NULL, 0, D3DGETDATA_FLUSH) == S_FALSE)
        {
            __asm pause;
        }
    }

    if (prerenderingFrameIndex != frameIndex)
    {
        if (!Render(fp, fpip, frameIndex))
        {
            ShowErrorMsg((TCHAR*)"Renderに失敗");
            FinalizeD3D();
            return FALSE;
        }
        needCacheClear = FALSE;
    }
    else if (frameIndex + timeRadius + 1 <= fpip->frame_n-1)
    {
        fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, frameIndex + timeRadius + 1, NULL, NULL);
    }
    
    TexReadWriteParam prm;
    if (fp->track[7] != 0 || mode != 3)
    {
        prm.src2 = (short*)(fp->exfunc->get_ycp_filtering_cache_ex(fp, fpip->editp, frameIndex, NULL, NULL));
        prm.srcpitch2 = ((srcWidth+7)&~7) * 6;
    }
    if (fp->track[7] != 0)
    {
        int threadcount = 0;
        fp->exfunc->exec_multi_thread_func(check_threadcount, &threadcount, NULL);
        if (threadcount != maxthread)
        {
            if(work != NULL) _mm_free(work);
            work = NULL;
            maxthread = threadcount;
        }
        if (before.srcwidth != srcWidth || before.srcheight != srcHeight)
        {
            releaseWaveletWork();
            before.srcwidth = srcWidth;
            before.srcheight = srcHeight;
            before.destwidth = ((srcWidth + 15) &~15) / 2;
            before.destheight = ((srcHeight + 15) &~15) / 2;
            before.src_bpl = prm.srcpitch2;
            before.dest_bpl = (before.destwidth * 12 + 63) &~63;
            if ((before.dest_bpl & 4095) == 0) before.dest_bpl += 64;
            
            after.srcwidth = srcWidth;
            after.srcheight = srcHeight;
            after.destwidth = (srcWidth + 15) &~15;
            after.destheight = (srcHeight + 15) &~15;
            after.src_bpl = fpip->max_w * 6;
            after.dest_bpl = after.destwidth * 12;
            if ((after.dest_bpl & 4095) == 0) after.dest_bpl += 64;
            
            before.dest = (BYTE*)_mm_malloc(before.dest_bpl * before.destheight, 64);
            after.dest = (BYTE*)_mm_malloc(after.dest_bpl * after.destheight, 64);
        }
        
        if (work == NULL)
        {
            int xworksize = (((((before.destwidth +15) &~15) + 1) * 48 + 63) &~63) * 2;
            worksize = 64 * 12 + xworksize;
            work = (BYTE*)_mm_malloc(worksize * maxthread, 64);
        }
        before.src = (BYTE*)prm.src2;
        before.atomic_counter = 0;
        fp->exfunc->exec_multi_thread_func(fwt53_LL, (void*)&before, NULL);
    }
    IDirect3DSurface9* src;
    IDirect3DSurface9* dest;
    DestTexture->GetSurfaceLevel(0, &src);
    DestSysTexture->GetSurfaceLevel(0, &dest);
    device->GetRenderTargetData(src, dest);
    SAFE_RELEASE(src);
    SAFE_RELEASE(dest);
    D3DLOCKED_RECT lockedRect = {0};
    if (FAILED(DestSysTexture->LockRect(0, &lockedRect, NULL, D3DLOCK_READONLY)))
    {
        FinalizeD3D();
        ShowErrorMsg((TCHAR*)"LockRectに失敗");
        return FALSE;
    }
    prm.src = (short*)lockedRect.pBits;
    prm.dest = (short*)fpip->ycp_edit;
    prm.srcpitch = lockedRect.Pitch;
    prm.destpitch = fpip->max_w * 6;
    prm.width = srcWidth;
    prm.height = srcHeight;
    
    switch(mode)
    {
    case 1:
        fp->exfunc->exec_multi_thread_func(ReadTextureMode1, (void*)&prm, NULL);
        break;
    case 2:
        fp->exfunc->exec_multi_thread_func(ReadTextureMode2, (void*)&prm, NULL);
        break;
    case 3:
        fp->exfunc->exec_multi_thread_func(ReadTextureMode3, (void*)&prm, NULL);
        break;
    }
    if (FAILED(DestSysTexture->UnlockRect(0))){
        FinalizeD3D();
        ShowErrorMsg((TCHAR*)"UnlockRectに失敗");
        return FALSE;
    }

    if (frameIndex != fpip->frame_n-1 && fp->exfunc->is_saving(fpip->editp) != 0)
    {
        if(!Render(fp, fpip, frameIndex+1))
        {
            FinalizeD3D();
            return FALSE;
        }
        prerenderingFrameIndex = frameIndex + 1;
        query->Issue(D3DISSUE_END);
        query->GetData(NULL, 0, D3DGETDATA_FLUSH);
    }
    else
    {
        prerenderingFrameIndex = -1;
    }
    if (fp->track[7] != 0)
    {
        after.src = (BYTE*)fpip->ycp_edit;
        after.atomic_counter = 0;
        fp->exfunc->exec_multi_thread_func(fwt53, (void*)&after, NULL);
        BLEND_PARAM blend;
        blend.src = before.dest;
        blend.dest = after.dest;
        blend.width = before.destwidth;
        blend.height = before.destheight;
        blend.src_bpl = before.dest_bpl;
        blend.dest_bpl = after.dest_bpl;
        blend.str = (float)fp->track[7];
        fp->exfunc->exec_multi_thread_func(blend_lo, &blend, NULL);
        
        WAVELET_PARAM iwt;
        iwt.src = after.dest;
        iwt.dest = (BYTE*)fpip->ycp_edit;
        iwt.srcwidth = after.destwidth;
        iwt.srcheight = after.destheight;
        iwt.destwidth = after.srcwidth;
        iwt.destheight = after.srcheight;
        iwt.src_bpl = after.dest_bpl;
        iwt.dest_bpl = fpip->max_w * 6;
        iwt.atomic_counter = 0;
        fp->exfunc->exec_multi_thread_func(iwt53, (void*)&iwt, NULL);
    }
    return TRUE;
}

BOOL func_exit( FILTER *fp )
{
    FinalizeD3D();
    return TRUE;
}

BOOL func_update( FILTER *fp,int status )
{
    double H = pow((double)fp->track[2] / 100, 2) * 40;
    lumaH2 = (float)(1.0 / (H * H));
    H = pow((double)fp->track[5] / 100, 2) * 40;
    chromaH2 = (float)(1.0 / (H * H));
    D3DADAPTER_IDENTIFIER9 ai;
    if (direct3D == NULL)
    {
        if (InitD3D(fp->hwnd, fp->dll_hinst, fp->track[6]) == false)
        {
            return FALSE;
        }
    }
    direct3D->GetAdapterIdentifier(fp->track[6], 0, &ai);
    char buf[512];
    sprintf_s(buf, sizeof(buf), "NL-Means Light[%s]", ai.Description);
    SetWindowText(fp->hwnd, buf);
    if (adapter != fp->track[6])
    {
        prerenderingFrameIndex = -1;
        FinalizeD3D();
        if (InitD3D(fp->hwnd, fp->dll_hinst, fp->track[6]) == false)
        {
            return FALSE;
        }
    }
    return TRUE;
}

BOOL func_WndProc(HWND hwnd,UINT message,WPARAM wparam,LPARAM lparam,void *editp,FILTER *fp)
{
    switch(message)
    {
    case WM_FILTER_CHANGE_ACTIVE:
        if (fp->exfunc->is_filter_active(fp) == FALSE)
        {
            FinalizeD3D();
            prerenderingFrameIndex = -1;
            srcWidth = 0;
            srcHeight = 0;
        }
        else
        {
            skipfilter = FALSE;
        }
        break;
    case WM_FILTER_UPDATE:
        if (prerenderingFrameIndex != -1)
        {
            needCacheClear = TRUE;
        }
        ClearCache(max(fp->track[1], fp->track[4]) * 2 + 1);
        clearAviUtlCache = TRUE;
        break;
    }
    return FALSE;
}
