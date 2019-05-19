#これなに  
http://nilposoft.info/aviutl-plugin/  
のNL-Means_Light_GPUをソースから簡単にAviUtl用のバイナリを生成出来るようにしたりAviSynthへの移植を目的としたRepositoryです。  
本当にビルドできるようにするだけだったりAviSynthでAviUtl同等に動かすことを目的としているので最適化などは期待しないでください。  

#How to Build  
  
1. VisualStudioとDirectX SDKを入れる  
2. これを落とす  
3. 以下の呪文をVS2019のコマンドプロンプトで唱える  
>SET BUILDSRC=ソースフォルダ  
>cd BUILDSRC  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=0 /Fo %BUILDSRC%/Resource/odd_t0.pso %BUILDSRC%/nlmeans_odd.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=1 /Fo %BUILDSRC%/Resource/odd_t1.pso %BUILDSRC%/nlmeans_odd_t.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=2 /Fo %BUILDSRC%/Resource/odd_t2.pso %BUILDSRC%/nlmeans_odd_t.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=3 /Fo %BUILDSRC%/Resource/odd_t3.pso %BUILDSRC%/nlmeans_odd_t.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=0 /Fo %BUILDSRC%/Resource/even_t0.pso %BUILDSRC%/nlmeans_even.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=1 /Fo %BUILDSRC%/Resource/even_t1.pso %BUILDSRC%/nlmeans_even_t.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=2 /Fo %BUILDSRC%/Resource/even_t2.pso %BUILDSRC%/nlmeans_even_t.hlsl  
>fxc /T ps_3_0 /E process /D TIME_RADIUS=3 /Fo %BUILDSRC%/Resource/even_t3.pso %BUILDSRC%/nlmeans_even_t.hlsl  
4. VS2019起動してソリューションをビルド  
5. ｳﾏｰ  