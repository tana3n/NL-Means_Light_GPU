SET BUILDSRC=E:\src\NL-Means_Light_GPU_bak
cd /d %BUILDSRC%  
mkdir resource
fxc /T ps_3_0 /E process /D TIME_RADIUS=0 /Fo resource/odd_t0.pso nlmeans_odd.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=1 /Fo resource/odd_t1.pso nlmeans_odd_t.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=2 /Fo resource/odd_t2.pso nlmeans_odd_t.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=3 /Fo resource/odd_t3.pso nlmeans_odd_t.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=0 /Fo resource/even_t0.pso nlmeans_even.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=1 /Fo resource/even_t1.pso nlmeans_even_t.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=2 /Fo resource/even_t2.pso nlmeans_even_t.hlsl  
fxc /T ps_3_0 /E process /D TIME_RADIUS=3 /Fo resource/even_t3.pso nlmeans_even_t.hlsl  