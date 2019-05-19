//fxc /Tps_3_0 /Eprocess /DTIME_RADIUS=0 /Foeven_t0.pso nlmeans_even.hlsl
sampler2D s[TIME_RADIUS*2+1];
float2 delta[12];
float2 inverseSize;
float2 spaceRadius;
float H2;
float offset;
int range;
int range2;

float4 getWeight(float4 v1, float4 v2, float4 v3, float4 v4, float4 t1, float4 t2, float4 t3, float4 t4)
{
	float4 d1, d2, d3, sum;
	d1 = t1 - v1;
	d2 = t2 - v2;
	d3 = t3 - v3;
	d1 *= d1;
	d2 *= d2;
	d3 *= d3;
	d1 += d3;
	sum.xy = (d1.xy + d1.zw) * 0.07 + (d1.yz + d2.xy + d2.zw) * 0.12 + d2.yz * 0.20;
	d1 = t4 - v4;
	d1 *= d1;
	d1 += d2;
	sum.zw = (d1.xy + d1.zw) * 0.07 + (d1.yz + d3.xy + d3.zw) * 0.12 + d3.yz * 0.20;
	return exp(-4096 * 4096 * sum * H2);
}

float4 process(float2 pos : VPOS) : COLOR0
{
	pos = float2(pos.x + offset + 0.25, pos.y * 2 + 0.25) * inverseSize;
	const float4 v1 = float4(tex2D(s[TIME_RADIUS], pos+delta[0]).y, tex2D(s[TIME_RADIUS], pos+delta[1]).xy, tex2D(s[TIME_RADIUS], pos+delta[2]).x);
	const float4 v2 = float4(tex2D(s[TIME_RADIUS], pos+delta[3]).y, tex2D(s[TIME_RADIUS], pos+delta[4]).xy, tex2D(s[TIME_RADIUS], pos+delta[5]).x);
	const float4 v3 = float4(tex2D(s[TIME_RADIUS], pos+delta[6]).y, tex2D(s[TIME_RADIUS], pos+delta[7]).xy, tex2D(s[TIME_RADIUS], pos+delta[8]).x);
	const float4 v4 = float4(tex2D(s[TIME_RADIUS], pos+delta[9]).y, tex2D(s[TIME_RADIUS], pos+delta[10]).xy, tex2D(s[TIME_RADIUS], pos+delta[11]).x);
	const float2 pos2 = pos - spaceRadius * inverseSize;
	float4 sum = 0;
	float4 value = 0;
	
	[unroll]for (int dt=0; dt<TIME_RADIUS*2+1; dt++)
	{
		pos.y = pos2.y;
		[loop]for (int i=0; i<range; i++)
		{	
			pos.x = pos2.x;
			float4 t1, t2, t3, t4, w;
			t1.yzw = float3(tex2D(s[dt], pos.xy+delta[0]).y, tex2D(s[dt], pos.xy+delta[1]).xy);
			t2.yzw = float3(tex2D(s[dt], pos.xy+delta[3]).y, tex2D(s[dt], pos.xy+delta[4]).xy);
			t3.yzw = float3(tex2D(s[dt], pos.xy+delta[6]).y, tex2D(s[dt], pos.xy+delta[7]).xy);
			t4.yzw = float3(tex2D(s[dt], pos.xy+delta[9]).y, tex2D(s[dt], pos.xy+delta[10]).xy);
			[loop]for(int j=0; j<range2; j++)
			{
				float4 t5 = float4(tex2D(s[dt], pos.xy+delta[2]).xy, tex2D(s[dt], pos.xy+delta[5]).xy);
				float4 t6 = float4(tex2D(s[dt], pos.xy+delta[8]).xy, tex2D(s[dt], pos.xy+delta[11]).xy);
				t1 = float4(t1.yzw, t5.x);
				t2 = float4(t2.yzw, t5.z);
				t3 = float4(t3.yzw, t6.x);
				t4 = float4(t4.yzw, t6.z);
				w = getWeight(v1, v2, v3, v4, t1, t2, t3, t4);
				sum += w;
				value += float4(t2.yz, t3.yz) * w;
				t1 = float4(t1.yzw, t5.y);
				t2 = float4(t2.yzw, t5.w);
				t3 = float4(t3.yzw, t6.y);
				t4 = float4(t4.yzw, t6.w);
				w = getWeight(v1, v2, v3, v4, t1, t2, t3, t4);
				sum += w;
				value += float4(t2.yz, t3.yz) * w;
				pos.x += inverseSize.x;
			}
			t1 = float4(t1.yzw, tex2D(s[dt], pos.xy+delta[2]).x);
			t2 = float4(t2.yzw, tex2D(s[dt], pos.xy+delta[5]).x);
			t3 = float4(t3.yzw, tex2D(s[dt], pos.xy+delta[8]).x);
			t4 = float4(t4.yzw, tex2D(s[dt], pos.xy+delta[11]).x);
			w = getWeight(v1, v2, v3, v4, t1, t2, t3, t4);
			sum += w;
			value += float4(t2.yz, t3.yz) * w;
			pos.y += inverseSize.y;
		}
	}
	return (value / sum).xzyw;
}
