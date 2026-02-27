// GPU probe volume shading from SH field (order 2).

Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gAlbedo;
Texture2D<float4> gEmissive;
Texture2D<float>  gDepth;
Texture2D<float2> gLinearZ;

StructuredBuffer<float4> gFieldSH; // packed as [cell*9 + coeff], xyz used
RWTexture2D<float4> gOutput;

float3 gBoundsMin;
float3 gBoundsMax;
uint3 gFieldRes;
uint2 gFrameDim;
float gExposure;
float gAOStrength;
int gNormalTransform; // 0 identity, 1 swap_yz, 2 swap_xz, 3 swap_xy, 4 flip_x, 5 flip_y, 6 flip_z
int gCosineMode;      // 0 irradiance(A_l on), 1 radiance(A_l off)

static const float PI = 3.14159265359f;
static const float INV_PI = 0.31830988618f;

float3 apply_normal_transform(float3 n)
{
    if (gNormalTransform == 1) return float3(n.x, n.z, n.y); // swap_yz
    if (gNormalTransform == 2) return float3(n.z, n.y, n.x); // swap_xz
    if (gNormalTransform == 3) return float3(n.y, n.x, n.z); // swap_xy
    if (gNormalTransform == 4) return float3(-n.x, n.y, n.z); // flip_x
    if (gNormalTransform == 5) return float3(n.x, -n.y, n.z); // flip_y
    if (gNormalTransform == 6) return float3(n.x, n.y, -n.z); // flip_z
    return n;
}

float3 eval_irradiance(float3 n, float3 sh[9])
{
    float x = n.x;
    float y = n.y;
    float z = n.z;

    float Y0 = 0.282095f;
    float Y1 = 0.488603f * y;
    float Y2 = 0.488603f * z;
    float Y3 = 0.488603f * x;
    float Y4 = 1.092548f * x * y;
    float Y5 = 1.092548f * y * z;
    float Y6 = 0.315392f * (3.0f * z * z - 1.0f);
    float Y7 = 1.092548f * x * z;
    float Y8 = 0.546274f * (x * x - y * y);

    float A0 = (gCosineMode == 0) ? PI : 1.0f;
    float A1 = (gCosineMode == 0) ? (2.0f * PI / 3.0f) : 1.0f;
    float A2 = (gCosineMode == 0) ? (PI / 4.0f) : 1.0f;

    float b0 = Y0 * A0;
    float b1 = Y1 * A1;
    float b2 = Y2 * A1;
    float b3 = Y3 * A1;
    float b4 = Y4 * A2;
    float b5 = Y5 * A2;
    float b6 = Y6 * A2;
    float b7 = Y7 * A2;
    float b8 = Y8 * A2;

    float3 c = 0.0f;
    c += sh[0] * b0;
    c += sh[1] * b1;
    c += sh[2] * b2;
    c += sh[3] * b3;
    c += sh[4] * b4;
    c += sh[5] * b5;
    c += sh[6] * b6;
    c += sh[7] * b7;
    c += sh[8] * b8;
    return c;
}

float3 load_coeff(uint cellIndex, uint coeffIndex)
{
    return gFieldSH[cellIndex * 9 + coeffIndex].xyz;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gFrameDim.x || tid.y >= gFrameDim.y) return;

    float depth = gDepth[tid.xy];
    if (depth <= 0.0f || depth >= 1.0f)
    {
        gOutput[tid.xy] = float4(0, 0, 0, 1);
        return;
    }

    float3 pos = gPosW[tid.xy].xyz;
    float3 n = gNormW[tid.xy].xyz;
    float nlen = length(n);
    if (nlen < 1e-4f)
    {
        gOutput[tid.xy] = float4(0, 0, 0, 1);
        return;
    }
    n /= nlen;
    n = apply_normal_transform(n);
    float nlen2 = length(n);
    if (nlen2 < 1e-6f)
    {
        gOutput[tid.xy] = float4(0, 0, 0, 1);
        return;
    }
    n /= nlen2;

    float3 albedo = gAlbedo[tid.xy].xyz;
    float3 emissive = gEmissive[tid.xy].xyz;
    float2 linZ = gLinearZ[tid.xy];

    float3 t = (pos - gBoundsMin) / max(gBoundsMax - gBoundsMin, float3(1e-6f));
    t = clamp(t, 0.0f, 0.999999f);

    float3 c = t * (float3(gFieldRes) - 1.0f);
    uint3 base = (uint3)floor(c);
    base = min(base, gFieldRes - 2);
    float3 f = c - (float3)base;

    uint strideY = gFieldRes.z;
    uint strideX = gFieldRes.y * gFieldRes.z;
    uint baseIdx = (base.x * strideX + base.y * strideY + base.z);

    float wx0 = 1.0f - f.x;
    float wy0 = 1.0f - f.y;
    float wz0 = 1.0f - f.z;
    float wx1 = f.x;
    float wy1 = f.y;
    float wz1 = f.z;

    float w000 = wx0 * wy0 * wz0;
    float w100 = wx1 * wy0 * wz0;
    float w010 = wx0 * wy1 * wz0;
    float w110 = wx1 * wy1 * wz0;
    float w001 = wx0 * wy0 * wz1;
    float w101 = wx1 * wy0 * wz1;
    float w011 = wx0 * wy1 * wz1;
    float w111 = wx1 * wy1 * wz1;

    float3 sh[9];
    [unroll] for (uint i = 0; i < 9; ++i) sh[i] = float3(0,0,0);

    uint idx000 = baseIdx;
    uint idx100 = baseIdx + strideX;
    uint idx010 = baseIdx + strideY;
    uint idx110 = baseIdx + strideX + strideY;
    uint idx001 = baseIdx + 1;
    uint idx101 = baseIdx + strideX + 1;
    uint idx011 = baseIdx + strideY + 1;
    uint idx111 = baseIdx + strideX + strideY + 1;

    [unroll] for (uint i = 0; i < 9; ++i)
    {
        sh[i] += load_coeff(idx000, i) * w000;
        sh[i] += load_coeff(idx100, i) * w100;
        sh[i] += load_coeff(idx010, i) * w010;
        sh[i] += load_coeff(idx110, i) * w110;
        sh[i] += load_coeff(idx001, i) * w001;
        sh[i] += load_coeff(idx101, i) * w101;
        sh[i] += load_coeff(idx011, i) * w011;
        sh[i] += load_coeff(idx111, i) * w111;
    }

    float3 irradiance = eval_irradiance(n, sh);
    float ao = 1.0f;
    if (gAOStrength > 0.0f)
    {
        // Simple depth-based AO proxy (relative depth slope).
        float z = max(linZ.x, 1e-4f);
        float slope = abs(linZ.y) * 4.0f;
        ao = saturate(1.0f - gAOStrength * slope / z);
    }
    float3 color = (((albedo * INV_PI) * irradiance) * ao + emissive) * gExposure;
    gOutput[tid.xy] = float4(color, 1.0f);
}
