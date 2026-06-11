// Live split-screen GT-vs-Pred SH probe volume comparison.

Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gAlbedo;
Texture2D<float>  gDepth;
Texture2D<float2> gLinearZ;
Texture2D<float4> gBaseColor;

StructuredBuffer<float4> gFieldSHGT;   // packed as [cell*9 + coeff], xyz used
StructuredBuffer<float4> gFieldSHPred; // packed as [cell*9 + coeff], xyz used
RWTexture2D<float4> gOutput;

float3 gBoundsMin;
float3 gBoundsMax;
uint3 gFieldRes;
uint2 gOutputDim;
uint2 gSourceDim;
uint2 gBaseDim;
float gExposure;
float gBaseScale;
float gAOStrength;
int gMode;             // 0 = SH/env only, 1 = shared base + SH/env
int gNormalTransform;  // 0 identity, 1 swap_yz, 2 swap_xz, 3 swap_xy, 4 flip_x, 5 flip_y, 6 flip_z
int gCosineMode;       // 0 irradiance(A_l on), 1 radiance(A_l off)
int gToneMap;          // 0 linear, 1 Reinhard + display gamma

static const float PI = 3.14159265359f;
static const float INV_PI = 0.31830988618f;

float3 apply_normal_transform(float3 n)
{
    if (gNormalTransform == 1) return float3(n.x, n.z, n.y);
    if (gNormalTransform == 2) return float3(n.z, n.y, n.x);
    if (gNormalTransform == 3) return float3(n.y, n.x, n.z);
    if (gNormalTransform == 4) return float3(-n.x, n.y, n.z);
    if (gNormalTransform == 5) return float3(n.x, -n.y, n.z);
    if (gNormalTransform == 6) return float3(n.x, n.y, -n.z);
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

    float3 c = 0.0f;
    c += sh[0] * (Y0 * A0);
    c += sh[1] * (Y1 * A1);
    c += sh[2] * (Y2 * A1);
    c += sh[3] * (Y3 * A1);
    c += sh[4] * (Y4 * A2);
    c += sh[5] * (Y5 * A2);
    c += sh[6] * (Y6 * A2);
    c += sh[7] * (Y7 * A2);
    c += sh[8] * (Y8 * A2);
    return c;
}

float3 load_coeff(bool usePred, uint cellIndex, uint coeffIndex)
{
    uint idx = cellIndex * 9 + coeffIndex;
    if (usePred) return gFieldSHPred[idx].xyz;
    return gFieldSHGT[idx].xyz;
}

float3 sample_field_irradiance(uint2 src, bool usePred)
{
    float3 pos = gPosW[src].xyz;
    float3 n = gNormW[src].xyz;
    float nlen = length(n);
    if (nlen < 1e-4f) return float3(0, 0, 0);
    n = apply_normal_transform(n / nlen);
    float nlen2 = length(n);
    if (nlen2 < 1e-6f) return float3(0, 0, 0);
    n /= nlen2;

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

    uint idx000 = baseIdx;
    uint idx100 = baseIdx + strideX;
    uint idx010 = baseIdx + strideY;
    uint idx110 = baseIdx + strideX + strideY;
    uint idx001 = baseIdx + 1;
    uint idx101 = baseIdx + strideX + 1;
    uint idx011 = baseIdx + strideY + 1;
    uint idx111 = baseIdx + strideX + strideY + 1;

    float3 sh[9];
    [unroll] for (uint i = 0; i < 9; ++i) sh[i] = float3(0, 0, 0);
    [unroll] for (uint i = 0; i < 9; ++i)
    {
        sh[i] += load_coeff(usePred, idx000, i) * w000;
        sh[i] += load_coeff(usePred, idx100, i) * w100;
        sh[i] += load_coeff(usePred, idx010, i) * w010;
        sh[i] += load_coeff(usePred, idx110, i) * w110;
        sh[i] += load_coeff(usePred, idx001, i) * w001;
        sh[i] += load_coeff(usePred, idx101, i) * w101;
        sh[i] += load_coeff(usePred, idx011, i) * w011;
        sh[i] += load_coeff(usePred, idx111, i) * w111;
    }
    return eval_irradiance(n, sh);
}

float3 display_map(float3 color)
{
    color = max(color * gExposure, 0.0f);
    if (gToneMap != 0)
    {
        color = color / (1.0f + color);
        color = pow(color, float3(1.0f / 2.2f));
    }
    return color;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gOutputDim.x || tid.y >= gOutputDim.y) return;

    uint halfW = max(1u, gOutputDim.x / 2u);
    bool isRight = tid.x >= halfW;
    uint srcX = min(gSourceDim.x - 1u, (tid.x * gSourceDim.x) / max(1u, gOutputDim.x));
    uint srcY = min(gSourceDim.y - 1u, (tid.y * gSourceDim.y) / max(1u, gOutputDim.y));
    uint2 src = uint2(srcX, srcY);
    uint baseX = min(gBaseDim.x - 1u, (tid.x * gBaseDim.x) / max(1u, gOutputDim.x));
    uint baseY = min(gBaseDim.y - 1u, (tid.y * gBaseDim.y) / max(1u, gOutputDim.y));
    uint2 baseSrc = uint2(baseX, baseY);

    uint divider = halfW;
    if (abs((int)tid.x - (int)divider) <= 1)
    {
        gOutput[tid.xy] = float4(1.0f, 0.85f, 0.1f, 1.0f);
        return;
    }

    float depth = gDepth[src];
    if (depth <= 0.0f || depth >= 1.0f)
    {
        gOutput[tid.xy] = float4(0, 0, 0, 1);
        return;
    }

    float3 albedo = gAlbedo[src].xyz;
    float3 irradiance = sample_field_irradiance(src, isRight);

    float ao = 1.0f;
    if (gAOStrength > 0.0f)
    {
        float2 linZ = gLinearZ[src];
        float z = max(linZ.x, 1e-4f);
        float slope = abs(linZ.y) * 4.0f;
        ao = saturate(1.0f - gAOStrength * slope / z);
    }

    float3 envColor = ((albedo * INV_PI) * irradiance) * ao;
    float3 color = envColor;
    if (gMode == 1)
    {
        color += gBaseColor[baseSrc].xyz * gBaseScale;
    }
    gOutput[tid.xy] = float4(display_map(color), 1.0f);
}
