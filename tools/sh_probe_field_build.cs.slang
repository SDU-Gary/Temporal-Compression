StructuredBuffer<float> gProbeSH;
StructuredBuffer<uint> gKnnIdx;
StructuredBuffer<float> gKnnW;
RWStructuredBuffer<float4> gFieldSH;

uint3 gFieldRes;
uint gNumProbes;
uint gK;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint cell = tid.x;
    uint numCells = gFieldRes.x * gFieldRes.y * gFieldRes.z;
    if (cell >= numCells) return;

    float coeff[27];
    [unroll] for (uint c = 0; c < 27; ++c) coeff[c] = 0.0f;

    uint base = cell * gK;
    for (uint j = 0; j < gK; ++j)
    {
        uint p = gKnnIdx[base + j];
        if (p >= gNumProbes) continue;
        float w = gKnnW[base + j];
        uint pBase = p * 27;
        [unroll] for (uint c = 0; c < 27; ++c)
        {
            coeff[c] += w * gProbeSH[pBase + c];
        }
    }

    uint outBase = cell * 9;
    [unroll] for (uint i = 0; i < 9; ++i)
    {
        gFieldSH[outBase + i] = float4(coeff[i], coeff[9 + i], coeff[18 + i], 0.0f);
    }
}
