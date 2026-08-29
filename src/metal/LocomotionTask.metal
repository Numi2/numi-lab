#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/task_program_types.h"

using namespace metal;

namespace {

constant float kPi = 3.14159265358979323846f;
constant float kTwoPi = 2.0f * kPi;
constant uint kImpactOrderMask = 0xffu;
constant uint kImpactSceneShift = 8u;
constant uint kImpactSceneMask = 0xffu << kImpactSceneShift;
constant uint kImpactEnabled = 1u << 16u;
constant uint kImpactOffsetShift = 17u;
constant uint kImpactOffsetMask = 0xffu << kImpactOffsetShift;
constant uint kImpactContactLatched = 1u << 25u;
constant uint kImpactContactPublished = 1u << 26u;

// ARDY_PHYSICS_GATED_REFERENCE_V4. status.w is available in ordinary
// interaction tracking and is compile-time excluded from impact-sequence tasks.
constant uint kInteractionPhaseFractionBits = 16u;
constant uint kInteractionPhaseScale = 1u << kInteractionPhaseFractionBits;
constant uint kInteractionPhaseMask = (1u << 29u) - 1u;
constant uint kInteractionRateShift = 29u;
constant uint kInteractionRateMask = 3u << kInteractionRateShift;
constant uint kInteractionFallLatched = 1u << 31u;

inline bool interactionPhysicsGated(
    device const MRTaskProgramHeaderGPU& program
) {
    return (program.schedule.w &
            MR_TASK_PROGRAM_INTERACTION_PHYSICS_GATED) != 0u;
}

inline float packedInteractionFramePosition(
    thread const MRTaskStateGPU& state
) {
    return float(state.status.w & kInteractionPhaseMask) /
        float(kInteractionPhaseScale);
}

inline uint interactionRateCode(thread const MRTaskStateGPU& state) {
    return (state.status.w & kInteractionRateMask) >>
        kInteractionRateShift;
}

inline float interactionPlaybackRate(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state
) {
    if (!interactionPhysicsGated(program)) {
        return 1.0f;
    }
    switch (interactionRateCode(state)) {
    case 1u: return 0.25f;
    case 2u: return 0.50f;
    case 3u: return 1.00f;
    default: return 0.0f;
    }
}

inline bool interactionFallIsLatched(
    thread const MRTaskStateGPU& state
) {
    return (state.status.w & kInteractionFallLatched) != 0u;
}

inline uint packInteractionClock(
    const float framePosition,
    const uint rateCode,
    const bool fallLatched
) {
    const float maximum =
        float(kInteractionPhaseMask) / float(kInteractionPhaseScale);
    const float bounded = clamp(framePosition, 0.0f, maximum);
    const uint fixed = min(
        uint(bounded * float(kInteractionPhaseScale) + 0.5f),
        kInteractionPhaseMask
    );
    return fixed |
        ((min(rateCode, 3u) << kInteractionRateShift) &
         kInteractionRateMask) |
        (fallLatched ? kInteractionFallLatched : 0u);
}


inline uint impactOrder(thread const MRTaskStateGPU& state) {
    return state.recoveryStats.w & kImpactOrderMask;
}

inline uint impactScene(thread const MRTaskStateGPU& state) {
    return
        (state.recoveryStats.w & kImpactSceneMask) >>
        kImpactSceneShift;
}

inline bool impactSequenceEnabled(
    thread const MRTaskStateGPU& state
) {
    return (state.recoveryStats.w & kImpactEnabled) != 0u;
}

inline uint impactOffset(thread const MRTaskStateGPU& state) {
    return
        (state.recoveryStats.w & kImpactOffsetMask) >>
        kImpactOffsetShift;
}

template <typename T>
inline device const T* taskTable(
    device const uchar* arena,
    const uint byteOffset
) {
    return reinterpret_cast<device const T*>(
        arena + byteOffset
    );
}

inline ulong mix64(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value =
        (value ^ (value >> 30u)) *
        0xbf58476d1ce4e5b9ul;
    value =
        (value ^ (value >> 27u)) *
        0x94d049bb133111ebul;
    return value ^ (value >> 31u);
}

inline float randomUnit(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel
) {
    ulong key = dispatch.seed;
    key ^= ulong(environment + 1u) *
        0xd2b74407b1ce6e93ul;
    key ^= ulong(episode + 1u) *
        0xca5a826395121157ul;
    key ^= ulong(controlStep + 1u) *
        0x9e3779b185ebca87ul;
    key ^= ulong(channel + 1u) *
        0x94d049bb133111ebul;
    return
        float(uint(mix64(key) >> 40u)) *
        (1.0f / 16777216.0f);
}

inline float randomSigned(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel
) {
    return
        2.0f *
        randomUnit(
            dispatch,
            environment,
            episode,
            controlStep,
            channel
        ) -
        1.0f;
}

inline float randomRange(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint controlStep,
    const uint channel,
    const float lower,
    const float upper
) {
    return lower +
        (upper - lower) *
        randomUnit(
            dispatch,
            environment,
            episode,
            controlStep,
            channel
        );
}

inline float3 rotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 tangent =
        2.0f * cross(quaternion.xyz, value);
    return
        value +
        quaternion.w * tangent +
        cross(quaternion.xyz, tangent);
}

inline float3 rotateInverse(
    const float4 quaternion,
    const float3 value
) {
    const float3 tangent =
        2.0f * cross(quaternion.xyz, value);
    return
        value -
        quaternion.w * tangent +
        cross(quaternion.xyz, tangent);
}

constant uint kCrowNavigationWaypointCount = 5u;
// The accepted V10 learner reliably reaches the gate and first slalom but
// commonly misses the alternating second slalom. Preview only that measured
// transition; applying it at gate acquisition was screened and regressed.
constant uint kCrowNavigationTurnPreviewWaypoint = 1u;

// Accepted autonomous V10 route states immediately after waypoints one
// through four. These are reset-only curriculum initializers. Stages one and
// four are sourced from CrowReplayPack payload SHA-256
// 4b62d9d53e8bcb57e6530a7764dd7d65542f5f741076acb9fac1c3a19d37a817
// (policy-rollout fingerprint 10801763159576431185). Stage two is refreshed
// from payload SHA-256
// dc643e50622a959926a62bdd56a24950002b52042e5ca73a09a848fb0d196523
// (policy-rollout fingerprint 2110342465157374846). Stage three is refreshed
// from payload SHA-256
// 0e5fa034d4685828d662486943ff0f1fc05286497ef3bb11a92dc437b9c088c3
// (policy-rollout fingerprint 17821064067433974087). Root translation is
// rebound to the active course below; articulation, attitude, and generalized
// velocity retain a physically accepted in-flight state.
constant float kCrowNavigationStageQ[4][20] = {
    {1.13109732f, 0.198652461f, 0.560217023f, -0.0400673859f,
     0.122633241f, 0.00768928742f, 0.99161309f, 0.00947896484f,
     0.0130759897f, 0.512515604f, 0.51802206f, -0.0287190545f,
     -0.0089144418f, 0.0458946526f, -0.12404716f, 0.11143811f,
     -0.210221276f, -0.111703545f, 0.132944226f, -0.205960587f},
    {2.09003043f, -0.0558142439f, 0.909064412f, 0.0159012675f,
     0.092480734f, 0.004613603f, 0.995576799f, -0.0171826165f,
     0.0421781205f, 0.621542394f, 0.606107652f, -0.0888328776f,
     -0.0858808309f, 0.0347312614f, -0.0739645511f, 0.272400469f,
     -0.163734809f, -0.23345077f, 0.478502274f, -0.213011384f},
    {3.54018831f, 0.140966445f, 0.683682203f, -0.0361278653f,
     0.155705854f, -0.0399651602f, 0.986333311f, -0.00464380207f,
     0.163432166f, 0.514467359f, 0.524634659f, -0.0260841846f,
     -0.0236391146f, 0.0145904692f, -0.152591467f, 0.330334008f,
     -0.226577878f, -0.171876743f, 0.345600665f, -0.243104815f},
    {4.13695097f, 0.439768612f, 0.915266156f, -0.0679064244f,
     0.145325199f, 0.0255004298f, 0.986721337f, 0.0103961304f,
     0.00122527662f, 0.00248511042f, 0.004241494f, -0.00907188933f,
     0.00870299526f, 0.0876418278f, -0.162871778f, 0.163604558f,
     -0.225277692f, -0.142725363f, 0.173754305f, -0.193642572f},
};

constant float kCrowNavigationStageV[4][19] = {
    {1.48302794f, 0.487196326f, 0.787553549f, -0.129934981f,
     -0.0500320196f, -0.0056963223f, 0.154043794f, 0.380192518f,
     9.52633858f, 9.98314095f, -0.0280686654f, 0.107957162f,
     0.140687004f, -0.0227479655f, 0.202757075f, 0.0659321472f,
     -0.0927779824f, 0.4192985f, -0.00824963674f},
    {1.21694458f, -0.108226597f, 0.302143395f, 0.0508630387f,
     0.0104958089f, 0.00470782351f, -0.0000900672094f,
     -0.0612537228f, -6.44103527f, -6.04118013f, 0.197533026f,
     -0.0242968667f, 0.0292119533f, 0.136108965f, 0.0354880281f,
     -0.0369888619f, -0.15328452f, 0.316795558f, -0.0739374161f},
    {0.657629788f, 0.862302959f, -0.516957819f, -0.542696059f,
     0.179450199f, 0.0122887995f, -0.00651729153f, -0.968211532f,
     7.75862885f, 8.41296291f, -0.898698092f, -0.798534751f,
     0.361390024f, 0.0572327971f, -0.139061913f, 0.0406738743f,
     -0.205706239f, 0.0832161382f, 0.0480266176f},
    {1.95027649f, 1.09029245f, -0.335707068f, 0.000370803609f,
     0.0730518326f, -0.00673860079f, 0.0169121493f, 0.00446468825f,
     5.35440922f, 5.26737833f, 0.0688800886f, 0.0805166662f,
     0.0783760399f, 0.0105284788f, -0.0350894667f, -0.0194789935f,
     0.0115323877f, -0.0463652983f, -0.0318701155f},
};

// Last accepted policy action at the same replay state as each refreshed
// pose template. Stages two and three use the current-parent captures below;
// stage four is refreshed from the historical accepted route state cited
// above. Stage one remains zero until it receives the same exact treatment.
// Restoring these values prevents an impossible one-step jump from an
// in-flight pose to a zero previous-action history.
constant float kCrowNavigationStageAction[4][15] = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.00481179683f, -0.0507596098f, -0.076870501f,
     0.191068873f, -0.281930894f, -0.282805979f, 0.0842511579f,
     0.072263442f, -0.00834729057f, -0.00888217427f,
     -0.168518454f, 0.276037514f, -0.085493125f,
     -0.286596894f, 0.0203184132f},
    {-0.155826539f, -0.0987512097f, -0.0195673537f,
     0.165421113f, -0.157934487f, -0.126775205f, 0.0681389496f,
     -0.0417104103f, 0.0599405393f, -0.10034842f,
     -0.0877058506f, 0.0921596512f, -0.125721052f,
     -0.197784081f, 0.0106463395f},
    {-0.0283845011f, -0.0303944517f, 0.0445872657f,
     -0.00336635858f, -0.0225012358f, 0.0392310098f,
     0.19899264f, -0.0624645576f, -0.15244481f,
     -0.10137862f, -0.0330022685f, -0.139661178f,
     -0.0533713326f, -0.00131072965f, -0.0315197296f},
};

// Command, wing-cycle phase, and normalized journey phase observed by the
// next policy decision after each refreshed replay state. Navigation rewrites
// these values from accepted root/target feedback every ordinary step; a
// mid-route reset must restore that already-accepted result once rather than
// exposing the unrelated sampled reset command.
constant float4 kCrowNavigationStageCommandAndPhase[4] = {
    float4(0.0f),
    float4(
        0.203664675f, -0.129399031f, -0.40016675f,
        -2.91540722f
    ),
    float4(
        0.00912886858f, 0.197559699f, 0.449999988f,
        1.08069674f
    ),
    float4(
        0.111874416f, -0.169116884f, -0.449999988f,
        0.050255771f
    ),
};

constant float kCrowNavigationStageJourneyPhase[4] = {
    0.0f, 0.254999995f, 0.30687499f, 0.264999986f,
};

// Three non-terminal waypoint-two arrivals from the transferred parent under
// the two-slalom preview task. Each source continues autonomously to waypoint
// three on the development fixture. Payload SHA-256 values are, in row order:
// 509692bf8c7c2d280578d8229a4c45fd05d6bfc7b0eb7da23c4c5236fabf5178,
// c54f367df4c7033c37a32e45c6f4a87ab4626f24d5271ac5cca1b89dea289afa,
// and e15583bcb4cc19f0c720b21e658b966dae78dc333acdd7e78e79e5cb35f38e80.
constant uint kCrowNavigationStageTwoArrivalCount = 3u;
constant uint kCrowNavigationStageTwoStep[3] = {431u, 466u, 458u};
constant float kCrowNavigationStageTwoQ[3][20] = {
    {2.13589787f, -0.0606460311f, 0.76040709f, 0.023641225f,
     0.120915644f, 0.00516379997f, 0.992367804f, -0.0102051618f,
     0.0338598639f, 0.227975532f, 0.230021283f, -0.0718962997f,
     -0.0739652142f, 0.042625729f, -0.0811514854f, 0.274013758f,
     -0.189021945f, -0.209642157f, 0.436182529f, -0.22018753f},
    {2.26540184f, 0.0601888113f, 0.720522761f, 0.0139866425f,
     0.179435238f, 0.0267356057f, 0.983306944f, -0.00575508131f,
     0.0240807552f, -0.240805537f, -0.212647438f, -0.0340705365f,
     -0.0341854729f, 0.0682462901f, -0.0757721066f, 0.226152867f,
     -0.193765819f, -0.155752674f, 0.320953548f, -0.194535419f},
    {2.2397058f, 0.043111708f, 0.731827736f, 0.0158401951f,
     0.170515537f, 0.0204968788f, 0.985014439f, 0.00608130265f,
     0.0316092558f, 0.419135928f, 0.41700238f, -0.0286769513f,
     -0.028405549f, 0.0753641054f, -0.0818556324f, 0.215299189f,
     -0.199235618f, -0.151489764f, 0.301145196f, -0.195205882f},
};
constant float kCrowNavigationStageTwoV[3][19] = {
    {1.50162017f, -0.0863834471f, 0.614991724f, 0.0589567199f,
     0.143911555f, -0.0746961534f, 0.224859759f, -0.354686528f,
     -9.83466911f, -9.33519363f, 0.773527265f, 0.693880379f,
     0.0655890927f, -0.321026683f, 0.114985533f, -0.622757852f,
     0.620057166f, -1.61909199f, -0.215321884f},
    {2.08387923f, 0.147922069f, 0.581576884f, 0.021635009f,
     0.0996586606f, -0.0280879028f, 0.0174259394f, -0.0432286896f,
     -2.29464746f, -2.04814291f, 0.324494541f, 0.50143522f,
     0.337073177f, 0.049996186f, -0.564962327f, 0.00599250104f,
     0.391232401f, -0.984157383f, 0.258066297f},
    {1.98232937f, 0.119606823f, 0.821961582f, 0.19816944f,
     0.0134700658f, -0.0598081797f, 0.556046367f, 0.374667495f,
     10.2543917f, 9.60196209f, -0.196155772f, -0.0824618638f,
     0.00140023034f, -0.0162287503f, -0.000297540508f,
     -0.0289120097f, -0.0751506314f, 0.0966891274f, -0.131716251f},
};
constant float3 kCrowNavigationStageTwoRootOffset[3] = {
    float3(-0.310068846f, 0.214902487f, -0.133697093f),
    float3(-0.18056488f, 0.335737329f, -0.173581421f),
    float3(-0.20626092f, 0.318660226f, -0.162276447f),
};
constant float kCrowNavigationStageTwoAction[3][15] = {
    {0.0009484521f, -0.0253348965f, -0.0116323726f, 0.112097196f,
     -0.186883822f, -0.188051417f, 0.107624233f, 0.0322972797f,
     -0.0022304093f, -0.0660860837f, -0.0929075256f,
     0.153825268f, -0.102304503f, -0.228072822f, -0.0168831609f},
    {0.0482066236f, -0.00675500231f, -0.0305139311f, 0.11446733f,
     -0.0770365819f, -0.0743829906f, 0.174834698f, 0.0687775537f,
     -0.0956863388f, -0.050864663f, -0.0268785376f,
     0.0154311098f, -0.0433609746f, -0.185750961f, -0.0792517662f},
    {0.0448262282f, -0.00549940206f, -0.0385215767f, 0.107643709f,
     -0.115237661f, -0.108170472f, 0.170662865f, 0.0525369011f,
     -0.0845380574f, -0.061651364f, -0.0514066815f,
     0.0347264521f, -0.0600540452f, -0.186564624f, -0.0623542666f},
};
constant float4 kCrowNavigationStageTwoCommandAndPhase[3] = {
    float4(0.12769641f, -0.100817353f, -0.347834498f, -2.18655806f),
    float4(0.0439169668f, -0.167902425f, -0.449999988f, -0.804258142f),
    float4(0.0611217655f, -0.155982122f, -0.449999988f, 0.854502756f),
};
constant float kCrowNavigationStageTwoJourneyPhase[3] = {
    0.269374996f, 0.29124999f, 0.286249995f,
};

// Three non-terminal waypoint-three arrivals from the retained revision-14
// parent on the development fixture. Payload SHA-256 values are, in row
// order: e32a7953886ab4b7b294cf511370c0bcd30318af2762851d8893b59e929319a8,
// a90a8058960f6743dcce98fe6f97628b147e150715a243fdf9ffed5ad0e72176,
// and 7320c03409333dad26f48ec2d67fc14132efb7dfaf19dd43adeee6b536d2ff94.
// Sampling the measured arrival distribution avoids fitting one lucky turn.
constant uint kCrowNavigationStageThreeArrivalCount = 3u;
constant uint kCrowNavigationStageThreeStep[3] = {495u, 489u, 1004u};
constant float kCrowNavigationStageThreeQ[3][20] = {
    {3.50210881f, 0.143385723f, 0.860585332f, -0.0284729004f,
     0.181488231f, 0.027799502f, 0.982587636f, -0.0166104902f,
     0.0424572751f, 0.559912026f, 0.576584041f, -0.0828051046f,
     0.00938363094f, 0.0300801098f, -0.138502494f, 0.294347584f,
     -0.204594299f, -0.212436348f, 0.370280713f, -0.207263514f},
    {3.52356148f, 0.138144687f, 0.756174803f, -0.0283913612f,
     0.155977055f, -0.0315639302f, 0.986847937f, -0.00545555167f,
     0.171603352f, 0.0208268017f, -0.00293723494f, -0.00863065757f,
     -0.0119381789f, 0.00445998972f, -0.154711649f, 0.341217637f,
     -0.225468725f, -0.157754138f, 0.345327407f, -0.246268839f},
    {3.4813745f, 0.13545619f, 0.839063704f, -0.0193401705f,
     0.173014671f, -0.00223507779f, 0.984726787f, -0.0134387277f,
     0.115799397f, 0.253562301f, 0.244876131f, -0.00632541161f,
     -0.0438632518f, 0.00235262187f, -0.166690424f, 0.341274083f,
     -0.239120021f, -0.165477619f, 0.326803952f, -0.236556605f},
};
constant float kCrowNavigationStageThreeV[3][19] = {
    {1.72582436f, 0.98553586f, -0.376300991f, -0.214206755f,
     -0.248915091f, -0.283669561f, 0.153804645f, 0.166299805f,
     -5.28743744f, -5.89879656f, 0.0499887653f, 3.163872f,
     -0.218357414f, 0.0648732036f, -0.0176161155f, -0.0490236655f,
     0.057137236f, 0.0410150699f, -0.0674742386f},
    {0.771366715f, 0.84849453f, -0.620381594f, -0.055142004f,
     0.0517907254f, -0.142391726f, -0.348081261f, -1.24816215f,
     3.63679957f, 3.83595014f, 0.168574363f, 0.906417906f,
     -0.0700905845f, -0.0838954374f, 0.00783813838f,
     -0.0961545855f, 0.00531622395f, -0.178670332f, 0.0154206008f},
    {0.724687696f, 0.919267118f, -0.356934875f, -0.197543144f,
     0.0764068961f, -0.0544071048f, 0.0980717391f, -0.421545297f,
     6.16408873f, 6.72851181f, -0.362095267f, 0.43627575f,
     0.117817059f, -0.149489164f, 0.162671268f, -0.0900872499f,
     -0.158793077f, 0.0985718891f, -0.176919371f},
};
constant float3 kCrowNavigationStageThreeRootOffset[3] = {
    float3(-0.135943413f, -0.375942126f, 0.0403103829f),
    float3(-0.114490747f, -0.381183162f, -0.0641001462f),
    float3(-0.156677723f, -0.383871659f, 0.018788755f),
};
constant float kCrowNavigationStageThreeAction[3][15] = {
    {-0.184603021f, -0.130792141f, -0.109433986f, 0.213777155f,
     -0.266108692f, -0.210807323f, 0.0503534004f, -0.0249408986f,
     0.0189650748f, -0.0717677251f, -0.127333611f, 0.119747244f,
     -0.0763419867f, -0.230368823f, -0.0687240809f},
    {-0.203347117f, -0.14121455f, -0.0542733781f, 0.204870641f,
     -0.014704342f, 0.00115435116f, 0.00889537018f, -0.0554910079f,
     0.0807471052f, -0.104144491f, -0.0554216132f, 0.0776916966f,
     -0.131588712f, -0.18050769f, -0.0105103031f},
    {-0.231401399f, -0.159631938f, -0.0471213348f, 0.211564347f,
     -0.0553034581f, -0.0292664394f, 0.0265433565f, -0.0761638731f,
     0.0877227709f, -0.124349721f, -0.0773988217f, 0.0671705008f,
     -0.124200098f, -0.211354509f, 0.00779330777f},
};
constant float4 kCrowNavigationStageThreeCommandAndPhase[3] = {
    float4(0.0895515382f, 0.208614454f, 0.449999988f, -2.89027647f),
    float4(0.0341645181f, 0.210502878f, 0.449999988f, -0.0754092364f),
    float4(0.0866046548f, 0.222513929f, 0.449999988f, 0.502643609f),
};
constant float kCrowNavigationStageThreeJourneyPhase[3] = {
    0.309374988f, 0.305624992f, 0.306250006f,
};

// Root offsets from each captured waypoint target. Stages two and three are
// refreshed from current deterministic parents; the remaining historical
// templates retain their conservative target-centered reset until refreshed.
constant float3 kCrowNavigationStageRootOffset[4] = {
    float3(0.0f),
    float3(-0.35593629f, 0.21973427f, 0.01496023f),
    float3(-0.097863915f, -0.378361404f, -0.136592746f),
    float3(-0.268202782f, 0.280686274f, -0.138459015f),
};

inline float3 crowNavigationWaypointTarget(
    device const MRBodyStateGPU* sceneBodies,
    const uint courseStart,
    const uint waypoint
) {
    const float3 gateLeft = sceneBodies[courseStart].position.xyz;
    const float3 gateRight = sceneBodies[courseStart + 1u].position.xyz;
    const float3 slalomA = sceneBodies[courseStart + 2u].position.xyz;
    const float3 slalomB = sceneBodies[courseStart + 3u].position.xyz;
    const float3 perch = sceneBodies[courseStart + 4u].position.xyz;
    if (waypoint == 1u) {
        return slalomA + float3(
            0.0f,
            slalomA.y >= 0.0f ? -0.75f : 0.75f,
            0.25f
        );
    }
    if (waypoint == 2u) {
        return slalomB + float3(
            0.0f,
            slalomB.y >= 0.0f ? -0.75f : 0.75f,
            0.25f
        );
    }
    if (waypoint == 3u) {
        return perch + float3(-0.50f, 0.0f, 0.30f);
    }
    if (waypoint == 4u) {
        // The authored perch spans 1.44 m across Y. The route exits the
        // second slalom on its positive-Y side, so the landing target stays
        // on that reachable half of the physical surface instead of asking
        // the bird to cross back through the perch centerline at touchdown.
        return perch + float3(0.0f, 0.55f, 0.12f);
    }
    return 0.5f * (gateLeft + gateRight);
}

inline float4 quaternionProduct(
    const float4 a,
    const float4 b
) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

inline float4 yawQuaternion(const float4 orientation) {
    const float yaw = atan2(
        2.0f * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0f - 2.0f * (orientation.y * orientation.y + orientation.z * orientation.z)
    );
    return float4(0.0f, 0.0f, sin(0.5f * yaw), cos(0.5f * yaw));
}

inline float3 quaternionWorldAngularVelocity(
    const float4 first,
    const float4 second,
    const float framesPerSecond
) {
    const float4 firstUnit = normalize(first);
    const float4 secondUnit = normalize(second);
    float4 delta = quaternionProduct(
        secondUnit,
        float4(-firstUnit.xyz, firstUnit.w)
    );
    delta *= delta.w < 0.0f ? -1.0f : 1.0f;
    const float sineHalfAngle = length(delta.xyz);
    if (sineHalfAngle <= 1.0e-7f) {
        return 2.0f * framesPerSecond * delta.xyz;
    }
    const float angle = 2.0f * atan2(
        sineHalfAngle,
        clamp(delta.w, -1.0f, 1.0f)
    );
    return delta.xyz * (framesPerSecond * angle / sineHalfAngle);
}

inline float3 normalizedOr(
    const float3 value,
    const float3 fallback
) {
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f &&
            isfinite(lengthSquared)
        ? value * rsqrt(lengthSquared)
        : fallback;
}

inline bool bodyMember(
    const uint body,
    device const MRTaskContactGroupGPU& group,
    device const uint* members
) {
    for (uint local = 0u;
         local < group.members.y;
         ++local) {
        if (members[group.members.x + local] == body) {
            return true;
        }
    }
    return false;
}

inline float3 stableContactTangent(const float3 normal) {
    const float3 absoluteNormal = abs(normal);
    const float3 reference =
        absoluteNormal.x <= absoluteNormal.y &&
        absoluteNormal.x <= absoluteNormal.z
        ? float3(1.0f, 0.0f, 0.0f)
        : absoluteNormal.y <= absoluteNormal.z
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(0.0f, 0.0f, 1.0f);
    return normalizedOr(
        cross(reference, normal),
        float3(1.0f, 0.0f, 0.0f)
    );
}

inline float surfaceHeight(
    device const MRTaskProgramHeaderGPU& program,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device const MRBodyStateGPU* sceneBodies,
    const float2 worldPosition
) {
    if (program.terrain.x == MR_INVALID_INDEX ||
        program.terrain.y == MR_INVALID_INDEX) {
        return 0.0f;
    }
    const MRBodyStateGPU scene =
        sceneBodies[program.terrain.x];
    const MRShapeGPU shape = shapes[program.terrain.y];
    if (shape.shapeType != MR_SHAPE_HEIGHTFIELD ||
        program.terrain.z == MR_INVALID_INDEX) {
        return
            scene.position.z +
            shape.localPosition.z;
    }
    const MRGeometryHeaderGPU geometry =
        geometryHeaders[program.terrain.z];
    if (geometry.kind != MR_GEOMETRY_HEIGHTFIELD ||
        geometry.vertexCount == 0u ||
        !(geometry.localLower.w > 0.0f)) {
        return scene.position.z;
    }

    const float3 sceneLocal = rotateInverse(
        scene.orientation,
        float3(worldPosition, scene.position.z) -
            scene.position.xyz
    );
    const float3 shapeLocal = rotateInverse(
        shape.localRotation,
        sceneLocal - shape.localPosition.xyz
    );
    const float spacing = geometry.localLower.w;
    const uint width = max(
        1u,
        uint(round(
            (geometry.localUpper.x -
             geometry.localLower.x) /
                spacing
        )) + 1u
    );
    const uint height = max(
        1u,
        uint(round(
            (geometry.localUpper.y -
             geometry.localLower.y) /
                spacing
        )) + 1u
    );
    if (width * height > geometry.vertexCount) {
        return scene.position.z;
    }
    const float gridX = clamp(
        (shapeLocal.x - geometry.localLower.x) /
            spacing,
        0.0f,
        float(width - 1u)
    );
    const float gridY = clamp(
        (shapeLocal.y - geometry.localLower.y) /
            spacing,
        0.0f,
        float(height - 1u)
    );
    const uint x0 = min(uint(floor(gridX)), width - 1u);
    const uint y0 = min(uint(floor(gridY)), height - 1u);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);
    const float tx = gridX - float(x0);
    const float ty = gridY - float(y0);
    const uint base = geometry.vertexOffset;
    const float h00 =
        geometryVertices[base + y0 * width + x0].z;
    const float h10 =
        geometryVertices[base + y0 * width + x1].z;
    const float h01 =
        geometryVertices[base + y1 * width + x0].z;
    const float h11 =
        geometryVertices[base + y1 * width + x1].z;
    const float localHeight = mix(
        mix(h00, h10, tx),
        mix(h01, h11, tx),
        ty
    );
    return
        scene.position.z +
        shape.localPosition.z +
        shape.dimensions.z * localHeight;
}

inline float4 rootOrientation(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return float4(
        q[program.root.z + 3u],
        q[program.root.z + 4u],
        q[program.root.z + 5u],
        q[program.root.z + 6u]
    );
}

inline float3 rootReferenceOffset(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    return rotate(
        rootOrientation(program, q),
        program.rootReference.xyz
    );
}

inline float3 rootWorldPosition(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
        return float3(0.0f);
    }
    return float3(
        q[program.root.z + 0u],
        q[program.root.z + 1u],
        q[program.root.z + 2u]
    ) + rootReferenceOffset(program, q);
}

inline float3 rootWorldLinearVelocity(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q,
    device const float* v
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
        return float3(0.0f);
    }
    const float3 offset =
        rootReferenceOffset(program, q);
    return float3(
        v[program.root.w + 0u],
        v[program.root.w + 1u],
        v[program.root.w + 2u]
    ) + cross(
        float3(
            v[program.root.w + 3u],
            v[program.root.w + 4u],
            v[program.root.w + 5u]
        ),
        offset
    );
}

inline float3 rootWorldAngularVelocity(
    device const MRTaskProgramHeaderGPU& program,
    device const float* v
) {
    if ((program.schedule.w & MR_TASK_PROGRAM_FIXED_ROOT) != 0u) {
        return float3(0.0f);
    }
    return float3(
        v[program.root.w + 3u],
        v[program.root.w + 4u],
        v[program.root.w + 5u]
    );
}

inline float rootHeight(
    device const MRTaskProgramHeaderGPU& program,
    device const float* q
) {
    return rootWorldPosition(program, q).z;
}

inline float interactionFramePosition(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    if (program.interaction.x == 0u) {
        return 0.0f;
    }
    if (interactionPhysicsGated(program)) {
        const float position = packedInteractionFramePosition(state);
        return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
            ? fmod(position, float(program.interaction.x))
            : min(position, float(program.interaction.x - 1u));
    }
    float elapsedSeconds =
        float(state.episode.x) * controlStepSeconds;
    if (program.threat.y != 0u &&
        state.threatMetadata.x != MR_INVALID_INDEX) {
        // ARDY supplies imagined outcome timing. Align its final pose with
        // native closest approach rather than replaying from episode reset.
        elapsedSeconds = max(
            program.interactionTiming.y - state.threatGeometry.y,
            0.0f
        );
    } else if (program.threat.y != 0u) {
        elapsedSeconds = 0.0f;
    }
    const float unbounded =
        elapsedSeconds * program.interactionTiming.x;
    return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
        ? fmod(unbounded, float(program.interaction.x))
        : min(unbounded, float(program.interaction.x - 1u));
}

inline uint interactionFrame(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    return uint(floor(interactionFramePosition(
        program,
        state,
        controlStepSeconds
    )));
}

inline float interactionFrameBlend(
    device const MRTaskProgramHeaderGPU& program,
    thread const MRTaskStateGPU& state,
    const float controlStepSeconds
) {
    const float position = interactionFramePosition(
        program,
        state,
        controlStepSeconds
    );
    return position - floor(position);
}

inline uint interactionNextFrame(
    device const MRTaskProgramHeaderGPU& program,
    const uint frame
) {
    return (program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u
        ? (frame + 1u) % program.interaction.x
        : min(frame + 1u, program.interaction.x - 1u);
}

inline float4 quaternionInterpolate(
    const float4 first,
    float4 second,
    const float amount
) {
    second *= dot(first, second) < 0.0f ? -1.0f : 1.0f;
    return normalize(mix(first, second, amount));
}

inline float supportPatchFeature(
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskContactGroupGPU group,
    device const float* compactContact,
    const uint component
) {
    if (component < 6u) {
        return compactContact[group.reference.y + component];
    }
    if (component < 8u) {
        return compactContact[
            group.members.w + 4u + (component - 6u)
        ];
    }
    const float2 extent =
        group.supportPatchBounds.zw -
        group.supportPatchBounds.xy;
    const float cellArea =
        extent.x * extent.y /
        max(float(group.supportPatch.z), 1.0f);
    if (component == 8u) {
        uint occupied = 0u;
        for (uint cell = 0u;
             cell < group.supportPatch.z;
             ++cell) {
            occupied += compactContact[
                group.supportPatch.w + cell
            ] > program.dynamics.y ? 1u : 0u;
        }
        return float(occupied) * cellArea;
    }
    const uint cell = component - 9u;
    return compactContact[
        group.supportPatch.w + cell
    ] / max(cellArea, 1.0e-9f);
}

inline float cleanObservation(
    device const MRTaskProgramHeaderGPU& program,
    const MRTaskObservationOperatorGPU operation,
    device const uchar* arena,
    const float controlStepSeconds,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices
) {
    const float4 orientation = rootOrientation(program, q);
    float value = 0.0f;
    switch (operation.source.x) {
    case MR_TASK_OBSERVE_ROOT_ANGULAR_VELOCITY_LOCAL:
        value = rotateInverse(
            orientation,
            rootWorldAngularVelocity(program, v)
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_PROJECTED_GRAVITY:
        value = normalizedOr(
            rotateInverse(
                orientation,
                float3(0.0f, 0.0f, -1.0f)
            ),
            float3(0.0f, 0.0f, -1.0f)
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_COMMAND:
        value = operation.source.z < 3u
            ? state.commandAndPhase[operation.source.z]
            : state.commandExtension[operation.source.z - 3u];
        break;
    case MR_TASK_OBSERVE_JOINT_POSITION_ERROR: {
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        value =
            q[binding.indices.z] -
            defaultQ[binding.indices.z];
        break;
    }
    case MR_TASK_OBSERVE_JOINT_VELOCITY:
        value = v[
            actions[operation.source.y].indices.w
        ];
        break;
    case MR_TASK_OBSERVE_JOINT_FINITE_DIFFERENCE_VELOCITY: {
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        value = (
            q[binding.indices.z] -
            previousJointPosition[operation.source.y]
        ) / controlStepSeconds;
        break;
    }
    case MR_TASK_OBSERVE_PREVIOUS_ACTION:
        value = previousAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_PREVIOUS_POLICY_ACTION:
        value = previousPolicyAction[operation.source.y];
        break;
    case MR_TASK_OBSERVE_DELAYED_ACTION:
        value = previousAction[
            operation.source.y -
            operation.source.z * program.counts0.x
        ];
        break;
    case MR_TASK_OBSERVE_INTERACTION_JOINT_POSITION_ERROR: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const MRTaskActionBindingGPU binding =
            actions[operation.source.y];
        const float reference = mix(
            targets[frame * program.interaction.y + operation.source.y],
            targets[nextFrame * program.interaction.y + operation.source.y],
            blend
        );
        value = q[binding.indices.z] - reference;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET:
    case MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET_VELOCITY: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const float current = targets[
            frame * program.interaction.y + operation.source.y
        ];
        const float next = targets[
            nextFrame * program.interaction.y + operation.source.y
        ];
        value = operation.source.x == MR_TASK_OBSERVE_INTERACTION_JOINT_TARGET
            ? mix(current, next, blend)
            : (next - current) * program.interactionTiming.x;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_ANCHOR_ORIENTATION: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* rootTargets = taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
        device const float* jointTargets = taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
        const uint rootBase = frame * 7u;
        const uint nextRootBase = nextFrame * 7u;
        float4 referenceRoot = quaternionInterpolate(
            float4(
                rootTargets[rootBase + 3u], rootTargets[rootBase + 4u],
                rootTargets[rootBase + 5u], rootTargets[rootBase + 6u]
            ),
            float4(
                rootTargets[nextRootBase + 3u], rootTargets[nextRootBase + 4u],
                rootTargets[nextRootBase + 5u], rootTargets[nextRootBase + 6u]
            ),
            blend
        );
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_ALIGN_REFERENCE_YAW) != 0u) {
            const float4 initialReferenceRoot = float4(
                rootTargets[3u], rootTargets[4u], rootTargets[5u], rootTargets[6u]
            );
            const float4 referenceYaw = yawQuaternion(initialReferenceRoot);
            const float4 alignment = quaternionProduct(
                yawQuaternion(rootOrientation(program, defaultQ)),
                float4(-referenceYaw.xyz, referenceYaw.w)
            );
            referenceRoot = quaternionProduct(alignment, referenceRoot);
        }
        const float referenceWaist = mix(
            jointTargets[frame * program.interaction.y + operation.source.y],
            jointTargets[nextFrame * program.interaction.y + operation.source.y],
            blend
        );
        const MRTaskActionBindingGPU binding = actions[operation.source.y];
        const float liveWaist = q[binding.indices.z];
        const float4 referenceTorso = quaternionProduct(
            referenceRoot,
            float4(0.0f, 0.0f, sin(0.5f * referenceWaist),
                   cos(0.5f * referenceWaist))
        );
        const float4 liveTorso = quaternionProduct(
            orientation,
            float4(0.0f, 0.0f, sin(0.5f * liveWaist),
                   cos(0.5f * liveWaist))
        );
        // The source publishes the transpose of this relative rotation's
        // first two columns as its six-dimensional anchor representation.
        const float4 relative = quaternionProduct(
            float4(-referenceTorso.xyz, referenceTorso.w), liveTorso
        );
        const float x = relative.x;
        const float y = relative.y;
        const float z = relative.z;
        const float w = relative.w;
        const float values[6] = {
            1.0f - 2.0f * (y * y + z * z),
            2.0f * (x * y + z * w),
            2.0f * (x * y - z * w),
            1.0f - 2.0f * (x * x + z * z),
            2.0f * (x * z + y * w),
            2.0f * (y * z - x * w),
        };
        value = values[operation.source.z];
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_MODE: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        const MRTaskInteractionSampleGPU sample = samples[
            frame * program.interaction.z + operation.source.y
        ];
        if (operation.source.z == 0u) {
            value = sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_STICK ||
                    sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_ROLL ||
                    sample.metadata.x ==
                        MR_TASK_INTERACTION_CONTACT_SLIDE
                ? 1.0f
                : 0.0f;
        } else {
            value = sample.confidence.x;
        }
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_TARGET: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint sampleIndex =
            frame * program.interaction.z + operation.source.y;
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        const uint feature = operation.source.z;
        if ((samples[sampleIndex].metadata.y &
             (1u << feature)) != 0u) {
            device const float* targets = taskTable<float>(
                arena,
                program.interactionOffsets1.x
            );
            value = targets[
                sampleIndex *
                    MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT +
                feature
            ];
        }
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_CONTACT_VALIDITY: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint sampleIndex =
            frame * program.interaction.z + operation.source.y;
        device const MRTaskInteractionSampleGPU* samples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
        value = (samples[sampleIndex].metadata.y &
                 (1u << operation.source.z)) != 0u
            ? 1.0f
            : 0.0f;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_PHASE: {
        const float framePosition = interactionFramePosition(
            program,
            state,
            controlStepSeconds
        );
        const float progress = program.interaction.x > 1u
            ? framePosition / float(program.interaction.x - 1u)
            : 0.0f;
        value = operation.source.z == 0u
            ? sin(kTwoPi * progress)
            : operation.source.z == 1u
            ? cos(kTwoPi * progress)
            : progress;
        break;
    }
    case MR_TASK_OBSERVE_INTERACTION_ROOT_TRACKING_ERROR: {
        const uint frame = interactionFrame(
            program,
            state,
            controlStepSeconds
        );
        const uint nextFrame = interactionNextFrame(program, frame);
        const float blend = interactionFrameBlend(
            program,
            state,
            controlStepSeconds
        );
        device const float* targets = taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
        const uint targetBase = frame * 7u;
        const uint nextBase = nextFrame * 7u;
        const float3 framePosition = float3(
            targets[targetBase + 0u],
            targets[targetBase + 1u],
            targets[targetBase + 2u]
        );
        const float4 frameOrientation = float4(
            targets[targetBase + 3u],
            targets[targetBase + 4u],
            targets[targetBase + 5u],
            targets[targetBase + 6u]
        );
        const float3 nextPosition = float3(
            targets[nextBase + 0u],
            targets[nextBase + 1u],
            targets[nextBase + 2u]
        );
        const float4 nextOrientation = float4(
            targets[nextBase + 3u],
            targets[nextBase + 4u],
            targets[nextBase + 5u],
            targets[nextBase + 6u]
        );
        const float3 targetPosition = mix(
            framePosition,
            nextPosition,
            blend
        );
        const float4 targetOrientation = quaternionInterpolate(
            frameOrientation,
            nextOrientation,
            blend
        );
        if (operation.source.z < 3u) {
            value = rotateInverse(
                orientation,
                targetPosition - rootWorldPosition(program, q)
            )[operation.source.z];
        } else if (operation.source.z < 6u) {
            float4 delta = quaternionProduct(
                float4(-orientation.xyz, orientation.w),
                targetOrientation
            );
            delta *= delta.w < 0.0f ? -1.0f : 1.0f;
            const float sineHalfAngle = length(delta.xyz);
            const float angle = sineHalfAngle > 1.0e-7f
                ? 2.0f * atan2(
                    sineHalfAngle,
                    clamp(delta.w, -1.0f, 1.0f)
                )
                : 2.0f * sineHalfAngle;
            const float3 orientationError = sineHalfAngle > 1.0e-7f
                ? delta.xyz * (angle / sineHalfAngle)
                : 2.0f * delta.xyz;
            value = orientationError[operation.source.z - 3u];
        } else if (operation.source.z < 9u) {
            const float3 targetVelocity =
                (nextPosition - framePosition) *
                program.interactionTiming.x;
            value = rotateInverse(
                orientation,
                targetVelocity - rootWorldLinearVelocity(program, q, v)
            )[operation.source.z - 6u];
        } else {
            const float3 targetAngularVelocity =
                quaternionWorldAngularVelocity(
                    frameOrientation,
                    nextOrientation,
                    program.interactionTiming.x
                );
            const float3 currentAngularVelocity = float3(
                v[program.root.w + 3u],
                v[program.root.w + 4u],
                v[program.root.w + 5u]
            );
            value = rotateInverse(
                orientation,
                targetAngularVelocity - currentAngularVelocity
            )[operation.source.z - 9u];
        }
        break;
    }
    case MR_TASK_OBSERVE_ROOT_LINEAR_VELOCITY_LOCAL:
        value = rotateInverse(
            orientation,
            rootWorldLinearVelocity(program, q, v)
        )[operation.source.z];
        break;
    case MR_TASK_OBSERVE_ROOT_HEIGHT:
        value = rootHeight(program, q);
        break;
    case MR_TASK_OBSERVE_CONTACT_METRIC: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = compactContact[
            group.members.w + operation.source.z
        ];
        break;
    }
    case MR_TASK_OBSERVE_TERRAIN_HEIGHT: {
        if ((program.schedule.w &
             MR_TASK_PROGRAM_TERRAIN) == 0u) {
            value = 0.0f;
            break;
        }
        const float3 offset = rotate(
            orientation,
            terrainSamples[operation.source.y].xyz
        );
        const float3 rootPosition =
            rootWorldPosition(program, q);
        value =
            surfaceHeight(
                program,
                shapes,
                geometryHeaders,
                geometryVertices,
                sceneBodies,
                rootPosition.xy + offset.xy
            ) -
            rootPosition.z;
        break;
    }
    case MR_TASK_OBSERVE_BODY_PARAMETER_MEAN: {
        float total = 0.0f;
        for (uint body = 0u;
             body < program.articulation.y;
             ++body) {
            total += bodyParameters[
                program.articulation.x + body
            ][operation.source.z];
        }
        value = total /
            max(float(program.articulation.y), 1.0f);
        break;
    }
    case MR_TASK_OBSERVE_BODY_PARAMETER:
        value = bodyParameters[
            operation.source.y
        ][operation.source.z];
        break;
    case MR_TASK_OBSERVE_CONTROLLER_PARAMETER:
        value =
            controllerParameters[0][operation.source.z];
        break;
    case MR_TASK_OBSERVE_CONTACT_WRENCH_LOCAL: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = compactContact[
            group.reference.y + operation.source.z
        ];
        break;
    }
    case MR_TASK_OBSERVE_SUPPORT_PATCH: {
        const MRTaskContactGroupGPU group =
            contactGroups[operation.source.y];
        value = supportPatchFeature(
            program,
            group,
            compactContact,
            operation.source.z
        );
        break;
    }
    case MR_TASK_OBSERVE_SUPPORT_SENSE: {
        float totalLoad = 0.0f;
        float signedLoad = 0.0f;
        float maximumSlip = 0.0f;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            const MRTaskContactGroupGPU group =
                contactGroups[groupIndex];
            if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                continue;
            }
            const uint metric = group.members.w;
            const float load = max(compactContact[metric], 0.0f);
            totalLoad += load;
            signedLoad += load * cos(group.gait.x);
            maximumSlip = max(
                maximumSlip,
                max(compactContact[metric + 1u], 0.0f)
            );
        }
        switch (operation.source.z) {
        case 0u:
            value = totalLoad;
            break;
        case 1u:
            value = totalLoad > 1.0e-6f
                ? signedLoad / totalLoad
                : 0.0f;
            break;
        default:
            value = maximumSlip;
            break;
        }
        break;
    }
    case MR_TASK_OBSERVE_GAIT_PHASE: {
        const float commandMagnitude = length(
            state.commandAndPhase.xyz
        );
        if (commandMagnitude >= 0.1f) {
            value = operation.source.z == 0u
                ? sin(state.commandAndPhase.w)
                : cos(state.commandAndPhase.w);
        }
        break;
    }
    case MR_TASK_OBSERVE_CYCLIC_PHASE:
        value = operation.source.z == 0u
            ? sin(state.commandAndPhase.w)
            : cos(state.commandAndPhase.w);
        break;
    case MR_TASK_OBSERVE_CROW_GROUND_CARRIER_PHASE: {
        // Task completion has advanced episode.x before the next policy
        // decision, so this is the exact phase of the following accepted
        // ground-carrier command rather than a stale action-phase label.
        const bool crowGroundCarrier = state.episode.z == 1u &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_GROUND_CARRIER_PHASE_OBSERVATION) !=
                0u;
        const float phase = kTwoPi * float(state.episode.x) *
            controlStepSeconds / 0.50f;
        value = crowGroundCarrier
            ? (operation.source.z == 0u ? sin(phase) : cos(phase))
            : 0.0f;
        break;
    }
    case MR_TASK_OBSERVE_AVIAN_JOURNEY_PHASE:
        value = (program.schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u
            ? state.commandExtension.w
            : 0.0f;
        break;
    case MR_TASK_OBSERVE_AVIAN_JOURNEY_STAGE:
        // Preserve the v2 values for the five qualified fundamentals. Every
        // later v3 segment uses the former full-journey value and is
        // distinguished by the observable phase and velocity/yaw command.
        value = (program.schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u
            ? (state.episode.z <= 4u
                ? 0.2f * float(state.episode.z)
                : 1.0f)
            : 0.0f;
        break;
    case MR_TASK_OBSERVE_NAVIGATION_TARGET: {
        const uint waypoint = min(
            uint(max(state.navigation.z, 0.0f)),
            kCrowNavigationWaypointCount
        );
        if (operation.source.z < 6u &&
            waypoint < kCrowNavigationWaypointCount) {
            const uint requested = operation.source.z < 3u
                ? waypoint
                : min(waypoint + 1u,
                      kCrowNavigationWaypointCount - 1u);
            const float3 relative = rotateInverse(
                orientation,
                crowNavigationWaypointTarget(
                    sceneBodies,
                    operation.source.y,
                    requested
                ) - rootWorldPosition(program, q)
            );
            value = relative[operation.source.z % 3u];
        } else if (operation.source.z == 6u) {
            value = float(waypoint) /
                float(kCrowNavigationWaypointCount);
        } else if (operation.source.z == 7u) {
            value = state.navigation.w;
        } else if (operation.source.z < 13u) {
            value = operation.source.z - 8u == waypoint ? 1.0f : 0.0f;
        } else {
            value = 0.0f;
        }
        break;
    }
    case MR_TASK_OBSERVE_RECOVERY_EVENT:
        switch (operation.source.z) {
        case 0u:
            value = state.recovery.w;
            break;
        case 1u:
            value = state.recovery.x;
            break;
        case 2u:
            value = state.recovery.y;
            break;
        default:
            value = state.recovery.z;
            break;
        }
        break;
    case MR_TASK_OBSERVE_OBJECT_TRACK: {
        const MRBodyStateGPU object =
            sceneBodies[operation.source.y];
        const uint launchStep =
            (object.flagsAndIndices[3] &
             MR_BODY_STATE_LAUNCH_STEP_MASK) >>
            MR_BODY_STATE_LAUNCH_STEP_SHIFT;
        const bool visible = impactSequenceEnabled(state)
            ? impactScene(state) == operation.source.y + 1u
            : launchStep == 0u || state.episode.x >= launchStep;
        if (operation.source.z == 0u) {
            value = visible ? 1.0f : 0.0f;
            break;
        }
        if (!visible) {
            value = 0.0f;
            break;
        }
        const float3 relativePosition = rotateInverse(
            orientation,
            object.position.xyz - rootWorldPosition(program, q)
        );
        if (operation.source.z <= 3u) {
            value = relativePosition[operation.source.z - 1u];
            break;
        }
        const float3 relativeVelocity = rotateInverse(
            orientation,
            object.linearVelocityAndInverseMass.xyz -
                rootWorldLinearVelocity(program, q, v)
        );
        value = relativeVelocity[operation.source.z - 4u];
        break;
    }
    default:
        // Compilation rejects unknown sources. Device visual sources are a
        // separate direct suffix and never enter this physical-state reader.
        value = 0.0f;
        break;
    }
    return
        operation.transform.x * value +
        operation.transform.y;
}

inline void writeFrame(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const uchar* arena,
    device const MRTaskObservationOperatorGPU* actorOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
    device const float* sensorBias,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* actor,
    device float* clean
) {
    float3 normalizedGravity =
        float3(0.0f, 0.0f, -1.0f);
    bool haveNormalizedGravity = false;
    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        if (operation.source.x !=
                MR_TASK_OBSERVE_PROJECTED_GRAVITY ||
            (operation.source.w &
             MR_TASK_OBSERVATION_NORMALIZE_VECTOR3) == 0u) {
            continue;
        }
        const uint component = operation.source.z;
        float value = cleanObservation(
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
        value +=
            operation.transform.z *
            randomSigned(
                dispatch,
                environment,
                episode,
                episodeStep,
                operation.auxiliary.y
            );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            value += sensorBias[operation.auxiliary.x];
        }
        normalizedGravity[component] = value;
        haveNormalizedGravity = true;
    }
    if (haveNormalizedGravity) {
        normalizedGravity = normalizedOr(
            normalizedGravity,
            float3(0.0f, 0.0f, -1.0f)
        );
    }

    for (uint index = 0u;
         index < program.counts0.y;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[index];
        const float value = cleanObservation(
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
        clean[index] = value;
        if (operation.source.x ==
                MR_TASK_OBSERVE_PROJECTED_GRAVITY &&
            (operation.source.w &
             MR_TASK_OBSERVATION_NORMALIZE_VECTOR3) != 0u) {
            actor[index] =
                normalizedGravity[operation.source.z];
            continue;
        }
        float corrupted =
            value +
            operation.transform.z *
                randomSigned(
                    dispatch,
                    environment,
                    episode,
                    episodeStep,
                    operation.auxiliary.y
                );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            corrupted += sensorBias[operation.auxiliary.x];
        }
        actor[index] = corrupted;
    }
}

inline void writeCurrentActor(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const uchar* arena,
    device const MRTaskObservationOperatorGPU* actorOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
    device const float* sensorBias,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* output
) {
    for (uint index = 0u; index < program.counts3.w; ++index) {
        const MRTaskObservationOperatorGPU operation =
            actorOperators[program.layout.x + index];
        float value = cleanObservation(
            program,
            operation,
            arena,
            dispatch.timing.x,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
        value +=
            operation.transform.z *
            randomSigned(
                dispatch,
                environment,
                episode,
                episodeStep,
                operation.auxiliary.y
            );
        if (operation.auxiliary.x != MR_INVALID_INDEX) {
            value += sensorBias[operation.auxiliary.x];
        }
        output[index] = value;
    }
}

inline void writeCriticFrame(
    device const MRTaskProgramHeaderGPU& program,
    device const uchar* arena,
    const float controlStepSeconds,
    device const MRTaskObservationOperatorGPU* criticOperators,
    device const MRTaskActionBindingGPU* actions,
    device const MRTaskContactGroupGPU* contactGroups,
    device const float4* terrainSamples,
    device const float* q,
    device const float* v,
    device const float* defaultQ,
    thread const MRTaskStateGPU& state,
    device const float* previousAction,
    device const float* previousPolicyAction,
    device const float* previousJointPosition,
    device const float* compactContact,
    device const float4* bodyParameters,
    device const float4* controllerParameters,
    device const MRBodyStateGPU* sceneBodies,
    device const MRShapeGPU* shapes,
    device const MRGeometryHeaderGPU* geometryHeaders,
    device const float4* geometryVertices,
    device float* output
) {
    for (uint index = 0u;
         index < program.counts0.z;
         ++index) {
        const MRTaskObservationOperatorGPU operation =
            criticOperators[index];
        output[index] = cleanObservation(
            program,
            operation,
            arena,
            controlStepSeconds,
            actions,
            contactGroups,
            terrainSamples,
            q,
            v,
            defaultQ,
            state,
            previousAction,
            previousPolicyAction,
            previousJointPosition,
            compactContact,
            bodyParameters,
            controllerParameters,
            sceneBodies,
            shapes,
            geometryHeaders,
            geometryVertices
        );
    }
}

inline void publishCritic(
    device const MRTaskProgramHeaderGPU& program,
    device const float* cleanHistory,
    device const float* criticHistory,
    device float* output
) {
    uint outputIndex = 0u;
    if ((program.schedule.w &
         MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u) {
        const uint historyElements =
            program.layout.x * program.layout.y;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            output[outputIndex++] = cleanHistory[index];
        }
    }
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    for (uint index = 0u;
         index < criticHistoryElements;
         ++index) {
        output[outputIndex++] = criticHistory[index];
    }
}

inline uint durationSteps(
    device const MRTaskDispatchGPU& dispatch,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const uint channel,
    const float lowerSeconds,
    const float upperSeconds
) {
    return max(
        1u,
        uint(floor(
            randomRange(
                dispatch,
                environment,
                episode,
                episodeStep,
                channel,
                lowerSeconds,
                upperSeconds
            ) /
            dispatch.timing.x
        ))
    );
}

inline float4 scheduleSecondsForBand(
    device const MRTaskProgramHeaderGPU& program,
    const uint curriculum
) {
    float4 seconds = program.scheduleSeconds;
    if ((program.schedule.w & MR_TASK_PROGRAM_CLOCK_STRESS) == 0u ||
        program.schedule.z <= 1u) {
        return seconds;
    }
    const float progress = clamp(
        float(curriculum) /
            max(float(program.schedule.z - 1u), 1.0f),
        0.0f,
        1.0f
    );
    // Preserve the quiet adult rung, then progressively shorten both the
    // command horizon and disturbance interval. This models a finite
    // reaction budget without changing the physical impulse magnitude.
    const float commandScale = mix(1.0f, 0.45f, progress);
    const float disturbanceScale = mix(1.0f, 0.50f, progress);
    seconds.x = max(0.5f, seconds.x * commandScale);
    seconds.y = max(0.5f, seconds.y * commandScale);
    seconds.z = max(0.5f, seconds.z * disturbanceScale);
    seconds.w = max(0.5f, seconds.w * disturbanceScale);
    return seconds;
}

inline uint sampledDifficultyBand(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    const uint environment,
    const uint episode
) {
    const uint bandCount = max(program.schedule.z, 1u);
    const uint minimumBand = min(
        dispatch.sampling.x,
        bandCount - 1u
    );
    const uint requestedMaximum = dispatch.sampling.y == MR_INVALID_INDEX
        ? bandCount - 1u
        : dispatch.sampling.y;
    const uint maximumBand = max(
        minimumBand,
        min(requestedMaximum, bandCount - 1u)
    );
    const uint sampledBandCount = maximumBand - minimumBand + 1u;
    if (sampledBandCount == 1u) {
        return minimumBand;
    }
    const float exponent = max(
        dispatch.assistance.y > 0.0f
            ? dispatch.assistance.y
            : program.commandUpper.w,
        0.01f
    );
    const float sample = pow(
        randomUnit(dispatch, environment, episode, 0u, 15u),
        exponent
    );
    return minimumBand + min(
        uint(floor(sample * float(sampledBandCount))),
        sampledBandCount - 1u
    );
}

inline float3 sampledCommand(
    device const MRTaskDispatchGPU& dispatch,
    device const MRTaskProgramHeaderGPU& program,
    device const float4* curriculumRange,
    const uint environment,
    const uint episode,
    const uint episodeStep,
    const uint curriculum
) {
    const float3 lower = max(
        program.commandLower.xyz -
            float(curriculum) *
                curriculumRange[2].xyz,
        curriculumRange[0].xyz
    );
    const float3 upper = min(
        program.commandUpper.xyz +
            float(curriculum) *
                curriculumRange[2].xyz,
        curriculumRange[1].xyz
    );
    float3 command;
    for (uint component = 0u;
         component < 3u;
         ++component) {
        command[component] = randomRange(
            dispatch,
            environment,
            episode,
            episodeStep,
            16u + component,
            lower[component],
            upper[component]
        );
    }
    if (randomUnit(
            dispatch,
            environment,
            episode,
            episodeStep,
            19u
        ) < program.commandLower.w) {
        command = float3(0.0f);
    }
    return command;
}

inline bool desiredSupportContact(
    const MRTaskContactGroupGPU group,
    const float phase
) {
    float normalized =
        fmod(phase + group.gait.x, kTwoPi);
    if (normalized < 0.0f) {
        normalized += kTwoPi;
    }
    normalized /= kTwoPi;
    return normalized < group.gait.y;
}

} // namespace

kernel void mr_locomotion_task_latch_impact_contact(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    device const MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    constant MRMetalWorldPassGPU& pass [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }
    MRTaskStateGPU state = taskStates[environment];
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||
        impactOrder(state) >= program.counts3.x) {
        return;
    }
    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    if (contactStatus.code != MR_STEP_SUCCESS) {
        return;
    }
    device const MRTaskImpactEventGPU* impactEvents =
        reinterpret_cast<device const MRTaskImpactEventGPU*>(
            arena + program.offsets3.z
        );
    const uint activeImpactEvent =
        (impactOrder(state) + impactOffset(state)) %
        program.counts3.x;
    const uint projectileBody =
        impactEvents[activeImpactEvent].binding.w;
    const uint articulationBodyBegin = program.articulation.x;
    const uint articulationBodyEnd =
        articulationBodyBegin + program.articulation.y;
    const uint activeContacts = min(
        contactStatus.activeContacts,
        contactDispatch.constraintCapacity
    );
    const uint contactBase =
        environment * contactDispatch.constraintStride;
    for (uint contact = 0u; contact < activeContacts; ++contact) {
        const MRContactConstraintGPU constraint =
            contacts[contactBase + contact];
        const bool projectileA =
            constraint.bodyA == projectileBody;
        const bool projectileB =
            constraint.bodyB == projectileBody;
        if (projectileA == projectileB) {
            continue;
        }
        const uint other = projectileA
            ? constraint.bodyB
            : constraint.bodyA;
        if (other >= articulationBodyBegin &&
            other < articulationBodyEnd) {
            state.recoveryStats.w |= kImpactContactLatched;
            taskStates[environment] = state;
            return;
        }
    }
}

kernel void mr_locomotion_task_select_threat_query(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const MRBodyStateGPU* bodyStates [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.threat.y == 0u ||
        program.threat.x >= program.counts0.w ||
        contactDispatch.pointQueryStride == 0u) {
        return;
    }
    device const MRTaskContactGroupGPU* groups =
        taskTable<MRTaskContactGroupGPU>(arena, program.offsets0.w);
    device const uint* members =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* memberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(arena, program.offsets3.z);

    MRTaskStateGPU state = taskStates[environment];
    const uint queryBase =
        environment * contactDispatch.pointQueryStride;
    MRArticulatedPointImpulseGPU query = {};
    query.bodyIndex = program.root.y;
    query.localPoint = float4(0.0f);
    pointQueries[queryBase] = query;

    const uint order = impactOrder(state);
    if (!impactSequenceEnabled(state) ||
        impactScene(state) == 0u ||
        order >= program.counts3.x) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }
    const uint activeEvent =
        (order + impactOffset(state)) % program.counts3.x;
    const MRTaskImpactEventGPU event = impactEvents[activeEvent];
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU projectile =
        bodyStates[bodyBase + event.binding.w];
    const float3 velocity =
        projectile.linearVelocityAndInverseMass.xyz;
    const float horizontalSpeedSquared = dot(velocity.xy, velocity.xy);
    if (length(velocity) < program.threatTiming.x ||
        horizontalSpeedSquared < 1.0e-6f) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }

    const MRTaskContactGroupGPU protectedGroup = groups[program.threat.x];
    float bestClearance = INFINITY;
    float bestTime = 0.0f;
    float bestStrikeHeight = 0.0f;
    uint bestBody = MR_INVALID_INDEX;
    const float projectileRadius = event.projectile.x;
    for (uint local = 0u; local < protectedGroup.members.y; ++local) {
        const uint memberOffset = protectedGroup.members.x + local;
        const uint body = members[memberOffset];
        const float radius = memberRadii[memberOffset];
        if (!(radius > 0.0f)) {
            continue;
        }
        const float3 linkPosition = bodyStates[bodyBase + body].position.xyz;
        const float2 horizontalRelative =
            projectile.position.xy - linkPosition.xy;
        const float time = clamp(
            -dot(horizontalRelative, velocity.xy) /
                horizontalSpeedSquared,
            0.0f,
            program.threatTiming.y
        );
        const float3 predictedProjectile =
            projectile.position.xyz + velocity * time +
            0.5f * program.projectileGravity.xyz * time * time;
        const float safeRadius =
            radius + projectileRadius + program.threatTiming.z;
        const float clearance =
            length(predictedProjectile - linkPosition) - safeRadius;
        if (clearance < bestClearance) {
            bestClearance = clearance;
            bestTime = time;
            bestBody = body;
            bestStrikeHeight = predictedProjectile.z -
                bodyStates[bodyBase + program.root.y].position.z;
        }
    }
    if (bestBody == MR_INVALID_INDEX ||
        bestTime <= 0.0f ||
        bestTime >= program.threatTiming.y) {
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        taskStates[environment] = state;
        return;
    }

    uint threatClass = MR_TASK_THREAT_DUCK;
    if (bestStrikeHeight <= program.threatClassification.x) {
        threatClass = MR_TASK_THREAT_STEP_OVER;
    } else if (bestStrikeHeight <= program.threatClassification.y) {
        threatClass = MR_TASK_THREAT_SIDESTEP;
    } else if (bestStrikeHeight <= program.threatClassification.z) {
        threatClass = MR_TASK_THREAT_LEAN;
    }
    uint escape = state.threatMetadata.z;
    if (state.threatMetadata.w != activeEvent || escape > 2u ||
        escape == 1u) {
        const float2 rootDelta =
            bodyStates[bodyBase + program.root.y].position.xy -
            projectile.position.xy;
        const float crossTrack =
            velocity.x * rootDelta.y - velocity.y * rootDelta.x;
        const float sign = abs(crossTrack) > 1.0e-4f
            ? (crossTrack > 0.0f ? 1.0f : -1.0f)
            : ((environment & 1u) == 0u ? 1.0f : -1.0f);
        escape = sign > 0.0f ? 2u : 0u;
    }
    query.bodyIndex = bestBody;
    pointQueries[queryBase] = query;
    state.threatGeometry = float4(
        bestClearance,
        bestTime,
        bestStrikeHeight,
        bestClearance
    );
    state.threatTeacher = float4(0.0f);
    state.threatMetadata = uint4(
        bestBody,
        threatClass,
        escape,
        activeEvent
    );
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_joint_cbf_teacher(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const float* vState [[buffer(6)]],
    device float* actionStream [[buffer(7)]],
    device const float* defaultQ [[buffer(8)]],
    device const MRBodyStateGPU* bodyStates [[buffer(9)]],
    device const float* pointJacobians [[buffer(10)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(11)]],
    device MRTaskStateGPU* taskStates [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.threat.y == 0u) {
        return;
    }
    MRTaskStateGPU state = taskStates[environment];
    if (state.threatMetadata.x == MR_INVALID_INDEX ||
        state.threatMetadata.w == MR_INVALID_INDEX ||
        operatorStatuses[environment].code !=
            MR_ARTICULATED_OPERATOR_SUCCESS) {
        state.threatTeacher = float4(0.0f);
        taskStates[environment] = state;
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(arena, program.offsets0.x);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(arena, program.offsets3.z);
    device const MRTaskContactGroupGPU* groups =
        taskTable<MRTaskContactGroupGPU>(arena, program.offsets0.w);
    device const uint* members =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* memberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    const MRTaskImpactEventGPU event =
        impactEvents[state.threatMetadata.w];
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU projectile =
        bodyStates[bodyBase + event.binding.w];
    const MRBodyStateGPU threatened =
        bodyStates[bodyBase + state.threatMetadata.x];
    const float time = state.threatGeometry.y;
    const float3 predictedProjectile =
        projectile.position.xyz +
        projectile.linearVelocityAndInverseMass.xyz * time +
        0.5f * program.projectileGravity.xyz * time * time;
    float3 separation = threatened.position.xyz - predictedProjectile;
    const float escapeSign = state.threatMetadata.z == 2u ? 1.0f : -1.0f;
    if (length(separation.xy) < 1.0e-3f) {
        const float2 horizontalVelocity =
            projectile.linearVelocityAndInverseMass.xy;
        const float horizontalSpeed = length(horizontalVelocity);
        const float2 flight = horizontalSpeed > 1.0e-6f
            ? horizontalVelocity / horizontalSpeed
            : float2(1.0f, 0.0f);
        separation.xy =
            escapeSign * float2(-flight.y, flight.x) * 1.0e-3f;
    }
    const float distance = max(length(separation), 1.0e-6f);
    float linkRadius = 0.0f;
    const MRTaskContactGroupGPU group = groups[program.threat.x];
    for (uint local = 0u; local < group.members.y; ++local) {
        const uint offset = group.members.x + local;
        if (members[offset] == state.threatMetadata.x) {
            linkRadius = memberRadii[offset];
            break;
        }
    }
    const float safeRadius =
        linkRadius + event.projectile.x + program.threatTiming.z;
    const float h = distance * distance - safeRadius * safeRadius;
    const uint jacobianBase =
        environment * contactDispatch.pointQueryStride *
        3u * dispatch.counts.w;
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint actionBase =
        pass.controlStep * dispatch.strides.x +
        environment * program.counts0.x;
    float3 desiredPointVelocity = float3(0.0f);
    for (uint dof = 0u; dof < dispatch.counts.w; ++dof) {
        const float velocity = vState[vBase + dof];
        desiredPointVelocity += float3(
            pointJacobians[jacobianBase + 0u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 1u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 2u * dispatch.counts.w + dof]
        ) * velocity;
    }
    float gradientNormSquared = 0.0f;
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );
    for (uint action = 0u; action < program.counts0.x; ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        const uint dof = binding.indices.w - program.root.w;
        if (dof >= dispatch.counts.w) {
            continue;
        }
        const float reference =
            (program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u
            ? mix(
                  interactionJointTargets[
                      referenceFrame * program.interaction.y + action
                  ],
                  interactionJointTargets[
                      nextReferenceFrame * program.interaction.y + action
                  ],
                  referenceBlend
              )
            : defaultQ[binding.indices.z];
        const float target = clamp(
            reference +
                binding.parameters.x * clamp(
                    actionStream[actionBase + action],
                    -1.0f,
                    1.0f
                ),
            binding.parameters.y,
            binding.parameters.z
        );
        const float desiredVelocity =
            (target - qState[qBase + binding.indices.z]) /
            program.threatTeacher.y;
        const float currentVelocity = vState[vBase + binding.indices.w];
        const float3 column = float3(
            pointJacobians[jacobianBase + 0u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 1u * dispatch.counts.w + dof],
            pointJacobians[jacobianBase + 2u * dispatch.counts.w + dof]
        );
        desiredPointVelocity += column *
            (desiredVelocity - currentVelocity);
        const float gradient = 2.0f * dot(separation, column);
        gradientNormSquared += gradient * gradient;
    }
    const float3 projectileVelocity =
        projectile.linearVelocityAndInverseMass.xyz +
        program.projectileGravity.xyz * time;
    const float barrierRate = 2.0f * dot(
        separation,
        desiredPointVelocity - projectileVelocity
    );
    const float urgencyFraction = clamp(
        (program.threatTeacher.x - time) /
            program.threatTeacher.x,
        0.0f,
        1.0f
    );
    const float urgency =
        urgencyFraction * safeRadius * safeRadius /
        program.threatTeacher.x;
    const float constraint =
        barrierRate + program.threatTiming.w * h;
    const float deficit = max(urgency - constraint, 0.0f);
    const float projectionScale = deficit /
        (gradientNormSquared + program.threatTeacher.z);
    const float correctionRms = projectionScale * sqrt(
        gradientNormSquared /
        max(float(program.counts0.x), 1.0f)
    );
    state.threatGeometry.w = constraint;
    state.threatTeacher = float4(
        correctionRms,
        max(safeRadius - distance, 0.0f),
        constraint + projectionScale * gradientNormSquared - urgency,
        urgency
    );
    if ((program.schedule.w &
         MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u) {
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            const MRTaskActionBindingGPU binding = actions[action];
            const uint dof = binding.indices.w - program.root.w;
            if (dof >= dispatch.counts.w ||
                !(binding.parameters.x > 0.0f)) {
                continue;
            }
            const float3 column = float3(
                pointJacobians[
                    jacobianBase + 0u * dispatch.counts.w + dof
                ],
                pointJacobians[
                    jacobianBase + 1u * dispatch.counts.w + dof
                ],
                pointJacobians[
                    jacobianBase + 2u * dispatch.counts.w + dof
                ]
            );
            const float gradient = 2.0f * dot(separation, column);
            const float reference = interactionJointTargets[
                referenceFrame * program.interaction.y + action
            ];
            const float requestedTarget = clamp(
                reference + binding.parameters.x * clamp(
                    actionStream[actionBase + action],
                    -1.0f,
                    1.0f
                ),
                binding.parameters.y,
                binding.parameters.z
            );
            const float correctedTarget = clamp(
                requestedTarget +
                    program.threatTeacher.y *
                    projectionScale * gradient,
                binding.parameters.y,
                binding.parameters.z
            );
            actionStream[actionBase + action] = clamp(
                (correctedTarget - reference) /
                    binding.parameters.x,
                -1.0f,
                1.0f
            );
        }
    }
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_motion_features(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const MRBodyStateGPU* bodyStates [[buffer(5)]],
    device float* features [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.motion.y == 0u ||
        program.motion.z != 9u * program.motion.y) {
        return;
    }
    device const uint* trackedBodies =
        taskTable<uint>(arena, program.motion.w);
    const uint bodyBase =
        environment * contactDispatch.bodyStateStride;
    const MRBodyStateGPU anchor =
        bodyStates[bodyBase + program.motion.x];
    const float4 inverseAnchor = float4(
        -anchor.orientation.xyz,
        anchor.orientation.w
    );
    const uint outputBase =
        (pass.controlStep * dispatch.counts.x + environment) *
        program.motion.z;
    for (uint index = 0u; index < program.motion.y; ++index) {
        const MRBodyStateGPU body =
            bodyStates[bodyBase + trackedBodies[index]];
        const float3 position = rotateInverse(
            anchor.orientation,
            body.position.xyz - anchor.position.xyz
        );
        float4 orientation = quaternionProduct(
            inverseAnchor,
            body.orientation
        );
        orientation *= orientation.w < 0.0f ? -1.0f : 1.0f;
        const float x = orientation.x;
        const float y = orientation.y;
        const float z = orientation.z;
        const float w = orientation.w;
        const uint base = outputBase + 9u * index;
        features[base + 0u] = position.x;
        features[base + 1u] = position.y;
        features[base + 2u] = position.z;
        // First two rotation-matrix columns, row-major, matching PAC-MAN's
        // continuous 6D orientation representation.
        features[base + 3u] = 1.0f - 2.0f * (y * y + z * z);
        features[base + 4u] = 2.0f * (x * y - z * w);
        features[base + 5u] = 2.0f * (x * y + z * w);
        features[base + 6u] = 1.0f - 2.0f * (x * x + z * z);
        features[base + 7u] = 2.0f * (x * z - y * w);
        features[base + 8u] = 2.0f * (y * z + x * w);
    }
}

kernel void mr_locomotion_task_observe(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device uint* resetMasks [[buffer(6)]],
    device float* resetQ [[buffer(7)]],
    device float* resetV [[buffer(8)]],
    device MRBodyStateGPU* resetScene [[buffer(9)]],
    device float* sourceQ [[buffer(10)]],
    device float* sourceV [[buffer(11)]],
    device const MRBodyStateGPU* initialScene [[buffer(12)]],
    device MRBodyStateGPU* sourceScene [[buffer(15)]],
    device const float* defaultQ [[buffer(16)]],
    device MRTaskStateGPU* taskStates [[buffer(17)]],
    device float* actionHistory [[buffer(18)]],
    device float* actorHistory [[buffer(19)]],
    device float* cleanHistory [[buffer(20)]],
    device float* previousJointVelocity [[buffer(21)]],
    device float* sensorBias [[buffer(22)]],
    device float4* bodyParameters [[buffer(23)]],
    device float4* controllerParameters [[buffer(24)]],
    device float* actorObservations [[buffer(25)]],
    device float* criticObservations [[buffer(26)]],
    device float* compactContact [[buffer(27)]],
    device const MRShapeGPU* shapes [[buffer(28)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(29)]],
    device const float4* geometryVertices [[buffer(30)]],
    device const MRTaskEvidenceStateGPU* evidenceState
        [[buffer(5)]],
    device float* criticHistory [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.counts.x !=
            worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION ||
        program.counts0.x == 0u ||
        program.counts0.y == 0u ||
        program.layout.y == 0u ||
        program.articulation.w == 0u ||
        program.layout.w < 2u) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    device const float* initialActionPositions = taskTable<float>(
        arena,
        program.actuatorTerms.z
    );
    device const float* interactionRootTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
    device const MRTaskObservationOperatorGPU*
        actorOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.y
            );
    device const MRTaskObservationOperatorGPU*
        criticOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.z
            );
    device const MRTaskContactGroupGPU* contactGroups =
        taskTable<MRTaskContactGroupGPU>(
            arena,
            program.offsets0.w
        );
    device const uint* contactMembers =
        taskTable<uint>(arena, program.offsets1.x);
    device const MRTaskRandomizationOperatorGPU*
        randomization =
            taskTable<MRTaskRandomizationOperatorGPU>(
                arena,
                program.offsets2.y
            );
    device const MRTaskBiasSpecGPU* biasSpecs =
        taskTable<MRTaskBiasSpecGPU>(
            arena,
            program.offsets2.z
        );
    device const float4* terrainSamples =
        taskTable<float4>(arena, program.offsets2.w);
    device const float4* terrainProfiles =
        taskTable<float4>(arena, program.offsets3.x);
    device const float4* commandCurriculum =
        taskTable<float4>(arena, program.offsets3.y);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(
            arena,
            program.offsets3.z
        );
    const uint maskIndex =
        pass.controlStep * dispatch.counts.x + environment;
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase =
        environment * historyElements;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint previousVelocityBase =
        environment * program.counts0.x;
    const uint biasBase =
        environment * program.counts2.z;
    const uint bodyParameterBase =
        environment * dispatch.strides.y;
    const uint contactBase =
        environment * program.layout.z;
    const uint sceneBase =
        environment * dispatch.strides.w;
    MRTaskStateGPU state = taskStates[environment];
    const bool reset =
        state.status.x == 0u ||
        state.status.y != 0u ||
        resetMasks[maskIndex] != 0u;
    resetMasks[maskIndex] = reset ? 1u : 0u;

    if (reset) {
        const uint episode = state.episode.y + 1u;
        const uint curriculum = sampledDifficultyBand(
            dispatch,
            program,
            environment,
            episode
        );
        state.episode.z = curriculum;
        const uint terrainLevel =
            program.terrain.w == 0u
            ? 0u
            : min(curriculum, program.terrain.w - 1u);
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.z;
             ++coordinate) {
            resetQ[qBase + coordinate] =
                defaultQ[coordinate];
        }
        if (program.actuatorTerms.w == program.counts0.x) {
            for (uint action = 0u; action < program.counts0.x; ++action) {
                resetQ[qBase + actions[action].indices.z] =
                    initialActionPositions[action];
            }
        }
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) != 0u) {
            for (uint action = 0u;
                 action < program.interaction.y;
                 ++action) {
                const MRTaskActionBindingGPU binding =
                    actions[action];
                resetQ[qBase + binding.indices.z] =
                    interactionJointTargets[action];
            }
        }
        for (uint coordinate = 0u;
             coordinate < dispatch.counts.w;
             ++coordinate) {
            resetV[vBase + coordinate] = 0.0f;
        }
        for (uint localScene = 0u;
             localScene < dispatch.strides.w;
             ++localScene) {
            MRBodyStateGPU scene =
                initialScene[sceneBase + localScene];
            const uint launchStep =
                (scene.flagsAndIndices[3] &
                 MR_BODY_STATE_LAUNCH_STEP_MASK) >>
                MR_BODY_STATE_LAUNCH_STEP_SHIFT;
            if (launchStep != 0u ||
                (scene.flagsAndIndices[3] &
                 MR_BODY_STATE_PRESERVE_RESET_VELOCITY) == 0u) {
                scene.linearVelocityAndInverseMass.xyz =
                    float3(0.0f);
                scene.angularVelocity = float4(0.0f);
            }
            resetScene[sceneBase + localScene] = scene;
        }
        if (program.terrain.x != MR_INVALID_INDEX &&
            program.terrain.w != 0u) {
            resetScene[
                sceneBase + program.terrain.x
            ].position.xyz =
                terrainProfiles[terrainLevel].xyz;
        }
        for (uint body = 0u;
             body < dispatch.strides.y;
             ++body) {
            bodyParameters[
                bodyParameterBase + body
            ] = float4(1.0f);
        }
        controllerParameters[environment] =
            float4(1.0f, 1.0f, 0.0f, 1.0f);
        for (uint bias = 0u;
             bias < program.counts2.z;
             ++bias) {
            const MRTaskBiasSpecGPU spec =
                biasSpecs[bias];
            sensorBias[biasBase + bias] = randomRange(
                dispatch,
                environment,
                episode,
                0u,
                spec.metadata.x,
                spec.range.x,
                spec.range.y
            );
        }
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            previousJointVelocity[
                previousVelocityBase + action
            ] = 0.0f;
            const uint qIndex = actions[action].indices.z;
            previousJointVelocity[
                previousVelocityBase + program.counts0.x + action
            ] = qIndex == MR_INVALID_INDEX
                ? 0.0f
                : resetQ[qBase + qIndex];
            for (uint delay = 0u;
                 delay < program.layout.w;
                 ++delay) {
                actionHistory[
                    delayBase +
                    delay * program.counts0.x +
                    action
                ] = 0.0f;
            }
            rawPolicyActions[
                environment * program.counts0.x + action
            ] = 0.0f;
        }
        for (uint index = 0u;
             index < program.layout.z;
             ++index) {
            compactContact[contactBase + index] = 0.0f;
        }

        uint actionDelay = 0u;
        uint observationDelay = 0u;
        const bool fixedCrowNavigationCourse =
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u &&
            dispatch.assistance.w != 0.0f;
        for (uint index = 0u;
             index < program.counts2.y;
             ++index) {
            const MRTaskRandomizationOperatorGPU operation =
                randomization[index];
            if (curriculum < operation.target.w) {
                continue;
            }
            if (fixedCrowNavigationCourse &&
                operation.target.x ==
                    MR_TASK_RANDOMIZE_SCENE_BODY_POSITION_OFFSET) {
                // The development reference is a paired course-regression
                // fixture. Preserve every other authored randomization so it
                // cannot masquerade as held-out qualification evidence.
                continue;
            }
            const uint channel = 2048u + index;
            switch (operation.target.x) {
            case MR_TASK_RANDOMIZE_ROOT_POSITION:
                for (uint component = 0u;
                     component < 3u;
                     ++component) {
                    resetQ[
                        qBase + program.root.z + component
                    ] +=
                        operation.parameters[component] *
                        randomSigned(
                            dispatch,
                            environment,
                            episode,
                            0u,
                            channel + component
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_ROOT_YAW: {
                const float yaw = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                const float halfYaw = 0.5f * yaw;
                const float4 authored = float4(
                    defaultQ[program.root.z + 3u],
                    defaultQ[program.root.z + 4u],
                    defaultQ[program.root.z + 5u],
                    defaultQ[program.root.z + 6u]
                );
                const float4 randomized =
                    quaternionProduct(
                        float4(
                            0.0f,
                            0.0f,
                            sin(halfYaw),
                            cos(halfYaw)
                        ),
                        authored
                    );
                resetQ[qBase + program.root.z + 3u] =
                    randomized.x;
                resetQ[qBase + program.root.z + 4u] =
                    randomized.y;
                resetQ[qBase + program.root.z + 5u] =
                    randomized.z;
                resetQ[qBase + program.root.z + 6u] =
                    randomized.w;
                break;
            }
            case MR_TASK_RANDOMIZE_ROOT_HEIGHT:
                resetQ[qBase + program.root.z + 2u] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_ROOT_ORIENTATION:
                for (uint component = 0u; component < 4u; ++component) {
                    resetQ[qBase + program.root.z + 3u + component] =
                        operation.parameters[component];
                }
                break;
            case MR_TASK_RANDOMIZE_JOINT_POSITION:
                resetQ[qBase + operation.target.y] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_POSITION:
                resetScene[
                    sceneBase + operation.target.y
                ].position[operation.target.z] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_POSITION_OFFSET:
                resetScene[
                    sceneBase + operation.target.y
                ].position[operation.target.z] += randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY:
                resetScene[
                    sceneBase + operation.target.y
                ].linearVelocityAndInverseMass[
                    operation.target.z
                ] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_SCENE_BODY_LAUNCH_STEP: {
                const uint lower = uint(operation.parameters.x);
                const uint upper = uint(operation.parameters.y);
                const uint launchStep = lower + uint(floor(
                    float(upper - lower + 1u) * randomUnit(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        channel
                    )
                ));
                device MRBodyStateGPU& scene = resetScene[
                    sceneBase + operation.target.y
                ];
                scene.flagsAndIndices[3] =
                    (scene.flagsAndIndices[3] &
                     ~MR_BODY_STATE_LAUNCH_STEP_MASK) |
                    ((launchStep << MR_BODY_STATE_LAUNCH_STEP_SHIFT) &
                     MR_BODY_STATE_LAUNCH_STEP_MASK);
                break;
            }
            case MR_TASK_RANDOMIZE_ACTION_POSITION:
                for (uint action = 0u;
                     action < program.counts0.x;
                     ++action) {
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference =
                        (program.schedule.w &
                         MR_TASK_PROGRAM_INTERACTION_RESET) != 0u
                        ? interactionJointTargets[action]
                        : defaultQ[binding.indices.z];
                    resetQ[qBase + binding.indices.z] =
                        clamp(
                            reference +
                                randomRange(
                                    dispatch,
                                    environment,
                                    episode,
                                    0u,
                                    channel + action,
                                    operation.parameters.x,
                                    operation.parameters.y
                                ),
                            binding.parameters.y,
                            binding.parameters.z
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_VELOCITY:
                for (uint coordinate = 0u;
                     coordinate < dispatch.counts.w;
                     ++coordinate) {
                    resetV[vBase + coordinate] =
                        randomRange(
                            dispatch,
                            environment,
                            episode,
                            0u,
                            channel + coordinate,
                            operation.parameters.x,
                            operation.parameters.y
                        );
                }
                break;
            case MR_TASK_RANDOMIZE_ACTION_VELOCITY:
                for (uint action = 0u;
                     action < program.counts0.x;
                     ++action) {
                    resetV[
                        vBase + actions[action].indices.w
                    ] = randomRange(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        channel + action,
                        operation.parameters.x,
                        operation.parameters.y
                    );
                }
                break;
            case MR_TASK_RANDOMIZE_BODY_PARAMETER: {
                const MRTaskContactGroupGPU group =
                    contactGroups[operation.target.y];
                const float sampled = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                for (uint local = 0u;
                     local < group.members.y;
                     ++local) {
                    const uint body =
                        contactMembers[
                            group.members.x + local
                        ];
                    bodyParameters[
                        bodyParameterBase + body
                    ][operation.target.z] = sampled;
                }
                break;
            }
            case MR_TASK_RANDOMIZE_WORLD_BODY_PARAMETER:
                bodyParameters[
                    bodyParameterBase + operation.target.y
                ][operation.target.z] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_BODY_PAYLOAD: {
                const uint body = operation.target.y;
                const float payload = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                bodyParameters[
                    bodyParameterBase + body
                ].x +=
                    payload * operation.parameters.z;
                bodyParameters[
                    bodyParameterBase + body
                ].x = max(
                    bodyParameters[
                        bodyParameterBase + body
                    ].x,
                    0.05f
                );
                break;
            }
            case MR_TASK_RANDOMIZE_CONTROLLER_PARAMETER:
                controllerParameters[environment][
                    operation.target.z
                ] = randomRange(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    channel,
                    operation.parameters.x,
                    operation.parameters.y
                );
                break;
            case MR_TASK_RANDOMIZE_ACTION_DELAY: {
                const uint lower =
                    uint(max(operation.parameters.x, 0.0f));
                const uint upper =
                    uint(max(
                        operation.parameters.y,
                        operation.parameters.x
                    ));
                actionDelay = min(
                    lower +
                        uint(floor(
                            float(upper - lower + 1u) *
                            randomUnit(
                                dispatch,
                                environment,
                                episode,
                                0u,
                                channel
                            )
                        )),
                    program.layout.w - 1u
                );
                break;
            }
            case MR_TASK_RANDOMIZE_OBSERVATION_DELAY: {
                const uint lower =
                    uint(max(operation.parameters.x, 0.0f));
                const uint upper =
                    uint(max(
                        operation.parameters.y,
                        operation.parameters.x
                    ));
                observationDelay = min(
                    lower +
                        uint(floor(
                            float(upper - lower + 1u) *
                            randomUnit(
                                dispatch,
                                environment,
                                episode,
                                0u,
                                channel
                            )
                        )),
                    program.schedule.y
                );
                break;
            }
            default:
                break;
            }
        }
        // Journey band three isolates cruise stabilization. Later v3 bands
        // reset into exact portions of the full 32-second journey.
        // fingerprinted, physically ordinary airborne state rather than
        // asking every cruise episode to rediscover launch from the ground.
        // Band four still begins on bilateral terrain support and remains the
        // authoritative combined ground-to-flight gate.
        if ((program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
            (curriculum == 3u ||
             (curriculum >= 5u && curriculum <= 8u))) {
            const float segmentHeight = curriculum == 8u
                ? 0.4167f
                : 0.85f;
            const float segmentForward = curriculum >= 7u
                ? 0.0f
                : 0.35f;
            resetQ[qBase + program.root.z + 2u] = segmentHeight;
            resetV[vBase + program.root.w + 0u] = segmentForward;
            resetV[vBase + program.root.w + 1u] = 0.0f;
            resetV[vBase + program.root.w + 2u] = curriculum == 8u
                ? -0.1083f
                : 0.0f;
        }
        uint interactionResetStep = 0u;
        if ((program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u) {
            interactionResetStep = curriculum == 5u ? 450u
                : curriculum == 6u ? 750u
                : curriculum == 7u ? 1050u
                : curriculum == 8u ? 1250u
                : curriculum == 9u ? 1350u
                : 0u;
        }
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_RESET) != 0u) {
            if (program.interactionCurriculum.x > 0.0f &&
                program.interactionCurriculum.y > 0.0f &&
                program.interaction.x > 1u &&
                program.interactionTiming.x > 0.0f &&
                dispatch.timing.x > 0.0f &&
                program.schedule.x > 1u) {
                const float clipControlSteps =
                    float(program.interaction.x - 1u) /
                    (program.interactionTiming.x * dispatch.timing.x);
                const uint maximumResetStep = uint(floor(
                    min(
                        float(program.schedule.x - 2u),
                        clipControlSteps
                    ) * program.interactionCurriculum.y
                ));
                const float curriculumSample = randomUnit(
                    dispatch,
                    environment,
                    episode,
                    0u,
                    4094u
                );
                const float canonicalFraction =
                    1.0f - program.interactionCurriculum.x;
                if (curriculumSample >= canonicalFraction) {
                    const float phaseSample =
                        (curriculumSample - canonicalFraction) /
                        program.interactionCurriculum.x;
                    interactionResetStep = min(
                        uint(floor(
                            phaseSample *
                            float(maximumResetStep + 1u)
                        )),
                        maximumResetStep
                    );
                }
            }
            const float resetFramePosition = min(
                float(interactionResetStep) *
                    dispatch.timing.x *
                    program.interactionTiming.x,
                float(program.interaction.x - 1u)
            );
            const uint resetFrame = uint(floor(resetFramePosition));
            const uint nextResetFrame = interactionNextFrame(
                program,
                resetFrame
            );
            const float resetBlend =
                resetFramePosition - floor(resetFramePosition);
            const uint rootBase = resetFrame * 7u;
            const uint nextRootBase = nextResetFrame * 7u;
            const float4 frameOrientation = float4(
                interactionRootTargets[rootBase + 3u],
                interactionRootTargets[rootBase + 4u],
                interactionRootTargets[rootBase + 5u],
                interactionRootTargets[rootBase + 6u]
            );
            const float4 nextOrientation = float4(
                interactionRootTargets[nextRootBase + 3u],
                interactionRootTargets[nextRootBase + 4u],
                interactionRootTargets[nextRootBase + 5u],
                interactionRootTargets[nextRootBase + 6u]
            );
            const float4 targetOrientation = quaternionInterpolate(
                frameOrientation,
                nextOrientation,
                resetBlend
            );
            const float3 frameRootLinkPosition = float3(
                interactionRootTargets[rootBase + 0u],
                interactionRootTargets[rootBase + 1u],
                interactionRootTargets[rootBase + 2u]
            );
            const float3 nextRootLinkPosition = float3(
                interactionRootTargets[nextRootBase + 0u],
                interactionRootTargets[nextRootBase + 1u],
                interactionRootTargets[nextRootBase + 2u]
            );
            // Generalized floating-root translation is the root body's COM,
            // while InteractionPack and task observations author the root-link
            // origin. rootReference is link-origin minus COM in body space.
            const float3 frameRootCOMPosition =
                frameRootLinkPosition - rotate(
                    frameOrientation,
                    program.rootReference.xyz
                );
            const float3 nextRootCOMPosition =
                nextRootLinkPosition - rotate(
                    nextOrientation,
                    program.rootReference.xyz
                );
            const float3 targetRootCOMPosition = mix(
                frameRootCOMPosition,
                nextRootCOMPosition,
                resetBlend
            );
            resetQ[qBase + program.root.z + 0u] =
                targetRootCOMPosition.x;
            resetQ[qBase + program.root.z + 1u] =
                targetRootCOMPosition.y;
            resetQ[qBase + program.root.z + 2u] =
                targetRootCOMPosition.z;
            resetQ[qBase + program.root.z + 3u] = targetOrientation.x;
            resetQ[qBase + program.root.z + 4u] = targetOrientation.y;
            resetQ[qBase + program.root.z + 5u] = targetOrientation.z;
            resetQ[qBase + program.root.z + 6u] = targetOrientation.w;
            resetV[vBase + program.root.w + 0u] =
                (nextRootCOMPosition.x - frameRootCOMPosition.x) *
                program.interactionTiming.x;
            resetV[vBase + program.root.w + 1u] =
                (nextRootCOMPosition.y - frameRootCOMPosition.y) *
                program.interactionTiming.x;
            resetV[vBase + program.root.w + 2u] =
                (nextRootCOMPosition.z - frameRootCOMPosition.z) *
                program.interactionTiming.x;
            const float3 rootAngularVelocity =
                quaternionWorldAngularVelocity(
                    frameOrientation,
                    nextOrientation,
                    program.interactionTiming.x
                );
            resetV[vBase + program.root.w + 3u] =
                rootAngularVelocity.x;
            resetV[vBase + program.root.w + 4u] =
                rootAngularVelocity.y;
            resetV[vBase + program.root.w + 5u] =
                rootAngularVelocity.z;
            for (uint action = 0u;
                 action < program.interaction.y;
                 ++action) {
                const MRTaskActionBindingGPU binding = actions[action];
                const uint jointIndex =
                    resetFrame * program.interaction.y + action;
                const uint nextJointIndex =
                    nextResetFrame * program.interaction.y + action;
                resetQ[qBase + binding.indices.z] =
                    mix(
                        interactionJointTargets[jointIndex],
                        interactionJointTargets[nextJointIndex],
                        resetBlend
                    );
                resetV[vBase + binding.indices.w] =
                    (
                        interactionJointTargets[nextJointIndex] -
                        interactionJointTargets[jointIndex]
                    ) * program.interactionTiming.x;
            }
        }
        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;

        const float4 scheduleSeconds = scheduleSecondsForBand(
            program,
            curriculum
        );

        state.episode = uint4(
            interactionResetStep,
            episode,
            curriculum,
            terrainLevel
        );
        state.schedule = uint4(
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                32u,
                scheduleSeconds.x,
                scheduleSeconds.y
            ),
            durationSteps(
                dispatch,
                environment,
                episode,
                0u,
                33u,
                scheduleSeconds.z,
                scheduleSeconds.w
            ),
            actionDelay,
            observationDelay
        );
        uint interactionClock = 0u;
        if (interactionPhysicsGated(program)) {
            const float resetFramePosition = min(
                float(interactionResetStep) *
                    dispatch.timing.x *
                    program.interactionTiming.x,
                float(program.interaction.x - 1u)
            );
            interactionClock = packInteractionClock(
                resetFramePosition,
                0u,
                false
            );
        }
        state.status = uint4(
            1u,
            0u,
            MR_TASK_TERMINATION_CONTINUING,
            interactionClock
        );
        state.commandAndPhase = float4(
            sampledCommand(
                dispatch,
                program,
                commandCurriculum,
                environment,
                episode,
                0u,
                curriculum
            ),
            0.0f
        );
        state.commandExtension = float4(0.0f);
        state.airReturnTracking = float4(
            0.0f,
            rootHeight(program, resetQ + qBase),
            0.0f,
            0.0f
        );
        state.recovery = float4(0.0f);
        state.recoveryStats = uint4(0u);
        state.threatGeometry = float4(0.0f);
        state.threatTeacher = float4(0.0f);
        state.threatMetadata = uint4(
            MR_INVALID_INDEX,
            MR_TASK_THREAT_NONE,
            0u,
            MR_INVALID_INDEX
        );
        state.navigation = float4(0.0f);
        // The invocation-scoped navigation curriculum exposes PPO to every
        // sequential route stage while leaving autonomous evaluation's
        // waypoint-zero reset unchanged. It supplies no actions or labels;
        // the policy retains full physical action authority.
        if ((program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u &&
            dispatch.assistance.z != 0.0f) {
            uint navigationCourseStart = MR_INVALID_INDEX;
            for (uint observation = 0u;
                 observation < program.counts0.y + program.counts3.w;
                 ++observation) {
                if (actorOperators[observation].source.x ==
                    MR_TASK_OBSERVE_NAVIGATION_TARGET) {
                    navigationCourseStart =
                        actorOperators[observation].source.y;
                    break;
                }
            }
            if (navigationCourseStart != MR_INVALID_INDEX) {
                // The default mixed curriculum retains all route stages.
                // Invocation modes two through six select one exact stage for
                // short, evidence-driven bottleneck training without changing
                // autonomous resets, geometry, rewards, or success criteria.
                constexpr uint curriculumStage[8] = {
                    0u, 1u, 2u, 2u, 2u, 2u, 3u, 4u,
                };
                const uint curriculumMode = uint(max(
                    dispatch.assistance.z,
                    0.0f
                ));
                const uint stage = curriculumMode >= 2u
                    ? min(curriculumMode - 2u, 4u)
                    : curriculumStage[(environment + episode) % 8u];
                const uint stageThreeArrival =
                    (environment + episode) %
                    kCrowNavigationStageThreeArrivalCount;
                const uint stageTwoArrival =
                    (environment + episode) %
                    kCrowNavigationStageTwoArrivalCount;
                // Full-route replay reaches stages one through four near
                // steps 316, 361, 407, and 423. Match that observable journey
                // phase for independent-stage resets; otherwise a stage-four
                // sample at phase zero teaches a different feed-forward state
                // than the final approach seen in autonomous completion.
                constexpr uint stageStep[5] = {
                    0u, 300u, 408u, 962u, 424u,
                };
                state.episode.x = stage == 2u
                    ? kCrowNavigationStageTwoStep[stageTwoArrival]
                    : stage == 3u
                    ? kCrowNavigationStageThreeStep[stageThreeArrival]
                    : stageStep[stage];
                state.navigation.z = float(stage);
                if (stage > 0u) {
                    const uint templateIndex = stage - 1u;
                    for (uint index = 3u;
                        index < min(dispatch.counts.z, 20u);
                         ++index) {
                        resetQ[qBase + program.root.z + index] =
                            stage == 2u
                            ? kCrowNavigationStageTwoQ[
                                  stageTwoArrival
                              ][index]
                            : stage == 3u
                            ? kCrowNavigationStageThreeQ[
                                  stageThreeArrival
                              ][index]
                            : kCrowNavigationStageQ[
                                  templateIndex
                              ][index];
                    }
                    for (uint index = 0u;
                         index < min(dispatch.counts.w, 19u);
                         ++index) {
                        resetV[vBase + program.root.w + index] =
                            stage == 2u
                            ? kCrowNavigationStageTwoV[
                                  stageTwoArrival
                              ][index]
                            : stage == 3u
                            ? kCrowNavigationStageThreeV[
                                  stageThreeArrival
                              ][index]
                            : kCrowNavigationStageV[
                                  templateIndex
                              ][index];
                    }
                    const float3 start = crowNavigationWaypointTarget(
                        resetScene + sceneBase,
                        navigationCourseStart,
                        stage - 1u
                    );
                    const float3 incomingStart = stage > 1u
                        ? crowNavigationWaypointTarget(
                              resetScene + sceneBase,
                              navigationCourseStart,
                              stage - 2u
                          )
                        : float3(0.0f);
                    const float3 direction = normalizedOr(
                        start - incomingStart,
                        float3(1.0f, 0.0f, 0.0f)
                    );
                    // Rebind the accepted flight state as one rigid yaw
                    // transform. Rotating only world linear velocity left the
                    // captured attitude and angular velocity on the old
                    // segment heading, creating a severe sideslip at the
                    // stage-two reset and poisoning the transition data.
                    const float4 acceptedOrientation = float4(
                        resetQ[qBase + program.root.z + 3u],
                        resetQ[qBase + program.root.z + 4u],
                        resetQ[qBase + program.root.z + 5u],
                        resetQ[qBase + program.root.z + 6u]
                    );
                    const float acceptedYaw = atan2(
                        2.0f * (
                            acceptedOrientation.w * acceptedOrientation.z +
                            acceptedOrientation.x * acceptedOrientation.y
                        ),
                        1.0f - 2.0f * (
                            acceptedOrientation.y * acceptedOrientation.y +
                            acceptedOrientation.z * acceptedOrientation.z
                        )
                    );
                    const float desiredYaw = atan2(
                        direction.y,
                        direction.x
                    );
                    // Stage two has a current accepted-state capture from the
                    // development fixture. Preserve its measured heading and
                    // sideslip relative to that fixture's incoming segment;
                    // apply only the course-frame delta on another split.
                    constexpr float stageTwoCapturedIncomingYaw =
                        -0.1807026444f;
                    constexpr float stageThreeCapturedIncomingYaw =
                        0.5880912777f;
                    constexpr float stageFourCapturedIncomingYaw =
                        -0.2927187926f;
                    const float referenceYaw = stage == 2u
                        ? stageTwoCapturedIncomingYaw
                        : stage == 3u
                        ? stageThreeCapturedIncomingYaw
                        : stage == 4u
                        ? stageFourCapturedIncomingYaw
                        : acceptedYaw;
                    const float yawDelta = desiredYaw - referenceYaw;
                    const float4 yawRebind = float4(
                        0.0f,
                        0.0f,
                        sin(0.5f * yawDelta),
                        cos(0.5f * yawDelta)
                    );
                    const float4 reboundOrientation = quaternionProduct(
                        yawRebind,
                        acceptedOrientation
                    );
                    const float3 reboundPosition = start + rotate(
                        yawRebind,
                        stage == 2u
                        ? kCrowNavigationStageTwoRootOffset[
                              stageTwoArrival
                          ]
                        : stage == 3u
                        ? kCrowNavigationStageThreeRootOffset[
                              stageThreeArrival
                          ]
                        : kCrowNavigationStageRootOffset[templateIndex]
                    );
                    resetQ[qBase + program.root.z + 0u] = reboundPosition.x;
                    resetQ[qBase + program.root.z + 1u] = reboundPosition.y;
                    resetQ[qBase + program.root.z + 2u] = reboundPosition.z;
                    for (uint component = 0u; component < 4u; ++component) {
                        resetQ[qBase + program.root.z + 3u + component] =
                            reboundOrientation[component];
                    }
                    // Rotate accepted linear and angular velocity with the
                    // same course-frame transform as pose and root offset.
                    // Reconstructing an on-centerline velocity erased the
                    // accepted sideslip that precedes the measured turn.
                    const float acceptedVelocityX =
                        resetV[vBase + program.root.w + 0u];
                    const float acceptedVelocityY =
                        resetV[vBase + program.root.w + 1u];
                    const float yawCos = cos(yawDelta);
                    const float yawSin = sin(yawDelta);
                    resetV[vBase + program.root.w + 0u] =
                        yawCos * acceptedVelocityX -
                        yawSin * acceptedVelocityY;
                    resetV[vBase + program.root.w + 1u] =
                        yawSin * acceptedVelocityX +
                        yawCos * acceptedVelocityY;
                    const float angularX =
                        resetV[vBase + program.root.w + 3u];
                    const float angularY =
                        resetV[vBase + program.root.w + 4u];
                    resetV[vBase + program.root.w + 3u] =
                        yawCos * angularX - yawSin * angularY;
                    resetV[vBase + program.root.w + 4u] =
                        yawSin * angularX + yawCos * angularY;

                    // The generic reset path ran before the replay template
                    // was selected, so refresh its temporal controller state
                    // from the accepted generalized state. Joint-acceleration
                    // features and rate rewards must not compare a mid-flight
                    // sample against an artificial zero-velocity predecessor.
                    for (uint action = 0u;
                         action < program.counts0.x;
                         ++action) {
                        const uint velocityIndex =
                            actions[action].indices.w;
                        previousJointVelocity[
                            previousVelocityBase + action
                        ] = velocityIndex == MR_INVALID_INDEX
                            ? 0.0f
                            : resetV[vBase + velocityIndex];
                        const uint positionIndex =
                            actions[action].indices.z;
                        previousJointVelocity[
                            previousVelocityBase +
                            program.counts0.x + action
                        ] = positionIndex == MR_INVALID_INDEX
                            ? 0.0f
                            : resetQ[qBase + positionIndex];
                    }
                    if (stage == 2u || stage == 3u || stage == 4u) {
                        state.commandAndPhase =
                            stage == 2u
                            ? kCrowNavigationStageTwoCommandAndPhase[
                                  stageTwoArrival
                              ]
                            : stage == 3u
                            ? kCrowNavigationStageThreeCommandAndPhase[
                                  stageThreeArrival
                              ]
                            : kCrowNavigationStageCommandAndPhase[
                                  templateIndex
                              ];
                        state.commandExtension.w =
                            stage == 2u
                            ? kCrowNavigationStageTwoJourneyPhase[
                                  stageTwoArrival
                              ]
                            : stage == 3u
                            ? kCrowNavigationStageThreeJourneyPhase[
                                  stageThreeArrival
                              ]
                            : kCrowNavigationStageJourneyPhase[
                                  templateIndex
                              ];
                        for (uint action = 0u;
                             action < min(program.counts0.x, 15u);
                             ++action) {
                            const float acceptedAction =
                                stage == 2u
                                ? kCrowNavigationStageTwoAction[
                                      stageTwoArrival
                                  ][action]
                                : stage == 3u
                                ? kCrowNavigationStageThreeAction[
                                      stageThreeArrival
                                  ][action]
                                : kCrowNavigationStageAction[
                                      templateIndex
                                  ][action];
                            for (uint delay = 0u;
                                 delay < program.layout.w;
                                 ++delay) {
                                actionHistory[
                                    delayBase +
                                    delay * program.counts0.x + action
                                ] = acceptedAction;
                            }
                            rawPolicyActions[
                                environment * program.counts0.x + action
                            ] = acceptedAction;
                        }
                    }
                }
            }
        }
        const bool projectileEpisode = randomUnit(
            dispatch,
            environment,
            episode,
            0u,
            3069u
        ) >= program.dynamics.z;
        for (uint impact = 0u;
             impact < program.counts3.x;
             ++impact) {
            const MRTaskImpactEventGPU event =
                impactEvents[impact];
            if (curriculum < event.binding.z ||
                !projectileEpisode) {
                continue;
            }
            state.recoveryStats.w |= kImpactEnabled;
            device MRBodyStateGPU& held = resetScene[
                sceneBase + event.binding.x
            ];
            held.linearVelocityAndInverseMass.xyz =
                float3(0.0f);
            held.angularVelocity = float4(0.0f);
        }
        if (impactSequenceEnabled(state)) {
            const uint count = program.counts3.x;
            const uint offset = min(
                uint(floor(
                    float(count) * randomUnit(
                        dispatch,
                        environment,
                        episode,
                        0u,
                        3072u
                    )
                )),
                count - 1u
            );
            state.recoveryStats.w |=
                offset << kImpactOffsetShift;
        }

        device float* firstActor =
            actorHistory + historyBase;
        device float* firstClean =
            cleanHistory + historyBase;
        writeFrame(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            episode,
            state.episode.x,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            sensorBias + biasBase,
            compactContact + contactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            resetScene + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            firstActor,
            firstClean
        );
        for (uint history = 1u;
             history < program.layout.y;
             ++history) {
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = firstActor[index];
                cleanHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = firstClean[index];
            }
        }
        device float* firstCritic =
            criticHistory + criticHistoryBase;
        writeCriticFrame(
            program,
            arena,
            dispatch.timing.x,
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            resetQ + qBase,
            resetV + vBase,
            defaultQ,
            state,
            actionHistory + delayBase,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            compactContact + contactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            resetScene + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            firstCritic
        );
        for (uint history = 1u;
             history < program.articulation.w;
             ++history) {
            for (uint index = 0u;
                 index < program.counts0.z;
                 ++index) {
                criticHistory[
                    criticHistoryBase +
                    history * program.counts0.z +
                    index
                ] = firstCritic[index];
            }
        }
    }

    if (!reset &&
        impactSequenceEnabled(state) &&
        program.counts3.x > 0u &&
        state.episode.z >= impactEvents[0].binding.z) {
        const uint order = impactOrder(state);
        uint activeScene = impactScene(state);
        bool newlyLaunched = false;
        if (order < program.counts3.x) {
            const uint eventIndex =
                (order + impactOffset(state)) %
                program.counts3.x;
            const MRTaskImpactEventGPU event =
                impactEvents[eventIndex];
            if (activeScene == 0u) {
                const bool stable =
                    state.recovery.w <= 0.5f &&
                    state.recovery.x <= event.gate.x &&
                    rootHeight(program, sourceQ + qBase) >=
                        event.gate.w;
                state.status.w = stable
                    ? state.status.w + 1u
                    : 0u;
                if (float(state.status.w) * dispatch.timing.x >=
                    event.gate.y) {
                    activeScene = event.binding.x + 1u;
                    state.recoveryStats.w =
                        (state.recoveryStats.w &
                         ~(kImpactSceneMask)) |
                        (activeScene << kImpactSceneShift);
                    state.status.w = 0u;
                    newlyLaunched = true;
                }
            } else {
                state.status.w += 1u;
            }
        }
        for (uint impact = 0u;
             impact < program.counts3.x;
             ++impact) {
            const MRTaskImpactEventGPU event =
                impactEvents[impact];
            const uint currentEvent =
                order < program.counts3.x
                ? (order + impactOffset(state)) %
                    program.counts3.x
                : MR_INVALID_INDEX;
            if (state.episode.z < event.binding.z ||
                (impact == currentEvent &&
                 activeScene != 0u &&
                 !newlyLaunched)) {
                continue;
            }
            MRBodyStateGPU scheduled = resetScene[
                sceneBase + event.binding.x
            ];
            scheduled.linearVelocityAndInverseMass.xyz =
                float3(0.0f);
            scheduled.angularVelocity = float4(0.0f);
            if (impact == currentEvent && newlyLaunched) {
                const MRBodyStateGPU initial = initialScene[
                    sceneBase + event.binding.x
                ];
                scheduled.linearVelocityAndInverseMass.xyz =
                    initial.linearVelocityAndInverseMass.xyz;
                scheduled.angularVelocity = initial.angularVelocity;
                for (uint index = 0u;
                     index < program.counts2.y;
                     ++index) {
                    const MRTaskRandomizationOperatorGPU operation =
                        randomization[index];
                    if (operation.target.x !=
                            MR_TASK_RANDOMIZE_SCENE_BODY_VELOCITY ||
                        operation.target.y != event.binding.x ||
                        state.episode.z < operation.target.w) {
                        continue;
                    }
                    scheduled.linearVelocityAndInverseMass[
                        operation.target.z
                    ] = randomRange(
                        dispatch,
                        environment,
                        state.episode.y,
                        0u,
                        2048u + index,
                        operation.parameters.x,
                        operation.parameters.y
                    );
                }
                if (program.projectile.y > 0.0f) {
                    const float targetRadius =
                        program.projectileGravity.w;
                    const float3 target = float3(
                        sourceQ[qBase + program.root.z] + randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4096u + impact * 4u,
                            -targetRadius,
                            targetRadius
                        ),
                        sourceQ[qBase + program.root.z + 1u] + randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4097u + impact * 4u,
                            -targetRadius,
                            targetRadius
                        ),
                        randomRange(
                            dispatch,
                            environment,
                            state.episode.y,
                            0u,
                            4098u + impact * 4u,
                            program.projectile.z,
                            program.projectile.w
                        )
                    );
                    // Scene-body velocity randomization owns the authored
                    // curriculum bands. Preserve its horizontal magnitude
                    // when retargeting the throw instead of replacing every
                    // level with the task-wide fallback range.
                    const float authoredHorizontalSpeed = length(
                        scheduled.linearVelocityAndInverseMass.xy
                    );
                    const float horizontalSpeed =
                        authoredHorizontalSpeed > 0.0f
                        ? authoredHorizontalSpeed
                        : randomRange(
                              dispatch,
                              environment,
                              state.episode.y,
                              0u,
                              4099u + impact * 4u,
                              program.projectile.x,
                              program.projectile.y
                          );
                    const float3 delta = target - scheduled.position.xyz;
                    const float flightSeconds = max(
                        length(delta.xy) / horizontalSpeed,
                        dispatch.timing.x
                    );
                    scheduled.linearVelocityAndInverseMass.xyz =
                        delta / flightSeconds -
                        0.5f * program.projectileGravity.xyz *
                            flightSeconds;
                }
            }
            sourceScene[
                sceneBase + event.binding.x
            ] = scheduled;
        }
    } else if (!reset) {
        for (uint localScene = 0u;
             localScene < dispatch.strides.w;
             ++localScene) {
            const MRBodyStateGPU authored =
                resetScene[sceneBase + localScene];
            const uint launchStep =
                (authored.flagsAndIndices[3] &
                 MR_BODY_STATE_LAUNCH_STEP_MASK) >>
                MR_BODY_STATE_LAUNCH_STEP_SHIFT;
            if (launchStep == 0u ||
                state.episode.x > launchStep) {
                continue;
            }
            MRBodyStateGPU scheduled = authored;
            if (state.episode.x < launchStep) {
                scheduled.linearVelocityAndInverseMass.xyz =
                    float3(0.0f);
                scheduled.angularVelocity = float4(0.0f);
            } else {
                // A delayed projectile is parked at reset, so resetScene
                // deliberately has zero velocity.  Restore the authored
                // launch state on its exact episode tick rather than letting
                // it fall vertically from its staging point.
                scheduled = initialScene[sceneBase + localScene];
            }
            sourceScene[sceneBase + localScene] = scheduled;
        }
    }

    taskStates[environment] = state;
    const uint actorObservationSize =
        dispatch.outputs.x / dispatch.counts.x;
    const uint actorOutputBase =
        pass.controlStep * dispatch.outputs.x +
        environment * actorObservationSize;
    const uint criticObservationSize =
        (
            (program.schedule.w &
             MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
            ? program.layout.x * program.layout.y
            : 0u
        ) +
        criticHistoryElements;
    const uint criticOutputBase =
        pass.controlStep * dispatch.outputs.y +
        environment * criticObservationSize;
    for (uint index = 0u;
         index < historyElements;
         ++index) {
        actorObservations[actorOutputBase + index] =
            actorHistory[historyBase + index];
    }
    device const float* observationQ = reset
        ? resetQ + qBase
        : sourceQ + qBase;
    device const float* observationV = reset
        ? resetV + vBase
        : sourceV + vBase;
    device const MRBodyStateGPU* observationScene = reset
        ? resetScene + sceneBase
        : sourceScene + sceneBase;
    writeCurrentActor(
        dispatch,
        program,
        arena,
        actorOperators,
        actions,
        contactGroups,
        terrainSamples,
        environment,
        state.episode.y,
        state.episode.x,
        observationQ,
        observationV,
        defaultQ,
        state,
        actionHistory + delayBase,
        rawPolicyActions + environment * program.counts0.x,
        previousJointVelocity + previousVelocityBase + program.counts0.x,
        sensorBias + biasBase,
        compactContact + contactBase,
        bodyParameters + bodyParameterBase,
        controllerParameters + environment,
        observationScene,
        shapes,
        geometryHeaders,
        geometryVertices,
        actorObservations + actorOutputBase + historyElements
    );
    publishCritic(
        program,
        cleanHistory + historyBase,
        criticHistory + criticHistoryBase,
        criticObservations + criticOutputBase
    );

    if (!reset && state.schedule.y == 0u &&
        program.dynamics.x > 0.0f) {
        const float progress = clamp(
            float(state.episode.z) /
                max(float(program.schedule.z - 1u), 1.0f),
            0.0f,
            1.0f
        );
        sourceV[vBase + program.root.w + 0u] +=
            progress * program.dynamics.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                48u
            );
        sourceV[vBase + program.root.w + 1u] +=
            progress * program.dynamics.x *
            randomSigned(
                dispatch,
                environment,
                state.episode.y,
                state.episode.x,
                49u
            );
    }

}

// A deliberately simple, observable showcase teacher. Every value is a
// normalized TaskPack action: the ordinary flapping joints, articulated leg
// drives, tail drive, and bounded yaw body-wrench remain the only physical
// authority. The continuous 32-second journey phase and wing clock are both
// actor observations, so a feed-forward student can actually distill this
// schedule instead of depending on hidden host time.
inline float birdFlowJourneyTeacherAction(
    const uint action,
    const MRTaskStateGPU state,
    const float controlStepSeconds,
    const bool navigation
) {
    const float seconds =
        float(state.episode.x) * controlStepSeconds;
    const bool groundOnly = state.episode.z <= 1u;
    const bool takeoffOnly = state.episode.z == 2u;
    const bool isolatedCruise = state.episode.z == 3u;
    const bool takeoffCruise = state.episode.z == 4u;
    const bool straightCruise =
        takeoffOnly || isolatedCruise || takeoffCruise || navigation;
    const float launchStart = groundOnly ? 32.0f
        : navigation ? 2.0f
        : takeoffOnly || isolatedCruise ? 0.0f
        : takeoffCruise ? 1.0f : 5.0f;
    const float approachStart = straightCruise ? 32.0f : 21.0f;
    const float landingStart = straightCruise ? 32.0f : 27.0f;
    const bool airborne = seconds >= launchStart && seconds < landingStart;
    float targetHeight = 0.1873f;
    if (isolatedCruise || navigation) {
        targetHeight = 0.85f;
    } else if (seconds >= launchStart && seconds < launchStart + 4.0f) {
        targetHeight = mix(
            0.1873f,
            0.85f,
            clamp((seconds - launchStart) / 4.0f, 0.0f, 1.0f)
        );
    } else if (seconds >= launchStart + 4.0f &&
               seconds < approachStart) {
        targetHeight = 0.85f;
    } else if (seconds >= approachStart && seconds < landingStart) {
        targetHeight = mix(
            0.85f,
            0.45f,
            clamp(
                (seconds - approachStart) /
                    max(landingStart - approachStart, 1.0f),
                0.0f,
                1.0f
            )
        );
    }
    const float heightError =
        targetHeight - state.airReturnTracking.y;
    const float forwardSpeedError = straightCruise
        ? state.commandExtension.z - 0.35f
        : state.threatGeometry.x;
    const float lateralSpeedError = state.threatGeometry.y;
    const float headingError = state.commandExtension.x;
    const float yawRate = straightCruise
        ? state.commandExtension.y
        : state.threatGeometry.z;
    if (action < 2u) {
        return airborne
            ? clamp(
                  -0.150f + 1.200f * heightError -
                      0.120f * forwardSpeedError,
                  -0.500f,
                  0.500f
              )
            : 0.0f;
    }
    if (action == 2u || action == 3u) {
        if (straightCruise) {
            const float correction = clamp(
                -0.10f * headingError - 0.04f * yawRate,
                -0.12f,
                0.12f
            );
            return action == 2u ? correction : -correction;
        }
        if (seconds < 9.0f || seconds >= 24.0f) {
            return 0.0f;
        }
        const float correction = clamp(
            -0.18f * lateralSpeedError - 0.10f * yawRate,
            -0.18f,
            0.18f
        );
        return action == 2u ? correction : -correction;
    }
    if (action == 4u || action == 5u) {
        const float pitchCorrection = 0.0f;
        return airborne
            ? clamp(
                  0.20f * sin(state.commandAndPhase.w + 2.62f) +
                      pitchCorrection,
                  -0.65f,
                  0.65f
              )
            : 0.0f;
    }
    if (action == 6u) {
        const float pitchError = straightCruise
            ? 0.0f
            : state.threatGeometry.w;
        return airborne
            ? clamp(
                  -0.20f + 0.60f * heightError - 2.00f * pitchError,
                  -0.80f,
                  0.60f
              )
            : 0.0f;
    }
    if (action >= 7u && action <= 12u) {
        const uint legAction = action - 7u;
        const bool right = legAction >= 3u;
        const uint joint = legAction % 3u;
        if ((state.episode.z == 1u || !straightCruise) &&
            seconds >= 2.0f && seconds < 5.0f) {
            const float phase = kTwoPi * seconds / 0.50f;
            const float swing = right ? -sin(phase) : sin(phase);
            const float lift = max(swing, 0.0f);
            return joint == 0u
                ? -0.014f * swing
                : joint == 1u
                ? 0.018f * lift
                : -0.010f * lift;
        }
        if (seconds >= launchStart &&
            seconds < (straightCruise ? 32.0f : 24.0f)) {
            return joint == 0u ? -0.10f : joint == 1u ? 0.18f : -0.08f;
        }
        if (!straightCruise && seconds >= 24.0f && seconds < landingStart) {
            const float tuck = (landingStart - seconds) /
                max(landingStart - 24.0f, 1.0f);
            return tuck *
                (joint == 0u ? -0.10f : joint == 1u ? 0.18f : -0.08f);
        }
        return 0.0f;
    }
    if (straightCruise && action == 13u && airborne) {
        return clamp(
            -0.45f * headingError - 0.18f * yawRate,
            -0.30f,
            0.30f
        );
    }
    if (!straightCruise && action == 13u &&
        seconds >= 9.0f && seconds < 24.0f) {
        return clamp(
            -0.30f * yawRate - 0.12f * lateralSpeedError,
            -0.35f,
            0.35f
        );
    }
    if (!straightCruise && action == 14u && airborne) {
        // The invocation-scoped teacher publishes the same live
        // proportional-and-rate target as the hierarchical supervisor, but
        // this action label passes through the ordinary actuator response
        // filter before reaching the body moment. Compensate for that lag in
        // the teacher gains; this remains a bounded training-time label and
        // neural deployment receives no supervisor authority.
        return clamp(
            -8.00f * state.threatGeometry.w -
                2.00f * state.threatTeacher.x,
            -1.0f,
            1.0f
        );
    }
    return 0.0f;
}

kernel void mr_locomotion_task_apply_actions(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch
        [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* actionStream [[buffer(5)]],
    device float* effortTrajectory [[buffer(6)]],
    device const float* defaultQ [[buffer(7)]],
    device const MRTaskStateGPU* taskStates [[buffer(8)]],
    device float* actionHistory [[buffer(9)]],
    device float* teacherActions [[buffer(10)]],
    device const float* rawPolicyLatents [[buffer(11)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.counts.x !=
            worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION ||
        program.counts0.x == 0u ||
        program.layout.w < 2u) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    const uint actionBase =
        pass.controlStep * dispatch.strides.x +
        environment * program.counts0.x;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
    const MRTaskStateGPU state = taskStates[environment];
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );

    for (uint coordinate = 0u;
         coordinate < dispatch.counts.w;
         ++coordinate) {
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            coordinate
        ] = 0.0f;
    }
    const uint filterSlot = program.layout.w - 1u;
    const uint rawLastSlot = filterSlot - 1u;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        for (uint delay = 0u;
             delay < rawLastSlot;
             ++delay) {
            actionHistory[
                delayBase +
                delay * program.counts0.x +
                action
            ] = actionHistory[
                delayBase +
                (delay + 1u) * program.counts0.x +
                action
            ];
        }
        // PolicyPack owns the action bound.  The task still clamps the
        // resulting position target to the joint range below, so policies
        // trained with residuals outside [-1, 1] retain their source action
        // semantics without weakening physical target safety.
        const bool avianJourneyTeacher =
            dispatch.sampling.w != 0u &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
            true;
        const bool avianJourneyApproachEnvelope =
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_APPROACH_ENVELOPE) != 0u &&
            state.commandExtension.w >= (18.0f / 32.0f);
        const float approachSupervisorBlend =
            avianJourneyApproachEnvelope
            ? smoothstep(
                  0.16f,
                  0.22f,
                  abs(state.threatGeometry.w)
              )
            : 0.0f;
        const float studentRequested = actionStream[actionBase + action];
        const float teacherRequested =
            avianJourneyTeacher || approachSupervisorBlend > 0.0f
            ? birdFlowJourneyTeacherAction(
                  action,
                  state,
                  dispatch.timing.x,
                  (program.schedule.w &
                   MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u
              )
            : 0.0f;
        // Full-journey distillation adds flight and approach to an actor that
        // has already qualified landed hold. Once the authored landing
        // boundary is reached, retain that actor as both the executed carrier
        // and the distillation label instead of blending it toward the
        // teacher's intentionally neutral post-flight action.
        const bool avianJourneyLandedActorHandoff =
            avianJourneyTeacher && state.episode.z >= 9u &&
            state.commandExtension.w >= (27.0f / 32.0f);
        const float effectiveTeacherRequested =
            avianJourneyLandedActorHandoff
            ? studentRequested
            : teacherRequested;
        const float requested = avianJourneyTeacher
            ? mix(
                  effectiveTeacherRequested,
                  studentRequested,
                  clamp(dispatch.assistance.x, 0.0f, 1.0f)
              )
            : mix(
                  studentRequested,
                  teacherRequested,
                  approachSupervisorBlend
              );
        if (avianJourneyTeacher) {
            teacherActions[actionBase + action] =
                effectiveTeacherRequested;
        }
        rawPolicyActions[
            environment * program.counts0.x + action
        ] = rawPolicyLatents[actionBase + action];
        const float previous = actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ];
        const MRTaskActionBindingGPU binding =
            actions[action];
        const float responseTimeSeconds = binding.parameters.w;
        const float responseFraction =
            responseTimeSeconds > 0.0f
            ? clamp(
                  1.0f - exp(
                      -dispatch.timing.x /
                      responseTimeSeconds
                  ),
                  0.0f,
                  1.0f
              )
            : 1.0f;
        actionHistory[
            delayBase +
            rawLastSlot * program.counts0.x +
            action
        ] = requested;
        const uint selected =
            rawLastSlot -
            min(state.schedule.z, rawLastSlot);
        const float delayed = actionHistory[
            delayBase +
            selected * program.counts0.x +
            action
        ];
        actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ] = mix(
            previous,
            delayed,
            responseFraction
        );
        const float filtered = actionHistory[
            delayBase +
            filterSlot * program.counts0.x +
            action
        ];
        if (binding.actuator.x != MR_TASK_ACTUATOR_JOINT_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_GRIPPER_POSITION &&
            binding.actuator.x != MR_TASK_ACTUATOR_FLAPPING_POSITION) {
            // Velocity, effort, tendon, and body-wrench commands are
            // evaluated from live microstep state by the generic actuator
            // pass. Rotor mixers have their own compiled robot program. A
            // task without an executable TeacherPack has no teacher-action
            // allocation, so none of these branches may touch that stream.
            continue;
        }
        const bool interactionReference =
            (program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u;
        const bool avianActionSet =
            program.counts0.x >= 9u &&
            actions[0].actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION &&
            actions[1].actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION;
        const bool avianJourney = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
            state.episode.z >= 4u;
        const bool avianJourneyGround = avianJourney &&
            (state.episode.z == 4u
                ? state.commandExtension.w < 0.03125f
                : (state.commandExtension.w < 0.15625f ||
                   state.commandExtension.w >= 0.84375f));
        const bool avianGroundCurriculum = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_GROUND_CURRICULUM) != 0u &&
            (state.episode.z < 2u || avianJourneyGround);
        const bool avianStandingCurriculum = avianGroundCurriculum &&
            (state.episode.z == 0u ||
             (avianJourney &&
              (state.episode.z == 4u
                  ? state.commandExtension.w < 0.03125f
                  : (state.commandExtension.w < 0.0625f ||
                     state.commandExtension.w >= 0.95f))));
        const bool avianCrowGroundGaitCarrier = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_GROUND_GAIT_CARRIER) != 0u &&
            state.episode.z == 1u;
        const bool avianCrowGroundLegResidual =
            avianCrowGroundGaitCarrier &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_GROUND_LEG_RESIDUAL) != 0u;
        // Crow action indices 7...12 are the bilateral hip, knee, and ankle
        // position drives. In the carrier-supported walking band, every
        // wing and tail position action must remain at its mechanism default:
        // flapping is already folded by the ground curriculum, and allowing
        // its sweep/pronation/tail siblings to learn residuals couples a leg
        // task to irrelevant airborne attitude authority.
        const bool avianCrowGroundLegAction =
            avianCrowGroundLegResidual &&
            binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION &&
            binding.indices.x >= 7u && binding.indices.x <= 12u;
        const bool avianCrowLiftoffTrimCarrier = avianActionSet &&
            (program.schedule.w &
             MR_TASK_PROGRAM_AVIAN_CROW_LIFTOFF_TRIM_CARRIER) != 0u &&
            state.episode.z == 2u;
        const bool avianCrowLiftoffPronationAction =
            avianCrowLiftoffTrimCarrier &&
            binding.actuator.x == MR_TASK_ACTUATOR_JOINT_POSITION &&
            (binding.indices.x == 4u || binding.indices.x == 5u);
        // The live action sweep brackets the estimated hybrid's transition:
        // +0.100 remains ground-bound whereas +0.125 repeatedly reaches the
        // altitude boundary.  A positive stroke-plane tilt then supplies
        // forward authority, so trim on the previous accepted root height,
        // vertical rate, and yaw-frame forward speed together.  The lower
        // wing limit must remain below the static-liftoff threshold: after a
        // real climb or forward overspeed the carrier must be able to remove
        // thrust, not merely reduce an always-positive stroke. The 0.10
        // forward-bias bracket still reached the height boundary under the
        // shallower controller, so altitude and vertical-rate feedback are
        // deliberately strong enough to cross through zero wing command.
        // This is a Metal-resident controller of the actual wing positions,
        // never an injected aerodynamic force or a prerecorded trajectory.
        float avianLiftoffWingCarrier = 0.0f;
        if (avianCrowLiftoffTrimCarrier &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION) {
            const float heightError = 0.85f - state.airReturnTracking.y;
            const float verticalRate = state.commandExtension.w;
            const float forwardSpeedError =
                state.commandExtension.z - 0.35f;
            avianLiftoffWingCarrier = clamp(
                -0.150f + 0.400f * heightError - 0.120f * verticalRate -
                    0.100f * forwardSpeedError,
                -1.000f,
                0.125f
            );
        }
        const float avianWingPolicyCommand = avianCrowLiftoffTrimCarrier
            ? clamp(
                avianLiftoffWingCarrier + 0.25f * filtered,
                -1.0f,
                1.0f
            )
            : filtered;
        const float avianWingAmplitude =
            avianGroundCurriculum &&
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? 0.0f
            : clamp(
                binding.drive.z + binding.drive.w * avianWingPolicyCommand,
                0.0f,
                1.0f
            );
        float avianGroundGaitCarrier = 0.0f;
        if (avianCrowGroundGaitCarrier) {
            const float phase = kTwoPi *
                float(state.episode.x) * dispatch.timing.x / 0.50f;
            const float leftSwing = sin(phase);
            const float rightSwing = -leftSwing;
            switch (binding.indices.x) {
            case 7u:
                avianGroundGaitCarrier = -0.014f * leftSwing;
                break;
            case 8u:
                avianGroundGaitCarrier = 0.018f * max(leftSwing, 0.0f);
                break;
            case 9u:
                avianGroundGaitCarrier = -0.010f * max(leftSwing, 0.0f);
                break;
            case 10u:
                avianGroundGaitCarrier = -0.014f * rightSwing;
                break;
            case 11u:
                avianGroundGaitCarrier = 0.018f * max(rightSwing, 0.0f);
                break;
            case 12u:
                avianGroundGaitCarrier = -0.010f * max(rightSwing, 0.0f);
                break;
            default:
                break;
            }
        }
        // Stage 1 is a residual-learning problem around the qualified gait
        // carrier.  The learned action has a deliberately narrower authority
        // than the carrier: early policy updates can improve timing and trim
        // without trivially cancelling the stable walking cycle. The isolated
        // lift-off band is likewise a residual problem around live wing and
        // tail control: full-range leg or tail actions otherwise bypass the
        // measured speed/altitude loop before a flight skill exists.
        const float avianGroundResidualScale =
            avianCrowGroundGaitCarrier ? 0.25f : 1.0f;
        const float avianGroundPolicyResidual =
            avianCrowGroundLegResidual && !avianCrowGroundLegAction
            ? 0.0f
            : filtered * avianGroundResidualScale;
        const float avianLiftoffNonWingResidualScale =
            avianCrowLiftoffTrimCarrier ? 0.25f : 1.0f;
        const float avianLiftoffTailResidualScale =
            avianCrowLiftoffTrimCarrier ? 0.10f : 1.0f;
        const float avianLiftoffPronationCarrier =
            avianCrowLiftoffPronationAction
            ? 0.20f * sin(state.commandAndPhase.w + 2.62f)
            : 0.0f;
        const float avianLiftoffTailCarrier =
            avianCrowLiftoffTrimCarrier && binding.indices.x == 6u
            ? clamp(
                // The long-horizon response sweep retains this qualified
                // speed-and-height trim: forcing lower tail pitch eventually
                // contacts. The lower bound is deliberately inside the
                // sampled joint range, not a force clamp.
                0.25f + 0.075f *
                    (state.commandExtension.z - 0.35f) +
                    0.300f * (0.85f - state.airReturnTracking.y),
                0.15f,
                0.50f
            )
            : 0.0f;
        const float avianNonWingCommand =
            avianLiftoffTailCarrier != 0.0f
            ? clamp(
                avianLiftoffTailCarrier +
                    avianLiftoffTailResidualScale * filtered,
                -1.0f,
                1.0f
            )
            : avianCrowLiftoffPronationAction
            ? clamp(
                // The long-horizon host position-drive sweep rejected an
                // in-phase feathering signal but retained the filtered pi
                // phase convention. Compensate its 20 ms first-order action
                // filter (about 0.52 rad at 4.6 Hz) here, where the carrier
                // directly targets the actual ABA pronation coordinates. The
                // 0.20 normalized amplitude maps to +/-0.060 rad, inside the
                // verified static +/-0.075-rad position-target envelope.
                // This never edits an aerodynamic coefficient or wrench;
                // the learner only adds its existing bounded residual.
                avianLiftoffPronationCarrier + 0.25f * filtered,
                -1.0f,
                1.0f
            )
            : avianGroundPolicyResidual *
                    avianLiftoffNonWingResidualScale +
                avianGroundGaitCarrier;
        const float studentTarget =
            binding.actuator.x == MR_TASK_ACTUATOR_FLAPPING_POSITION
            ? defaultQ[binding.indices.z] +
                binding.parameters.x *
                    // The clock is robot-owned, while each policy output is
                    // a bilateral stroke-amplitude residual. Resolved
                    // aerodynamic loads, not this kinematic carrier, decide
                    // the resulting height and attitude.
                    // The compiled flapping binding supplies its own
                    // zero-action trim and bounded residual span, so initial
                    // policy exploration begins in the viable wingbeat band.
                    avianWingAmplitude *
                    sin(state.commandAndPhase.w)
            : defaultQ[binding.indices.z] +
                binding.parameters.x *
                    avianNonWingCommand *
                    (avianStandingCurriculum ? 0.0f : 1.0f);
        float targetCandidate = studentTarget;
        if (interactionReference) {
            const float frameReference = interactionJointTargets[
                referenceFrame * program.interaction.y + action
            ];
            const float nextReference = interactionJointTargets[
                nextReferenceFrame * program.interaction.y + action
            ];
            const float reference = mix(
                frameReference,
                nextReference,
                referenceBlend
            );
            const float referenceVelocity =
                (nextReference - frameReference) *
                program.interactionTiming.x *
                interactionPlaybackRate(program, state);
            // MetalWorld's implicit drive evaluates position at q + h*v and
            // damps v toward zero. Lead the position target by
            // (h + kd/kp)*v_ref so the same physical drive instead tracks
            // both ARDY's q_ref and v_ref. Gravity, contact, effort limits,
            // and the articulated solve remain authoritative.
            const float velocityLeadSeconds =
                dispatch.timing.y +
                (binding.drive.x > 0.0f
                    ? binding.drive.y / binding.drive.x
                    : 0.0f);
            targetCandidate =
                reference +
                velocityLeadSeconds * referenceVelocity +
                (interactionPhysicsGated(program)
                    ? 1.0f
                    : program.interactionTiming.z) *
                    (studentTarget - defaultQ[binding.indices.z]);
        }
        const float target = clamp(
            targetCandidate,
            binding.parameters.y,
            binding.parameters.z
        );
        effortTrajectory[
            pass.controlStep *
                worldDispatch.effortStepStride +
            environment *
                worldDispatch.effortEnvironmentStride +
            binding.indices.w
        ] = target;
        if ((program.schedule.w &
             MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u) {
            // With nonzero student authority the policy owns a residual on
            // top of ARDY's motion, so the neutral teacher residual is zero.
            // At zero authority this is pure teacher collection for an
            // autonomous student and the absolute interaction action remains
            // the correct distillation label.
            teacherActions[actionBase + action] =
                program.interactionTiming.z > 0.0f
                ? 0.0f
                : clamp(
                    (target - defaultQ[binding.indices.z]) /
                        binding.parameters.x,
                    -1.0f,
                    1.0f
                );
        }
    }
}

inline float actuatorEffortEnvelope(
    device const MRDofPropertiesGPU& dof,
    device const MRActuatorProfileGPU& actuator,
    const float velocity
) {
    const float speedFraction = clamp(
        abs(velocity) /
            max(actuator.motorAndSpeed.z, 1.175494351e-38f),
        0.0f,
        1.0f
    );
    const float dofLimit =
        (dof.flags & MR_DOF_FLAG_EFFORT_LIMIT) != 0u &&
            dof.limits.w > 0.0f
        ? dof.limits.w
        : 3.402823466e+38f;
    return min(
        dofLimit,
        actuator.transmissionAndEnvelope.z *
            actuator.motorAndSpeed.w *
            (1.0f - speedFraction)
    );
}

inline float actuatorDryFriction(
    device const MRDofPropertiesGPU& dof,
    const float velocity,
    const float requested
) {
    const float friction = dof.drive.w;
    if (!(friction > 0.0f)) {
        return 0.0f;
    }
    return abs(velocity) > 1.0e-4f
        ? -copysign(friction, velocity)
        : -clamp(requested, -friction, friction);
}

// Executes every effort-producing RobotPack actuator from the live accepted
// microstep state. Position/gripper targets remain in the implicit-drive
// trajectory; rotor mixers remain in their robot-authored program. This pass
// adds no integration or kinematic override: it only writes generalized
// effort and world-frame body wrench consumed by the ordinary ABA solve.
kernel void mr_locomotion_task_apply_native_actuators(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const float* vState [[buffer(6)]],
    device const float* actionHistory [[buffer(7)]],
    device const MRDofPropertiesGPU* dofs [[buffer(8)]],
    device const MRActuatorProfileGPU* actuatorProfiles [[buffer(9)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(10)]],
    device float* workingEffort [[buffer(11)]],
    device MRABABodyWrenchGPU* bodyWrenches [[buffer(12)]],
    device const MRTaskStateGPU* taskStates [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        pass.physicsSubstep >= worldDispatch.physicsSubsteps ||
        dispatch.counts.x != worldDispatch.environmentCount ||
        dispatch.counts.z != worldDispatch.nq ||
        dispatch.counts.w != worldDispatch.nv ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        dispatch.worldFingerprint != program.worldFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.layout.w < 2u) {
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(arena, program.offsets0.x);
    device const MRTaskActuatorTermGPU* terms =
        taskTable<MRTaskActuatorTermGPU>(arena, program.actuatorTerms.x);
    const uint bodyCount = dispatch.sampling.z;
    // MetalWorld clears the complete global wrench arena exactly once before
    // all actuator and multiphysics producers execute. Every producer is
    // therefore additive and cannot erase a reaction authored by another
    // subsystem.
    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint filterSlot = program.layout.w - 1u;
    const uint historyBase =
        environment * program.layout.w * program.counts0.x +
        filterSlot * program.counts0.x;
    for (uint action = 0u; action < program.counts0.x; ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        const uint kind = binding.actuator.x;
        if (kind == MR_TASK_ACTUATOR_JOINT_POSITION ||
            kind == MR_TASK_ACTUATOR_GRIPPER_POSITION ||
            kind == MR_TASK_ACTUATOR_FLAPPING_POSITION ||
            kind == MR_TASK_ACTUATOR_ROTOR_MIXER ||
            kind == MR_TASK_ACTUATOR_MILLARD_EXCITATION) {
            continue;
        }
        const float filtered = actionHistory[historyBase + action];
        if (kind == MR_TASK_ACTUATOR_JOINT_VELOCITY ||
            kind == MR_TASK_ACTUATOR_JOINT_EFFORT) {
            const uint dofIndex = binding.indices.y;
            const uint velocityIndex = binding.indices.w;
            if (dofIndex >= dispatch.counts.w ||
                velocityIndex >= dispatch.counts.w) {
                continue;
            }
            device const MRDofPropertiesGPU& dof = dofs[dofIndex];
            const float velocity = vState[vBase + velocityIndex];
            const float target = clamp(
                binding.parameters.x * filtered,
                binding.parameters.y,
                binding.parameters.z
            );
            float effort = kind == MR_TASK_ACTUATOR_JOINT_VELOCITY
                ? binding.drive.y * (target - velocity)
                : target;
            effort += actuatorDryFriction(dof, velocity, effort);
            const float envelope = actuatorEffortEnvelope(
                dof, actuatorProfiles[dofIndex], velocity);
            workingEffort[vBase + velocityIndex] =
                clamp(effort, -envelope, envelope);
            continue;
        }
        if (kind == MR_TASK_ACTUATOR_TENDON_POSITION) {
            const uint first = binding.indices.y;
            const uint count = binding.indices.z;
            if (first > program.actuatorTerms.y ||
                count > program.actuatorTerms.y - first) {
                continue;
            }
            float length = 0.0f;
            float rate = 0.0f;
            for (uint termIndex = 0u; termIndex < count; ++termIndex) {
                const MRTaskActuatorTermGPU term = terms[first + termIndex];
                length += term.coefficient.x *
                    qState[qBase + term.indices.y];
                rate += term.coefficient.x *
                    vState[vBase + term.indices.z];
            }
            const float target =
                binding.drive.w + binding.parameters.x * filtered;
            const float tension = clamp(
                binding.drive.x * (target - length) -
                    binding.drive.y * rate,
                -binding.drive.z,
                binding.drive.z
            );
            for (uint termIndex = 0u; termIndex < count; ++termIndex) {
                const MRTaskActuatorTermGPU term = terms[first + termIndex];
                const uint dofIndex = term.indices.x;
                const uint velocityIndex = term.indices.z;
                const float velocity = vState[vBase + velocityIndex];
                device const MRDofPropertiesGPU& dof = dofs[dofIndex];
                float effort = term.coefficient.x * tension;
                effort += actuatorDryFriction(dof, velocity, effort);
                effort += workingEffort[vBase + velocityIndex];
                const float envelope = actuatorEffortEnvelope(
                    dof, actuatorProfiles[dofIndex], velocity);
                workingEffort[vBase + velocityIndex] =
                    clamp(effort, -envelope, envelope);
            }
            continue;
        }
        if (kind == MR_TASK_ACTUATOR_BODY_WRENCH &&
            (worldDispatch.flags & MR_METAL_WORLD_HAS_BODY_WRENCHES) != 0u &&
            binding.actuator.y < bodyCount && binding.actuator.z < 6u) {
            const uint bodyIndex = binding.actuator.y;
            const uint wrenchIndex = environment * bodyCount + bodyIndex;
            const MRArticulatedBodyPoseGPU pose =
                bodyPoses[wrenchIndex];
            const uint component = binding.actuator.z;
            float requested = filtered;
            if ((program.schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_APPROACH_ENVELOPE) != 0u &&
                action == 14u && component == 4u &&
                taskStates[environment].commandExtension.w >=
                    (21.0f / 32.0f)) {
                const float4 orientation = rootOrientation(
                    program,
                    qState + qBase
                );
                const float pitch = asin(clamp(
                    2.0f * (
                        orientation.w * orientation.y -
                        orientation.z * orientation.x
                    ),
                    -1.0f,
                    1.0f
                ));
                const float bodyPitchRate = rotateInverse(
                    orientation,
                    rootWorldAngularVelocity(program, vState + vBase)
                ).y;
                // Rejected soft and late coupled envelopes activated too late
                // to remove every failure in 640 full journeys. V7 makes the
                // coupled state trigger eligible at 18 seconds and retains
                // this direct pitch loop from the authored approach boundary
                // through landed hold. The only physical authority remains
                // the existing 0.020 N m actuator.
                const float safetyRequested = clamp(
                    -2.40f * pitch - 0.25f * bodyPitchRate,
                    -1.0f,
                    1.0f
                );
                requested = clamp(
                    safetyRequested + 0.15f * filtered,
                    -1.0f,
                    1.0f
                );
            }
            float3 local = float3(0.0f);
            local[component % 3u] = binding.parameters.x * requested;
            const float3 world = rotate(pose.orientation, local);
            MRABABodyWrenchGPU wrench = bodyWrenches[wrenchIndex];
            if (component < 3u) {
                wrench.force.xyz += world;
            } else {
                wrench.torque.xyz += world;
            }
            bodyWrenches[wrenchIndex] = wrench;
        }
    }
}

kernel void mr_locomotion_task_measure_effort(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    constant MRMetalWorldPassGPU& pass [[buffer(3)]],
    device const float* vState [[buffer(4)]],
    device const float* workingEffort [[buffer(5)]],
    device MRTaskStateGPU* taskStates [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }
    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    const uint vBase = environment * dispatch.counts.w;
    float mechanicalPower = 0.0f;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        const MRTaskActionBindingGPU binding = actions[action];
        if (binding.indices.w == MR_INVALID_INDEX) {
            continue;
        }
        const uint velocityIndex = binding.indices.w;
        mechanicalPower += abs(
            workingEffort[vBase + velocityIndex] *
            vState[vBase + velocityIndex]
        );
    }
    MRTaskStateGPU state = taskStates[environment];
    state.airReturnTracking.x = mechanicalPower;
    taskStates[environment] = state;
}

kernel void mr_locomotion_task_complete(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device const uchar* arena [[buffer(2)]],
    device const MRTaskEvidenceStateGPU* evidenceState
        [[buffer(3)]],
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(4)]],
    constant MRMetalWorldPassGPU& pass [[buffer(5)]],
    device const float* qState [[buffer(6)]],
    device const float* vState [[buffer(7)]],
    device const MRBodyStateGPU* bodyStates [[buffer(8)]],
    device const MRContactConstraintGPU* contacts [[buffer(9)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses
        [[buffer(10)]],
    device const MRMetalWorldStatusGPU* worldStatuses
        [[buffer(11)]],
    device const MRBodyStateGPU* sceneState [[buffer(12)]],
    device const MRDofPropertiesGPU* dofs [[buffer(13)]],
    device const float* defaultQ [[buffer(14)]],
    device MRTaskStateGPU* taskStates [[buffer(15)]],
    device float* actionHistory [[buffer(16)]],
    device float* actorHistory [[buffer(17)]],
    device float* cleanHistory [[buffer(18)]],
    device float* previousJointVelocity [[buffer(19)]],
    device const float* sensorBias [[buffer(20)]],
    device const float4* bodyParameters [[buffer(21)]],
    device const float4* controllerParameters [[buffer(22)]],
    device float* compactContact [[buffer(23)]],
    device MRTaskTransitionGPU* transitions [[buffer(24)]],
    device const MRShapeGPU* shapes [[buffer(25)]],
    device const MRGeometryHeaderGPU* geometryHeaders
        [[buffer(26)]],
    device const float4* geometryVertices [[buffer(27)]],
    device float* actorObservations [[buffer(28)]],
    device float* criticObservations [[buffer(29)]],
    device float* criticHistory [[buffer(30)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.counts.x ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint !=
            program.taskFingerprint ||
        dispatch.worldFingerprint !=
            program.worldFingerprint ||
        program.articulation.z !=
            MR_TASK_PROGRAM_ABI_VERSION) {
        return;
    }

    device const MRTaskActionBindingGPU* actions =
        taskTable<MRTaskActionBindingGPU>(
            arena,
            program.offsets0.x
        );
    device const MRTaskObservationOperatorGPU*
        actorOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.y
            );
    device const MRTaskObservationOperatorGPU*
        criticOperators =
            taskTable<MRTaskObservationOperatorGPU>(
                arena,
                program.offsets0.z
            );
    device const MRTaskContactGroupGPU* contactGroups =
        taskTable<MRTaskContactGroupGPU>(
            arena,
            program.offsets0.w
        );
    device const uint* contactMembers =
        taskTable<uint>(arena, program.offsets1.x);
    device const float* contactMemberRadii =
        taskTable<float>(arena, program.offsets3.w);
    device const MRTaskIndexGroupGPU* jointGroups =
        taskTable<MRTaskIndexGroupGPU>(
            arena,
            program.offsets1.y
        );
    device const uint* jointMembers =
        taskTable<uint>(arena, program.offsets1.z);
    device const MRTaskRewardOperatorGPU* rewards =
        taskTable<MRTaskRewardOperatorGPU>(
            arena,
            program.offsets1.w
        );
    device const MRTaskTerminationOperatorGPU*
        terminations =
            taskTable<MRTaskTerminationOperatorGPU>(
                arena,
                program.offsets2.x
            );
    device const float4* terrainSamples =
        taskTable<float4>(arena, program.offsets2.w);
    device const float4* commandCurriculum =
        taskTable<float4>(arena, program.offsets3.y);
    device const MRTaskImpactEventGPU* impactEvents =
        taskTable<MRTaskImpactEventGPU>(
            arena,
            program.offsets3.z
        );
    device const float* interactionJointTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.y
        );
    device const float* interactionRootTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets0.x
        );
    device const MRTaskInteractionContactGPU*
        interactionContacts =
            taskTable<MRTaskInteractionContactGPU>(
                arena,
                program.interactionOffsets0.z
            );
    device const MRTaskInteractionSampleGPU*
        interactionSamples =
            taskTable<MRTaskInteractionSampleGPU>(
                arena,
                program.interactionOffsets0.w
            );
    device const float* interactionContactTargets =
        taskTable<float>(
            arena,
            program.interactionOffsets1.x
        );
    device const float* interactionContactTolerances =
        taskTable<float>(
            arena,
            program.interactionOffsets1.y
        );
    device const uint* outcomeRewardOperations =
        taskTable<uint>(arena, program.interactionOffsets1.z);

    const uint qBase = environment * dispatch.counts.z;
    const uint vBase = environment * dispatch.counts.w;
    const uint bodyBase =
        environment * dispatch.strides.y;
    const uint sceneBase =
        environment * dispatch.strides.w;
    const uint delayBase =
        environment *
        program.layout.w *
        program.counts0.x;
    device const float* rawPolicyActions =
        actionHistory +
        dispatch.counts.x * program.layout.w * program.counts0.x;
    const uint historyElements =
        program.layout.x * program.layout.y;
    const uint historyBase =
        environment * historyElements;
    const uint criticHistoryElements =
        program.counts0.z * program.articulation.w;
    const uint criticHistoryBase =
        environment * criticHistoryElements;
    const uint previousVelocityBase =
        environment * program.counts0.x;
    const uint biasBase =
        environment * program.counts2.z;
    const uint bodyParameterBase =
        environment * dispatch.strides.y;
    const uint compactBase =
        environment * program.layout.z;
    const uint contactBase =
        environment * contactDispatch.constraintStride;
    MRTaskStateGPU state = taskStates[environment];
    const uint referenceFrame = interactionFrame(
        program,
        state,
        dispatch.timing.x
    );
    const uint nextReferenceFrame = interactionNextFrame(
        program,
        referenceFrame
    );
    const float referenceBlend = interactionFrameBlend(
        program,
        state,
        dispatch.timing.x
    );
    const uint curriculum = min(
        state.episode.z,
        program.schedule.z - 1u
    );
    const MRMetalWorldStatusGPU worldStatus =
        worldStatuses[environment];
    const MRMetalWorldContactStatusGPU contactStatus =
        contactStatuses[environment];
    const bool physicsError =
        worldStatus.code != MR_STEP_SUCCESS ||
        contactStatus.code != MR_STEP_SUCCESS;
    const uint activeContacts = physicsError
        ? 0u
        : min(
              contactStatus.activeContacts,
              contactDispatch.constraintCapacity
          );

    for (uint groupIndex = 0u;
         groupIndex < program.counts0.w;
         ++groupIndex) {
        const MRTaskContactGroupGPU group =
            contactGroups[groupIndex];
        const uint metric = compactBase + group.members.w;
        if ((group.members.z &
             MR_TASK_CONTACT_SUPPORT) != 0u) {
            // Preserve previous air time in slot 2. Other slots become
            // impulse/COP accumulators until the reduction is finalized.
            compactContact[metric + 0u] = 0.0f;
            compactContact[metric + 1u] = 0.0f;
            compactContact[metric + 3u] = 0.0f;
            compactContact[metric + 4u] = 0.0f;
            compactContact[metric + 5u] = 0.0f;
            for (uint cell = 0u;
                 cell < group.supportPatch.z;
                 ++cell) {
                compactContact[
                    compactBase + group.supportPatch.w + cell
                ] = 0.0f;
            }
        } else if ((group.members.z &
                    MR_TASK_CONTACT_FORBIDDEN) != 0u) {
            compactContact[metric] = 0.0f;
        }
        const uint wrench =
            compactBase + group.reference.y;
        for (uint component = 0u;
             component < 6u;
             ++component) {
            compactContact[wrench + component] = 0.0f;
        }
    }

    const uint robotFirst = program.articulation.x;
    const uint robotEnd =
        robotFirst + program.articulation.y;
    for (uint contact = 0u;
         contact < activeContacts;
         ++contact) {
        const MRContactConstraintGPU constraint =
            contacts[contactBase + contact];
        const bool robotA =
            constraint.bodyA >= robotFirst &&
            constraint.bodyA < robotEnd;
        const bool robotB =
            constraint.bodyB >= robotFirst &&
            constraint.bodyB < robotEnd;
        if (robotA == robotB) {
            continue;
        }
        const uint robotBody =
            robotA ? constraint.bodyA : constraint.bodyB;
        const float impulse = abs(constraint.impulses.x);
        const float3 normal = normalizedOr(
            constraint.normal.xyz,
            float3(0.0f, 0.0f, 1.0f)
        );
        const float3 authoredTangent =
            constraint.tangent.xyz -
            normal * dot(normal, constraint.tangent.xyz);
        const float3 tangent = normalizedOr(
            authoredTangent,
            stableContactTangent(normal)
        );
        const float3 bitangent = cross(normal, tangent);
        const float3 impulseOnA = -(
            normal * constraint.impulses.x +
            tangent * constraint.impulses.y +
            bitangent * constraint.impulses.z
        );
        const float3 robotImpulse =
            robotA ? impulseOnA : -impulseOnA;
        const float3 forceWorld =
            robotImpulse / dispatch.timing.y;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            device const MRTaskContactGroupGPU& group =
                contactGroups[groupIndex];
            if (!bodyMember(
                    robotBody,
                    group,
                    contactMembers
                )) {
                continue;
            }
            const uint metric =
                compactBase + group.members.w;
            const MRBodyStateGPU referenceBody =
                bodyStates[
                    bodyBase + group.reference.x
                ];
            const float3 referencePosition =
                referenceBody.position.xyz +
                rotate(
                    referenceBody.orientation,
                    group.localReference.xyz
                );
            const float torsionalSign =
                robotA ? -1.0f : 1.0f;
            const float3 torqueWorld =
                cross(
                    constraint.pointAndSeparation.xyz -
                        referencePosition,
                    forceWorld
                ) +
                torsionalSign *
                    normal *
                    constraint.impulses.w /
                    dispatch.timing.y;
            const float3 forceLocal = rotateInverse(
                referenceBody.orientation,
                forceWorld
            );
            const float3 contactLocal = rotateInverse(
                referenceBody.orientation,
                constraint.pointAndSeparation.xyz -
                    referencePosition
            );
            const float3 torqueLocal = rotateInverse(
                referenceBody.orientation,
                torqueWorld
            );
            const uint wrench =
                compactBase + group.reference.y;
            compactContact[wrench + 0u] += forceLocal.x;
            compactContact[wrench + 1u] += forceLocal.y;
            compactContact[wrench + 2u] += forceLocal.z;
            compactContact[wrench + 3u] += torqueLocal.x;
            compactContact[wrench + 4u] += torqueLocal.y;
            compactContact[wrench + 5u] += torqueLocal.z;
            if ((group.members.z &
                 MR_TASK_CONTACT_SUPPORT) != 0u) {
                compactContact[metric + 0u] += impulse;
                compactContact[metric + 4u] +=
                    impulse * contactLocal.x;
                compactContact[metric + 5u] +=
                    impulse * contactLocal.y;
                if (group.supportPatch.z != 0u &&
                    contactLocal.x >=
                        group.supportPatchBounds.x &&
                    contactLocal.x <=
                        group.supportPatchBounds.z &&
                    contactLocal.y >=
                        group.supportPatchBounds.y &&
                    contactLocal.y <=
                        group.supportPatchBounds.w) {
                    const float2 normalized = clamp(
                        (
                            contactLocal.xy -
                            group.supportPatchBounds.xy
                        ) /
                        (
                            group.supportPatchBounds.zw -
                            group.supportPatchBounds.xy
                        ),
                        float2(0.0f),
                        float2(0.999999f)
                    );
                    const uint column = min(
                        uint(normalized.x *
                            float(group.supportPatch.x)),
                        group.supportPatch.x - 1u
                    );
                    const uint row = min(
                        uint(normalized.y *
                            float(group.supportPatch.y)),
                        group.supportPatch.y - 1u
                    );
                    compactContact[
                        compactBase + group.supportPatch.w +
                        row * group.supportPatch.x + column
                    ] += impulse;
                }
            }
            if ((group.members.z &
                 MR_TASK_CONTACT_FORBIDDEN) != 0u) {
                compactContact[metric] = max(
                    compactContact[metric],
                    float(
                        impulse / dispatch.timing.y >
                        program.dynamics.y
                    )
                );
            }
        }
    }

    for (uint groupIndex = 0u;
         groupIndex < program.counts0.w;
         ++groupIndex) {
        const MRTaskContactGroupGPU group =
            contactGroups[groupIndex];
        if ((group.members.z &
             MR_TASK_CONTACT_SUPPORT) == 0u) {
            continue;
        }
        const uint metric =
            compactBase + group.members.w;
        const float impulse =
            compactContact[metric + 0u];
        const float previousAir =
            compactContact[metric + 2u];
        const MRBodyStateGPU body =
            bodyStates[
                bodyBase + group.reference.x
            ];
        const float3 offset = rotate(
            body.orientation,
            group.localReference.xyz
        );
        const float3 kinematicOffset = rotate(
            body.orientation,
            group.kinematicReference.xyz
        );
        const float3 position =
            body.position.xyz + offset;
        const float3 kinematicVelocity =
            body.linearVelocityAndInverseMass.xyz +
            cross(
                body.angularVelocity.xyz,
                kinematicOffset
            );
        const float force =
            impulse / dispatch.timing.y;
        const bool contact =
            force > program.dynamics.y;
        const float slip =
            contact
            ? length(kinematicVelocity.xy)
            : 0.0f;
        const float airTime =
            contact
            ? 0.0f
            : previousAir + dispatch.timing.x;
        const float clearance =
            position.z -
            group.localReference.w -
            surfaceHeight(
                program,
                shapes,
                geometryHeaders,
                geometryVertices,
                sceneState + sceneBase,
                position.xy
            );
        const float2 cop =
            impulse > 1.0e-6f
            ? float2(
                  compactContact[metric + 4u],
                  compactContact[metric + 5u]
              ) / impulse
            : float2(0.0f);
        compactContact[metric + 0u] = force;
        compactContact[metric + 1u] = slip;
        compactContact[metric + 2u] = airTime;
        compactContact[metric + 3u] = clearance;
        compactContact[metric + 4u] = cop.x;
        compactContact[metric + 5u] = cop.y;
        for (uint cell = 0u;
             cell < group.supportPatch.z;
             ++cell) {
            compactContact[
                compactBase + group.supportPatch.w + cell
            ] /= dispatch.timing.y;
        }
    }

    device const float* q = qState + qBase;
    device const float* v = vState + vBase;
    const float4 orientation = rootOrientation(program, q);
    const float3 rootLinearVelocity =
        rootWorldLinearVelocity(program, q, v);
    const float3 baseLinear = rotateInverse(
        orientation,
        rootLinearVelocity
    );
    const float3 baseAngular = rotateInverse(
        orientation,
        rootWorldAngularVelocity(program, v)
    );
    const float3 gravity = normalizedOr(
        rotateInverse(
            orientation,
            float3(0.0f, 0.0f, -1.0f)
        ),
        float3(0.0f, 0.0f, -1.0f)
    );
    const float height = rootHeight(
        program,
        q
    );
    const float tilt = atan2(
        length(gravity.xy),
        max(-gravity.z, 1.0e-6f)
    );
    float2 yawBasis = float2(
        1.0f -
            2.0f *
                (
                    orientation.y * orientation.y +
                    orientation.z * orientation.z
                ),
        2.0f *
            (
                orientation.w * orientation.z +
                orientation.x * orientation.y
            )
    );
    yawBasis *= rsqrt(
        max(dot(yawBasis, yawBasis), 1.0e-12f)
    );
    bool figureEightConfigured = false;
    bool figureEightActive = false;
    float figureEightPathErrorSquared = 0.0f;
    const bool avianGroundCurriculum =
        (program.schedule.w & MR_TASK_PROGRAM_AVIAN_GROUND_CURRICULUM) != 0u;
    const uint avianCurriculumBand = state.episode.z;
    const bool avianJourneyShowcase =
        (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
        avianCurriculumBand >= 4u;
    const bool avianJourneyTakeoffCruise =
        avianJourneyShowcase && avianCurriculumBand == 4u;
    const float avianJourneyPhase = avianJourneyShowcase
        ? state.commandExtension.w
        : 0.0f;
    // The first two avian bands are supported standing and walking, not
    // flight.  The 0.1873 m target is the measured mean root height from the
    // held default-pose standing rollout on the imported hybrid; it prevents
    // an airborne height objective from competing with legged locomotion.
    constexpr float avianGroundRootHeightTarget = 0.1873f;
    const bool avianSupportedGroundStage =
        avianGroundCurriculum &&
        (avianCurriculumBand < 2u ||
         (avianJourneyShowcase &&
          (avianJourneyTakeoffCruise
              ? avianJourneyPhase < 0.03125f
              : (avianJourneyPhase < 0.15625f ||
                 avianJourneyPhase >= 0.84375f))));
    const bool avianCrowGroundTiltEnvelope =
        avianGroundCurriculum && avianCurriculumBand == 1u &&
        (program.schedule.w &
         MR_TASK_PROGRAM_AVIAN_CROW_GROUND_TILT_ENVELOPE) != 0u;
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation = rewards[rewardIndex];
        if (operation.source.x !=
                MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING) {
            continue;
        }
        figureEightConfigured = true;
        const float extentX = operation.parameters.y;
        const float extentY = operation.parameters.z;
        const float cycleSeconds = operation.parameters.w;
        const float takeoffSeconds = operation.auxiliary.x;
        const float episodeSeconds =
            float(state.episode.x + 1u) * dispatch.timing.x;
        const float avianJourneyNormalizedPhase =
            clamp(episodeSeconds / 32.0f, 0.0f, 1.0f);
        if (avianJourneyTakeoffCruise && episodeSeconds < 1.0f) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                avianJourneyNormalizedPhase
            );
            state.commandAndPhase.xyz = float3(0.0f);
            break;
        }
        if (avianJourneyTakeoffCruise) {
            state.commandExtension.w = avianJourneyNormalizedPhase;
            state.commandAndPhase.xyz = float3(0.35f, 0.0f, 0.0f);
            break;
        }
        if (avianJourneyShowcase && episodeSeconds < 2.0f) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                avianJourneyNormalizedPhase
            );
            state.commandAndPhase.xyz = float3(0.0f);
            break;
        }
        if (avianJourneyShowcase && episodeSeconds < 5.0f) {
            state.commandExtension.w = avianJourneyNormalizedPhase;
            state.commandAndPhase.xyz = float3(0.22f, 0.0f, 0.0f);
            break;
        }
        if (avianJourneyShowcase && episodeSeconds < 9.0f) {
            state.commandExtension.w = avianJourneyNormalizedPhase;
            state.commandAndPhase.xyz = float3(0.35f, 0.0f, 0.0f);
            break;
        }
        if (avianJourneyShowcase && episodeSeconds < 21.0f) {
            // Two independently qualified constant-curvature arcs form the
            // journey's signed turn sequence. This stays inside the crow's
            // qualified 0.35 m/s envelope and avoids hiding an infeasible
            // high-speed lemniscate behind a path reward.
            const float turnDirection = episodeSeconds < 15.0f
                ? 1.0f
                : -1.0f;
            state.commandExtension.w = avianJourneyNormalizedPhase;
            state.commandAndPhase.xyz = float3(
                0.35f,
                0.0f,
                0.12f * turnDirection
            );
            break;
        }
        if (avianJourneyShowcase && episodeSeconds >= 21.0f) {
            const bool landedHold = episodeSeconds >= 27.0f;
            state.commandExtension.w = avianJourneyNormalizedPhase;
            state.commandAndPhase.xyz = landedHold
                ? float3(0.0f)
                : float3(0.20f, 0.0f, 0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 0u) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            state.commandAndPhase.xyz = float3(0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 1u) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            // A modest target leaves forward progress, bilateral support,
            // and uprightness as the learning signal while wings remain
            // folded in the ground curriculum.
            state.commandAndPhase.xyz = float3(0.22f, 0.0f, 0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 2u) {
            const float verticalRate = clamp(
                (height - state.airReturnTracking.y) / dispatch.timing.x,
                -6.0f,
                6.0f
            );
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                verticalRate
            );
            // Learn the vertical transition and a modest forward airspeed
            // before asking the same policy to solve a curved flight path.
            // The 0.35 m/s target matches the authored launch-tracking scale;
            // the figure-eight command is reserved for the later flight band.
            state.commandAndPhase.xyz = float3(0.35f, 0.0f, 0.0f);
            break;
        }
        if (avianGroundCurriculum && avianCurriculumBand == 3u) {
            // Isolated cruise is straight and command-conditioned. Curvature
            // is deliberately deferred until the full-journey band so the
            // first neural gate diagnoses stabilization rather than path
            // planning and launch simultaneously.
            state.commandExtension.w = 0.0f;
            state.commandAndPhase.xyz = float3(0.35f, 0.0f, 0.0f);
            break;
        }
        if (episodeSeconds <= takeoffSeconds) {
            state.commandExtension = float4(
                rootWorldPosition(program, q).xy,
                0.0f,
                0.0f
            );
            state.commandAndPhase.xyz = float3(0.9f, 0.0f, 0.0f);
            break;
        }
        figureEightActive = true;
        const float omega = kTwoPi / max(cycleSeconds, 1.0e-4f);
        const float theta = fmod(
            state.commandExtension.z + omega * dispatch.timing.x,
            kTwoPi
        );
        // Rotate the Gerono tangent so the first post-takeoff command is
        // forward in world X instead of diagonally across the crossing.
        const float initialAngle = atan2(2.0f * extentY, extentX);
        const float c = cos(-initialAngle);
        const float s = sin(-initialAngle);
        const float2 rawVelocity = float2(
            extentX * cos(theta) * omega,
            2.0f * extentY * cos(2.0f * theta) * omega
        );
        const float2 rawAcceleration = float2(
            -extentX * sin(theta) * omega * omega,
            -4.0f * extentY * sin(2.0f * theta) * omega * omega
        );
        const float2 targetVelocity = float2(
            c * rawVelocity.x - s * rawVelocity.y,
            s * rawVelocity.x + c * rawVelocity.y
        );
        const float2 targetAcceleration = float2(
            c * rawAcceleration.x - s * rawAcceleration.y,
            s * rawAcceleration.x + c * rawAcceleration.y
        );
        state.commandExtension.xy += targetVelocity * dispatch.timing.x;
        state.commandExtension.zw = float2(
            theta,
            avianJourneyShowcase
                ? avianJourneyNormalizedPhase
                : 1.0f
        );
        const float2 pathError =
            state.commandExtension.xy - rootWorldPosition(program, q).xy;
        const float2 correctedVelocity =
            targetVelocity + clamp(pathError * 0.70f, -2.0f, 2.0f);
        state.commandAndPhase.xy = float2(
            yawBasis.x * correctedVelocity.x + yawBasis.y * correctedVelocity.y,
            -yawBasis.y * correctedVelocity.x + yawBasis.x * correctedVelocity.y
        );
        state.commandAndPhase.z = clamp(
            (targetVelocity.x * targetAcceleration.y -
             targetVelocity.y * targetAcceleration.x) /
                max(dot(targetVelocity, targetVelocity), 1.0e-4f),
            -2.0f,
            2.0f
        );
        figureEightPathErrorSquared = dot(pathError, pathError);
        break;
    }
    const float2 yawFrameLinear = float2(
        yawBasis.x *
            rootLinearVelocity.x +
            yawBasis.y *
                rootLinearVelocity.y,
        -yawBasis.y *
            rootLinearVelocity.x +
            yawBasis.x *
                rootLinearVelocity.y
    );
    if ((program.schedule.w &
         MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION) != 0u) {
        uint navigationCourseStart = MR_INVALID_INDEX;
        for (uint rewardIndex = 0u;
             rewardIndex < program.counts1.w;
             ++rewardIndex) {
            const MRTaskRewardOperatorGPU operation = rewards[rewardIndex];
            if (operation.source.x ==
                    MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS ||
                operation.source.x ==
                    MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) {
                navigationCourseStart = operation.source.z;
                break;
            }
        }
        const uint waypoint = min(
            uint(max(state.navigation.z, 0.0f)),
            kCrowNavigationWaypointCount
        );
        if (navigationCourseStart != MR_INVALID_INDEX &&
            waypoint < kCrowNavigationWaypointCount) {
            const float3 target = crowNavigationWaypointTarget(
                sceneState + sceneBase,
                navigationCourseStart,
                waypoint
            );
            const float3 rootLocalDelta = rotateInverse(
                orientation,
                target - rootWorldPosition(program, q)
            );
            const float planarDistance = length(rootLocalDelta.xy);
            const float2 direction = planarDistance > 1.0e-5f
                ? rootLocalDelta.xy / planarDistance
                : float2(1.0f, 0.0f);
            const float speed = 0.35f * smoothstep(
                0.10f,
                0.60f,
                planarDistance
            );
            const float headingError = atan2(
                rootLocalDelta.y,
                rootLocalDelta.x
            );
            float commandedHeadingError = headingError;
            if ((program.schedule.w &
                 MR_TASK_PROGRAM_AVIAN_CROW_NAVIGATION_TURN_PREVIEW_SLALOMS) !=
                    0u &&
                (waypoint == kCrowNavigationTurnPreviewWaypoint ||
                 waypoint == kCrowNavigationTurnPreviewWaypoint + 1u)) {
                // Hold the active waypoint as the positional target so the
                // authored 0.42 m reach sphere and its native progress reward
                // remain unchanged. Near either slalom, blend only the yaw
                // reference toward the alternating segment. This gives the
                // learned flier time to bank rather than demanding a
                // point-to-point heading reversal at either slalom crossing.
                const float3 nextTarget = crowNavigationWaypointTarget(
                    sceneState + sceneBase,
                    navigationCourseStart,
                    waypoint + 1u
                );
                const float3 localOutgoing = rotateInverse(
                    orientation,
                    nextTarget - target
                );
                const float outgoingLength = length(localOutgoing.xy);
                if (outgoingLength > 1.0e-5f) {
                    const float nearWaypoint = 1.0f - smoothstep(
                        0.16f,
                        0.70f,
                        planarDistance
                    );
                    const float outgoingHeading = atan2(
                        localOutgoing.y,
                        localOutgoing.x
                    );
                    const float headingDelta = atan2(
                        sin(outgoingHeading - headingError),
                        cos(outgoingHeading - headingError)
                    );
                    commandedHeadingError = headingError +
                        0.45f * nearWaypoint * headingDelta;
                }
            }
            state.commandAndPhase.xyz = float3(
                speed * direction.x,
                speed * direction.y,
                clamp(1.40f * commandedHeadingError, -0.45f, 0.45f)
            );
            // The teacher reads this accepted-state feedback on the next
            // transaction. It remains a label and action blend only when the
            // rollout explicitly enables teacher sampling.
            state.commandExtension.xyz = float3(
                commandedHeadingError,
                baseAngular.z,
                yawFrameLinear.x
            );
        } else {
            state.commandAndPhase.xyz = float3(0.0f);
            state.commandExtension.xyz = float3(0.0f);
        }
    }
    if (avianJourneyTakeoffCruise) {
        // Feed the next teacher action from accepted state without a host
        // loop. xyz are heading error, yaw rate, and forward airspeed; w
        // remains the observable normalized journey phase.
        state.commandExtension.xyz = float3(
            atan2(yawBasis.y, yawBasis.x),
            baseAngular.z,
            yawFrameLinear.x
        );
    }
    if (avianGroundCurriculum && avianCurriculumBand == 2u) {
        // The action kernel consumes this accepted-state value on the next
        // control step.  Keeping the feedback inside task state avoids a
        // host readback loop and preserves transactional replay semantics.
        state.commandExtension.z = yawFrameLinear.x;
    }
    const float2 trackingDelta =
        yawFrameLinear - state.commandAndPhase.xy;
    const float trackingError =
        dot(trackingDelta, trackingDelta);
    const float yawDelta =
        baseAngular.z - state.commandAndPhase.z;
    const float yawError = yawDelta * yawDelta;
    if (avianJourneyShowcase && !avianJourneyTakeoffCruise) {
        // Accepted-state tracking feedback for the next native teacher action.
        // These lanes are task-private and never replace the actor's command,
        // phase, root-state, or velocity observations.
        const float pitchAngle = asin(clamp(
            2.0f * (
                orientation.w * orientation.y -
                orientation.z * orientation.x
            ),
            -1.0f,
            1.0f
        ));
        state.threatGeometry = float4(
            trackingDelta.x,
            trackingDelta.y,
            yawDelta,
            pitchAngle
        );
        // Internal accepted-state feedback for the invocation-scoped journey
        // teacher's pitch-rate term. Crow observation operators do not expose
        // this threat-teacher lane, and autonomous execution does not consume
        // the teacher action.
        state.threatTeacher.x = baseAngular.y;
    }

    float velocitySquared = 0.0f;
    float accelerationSquared = 0.0f;
    float actionRateSquared = 0.0f;
    float limitViolationSquared = 0.0f;
    const float mechanicalPower =
        state.airReturnTracking.x;
    const uint rawLastSlot = program.layout.w - 2u;
    const uint previousRawSlot = rawLastSlot - 1u;
    for (uint action = 0u;
         action < program.counts0.x;
         ++action) {
        const MRTaskActionBindingGPU binding =
            actions[action];
        const float currentAction = actionHistory[
            delayBase +
            rawLastSlot * program.counts0.x +
            action
        ];
        const float previousAction = actionHistory[
            delayBase +
            previousRawSlot * program.counts0.x +
            action
        ];
        const float actionDelta = currentAction - previousAction;
        actionRateSquared += actionDelta * actionDelta;
        if (binding.indices.y == MR_INVALID_INDEX ||
            binding.indices.z == MR_INVALID_INDEX ||
            binding.indices.w == MR_INVALID_INDEX) {
            continue;
        }
        const MRDofPropertiesGPU dof =
            dofs[binding.indices.y];
        const float position =
            qState[qBase + binding.indices.z];
        const float velocity =
            vState[vBase + binding.indices.w];
        const float acceleration =
            (
                velocity -
                previousJointVelocity[
                    previousVelocityBase + action
                ]
            ) /
            dispatch.timing.x;
        const float lower =
            max(dof.limits.x - position, 0.0f);
        const float upper =
            max(position - dof.limits.y, 0.0f);
        velocitySquared += velocity * velocity;
        accelerationSquared +=
            acceleration * acceleration;
        limitViolationSquared +=
            lower * lower + upper * upper;
    }

    const float commandMagnitude = length(
        state.commandAndPhase.xyz
    );
    const bool moving = commandMagnitude > 0.1f;
    float phase =
        state.commandAndPhase.w +
        kTwoPi * dispatch.timing.x /
            program.locomotion.y;
    phase = fmod(phase, kTwoPi);

    bool recoveryConfigured = false;
    float recoveryActivationTilt = 0.0f;
    float recoveryStableTilt = 0.0f;
    float recoveryStableDuration = 0.0f;
    uint recoveryContactGroup = MR_INVALID_INDEX;
    bool standingConfigured = false;
    float standingHeight = program.locomotion.x;
    float standingCosine = 0.8f;
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        if (!recoveryConfigured &&
            (operation.source.x ==
                MR_TASK_REWARD_RECOVERY_TILT_PROGRESS ||
            operation.source.x ==
                MR_TASK_REWARD_RECOVERY_COMPLETION)) {
            recoveryConfigured = true;
            recoveryActivationTilt = operation.parameters.y;
            recoveryStableTilt = operation.parameters.z;
            recoveryStableDuration = operation.parameters.w;
            recoveryContactGroup = operation.source.y;
        }
        if (operation.source.x ==
                MR_TASK_REWARD_STANDING_COMPLETION) {
            standingConfigured = true;
            standingHeight = operation.parameters.y;
            standingCosine = operation.parameters.z;
        }
    }
    bool recoveryTouch = false;
    if (recoveryContactGroup != MR_INVALID_INDEX) {
        const MRTaskContactGroupGPU group =
            contactGroups[recoveryContactGroup];
        const uint wrench = compactBase + group.reference.y;
        recoveryTouch = length(float3(
            compactContact[wrench + 0u],
            compactContact[wrench + 1u],
            compactContact[wrench + 2u]
        )) > program.dynamics.y;
    }
    const bool recoveryActiveBefore = state.recovery.w > 0.5f;
    const bool recoveryTouchBefore = state.recoveryStats.z != 0u;
    const bool eventSequenceAvailable =
        impactSequenceEnabled(state) &&
        program.counts3.x > 0u &&
        curriculum >= impactEvents[0].binding.z;
    const bool recoveryActivated =
        recoveryConfigured &&
        !recoveryActiveBefore &&
        (eventSequenceAvailable
            ? impactScene(state) != 0u &&
                recoveryTouch && !recoveryTouchBefore
            : (recoveryTouch && !recoveryTouchBefore) ||
                (state.recovery.x < recoveryActivationTilt &&
                 tilt >= recoveryActivationTilt));
    const bool recoveryActive =
        recoveryActiveBefore || recoveryActivated;
    const float recoveryPeakTilt = recoveryActive
        ? max(
              recoveryActiveBefore ? state.recovery.y : tilt,
              tilt
          )
        : 0.0f;
    const float recoveryStableTime = recoveryActive
        ? (tilt <= recoveryStableTilt
            ? (recoveryActiveBefore ? state.recovery.z : 0.0f) +
                dispatch.timing.x
            : 0.0f)
        : 0.0f;
    const bool recoveryCompleted =
        recoveryActive &&
        recoveryStableTime >= recoveryStableDuration;
    const uint recoveryEventCount =
        state.recoveryStats.x + uint(recoveryActivated);
    const uint recoveryCompletionCount =
        state.recoveryStats.y + uint(recoveryCompleted);
    const uint activeImpactScene = impactScene(state);
    const uint activeImpactOrder = impactOrder(state);
    uint activeImpactEvent = MR_INVALID_INDEX;
    if (activeImpactScene != 0u &&
        activeImpactOrder < program.counts3.x) {
        activeImpactEvent =
            (activeImpactOrder + impactOffset(state)) %
            program.counts3.x;
    }
    const bool impactContactLatchedBefore =
        (state.recoveryStats.w & kImpactContactLatched) != 0u;
    const bool impactContactPublishedBefore =
        (state.recoveryStats.w & kImpactContactPublished) != 0u;
    bool projectileContact = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        activeImpactEvent != MR_INVALID_INDEX) {
        const uint projectileBody =
            impactEvents[activeImpactEvent].binding.w;
        const uint articulationBodyBegin = program.articulation.x;
        const uint articulationBodyEnd =
            articulationBodyBegin + program.articulation.y;
        for (uint contact = 0u;
             contact < activeContacts && !projectileContact;
             ++contact) {
            const MRContactConstraintGPU constraint =
                contacts[contactBase + contact];
            const bool projectileA =
                constraint.bodyA == projectileBody;
            const bool projectileB =
                constraint.bodyB == projectileBody;
            if (projectileA == projectileB) {
                continue;
            }
            const uint other = projectileA
                ? constraint.bodyB
                : constraint.bodyA;
            projectileContact =
                other >= articulationBodyBegin &&
                other < articulationBodyEnd;
        }
    }
    const bool impactContactLatched =
        impactContactLatchedBefore || projectileContact;
    const bool newProjectileContact =
        impactContactLatched && !impactContactPublishedBefore;
    uint impactTransitionFlags =
        eventSequenceAvailable
        ? MR_TASK_IMPACT_SEQUENCE_ENABLED
        : 0u;
    impactTransitionFlags |=
        activeImpactScene != 0u &&
        recoveryActivated
        ? MR_TASK_IMPACT_TOUCH
        : 0u;
    impactTransitionFlags |= newProjectileContact
        ? MR_TASK_IMPACT_CONTACT
        : 0u;
    bool impactWindowElapsed = false;
    bool missedImpact = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        impactOrder(state) < program.counts3.x) {
        const MRTaskImpactEventGPU event =
            impactEvents[activeImpactEvent];
        impactWindowElapsed =
            float(state.status.w) * dispatch.timing.x >=
            event.gate.z;
        missedImpact =
            !recoveryActive &&
            impactWindowElapsed &&
            !impactContactLatched;
    }
    if (recoveryCompleted) {
        impactTransitionFlags |= MR_TASK_IMPACT_RECOVERED;
    }
    if (missedImpact) {
        impactTransitionFlags |= MR_TASK_IMPACT_MISSED;
    }
    bool projectileThreat = false;
    if (eventSequenceAvailable &&
        activeImpactScene != 0u &&
        activeImpactEvent != MR_INVALID_INDEX) {
        const MRBodyStateGPU projectile = sceneState[
            sceneBase + activeImpactScene - 1u
        ];
        const float3 rootPosition = float3(
            qState[qBase + program.root.z],
            qState[qBase + program.root.z + 1u],
            qState[qBase + program.root.z + 2u]
        );
        const float3 relativePosition =
            projectile.position.xyz - rootPosition;
        const float distance = max(
            length(relativePosition),
            1.0e-4f
        );
        const float3 relativeVelocity =
            projectile.linearVelocityAndInverseMass.xyz -
            rootLinearVelocity;
        const float closingSpeed =
            -dot(relativePosition, relativeVelocity) / distance;
        const float timeToClosestApproach =
            distance / max(closingSpeed, 1.0e-4f);
        projectileThreat =
            closingSpeed > 0.25f &&
            timeToClosestApproach <=
                impactEvents[activeImpactEvent].gate.z &&
            projectile.position.z > 0.10f;
    }
    float reward = 0.0f;
    float4 rewardBreakdown0 = float4(0.0f);
    float4 rewardBreakdown1 = float4(0.0f);
    float interactionTrackingSum = 0.0f;
    float interactionTrackingWeight = 0.0f;
    bool standingCompleted = false;
    bool restoredCompleted = false;
    uint recoveryOutcomeFlags = 0u;
    float4 outcomeChannels0 = float4(0.0f);
    float4 outcomeChannels1 = float4(0.0f);
    const bool neuralOnlyAvianJourney =
        (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
        (program.schedule.w &
         MR_TASK_PROGRAM_AVIAN_CROW_APPROACH_ENVELOPE) == 0u;
    const bool shadowApproachActive = neuralOnlyAvianJourney &&
        state.commandExtension.w >= (18.0f / 32.0f);
    const float shadowAbsolutePitch = abs(state.threatGeometry.w);
    bool navigationConfigured = false;
    uint navigationCourseStart = MR_INVALID_INDEX;
    float navigationReachRadius = 0.0f;
    float navigationMaximumStep = 0.0f;
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation = rewards[rewardIndex];
        if (operation.source.x !=
                MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS &&
            operation.source.x !=
                MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH) {
            continue;
        }
        navigationConfigured = true;
        navigationCourseStart = operation.source.z;
        navigationReachRadius = operation.parameters.y;
        navigationMaximumStep = operation.parameters.z;
        break;
    }
    float navigationStepProgress = 0.0f;
    bool navigationWaypointReached = false;
    if (navigationConfigured &&
        navigationCourseStart != MR_INVALID_INDEX) {
        const uint waypoint = min(
            uint(max(state.navigation.z, 0.0f)),
            kCrowNavigationWaypointCount
        );
        if (waypoint < kCrowNavigationWaypointCount) {
            const float3 target = crowNavigationWaypointTarget(
                sceneState + sceneBase,
                navigationCourseStart,
                waypoint
            );
            const float distance = length(
                target - rootWorldPosition(program, q)
            );
            if (state.navigation.x > 0.0f) {
                navigationStepProgress = clamp(
                    state.navigation.x - distance,
                    -navigationMaximumStep,
                    navigationMaximumStep
                );
            }
            const uint nextWaypoint = distance <= navigationReachRadius
                ? waypoint + 1u
                : waypoint;
            navigationWaypointReached = nextWaypoint != waypoint;
            state.navigation = float4(
                navigationWaypointReached ? 0.0f : distance,
                state.navigation.y + navigationStepProgress,
                float(nextWaypoint),
                nextWaypoint == kCrowNavigationWaypointCount ? 1.0f : 0.0f
            );
        }
    }
    for (uint rewardIndex = 0u;
         rewardIndex < program.counts1.w;
         ++rewardIndex) {
        const MRTaskRewardOperatorGPU operation =
            rewards[rewardIndex];
        float value = 0.0f;
        float interactionMetric = 0.0f;
        float interactionMetricWeight = 0.0f;
        switch (operation.source.x) {
        case MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS:
            value = navigationStepProgress / dispatch.timing.x;
            break;
        case MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH:
            value = navigationWaypointReached
                ? 1.0f / dispatch.timing.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING:
            value = figureEightActive
                ? exp(
                    -figureEightPathErrorSquared /
                    max(
                        0.0625f *
                            min(operation.parameters.y,
                                operation.parameters.z) *
                            min(operation.parameters.y,
                                operation.parameters.z),
                        0.25f
                    )
                )
                : 0.0f;
            break;
        case MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING:
            value = exp(
                -trackingError /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        case MR_TASK_REWARD_YAW_VELOCITY_TRACKING:
            value = exp(
                -yawError /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        case MR_TASK_REWARD_CONSTANT:
            value = 1.0f;
            break;
        case MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED:
            value = baseLinear.z * baseLinear.z;
            break;
        case MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED:
            value = dot(baseAngular.xy, baseAngular.xy);
            break;
        case MR_TASK_REWARD_TILT_SQUARED:
            if (neuralOnlyAvianJourney &&
                operation.parameters.y > 0.0f) {
                // A second, fingerprinted neural-journey tilt operator aligns
                // optimization with the held-out pitch envelope. It is a
                // reward only: no action or state is corrected here.
                const float excessPitch = shadowApproachActive
                    ? max(
                          shadowAbsolutePitch - operation.parameters.y,
                          0.0f
                      )
                    : 0.0f;
                value = max(operation.parameters.z, 1.0f) *
                    excessPitch * excessPitch;
                break;
            }
            // The isolated liftoff band is not yet a banking task.  Keep a
            // modest launch-pitch allowance, then penalize excess attitude so
            // height reward cannot be collected by an uncontrolled ballistic
            // climb.  The threshold comes from the held-out zero-policy mean
            // (about 0.29 rad) with margin for a physical push-off; later
            // figure-eight flight retains unconstrained learned banking.
            if (avianCrowGroundTiltEnvelope) {
                // Band-one policy selection rejects mean tilt above 0.0089
                // rad. Start this Crow-only hinge below that gate, so the
                // learned walking objective cannot buy tracking with the
                // large attitude excursion the held-out criterion forbids.
                // The -0.50 authored reward weight yields a -2.0 quadratic
                // coefficient above the 0.0075-rad training envelope.
                const float excessTilt = max(tilt - 0.0075f, 0.0f);
                value = 4.0f * excessTilt * excessTilt;
            } else if (avianGroundCurriculum && avianCurriculumBand == 2u) {
                const float excessTilt = max(tilt - 0.35f, 0.0f);
                // The task's existing -0.50 reward weight yields a -3.0
                // quadratic coefficient beyond the launch envelope.
                value = 6.0f * excessTilt * excessTilt;
            } else {
                value = avianSupportedGroundStage ? tilt * tilt : 0.0f;
            }
            break;
        case MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED:
            value = dot(gravity.xy, gravity.xy);
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED: {
            const float error =
                height - (avianSupportedGroundStage
                    ? avianGroundRootHeightTarget
                    : program.locomotion.x);
            value = error * error;
            break;
        }
        case MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED:
            if (avianSupportedGroundStage) {
                const float error =
                    height - avianGroundRootHeightTarget;
                value = exp(-error * error / 0.0025f);
            } else {
                value = clamp(
                    height / max(program.locomotion.x, 1.0e-6f),
                    0.0f,
                    1.0f
                );
            }
            break;
        case MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS:
            // Signed potential progress: rising earns exactly what settling
            // back down loses, preventing repeated bounce from manufacturing
            // height reward without a higher final physical state.
            value = avianSupportedGroundStage
                ? 0.0f
                : clamp(
                      (height - state.airReturnTracking.y) /
                          dispatch.timing.x,
                      -2.0f,
                      2.0f
                  );
            break;
        case MR_TASK_REWARD_UPRIGHTNESS:
            // Horizontal is not half-standing. Reward only the component of
            // body-up aligned with world-up; squaring retains a smooth slope
            // while concentrating value near an actually upright posture.
            value = pow(clamp(-gravity.z, 0.0f, 1.0f), 2.0f);
            break;
        case MR_TASK_REWARD_SUPPORT_CONTACT_COUNT: {
            float supportCount = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                supportCount += float(
                    compactContact[
                        compactBase + group.members.w
                    ] > program.dynamics.y
                );
                supportTotal += 1.0f;
            }
            value = supportCount / max(supportTotal, 1.0f);
            break;
        }
        case MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const MRBodyStateGPU body = bodyStates[
                bodyBase + group.reference.x
            ];
            const float bodyHeight =
                body.position.z +
                rotate(
                    body.orientation,
                    group.kinematicReference.xyz
                ).z;
            value = exp(
                clamp(
                    bodyHeight,
                    0.0f,
                    operation.parameters.y
                )
            ) - 1.0f;
            break;
        }
        case MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL: {
            float supportHeight = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const MRBodyStateGPU body = bodyStates[
                    bodyBase + group.reference.x
                ];
                supportHeight += max(
                    body.position.z +
                        rotate(
                            body.orientation,
                            group.kinematicReference.xyz
                        ).z,
                    0.0f
                );
                supportTotal += 1.0f;
            }
            value = exp(
                -operation.parameters.y *
                supportHeight / max(supportTotal, 1.0f)
            );
            break;
        }
        case MR_TASK_REWARD_BODY_UP_EXPONENTIAL:
            value = exp(-gravity.z);
            break;
        case MR_TASK_REWARD_STANDING_COMPLETION: {
            bool anySupported = false;
            uint supportTotal = 0u;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                anySupported = anySupported || compactContact[
                    compactBase + group.members.w
                ] > program.dynamics.y;
                ++supportTotal;
            }
            value = float(
                supportTotal > 0u &&
                anySupported &&
                height >= operation.parameters.y &&
                gravity.z <= -operation.parameters.z
            );
            standingCompleted = standingCompleted || value > 0.5f;
            break;
        }
        case MR_TASK_REWARD_RESTORATION: {
            bool anySupported = false;
            float supported = 0.0f;
            float supportTotal = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                const bool contact = compactContact[
                    compactBase + group.members.w
                ] > program.dynamics.y;
                anySupported = anySupported || contact;
                supported += float(contact);
                supportTotal += 1.0f;
            }
            float jointErrorSquared = 0.0f;
            for (uint action = 0u;
                 action < program.counts0.x;
                 ++action) {
                const MRTaskActionBindingGPU binding = actions[action];
                const float error =
                    q[binding.indices.z] - defaultQ[binding.indices.z];
                jointErrorSquared += error * error;
            }
            const float jointRms = sqrt(
                jointErrorSquared /
                max(float(program.counts0.x), 1.0f)
            );
            const float3 position = rootWorldPosition(program, q);
            const float3 targetPosition =
                rootWorldPosition(program, defaultQ);
            const float rootError = length(
                position.xy - targetPosition.xy
            );
            const float4 targetOrientation =
                rootOrientation(program, defaultQ);
            const float orientationCosine = clamp(
                abs(dot(orientation, targetOrientation)),
                0.0f,
                1.0f
            );
            const float generalizedSpeedRms = sqrt(
                (
                    velocitySquared +
                    dot(baseLinear, baseLinear) +
                    dot(baseAngular, baseAngular)
                ) /
                max(float(program.counts0.x + 6u), 1.0f)
            );
            const float heightScore = smoothstep(
                0.15f,
                max(standingHeight, 0.1501f),
                height
            );
            const float uprightScore = smoothstep(
                0.0f,
                max(standingCosine, 1.0e-4f),
                -gravity.z
            );
            const float supportScore = supportTotal > 0.0f
                ? supported / supportTotal
                : 0.0f;
            const float postureScore = exp(
                -jointErrorSquared /
                max(
                    float(program.counts0.x) *
                        operation.parameters.y *
                        operation.parameters.y,
                    1.0e-8f
                )
            );
            const float positionScore = exp(
                -(rootError * rootError) /
                max(
                    operation.parameters.z * operation.parameters.z,
                    1.0e-8f
                )
            );
            const float orientationScore = smoothstep(
                0.0f,
                operation.parameters.w,
                orientationCosine
            );
            const float stillnessScore = exp(
                -(generalizedSpeedRms * generalizedSpeedRms) /
                max(
                    operation.auxiliary.x * operation.auxiliary.x,
                    1.0e-8f
                )
            );
            const float standingQuality =
                heightScore * uprightScore * supportScore;
            value =
                0.20f * heightScore +
                0.20f * uprightScore +
                0.20f * supportScore +
                standingQuality *
                    (
                        0.15f * postureScore +
                        0.10f * positionScore +
                        0.10f * orientationScore +
                        0.05f * stillnessScore
                    );
            restoredCompleted = restoredCompleted ||
                standingConfigured &&
                supportTotal > 0.0f &&
                anySupported &&
                height >= standingHeight &&
                gravity.z <= -standingCosine &&
                jointRms <= operation.parameters.y &&
                rootError <= operation.parameters.z &&
                orientationCosine >= operation.parameters.w &&
                generalizedSpeedRms <= operation.auxiliary.x;
            break;
        }
        case MR_TASK_REWARD_WHOLE_BODY_RECOVERY: {
            const MRTaskContactGroupGPU assistGroup =
                contactGroups[operation.source.y];
            const MRTaskContactGroupGPU trunkGroup =
                contactGroups[operation.source.z];
            const uint assistWrench =
                compactBase + assistGroup.reference.y;
            const uint trunkWrench =
                compactBase + trunkGroup.reference.y;
            const bool assistContact = length(float3(
                compactContact[assistWrench + 0u],
                compactContact[assistWrench + 1u],
                compactContact[assistWrench + 2u]
            )) > program.dynamics.y;
            const bool trunkContact = length(float3(
                compactContact[trunkWrench + 0u],
                compactContact[trunkWrench + 1u],
                compactContact[trunkWrench + 2u]
            )) > program.dynamics.y;

            float supported = 0.0f;
            float supportTotal = 0.0f;
            float2 supportCenter = float2(0.0f);
            float copMarginSum = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z & MR_TASK_CONTACT_SUPPORT) == 0u) {
                    continue;
                }
                supportTotal += 1.0f;
                const uint metric = compactBase + group.members.w;
                const bool contact = compactContact[metric] >
                    program.dynamics.y;
                if (!contact) {
                    continue;
                }
                supported += 1.0f;
                const MRBodyStateGPU body = bodyStates[
                    bodyBase + group.reference.x
                ];
                supportCenter += (
                    body.position.xyz +
                    rotate(body.orientation, group.localReference.xyz)
                ).xy;
                const float2 cop = float2(
                    compactContact[metric + 4u],
                    compactContact[metric + 5u]
                );
                const float margin = min(
                    min(
                        cop.x - group.supportPatchBounds.x,
                        group.supportPatchBounds.z - cop.x
                    ),
                    min(
                        cop.y - group.supportPatchBounds.y,
                        group.supportPatchBounds.w - cop.y
                    )
                );
                const float halfMinimumExtent = 0.5f * min(
                    group.supportPatchBounds.z -
                        group.supportPatchBounds.x,
                    group.supportPatchBounds.w -
                        group.supportPatchBounds.y
                );
                copMarginSum += clamp(
                    margin / max(halfMinimumExtent, 1.0e-5f),
                    0.0f,
                    1.0f
                );
            }

            const bool anyFootSupport = supported > 0.0f;
            const float supportScore = supportTotal > 0.0f
                ? supported / supportTotal
                : 0.0f;
            const float copMarginScore = supported > 0.0f
                ? copMarginSum / supported
                : 0.0f;
            const float2 rootXY = rootWorldPosition(program, q).xy;
            const float supportDistance = anyFootSupport
                ? length(rootXY - supportCenter / supported)
                : operation.parameters.w;
            const float baseOverSupportScore = anyFootSupport
                ? exp(
                      -(supportDistance * supportDistance) /
                      max(
                          operation.parameters.w *
                              operation.parameters.w,
                          1.0e-8f
                      )
                  )
                : 0.0f;
            // Continuous from the floor: a crouch at 0.14 m is not placed in
            // a zero-gradient dead zone merely because standing is 0.65 m.
            const float heightScore = clamp(
                height / max(operation.parameters.y, 1.0e-4f),
                0.0f,
                1.0f
            );
            const float uprightScore = clamp(
                -gravity.z / max(operation.parameters.z, 1.0e-4f),
                0.0f,
                1.0f
            );
            const float generalizedSpeedRms = sqrt(
                (
                    velocitySquared +
                    dot(baseLinear, baseLinear) +
                    dot(baseAngular, baseAngular)
                ) /
                max(float(program.counts0.x + 6u), 1.0f)
            );
            const float stillnessScore = exp(
                -(generalizedSpeedRms * generalizedSpeedRms) /
                max(
                    operation.auxiliary.x * operation.auxiliary.x,
                    1.0e-8f
                )
            );
            const float trunkClear = float(!trunkContact);
            const float transferQuality =
                trunkClear * supportScore * baseOverSupportScore;
            const float braceQuality = float(assistContact) *
                (1.0f - transferQuality);
            const float riseQuality = transferQuality *
                sqrt(max(heightScore * uprightScore, 0.0f));
            const float quietStandQuality = riseQuality *
                copMarginScore * stillnessScore;

            // Additive physical qualities expose partial progress. Bracing
            // fades continuously only after load transfer, preventing it
            // from becoming the final local optimum.
            value =
                0.05f * braceQuality +
                0.10f * trunkClear +
                0.15f * supportScore +
                0.15f * baseOverSupportScore +
                0.10f * copMarginScore +
                0.25f * riseQuality +
                0.20f * quietStandQuality;

            recoveryOutcomeFlags |= assistContact
                ? MR_TASK_OUTCOME_RECOVERY_BRACE : 0u;
            recoveryOutcomeFlags |= !trunkContact
                ? MR_TASK_OUTCOME_TRUNK_CLEAR : 0u;
            recoveryOutcomeFlags |= anyFootSupport
                ? MR_TASK_OUTCOME_FOOT_SUPPORT : 0u;
            recoveryOutcomeFlags |=
                !trunkContact && anyFootSupport &&
                    baseOverSupportScore >= 0.50f &&
                    copMarginScore > 0.0f
                ? MR_TASK_OUTCOME_SUPPORT_TRANSFER : 0u;
            recoveryOutcomeFlags |=
                riseQuality >= 0.35f
                ? MR_TASK_OUTCOME_RECOVERY_RISE : 0u;
            recoveryOutcomeFlags |=
                height >= operation.parameters.y &&
                    -gravity.z >= operation.parameters.z &&
                    supportScore >= 0.999f &&
                    baseOverSupportScore >= 0.50f &&
                    copMarginScore > 0.0f &&
                    generalizedSpeedRms <= operation.auxiliary.x
                ? MR_TASK_OUTCOME_QUIET_STAND : 0u;
            break;
        }
        case MR_TASK_REWARD_RECOVERY_TILT_PROGRESS:
            value = recoveryActiveBefore
                ? clamp(
                      (state.recovery.x - tilt) /
                          dispatch.timing.x,
                      -2.0f,
                      2.0f
                  )
                : 0.0f;
            break;
        case MR_TASK_REWARD_RECOVERY_COMPLETION:
            // Reward weights are rates. Dividing the one-shot event by dt
            // makes its authored weight the integrated completion bonus.
            value = recoveryCompleted
                ? 1.0f / dispatch.timing.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_LINK_CLEARANCE_BARRIER: {
            const uint projectileScene = operation.source.z;
            const bool selected = !impactSequenceEnabled(state) ||
                impactScene(state) == projectileScene + 1u;
            if (!selected) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU projectile = sceneState[
                sceneBase + projectileScene
            ];
            const float3 projectileVelocity =
                projectile.linearVelocityAndInverseMass.xyz;
            const bool live = length(projectileVelocity) > 0.5f &&
                projectile.position.z > operation.parameters.z;
            if (!live) {
                value = 0.0f;
                break;
            }
            const MRTaskContactGroupGPU protectedGroup =
                contactGroups[operation.source.y];
            float mostBinding = 0.0f;
            for (uint local = 0u;
                 local < protectedGroup.members.y;
                 ++local) {
                const uint bodyIndex = contactMembers[
                    protectedGroup.members.x + local
                ];
                const float linkRadius = contactMemberRadii[
                    protectedGroup.members.x + local
                ];
                if (!(linkRadius > 0.0f)) {
                    continue;
                }
                const MRBodyStateGPU link = bodyStates[
                    bodyBase + bodyIndex
                ];
                const float3 relative =
                    projectile.position.xyz - link.position.xyz;
                const float distance = max(length(relative), 1.0e-6f);
                const float clearance =
                    distance -
                    (linkRadius + operation.parameters.z);
                const float closingRate = dot(
                    relative,
                    projectileVelocity -
                        link.linearVelocityAndInverseMass.xyz
                ) / distance;
                const float constraint = clamp(
                    closingRate +
                        operation.parameters.y * clearance,
                    -operation.parameters.w,
                    0.0f
                );
                mostBinding = min(mostBinding, constraint);
            }
            value = mostBinding;
            break;
        }
        case MR_TASK_REWARD_PROJECTILE_MISS:
            // Convert the event into a one-shot integrated bonus despite the
            // continuous-time TaskPack reward convention.
            value = missedImpact
                ? 1.0f / dispatch.timing.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_EVASION: {
            const uint projectileScene = operation.source.z;
            const bool selected = !impactSequenceEnabled(state) ||
                impactScene(state) == projectileScene + 1u;
            if (!selected) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU projectile = sceneState[
                sceneBase + projectileScene
            ];
            const bool live =
                length(projectile.linearVelocityAndInverseMass.xyz) > 0.5f;
            if (!live) {
                value = 0.0f;
                break;
            }
            const MRBodyStateGPU rootBody = bodyStates[
                bodyBase + program.root.y
            ];
            const float distance = length(
                projectile.position.xyz - rootBody.position.xyz
            );
            const float horizontalSpeedSquared = dot(
                rootBody.linearVelocityAndInverseMass.xy,
                rootBody.linearVelocityAndInverseMass.xy
            );
            const float positionBlend = operation.parameters.w;
            value =
                positionBlend *
                    (1.0f - exp(-operation.parameters.y * distance)) +
                (1.0f - positionBlend) *
                    exp(
                        -operation.parameters.z *
                            horizontalSpeedSquared
                    );
            break;
        }
        case MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS:
            value = !projectileThreat
                ? exp(
                      -operation.parameters.y *
                          dot(baseLinear.xy, baseLinear.xy)
                  )
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE:
            value = !projectileThreat ? actionRateSquared : 0.0f;
            break;
        case MR_TASK_REWARD_JOINT_CBF_CORRECTION:
            value = state.threatMetadata.x != MR_INVALID_INDEX
                ? state.threatTeacher.x
                : 0.0f;
            break;
        case MR_TASK_REWARD_JOINT_CBF_BUFFER:
            value = state.threatMetadata.x != MR_INVALID_INDEX
                ? state.threatTeacher.y
                : 0.0f;
            break;
        case MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE:
            if (state.threatMetadata.x != MR_INVALID_INDEX &&
                isfinite(state.threatGeometry.x) &&
                isfinite(state.threatGeometry.y)) {
                const float urgency = 1.0f - clamp(
                    state.threatGeometry.y /
                        max(program.threatTiming.y, 1.0e-6f),
                    0.0f,
                    1.0f
                );
                value = clamp(
                    urgency * tanh(
                        state.threatGeometry.x /
                            max(operation.parameters.y, 1.0e-4f)
                    ),
                    -1.0f,
                    1.0f
                );
            }
            break;
        case MR_TASK_REWARD_JOINT_VELOCITY_SQUARED:
            value = velocitySquared;
            break;
        case MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED:
            value = accelerationSquared;
            break;
        case MR_TASK_REWARD_ACTION_RATE_SQUARED:
            value = actionRateSquared;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED:
            value = limitViolationSquared;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE: {
            const float softFactor = clamp(
                operation.parameters.y,
                1.0e-6f,
                1.0f
            );
            for (uint action = 0u;
                 action < program.counts0.x;
                 ++action) {
                const MRTaskActionBindingGPU binding =
                    actions[action];
                const MRDofPropertiesGPU dof =
                    dofs[binding.indices.y];
                const float center =
                    0.5f * (dof.limits.x + dof.limits.y);
                const float halfRange =
                    0.5f *
                    (dof.limits.y - dof.limits.x) *
                    softFactor;
                const float position =
                    qState[qBase + binding.indices.z];
                value +=
                    max(center - halfRange - position, 0.0f) +
                    max(position - center - halfRange, 0.0f);
            }
            break;
        }
        case MR_TASK_REWARD_MECHANICAL_POWER:
            value = mechanicalPower;
            break;
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED: {
            const MRTaskIndexGroupGPU group =
                jointGroups[operation.source.y];
            float sum = 0.0f;
            for (uint local = 0u;
                 local < group.members.y;
                 ++local) {
                const uint action =
                    jointMembers[
                        group.members.x + local
                    ];
                const MRTaskActionBindingGPU binding =
                    actions[action];
                const float error =
                    qState[qBase + binding.indices.z] -
                    defaultQ[binding.indices.z];
                sum += error * error;
            }
            value = sum /
                max(float(group.members.y), 1.0f);
            break;
        }
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE: {
            const MRTaskIndexGroupGPU group =
                jointGroups[operation.source.y];
            float sum = 0.0f;
            for (uint local = 0u;
                 local < group.members.y;
                 ++local) {
                const uint action =
                    jointMembers[
                        group.members.x + local
                    ];
                const MRTaskActionBindingGPU binding =
                    actions[action];
                sum += abs(
                    qState[qBase + binding.indices.z] -
                    defaultQ[binding.indices.z]
                );
            }
            value = sum;
            break;
        }
        case MR_TASK_REWARD_GAIT_CONTACT_MATCH: {
            if (!moving) {
                value = 0.0f;
                break;
            }
            float matched = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const bool desired =
                    desiredSupportContact(group, phase);
                const bool actual =
                    compactContact[
                        compactBase + group.members.w
                    ] > program.dynamics.y;
                matched += float(desired == actual);
            }
            value = matched;
            break;
        }
        case MR_TASK_REWARD_SWING_CLEARANCE: {
            if (!moving) {
                value = 0.0f;
                break;
            }
            float clearanceReward = 0.0f;
            float count = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    desiredSupportContact(group, phase) ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const float error =
                    compactContact[
                        compactBase +
                        group.members.w + 3u
                    ] -
                    program.locomotion.z;
                clearanceReward += exp(
                    -(error * error) /
                    max(
                        operation.parameters.y,
                        1.0e-8f
                    )
                );
                count += 1.0f;
            }
            value =
                clearanceReward / max(count, 1.0f);
            break;
        }
        case MR_TASK_REWARD_FOOT_CLEARANCE: {
            float errorVelocity = 0.0f;
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) == 0u ||
                    (operation.source.y != MR_INVALID_INDEX &&
                     operation.source.y != groupIndex)) {
                    continue;
                }
                const MRBodyStateGPU foot =
                    bodyStates[
                        bodyBase + group.reference.x
                    ];
                const float3 offset = rotate(
                    foot.orientation,
                    group.kinematicReference.xyz
                );
                const float3 position =
                    foot.position.xyz + offset;
                const float3 velocity =
                    foot.linearVelocityAndInverseMass.xyz +
                    cross(
                        foot.angularVelocity.xyz,
                        offset
                    );
                const float heightError =
                    position.z -
                    program.locomotion.z;
                const float velocityWeight = tanh(
                    operation.parameters.z *
                    length(velocity.xy)
                );
                errorVelocity +=
                    heightError * heightError *
                    velocityWeight;
            }
            value = exp(
                -errorVelocity /
                max(operation.parameters.y, 1.0e-8f)
            );
            break;
        }
        case MR_TASK_REWARD_SUPPORT_SLIP:
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_SUPPORT) != 0u &&
                    (operation.source.y == MR_INVALID_INDEX ||
                     operation.source.y == groupIndex)) {
                    value += compactContact[
                        compactBase +
                        group.members.w + 1u
                    ];
                }
            }
            break;
        case MR_TASK_REWARD_FORBIDDEN_CONTACT:
            for (uint groupIndex = 0u;
                 groupIndex < program.counts0.w;
                 ++groupIndex) {
                const MRTaskContactGroupGPU group =
                    contactGroups[groupIndex];
                if ((group.members.z &
                     MR_TASK_CONTACT_FORBIDDEN) != 0u &&
                    (operation.source.y == MR_INVALID_INDEX ||
                     operation.source.y == groupIndex)) {
                    value = max(
                        value,
                        compactContact[
                            compactBase +
                            group.members.w
                        ]
                    );
                }
            }
            break;
        case MR_TASK_REWARD_INTERACTION_JOINT_TRACKING: {
            float squaredError = 0.0f;
            float squaredVelocityError = 0.0f;
            float jointCount = 0.0f;
            if (operation.source.y == MR_INVALID_INDEX) {
                for (uint action = 0u;
                     action < program.interaction.y;
                     ++action) {
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference = mix(
                        interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ],
                        interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ],
                        referenceBlend
                    );
                    const float delta =
                        qState[qBase + binding.indices.z] -
                        reference;
                    const float referenceVelocity =
                        (interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ] - interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ]) * program.interactionTiming.x;
                    const float velocityDelta =
                        vState[vBase + binding.indices.w] -
                        referenceVelocity;
                    squaredError += delta * delta;
                    squaredVelocityError +=
                        velocityDelta * velocityDelta;
                    jointCount += 1.0f;
                }
            } else {
                const MRTaskIndexGroupGPU group =
                    jointGroups[operation.source.y];
                for (uint member = 0u;
                     member < group.members.y;
                     ++member) {
                    const uint action = jointMembers[
                        group.members.x + member
                    ];
                    const MRTaskActionBindingGPU binding =
                        actions[action];
                    const float reference = mix(
                        interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ],
                        interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ],
                        referenceBlend
                    );
                    const float delta =
                        qState[qBase + binding.indices.z] -
                        reference;
                    const float referenceVelocity =
                        (interactionJointTargets[
                            nextReferenceFrame * program.interaction.y + action
                        ] - interactionJointTargets[
                            referenceFrame * program.interaction.y + action
                        ]) * program.interactionTiming.x;
                    const float velocityDelta =
                        vState[vBase + binding.indices.w] -
                        referenceVelocity;
                    squaredError += delta * delta;
                    squaredVelocityError +=
                        velocityDelta * velocityDelta;
                    jointCount += 1.0f;
                }
            }
            const float positionScore = exp(
                -(squaredError / max(jointCount, 1.0f)) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float velocityScore = operation.parameters.z > 0.0f
                ? exp(
                      -(squaredVelocityError /
                          max(jointCount, 1.0f)) /
                      operation.parameters.z
                  )
                : positionScore;
            value = 0.5f * (positionScore + velocityScore);
            interactionMetric = value;
            interactionMetricWeight = 1.0f;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_ROOT_TRACKING: {
            const uint targetBase = referenceFrame * 7u;
            const uint nextTargetBase = nextReferenceFrame * 7u;
            const float3 targetPosition = mix(
                float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                ),
                float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ),
                referenceBlend
            );
            const float3 positionDelta =
                rootWorldPosition(program, qState + qBase) -
                targetPosition;
            const float4 rawOrientation = float4(
                qState[qBase + program.root.z + 3u],
                qState[qBase + program.root.z + 4u],
                qState[qBase + program.root.z + 5u],
                qState[qBase + program.root.z + 6u]
            );
            const float4 orientation = rawOrientation * rsqrt(
                max(dot(rawOrientation, rawOrientation), 1.0e-12f)
            );
            const float4 rawTargetOrientation = float4(
                interactionRootTargets[targetBase + 3u],
                interactionRootTargets[targetBase + 4u],
                interactionRootTargets[targetBase + 5u],
                interactionRootTargets[targetBase + 6u]
            );
            const float4 rawNextTargetOrientation = float4(
                interactionRootTargets[nextTargetBase + 3u],
                interactionRootTargets[nextTargetBase + 4u],
                interactionRootTargets[nextTargetBase + 5u],
                interactionRootTargets[nextTargetBase + 6u]
            );
            const float4 targetOrientation = quaternionInterpolate(
                normalize(rawTargetOrientation),
                normalize(rawNextTargetOrientation),
                referenceBlend
            );
            const float orientationError =
                1.0f - abs(dot(orientation, targetOrientation));
            const float positionScore = exp(
                -dot(positionDelta, positionDelta) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float orientationScore = exp(
                -(orientationError * orientationError) /
                max(operation.parameters.z, 1.0e-8f)
            );
            const float3 targetLinearVelocity =
                (float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ) - float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                )) * program.interactionTiming.x;
            const float3 linearVelocityDelta =
                rootWorldLinearVelocity(
                    program,
                    qState + qBase,
                    vState + vBase
                ) - targetLinearVelocity;
            const float linearVelocityScore =
                operation.parameters.w > 0.0f
                ? exp(
                      -dot(linearVelocityDelta, linearVelocityDelta) /
                      operation.parameters.w
                  )
                : positionScore;
            const float3 targetAngularVelocity =
                quaternionWorldAngularVelocity(
                    rawTargetOrientation,
                    rawNextTargetOrientation,
                    program.interactionTiming.x
                );
            const float3 angularVelocityDelta = float3(
                vState[vBase + program.root.w + 3u],
                vState[vBase + program.root.w + 4u],
                vState[vBase + program.root.w + 5u]
            ) - targetAngularVelocity;
            const float angularVelocityScore =
                operation.auxiliary.x > 0.0f
                ? exp(
                      -dot(angularVelocityDelta, angularVelocityDelta) /
                      operation.auxiliary.x
                  )
                : orientationScore;
            value = 0.25f * (
                positionScore + orientationScore +
                linearVelocityScore + angularVelocityScore
            );
            interactionMetric = value;
            interactionMetricWeight = 1.0f;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_ROOT_LINEAR_VELOCITY_ERROR: {
            const uint targetBase = referenceFrame * 7u;
            const uint nextTargetBase = nextReferenceFrame * 7u;
            const float3 targetLinearVelocity =
                (float3(
                    interactionRootTargets[nextTargetBase + 0u],
                    interactionRootTargets[nextTargetBase + 1u],
                    interactionRootTargets[nextTargetBase + 2u]
                ) - float3(
                    interactionRootTargets[targetBase + 0u],
                    interactionRootTargets[targetBase + 1u],
                    interactionRootTargets[targetBase + 2u]
                )) * program.interactionTiming.x;
            const float3 linearVelocityDelta =
                rootWorldLinearVelocity(
                    program,
                    qState + qBase,
                    vState + vBase
                ) - targetLinearVelocity;
            const float scaledError = length(linearVelocityDelta) /
                max(operation.parameters.y, 1.0e-8f);
            // Pseudo-Huber growth stays informative far from a fast teacher
            // without the unbounded quadratic leverage of a squared penalty.
            value = sqrt(1.0f + scaledError * scaledError) - 1.0f;
            break;
        }
        case MR_TASK_REWARD_OBJECT_GRASP: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const uint objectBody = object.flagsAndIndices[2];
            uint distinctMembers = 0u;
            float normalForce = 0.0f;
            for (uint member = 0u;
                 member < group.members.y;
                 ++member) {
                const uint memberBody =
                    contactMembers[group.members.x + member];
                bool touching = false;
                for (uint contact = 0u;
                     contact < activeContacts;
                     ++contact) {
                    const MRContactConstraintGPU constraint =
                        contacts[contactBase + contact];
                    const bool matched =
                        (constraint.bodyA == memberBody &&
                         constraint.bodyB == objectBody) ||
                        (constraint.bodyB == memberBody &&
                         constraint.bodyA == objectBody);
                    if (!matched) {
                        continue;
                    }
                    touching = true;
                    normalForce += abs(constraint.impulses.x) /
                        dispatch.timing.y;
                }
                distinctMembers += uint(touching);
            }
            const float memberScore = smoothstep(
                0.0f,
                max(operation.parameters.y, 1.0f),
                float(distinctMembers)
            );
            const float forceScore = 1.0f - exp(
                -normalForce /
                max(operation.parameters.z, 1.0e-6f)
            );
            value = memberScore * forceScore;
            break;
        }
        case MR_TASK_REWARD_OBJECT_LIFT: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            value = smoothstep(
                operation.parameters.y,
                operation.parameters.z,
                object.position.z
            );
            break;
        }
        case MR_TASK_REWARD_OBJECT_POSITION: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const float3 delta = object.position.xyz - float3(
                operation.parameters.y,
                operation.parameters.z,
                operation.parameters.w
            );
            value = exp(
                -dot(delta, delta) /
                max(operation.auxiliary.x, 1.0e-8f)
            );
            break;
        }
        case MR_TASK_REWARD_OBJECT_PLACEMENT: {
            const MRBodyStateGPU object =
                sceneState[sceneBase + operation.source.z];
            const MRBodyStateGPU goal =
                sceneState[sceneBase + operation.source.y];
            const float3 positionDelta =
                object.position.xyz - goal.position.xyz;
            const float positionScore = exp(
                -dot(positionDelta, positionDelta) /
                max(operation.parameters.y, 1.0e-8f)
            );
            const float quietScore = 0.5f * (
                exp(
                    -dot(
                        object.linearVelocityAndInverseMass.xyz,
                        object.linearVelocityAndInverseMass.xyz
                    ) / max(operation.parameters.z, 1.0e-8f)
                ) +
                exp(
                    -dot(
                        object.angularVelocity.xyz,
                        object.angularVelocity.xyz
                    ) / max(operation.parameters.w, 1.0e-8f)
                )
            );
            value = positionScore * quietScore;
            break;
        }
        case MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING: {
            const uint trackIndex = operation.source.z;
            const MRTaskInteractionContactGPU track =
                interactionContacts[trackIndex];
            if (track.binding.x != operation.source.y) {
                value = 0.0f;
                break;
            }
            const uint sampleIndex =
                referenceFrame * program.interaction.z + trackIndex;
            const MRTaskInteractionSampleGPU sample =
                interactionSamples[sampleIndex];
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            const bool expectedContact =
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_STICK ||
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_ROLL ||
                sample.metadata.x ==
                    MR_TASK_INTERACTION_CONTACT_SLIDE;
            const bool supportGroup =
                (group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u;
            const uint wrench = compactBase + group.reference.y;
            const bool actualContact = supportGroup
                ? compactContact[
                      compactBase + group.members.w
                  ] > program.dynamics.y
                : length(float3(
                      compactContact[wrench + 0u],
                      compactContact[wrench + 1u],
                      compactContact[wrench + 2u]
                  )) > program.dynamics.y;
            const float modeScore = float(
                expectedContact == actualContact
            );
            float normalizedSquaredError = 0.0f;
            float validFeatureCount = 0.0f;
            device const float* compact =
                compactContact + compactBase;
            for (uint feature = 0u;
                 feature <
                    MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT;
                 ++feature) {
                if ((sample.metadata.y & (1u << feature)) == 0u) {
                    continue;
                }
                const uint targetIndex =
                    sampleIndex *
                        MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT +
                    feature;
                const float tolerance = max(
                    interactionContactTolerances[targetIndex],
                    1.0e-8f
                );
                const float delta =
                    (
                        supportPatchFeature(
                            program,
                            group,
                            compact,
                            feature
                        ) -
                        interactionContactTargets[targetIndex]
                    ) /
                    tolerance;
                normalizedSquaredError += min(
                    delta * delta,
                    1.0e6f
                );
                validFeatureCount += 1.0f;
            }
            const float fieldScore = validFeatureCount > 0.0f
                ? exp(
                      -(normalizedSquaredError / validFeatureCount) /
                      max(operation.parameters.z, 1.0e-8f)
                  )
                : modeScore;
            const float confidence = clamp(
                sample.confidence.x,
                0.0f,
                1.0f
            );
            const float contactScore = mix(
                fieldScore,
                modeScore,
                clamp(operation.parameters.y, 0.0f, 1.0f)
            );
            value = confidence * contactScore;
            interactionMetric = value;
            interactionMetricWeight = confidence;
            break;
        }
        default:
            value = 0.0f;
            break;
        }
        const float contribution =
            operation.parameters.x * value;
        interactionTrackingSum += interactionMetric;
        interactionTrackingWeight += interactionMetricWeight;
        reward += contribution;
        for (uint channel = 0u;
             channel < program.interactionOffsets1.w;
             ++channel) {
            if (outcomeRewardOperations[channel] != operation.source.x) {
                continue;
            }
            if (channel < 4u) {
                outcomeChannels0[channel] += contribution;
            } else {
                outcomeChannels1[channel - 4u] += contribution;
            }
        }
        switch (operation.source.x) {
        case MR_TASK_REWARD_NAVIGATION_WAYPOINT_PROGRESS:
        case MR_TASK_REWARD_NAVIGATION_WAYPOINT_REACH:
        case MR_TASK_REWARD_LINEAR_VELOCITY_TRACKING:
        case MR_TASK_REWARD_FIGURE_EIGHT_PATH_TRACKING:
        case MR_TASK_REWARD_YAW_VELOCITY_TRACKING:
        case MR_TASK_REWARD_CONSTANT:
        case MR_TASK_REWARD_GAIT_CONTACT_MATCH:
        case MR_TASK_REWARD_SWING_CLEARANCE:
        case MR_TASK_REWARD_FOOT_CLEARANCE:
        case MR_TASK_REWARD_INTERACTION_JOINT_TRACKING:
            rewardBreakdown0.x += contribution;
            break;
        case MR_TASK_REWARD_ROOT_VERTICAL_VELOCITY_SQUARED:
        case MR_TASK_REWARD_ROOT_ROLL_PITCH_VELOCITY_SQUARED:
        case MR_TASK_REWARD_TILT_SQUARED:
        case MR_TASK_REWARD_PROJECTED_GRAVITY_HORIZONTAL_SQUARED:
        case MR_TASK_REWARD_ROOT_HEIGHT_ERROR_SQUARED:
        case MR_TASK_REWARD_ROOT_HEIGHT_NORMALIZED:
        case MR_TASK_REWARD_ROOT_HEIGHT_PROGRESS:
        case MR_TASK_REWARD_UPRIGHTNESS:
        case MR_TASK_REWARD_SUPPORT_CONTACT_COUNT:
        case MR_TASK_REWARD_BODY_HEIGHT_EXPONENTIAL:
        case MR_TASK_REWARD_SUPPORT_HEIGHT_EXPONENTIAL:
        case MR_TASK_REWARD_BODY_UP_EXPONENTIAL:
        case MR_TASK_REWARD_STANDING_COMPLETION:
        case MR_TASK_REWARD_RESTORATION:
        case MR_TASK_REWARD_INTERACTION_ROOT_TRACKING:
        case MR_TASK_REWARD_INTERACTION_ROOT_LINEAR_VELOCITY_ERROR:
        case MR_TASK_REWARD_OBJECT_GRASP:
        case MR_TASK_REWARD_OBJECT_LIFT:
        case MR_TASK_REWARD_OBJECT_POSITION:
        case MR_TASK_REWARD_OBJECT_PLACEMENT:
        case MR_TASK_REWARD_RECOVERY_TILT_PROGRESS:
        case MR_TASK_REWARD_RECOVERY_COMPLETION:
        case MR_TASK_REWARD_WHOLE_BODY_RECOVERY:
        case MR_TASK_REWARD_LINK_CLEARANCE_BARRIER:
        case MR_TASK_REWARD_PROJECTILE_MISS:
        case MR_TASK_REWARD_PROJECTILE_EVASION:
        case MR_TASK_REWARD_PROJECTILE_SAFE_STILLNESS:
            rewardBreakdown0.y += contribution;
            break;
        case MR_TASK_REWARD_JOINT_VELOCITY_SQUARED:
            rewardBreakdown0.z += contribution;
            break;
        case MR_TASK_REWARD_JOINT_ACCELERATION_SQUARED:
            rewardBreakdown0.w += contribution;
            break;
        case MR_TASK_REWARD_ACTION_RATE_SQUARED:
        case MR_TASK_REWARD_PROJECTILE_SAFE_ACTION_RATE:
        case MR_TASK_REWARD_JOINT_CBF_CORRECTION:
        case MR_TASK_REWARD_JOINT_CBF_BUFFER:
        case MR_TASK_REWARD_PROJECTILE_PREDICTED_CLEARANCE:
            rewardBreakdown1.x += contribution;
            break;
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_SQUARED:
        case MR_TASK_REWARD_JOINT_LIMIT_VIOLATION_ABSOLUTE:
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_SQUARED:
        case MR_TASK_REWARD_JOINT_GROUP_POSTURE_ABSOLUTE:
            rewardBreakdown1.y += contribution;
            break;
        case MR_TASK_REWARD_MECHANICAL_POWER:
            rewardBreakdown1.z += contribution;
            break;
        case MR_TASK_REWARD_SUPPORT_SLIP:
        case MR_TASK_REWARD_FORBIDDEN_CONTACT:
        case MR_TASK_REWARD_INTERACTION_CONTACT_TRACKING:
            rewardBreakdown1.w += contribution;
            break;
        default:
            break;
        }
    }
    // TaskPack weights are rates. Integrating at the control boundary keeps
    // reward magnitude and PPO critic targets independent of control rate.
    reward *= dispatch.timing.x;
    rewardBreakdown0 *= dispatch.timing.x;
    rewardBreakdown1 *= dispatch.timing.x;
    outcomeChannels0 *= dispatch.timing.x;
    outcomeChannels1 *= dispatch.timing.x;

    // V8 omits the actuating approach-envelope flag but retains these two
    // direct shadow diagnostics. They are per-transition indicators, not
    // reward terms, and occupy the two channels unused by the inherited Crow
    // journey outcome schema.
    outcomeChannels1.z = shadowApproachActive &&
        shadowAbsolutePitch > 0.16f ? 1.0f : 0.0f;
    outcomeChannels1.w = shadowApproachActive &&
        shadowAbsolutePitch > 0.22f ? 1.0f : 0.0f;

    const float tracking = exp(-trackingError / 0.25f);
    const float yawTracking = exp(-yawError / 0.25f);
    const uint episodeSteps = state.episode.x + 1u;
    const bool avianJourney =
        (program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u;
    const uint episodeLimit = !avianJourney ? program.schedule.x
        : avianCurriculumBand == 0u ? 250u
        : avianCurriculumBand == 1u ? 400u
        : avianCurriculumBand == 2u ? 250u
        : avianCurriculumBand == 3u ? 400u
        : avianCurriculumBand == 4u ? 650u
        : avianCurriculumBand == 5u ? 750u
        : avianCurriculumBand == 6u ? 1050u
        : avianCurriculumBand == 7u ? 1250u
        : avianCurriculumBand == 8u ? 1400u
        : avianCurriculumBand == 9u ? 1600u
        : program.schedule.x;
    const bool timeout =
        episodeSteps >= episodeLimit;
    bool done = false;
    uint reason = MR_TASK_TERMINATION_CONTINUING;
    uint selectedPriority = 0u;
    float failurePenalty = 0.0f;
    for (uint index = 0u;
         index < program.counts2.x;
         ++index) {
        const MRTaskTerminationOperatorGPU operation =
            terminations[index];
        const uint maximumBand = operation.schedule.y == MR_INVALID_INDEX
            ? program.schedule.z - 1u
            : operation.schedule.y;
        if (curriculum < operation.schedule.x ||
            curriculum > maximumBand) {
            continue;
        }
        bool triggered = false;
        switch (operation.source.x) {
        case MR_TASK_TERMINATE_MINIMUM_ROOT_HEIGHT:
            triggered =
                height < operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_MAXIMUM_ROOT_HEIGHT:
            triggered =
                height > operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_MAXIMUM_TILT:
            triggered =
                tilt > operation.parameters.x;
            break;
        case MR_TASK_TERMINATE_CONTACT_GROUP: {
            const MRTaskContactGroupGPU group =
                contactGroups[operation.source.y];
            triggered =
                compactContact[
                    compactBase + group.members.w
                ] > operation.parameters.x;
            break;
        }
        case MR_TASK_TERMINATE_PROJECTILE_CONTACT: {
            device const MRTaskContactGroupGPU& group =
                contactGroups[operation.source.y];
            if (!eventSequenceAvailable ||
                activeImpactEvent == MR_INVALID_INDEX ||
                state.status.w == 0u) {
                break;
            }
            const uint projectileBody =
                impactEvents[activeImpactEvent].binding.w;
            for (uint contact = 0u;
                 contact < activeContacts && !triggered;
                 ++contact) {
                const MRContactConstraintGPU constraint =
                    contacts[contactBase + contact];
                const bool projectileA =
                    constraint.bodyA == projectileBody;
                const bool projectileB =
                    constraint.bodyB == projectileBody;
                if (projectileA == projectileB) {
                    continue;
                }
                const uint other = projectileA
                    ? constraint.bodyB
                    : constraint.bodyA;
                triggered = bodyMember(
                    other,
                    group,
                    contactMembers
                ) && abs(constraint.impulses.x) /
                    dispatch.timing.y > operation.parameters.x;
            }
            break;
        }
        default:
            break;
        }
        if (triggered &&
            (!done ||
             operation.source.w >= selectedPriority)) {
            done = true;
            reason = operation.source.z;
            selectedPriority = operation.source.w;
            failurePenalty = operation.parameters.y;
        }
    }
    if (!done &&
        navigationConfigured &&
        dispatch.assistance.z != 0.0f &&
        state.navigation.w != 0.0f) {
        // The navigation curriculum samples independent route segments. End
        // a successful sample immediately so the next accepted transaction
        // resamples a useful stage instead of collecting an arbitrarily long
        // zero-command tail after waypoint five. Autonomous evaluation keeps
        // its completion latch and therefore remains unchanged.
        done = true;
        reason = MR_TASK_TERMINATION_NAVIGATION_COMPLETION;
    }
    if (timeout) {
        done = true;
        reason = MR_TASK_TERMINATION_TIMEOUT;
    }
    if (physicsError) {
        done = true;
        reason = MR_TASK_TERMINATION_PHYSICS_ERROR;
    }
    if (done &&
        reason != MR_TASK_TERMINATION_TIMEOUT &&
        reason != MR_TASK_TERMINATION_PHYSICS_ERROR) {
        reward += failurePenalty;
        rewardBreakdown0.y += failurePenalty;
    }

    // ARDY_CLOSED_LOOP_BALANCE_POLICY_V5. The reference clock is
    // subordinate to accepted physical support. The policy owns balance and
    // unloading; ARDY owns the whole-body motion reference.
    if (interactionPhysicsGated(program)) {
        float gatedFramePosition = interactionFramePosition(
            program,
            state,
            dispatch.timing.x
        );
        bool fallLatched = interactionFallIsLatched(state);
        bool forbiddenContact = false;
        uint actualSupportCount = 0u;
        for (uint groupIndex = 0u;
             groupIndex < program.counts0.w;
             ++groupIndex) {
            const MRTaskContactGroupGPU group = contactGroups[groupIndex];
            if ((group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u &&
                compactContact[compactBase + group.members.w] >
                    program.dynamics.y) {
                ++actualSupportCount;
            }
            if ((group.members.z & MR_TASK_CONTACT_FORBIDDEN) != 0u &&
                compactContact[compactBase + group.members.w] > 0.5f) {
                forbiddenContact = true;
            }
        }

        uint comparedContacts = 0u;
        uint expectedContacts = 0u;
        uint strictContactMismatches = 0u;
        bool transitionMode = false;
        for (uint contactIndex = 0u;
             contactIndex < program.interaction.z;
             ++contactIndex) {
            const MRTaskInteractionContactGPU binding =
                interactionContacts[contactIndex];
            const uint sampleIndex =
                referenceFrame * program.interaction.z + contactIndex;
            const MRTaskInteractionSampleGPU sample =
                interactionSamples[sampleIndex];
            if (sample.confidence.x < 0.35f) {
                continue;
            }
            const uint mode = sample.metadata.x;
            const bool expectedContact =
                mode == MR_TASK_INTERACTION_CONTACT_STICK ||
                mode == MR_TASK_INTERACTION_CONTACT_ROLL ||
                mode == MR_TASK_INTERACTION_CONTACT_SLIDE;
            const bool transitional =
                mode == MR_TASK_INTERACTION_CONTACT_RELEASE ||
                mode == MR_TASK_INTERACTION_CONTACT_APPROACH;
            const MRTaskContactGroupGPU group =
                contactGroups[binding.binding.x];
            const uint wrench = compactBase + group.reference.y;
            const bool supportGroup =
                (group.members.z & MR_TASK_CONTACT_SUPPORT) != 0u;
            const bool actualContact = supportGroup
                ? compactContact[compactBase + group.members.w] >
                    program.dynamics.y
                : length(float3(
                      compactContact[wrench + 0u],
                      compactContact[wrench + 1u],
                      compactContact[wrench + 2u]
                  )) > program.dynamics.y;
            bool contactFeaturesAccepted = true;
            if (expectedContact && actualContact) {
                device const float* compact =
                    compactContact + compactBase;
                for (uint feature = 0u;
                     feature <
                         MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT;
                     ++feature) {
                    if ((sample.metadata.y & (1u << feature)) == 0u) {
                        continue;
                    }
                    const uint targetIndex =
                        sampleIndex *
                            MR_TASK_INTERACTION_CONTACT_FEATURE_COUNT +
                        feature;
                    const float target =
                        interactionContactTargets[targetIndex];
                    const float tolerance =
                        interactionContactTolerances[targetIndex];
                    const float actual = supportPatchFeature(
                        program,
                        group,
                        compact,
                        feature
                    );
                    contactFeaturesAccepted =
                        contactFeaturesAccepted &&
                        isfinite(actual) &&
                        isfinite(target) &&
                        isfinite(tolerance) &&
                        tolerance > 0.0f &&
                        abs(actual - target) <= tolerance;
                }
            }
            ++comparedContacts;
            expectedContacts += expectedContact ? 1u : 0u;
            transitionMode = transitionMode || transitional;
            if (!transitional &&
                (expectedContact != actualContact ||
                 !contactFeaturesAccepted)) {
                ++strictContactMismatches;
            }
        }

        float jointErrorSquared = 0.0f;
        for (uint action = 0u;
             action < program.interaction.y;
             ++action) {
            const MRTaskActionBindingGPU binding = actions[action];
            const float reference = mix(
                interactionJointTargets[
                    referenceFrame * program.interaction.y + action
                ],
                interactionJointTargets[
                    nextReferenceFrame * program.interaction.y + action
                ],
                referenceBlend
            );
            const float error = qState[qBase + binding.indices.z] - reference;
            jointErrorSquared += error * error;
        }
        const float jointRms = sqrt(
            jointErrorSquared /
            max(float(program.interaction.y), 1.0f)
        );

        const bool physicalFall =
            height < 0.55f ||
            tilt > 0.50f ||
            forbiddenContact ||
            reason == MR_TASK_TERMINATION_HEIGHT ||
            reason == MR_TASK_TERMINATION_TILT ||
            reason == MR_TASK_TERMINATION_CONTACT;
        fallLatched = fallLatched || physicalFall;

        // Bootstrap only after 0.30 s of actual quiet bilateral support. The
        // spare training-only lane is unused by this non-threat task.
        const bool bootstrapFrame = gatedFramePosition <= 0.50f;
        float quietSupportSeconds = bootstrapFrame
            ? state.threatTeacher.w
            : 0.30f;
        const bool quietSupport =
            actualSupportCount >= 2u &&
            height > 0.68f &&
            tilt < 0.08f &&
            length(baseLinear.xy) < 0.08f &&
            abs(baseLinear.z) < 0.10f &&
            length(baseAngular) < 0.20f;
        if (bootstrapFrame) {
            quietSupportSeconds = quietSupport
                ? quietSupportSeconds + dispatch.timing.x
                : 0.0f;
        }
        state.threatTeacher.w = quietSupportSeconds;
        const bool bootstrapped =
            !bootstrapFrame || quietSupportSeconds >= 0.30f;

        uint nextRateCode = 0u;
        const bool atEnd =
            (program.interaction.w & MR_TASK_INTERACTION_LOOP) == 0u &&
            gatedFramePosition >=
                float(program.interaction.x - 1u) - 1.0e-4f;
        const bool contactsReady =
            comparedContacts == 0u || strictContactMismatches == 0u;
        const bool expectedFlight =
            comparedContacts > 0u && expectedContacts == 0u &&
            !transitionMode;
        const bool supportSafe =
            actualSupportCount > 0u &&
            height > 0.64f &&
            tilt < 0.25f;

        if (!fallLatched && !done && !atEnd && bootstrapped && contactsReady) {
            if (transitionMode) {
                // RELEASE must move far enough to unload; APPROACH must move
                // far enough to make contact. Requiring equality here would
                // deadlock both transitions on one static pose.
                if (supportSafe && jointRms <= 0.60f) {
                    nextRateCode = 1u;
                }
            } else if (expectedFlight) {
                if (actualSupportCount == 0u &&
                    height > 0.60f && tilt < 0.38f) {
                    nextRateCode = 3u;
                }
            } else if (supportSafe) {
                // The staged preload runs slowly. Once the ARDY stride begins,
                // tight tracking may use half rate; lagging support stays slow.
                const bool stagedPrefix = gatedFramePosition < 24.0f;
                nextRateCode = stagedPrefix
                    ? 1u
                    : jointRms <= 0.24f
                    ? 2u
                    : jointRms <= 0.55f
                    ? 1u
                    : 0u;
            }
        }

        const float nextRate = nextRateCode == 1u
            ? 0.25f
            : nextRateCode == 2u
            ? 0.50f
            : nextRateCode == 3u
            ? 1.00f
            : 0.0f;
        gatedFramePosition +=
            dispatch.timing.x * program.interactionTiming.x * nextRate;
        if ((program.interaction.w & MR_TASK_INTERACTION_LOOP) != 0u) {
            gatedFramePosition = fmod(
                gatedFramePosition,
                float(program.interaction.x)
            );
        } else {
            gatedFramePosition = min(
                gatedFramePosition,
                float(program.interaction.x - 1u)
            );
        }
        state.status.w = packInteractionClock(
            gatedFramePosition,
            nextRateCode,
            fallLatched
        );
    }

    const float episodeReturn =
        state.airReturnTracking.z + reward;
    // Report continuous episode evidence; it never decides whether another
    // difficulty band or learner update is allowed.
    const float interactionTrackingScore =
        interactionTrackingWeight > 0.0f
        ? interactionTrackingSum / interactionTrackingWeight
        : 0.0f;
    const bool interactionReference =
        (program.schedule.w &
         MR_TASK_PROGRAM_INTERACTION_RESET) != 0u;
    const float evidenceTracking = interactionReference
        ? interactionTrackingScore
        : tracking;
    const float episodeTracking =
        state.airReturnTracking.w + evidenceTracking;
    const float episodeMeanTrackingScore =
        episodeTracking /
        max(float(episodeSteps), 1.0f);
    const float episodeTrackingScore = episodeMeanTrackingScore;
    const float trackingScore = interactionReference
        ? interactionTrackingScore
        : 0.5f * (tracking + yawTracking);
    const uint terrainLevel = state.episode.w;
    const float4 scheduleSeconds = scheduleSecondsForBand(
        program,
        curriculum
    );

    if (!figureEightConfigured && !done && state.schedule.x <= 1u) {
        state.commandAndPhase.xyz = sampledCommand(
            dispatch,
            program,
            commandCurriculum,
            environment,
            state.episode.y,
            episodeSteps,
            curriculum
        );
        state.schedule.x = durationSteps(
            dispatch,
            environment,
            state.episode.y,
            episodeSteps,
            64u,
            scheduleSeconds.x,
            scheduleSeconds.y
        );
    } else if (!figureEightConfigured && !done) {
        --state.schedule.x;
    }
    if (state.schedule.y == 0u || done) {
        state.schedule.y = durationSteps(
            dispatch,
            environment,
            state.episode.y,
            episodeSteps,
            65u,
            scheduleSeconds.z,
            scheduleSeconds.w
        );
    } else {
        --state.schedule.y;
    }

    state.episode = uint4(
        episodeSteps,
        state.episode.y,
        curriculum,
        terrainLevel
    );
    state.status.y = done ? 1u : 0u;
    state.status.z = reason;
    state.commandAndPhase.w = phase;
    state.airReturnTracking = float4(
        0.0f,
        height,
        done ? 0.0f : episodeReturn,
        done ? 0.0f : episodeTracking
    );
    state.recovery = done
        ? float4(0.0f)
        : float4(
              tilt,
              recoveryCompleted ? 0.0f : recoveryPeakTilt,
              recoveryCompleted ? 0.0f : recoveryStableTime,
              recoveryActive && !recoveryCompleted ? 1.0f : 0.0f
          );
    uint impactState = state.recoveryStats.w;
    if (impactContactLatched) {
        impactState |= kImpactContactLatched;
    }
    if (newProjectileContact) {
        impactState |= kImpactContactPublished;
    }
    const bool completedImpactWindow =
        recoveryCompleted ||
        (!recoveryActive && impactWindowElapsed);
    if (completedImpactWindow) {
        const uint nextOrder = min(
            impactOrder(state) + 1u,
            kImpactOrderMask
        );
        impactState =
            (impactState & kImpactOffsetMask) |
            kImpactEnabled |
            nextOrder;
        state.status.w = 0u;
    }
    state.recoveryStats = done
        ? uint4(0u)
        : uint4(
              recoveryEventCount,
              recoveryCompletionCount,
              recoveryTouch ? 1u : 0u,
              impactState
          );

    if (!done || (timeout && !physicsError)) {
        for (uint history = 0u;
             history + 1u < program.layout.y;
             ++history) {
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = actorHistory[
                    historyBase +
                    (history + 1u) *
                        program.layout.x +
                    index
                ];
                cleanHistory[
                    historyBase +
                    history * program.layout.x +
                    index
                ] = cleanHistory[
                    historyBase +
                    (history + 1u) *
                        program.layout.x +
                    index
                ];
            }
        }
        device float* actorTail =
            actorHistory +
            historyBase +
            (program.layout.y - 1u) *
                program.layout.x;
        device float* cleanTail =
            cleanHistory +
            historyBase +
            (program.layout.y - 1u) *
                program.layout.x;
        device const float* currentAction =
            actionHistory +
            delayBase +
            (program.layout.w - 2u) * program.counts0.x;
        writeFrame(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            state.episode.y,
            episodeSteps,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            actionHistory + delayBase +
                (program.layout.w - 2u) * program.counts0.x,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            sensorBias + biasBase,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            actorTail,
            cleanTail
        );
        if (state.schedule.w != 0u &&
            program.layout.y > 1u) {
            const uint delay = min(
                state.schedule.w,
                program.layout.y - 1u
            );
            const uint sourceHistory =
                program.layout.y - 1u - delay;
            for (uint index = 0u;
                 index < program.layout.x;
                 ++index) {
                actorTail[index] = actorHistory[
                    historyBase +
                    sourceHistory * program.layout.x +
                    index
                ];
            }
        }
        for (uint history = 0u;
             history + 1u < program.articulation.w;
             ++history) {
            for (uint index = 0u;
                 index < program.counts0.z;
                 ++index) {
                criticHistory[
                    criticHistoryBase +
                    history * program.counts0.z +
                    index
                ] = criticHistory[
                    criticHistoryBase +
                    (history + 1u) *
                        program.counts0.z +
                    index
                ];
            }
        }
        device float* criticTail =
            criticHistory +
            criticHistoryBase +
            (program.articulation.w - 1u) *
                program.counts0.z;
        writeCriticFrame(
            program,
            arena,
            dispatch.timing.x,
            criticOperators,
            actions,
            contactGroups,
            terrainSamples,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            currentAction,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            criticTail
        );
        for (uint action = 0u;
             action < program.counts0.x;
             ++action) {
            const uint velocityIndex = actions[action].indices.w;
            previousJointVelocity[previousVelocityBase + action] =
                velocityIndex == MR_INVALID_INDEX
                ? 0.0f
                : vState[vBase + velocityIndex];
            const uint qIndex = actions[action].indices.z;
            previousJointVelocity[
                previousVelocityBase + program.counts0.x + action
            ] =
                qIndex == MR_INVALID_INDEX
                ? 0.0f
                : qState[qBase + qIndex];
        }
    }

    const bool finalPolicyObservation =
        dispatch.timing.z != 0.0f &&
        pass.controlStep + 1u == dispatch.counts.y;
    if (finalPolicyObservation) {
        const uint actorObservationSize =
            dispatch.outputs.x / dispatch.counts.x;
        const uint actorOutputBase =
            dispatch.counts.y * dispatch.outputs.x +
            environment * actorObservationSize;
        for (uint index = 0u;
             index < historyElements;
             ++index) {
            actorObservations[actorOutputBase + index] =
                actorHistory[historyBase + index];
        }
        writeCurrentActor(
            dispatch,
            program,
            arena,
            actorOperators,
            actions,
            contactGroups,
            terrainSamples,
            environment,
            state.episode.y,
            episodeSteps,
            qState + qBase,
            vState + vBase,
            defaultQ,
            state,
            actionHistory + delayBase +
                (program.layout.w - 2u) * program.counts0.x,
            rawPolicyActions + environment * program.counts0.x,
            previousJointVelocity + previousVelocityBase + program.counts0.x,
            sensorBias + biasBase,
            compactContact + compactBase,
            bodyParameters + bodyParameterBase,
            controllerParameters + environment,
            sceneState + sceneBase,
            shapes,
            geometryHeaders,
            geometryVertices,
            actorObservations + actorOutputBase + historyElements
        );
    }
    if (dispatch.timing.w != 0.0f) {
        const uint criticObservationSize =
            (
                (program.schedule.w &
                 MR_TASK_PROGRAM_CRITIC_INCLUDES_CLEAN_HISTORY) != 0u
                ? historyElements
                : 0u
            ) +
            criticHistoryElements;
        const uint criticOutputBase =
            dispatch.counts.y * dispatch.outputs.y +
            environment * criticObservationSize;
        publishCritic(
            program,
            cleanHistory + historyBase,
            criticHistory + criticHistoryBase,
            criticObservations + criticOutputBase
        );
    }
    taskStates[environment] = state;

    const uint transitionIndex =
        pass.controlStep * dispatch.outputs.z + environment;
    MRTaskTransitionGPU transition{};
    transition.rewardAndState =
        float4(reward, trackingScore, height, tilt);
    transition.termination = uint4(
        done ? 1u : 0u,
        timeout ? 1u : 0u,
        physicsError ? 1u : 0u,
        reason
    );
    transition.rewardBreakdown0 = rewardBreakdown0;
    transition.rewardBreakdown1 = rewardBreakdown1;
    transition.outcomeChannels0 = outcomeChannels0;
    transition.outcomeChannels1 = outcomeChannels1;
    transition.policyRevision =
        dispatch.policyRevision;
    transition.episodeTrackingScore =
        done && !physicsError
        ? episodeTrackingScore
        : 0.0f;
    transition.taskProgress = uint4(
        curriculum,
        terrainLevel,
        activeImpactEvent == MR_INVALID_INDEX
            ? 0u
            : activeImpactEvent + 1u,
        impactTransitionFlags |
            recoveryOutcomeFlags |
            (standingCompleted ? MR_TASK_OUTCOME_STANDING : 0u) |
            (restoredCompleted ? MR_TASK_OUTCOME_RESTORED : 0u) |
            (
                (program.schedule.w &
                 MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u &&
                    program.interactionTiming.z > 0.0f
                ? MR_TASK_OUTCOME_POLICY_RESIDUAL : 0u
            ) |
            (
                ((program.schedule.w &
                  MR_TASK_PROGRAM_INTERACTION_REFERENCE) != 0u &&
                     program.interactionTiming.z == 0.0f) ||
                (dispatch.sampling.w != 0u &&
                 dispatch.assistance.x < 1.0f &&
                 (program.schedule.w &
                  MR_TASK_PROGRAM_AVIAN_CROW_JOURNEY) != 0u &&
                 true)
                ? MR_TASK_OUTCOME_INTERACTION_TEACHER : 0u
            )
    );
    transition.navigation = float4(
        navigationStepProgress,
        state.navigation.y,
        state.navigation.z,
        state.navigation.w
    );
    transitions[transitionIndex] = transition;
}

// One native thread reduces task-wide physical evidence. It never changes the
// episode sampling distribution or emits a pass/fail decision.
kernel void mr_locomotion_task_update_evidence(
    device const MRTaskDispatchGPU& dispatch [[buffer(0)]],
    device const MRTaskProgramHeaderGPU& program [[buffer(1)]],
    device MRTaskEvidenceStateGPU* evidenceState
        [[buffer(2)]],
    device MRTaskTransitionGPU* transitions [[buffer(3)]],
    constant MRMetalWorldPassGPU& pass [[buffer(4)]],
    const uint gridIndex [[thread_position_in_grid]]
) {
    if (gridIndex != 0u ||
        pass.controlStep >= dispatch.counts.y ||
        dispatch.taskFingerprint != program.taskFingerprint ||
        program.articulation.z != MR_TASK_PROGRAM_ABI_VERSION ||
        program.schedule.x == 0u ||
        program.schedule.z == 0u) {
        return;
    }

    MRTaskEvidenceStateGPU state = evidenceState[0];
    const ulong completedSteps = state.controlSteps + 1ul;
    const uint transitionBase =
        pass.controlStep * dispatch.outputs.z;
    for (uint environment = 0u;
         environment < dispatch.counts.x;
         ++environment) {
        const MRTaskTransitionGPU transition =
            transitions[transitionBase + environment];
        const uint impact = transition.taskProgress.w;
        state.impactContactCount +=
            (impact & MR_TASK_IMPACT_CONTACT) != 0u ? 1u : 0u;
        state.impactCleanMissCount +=
            (impact & MR_TASK_IMPACT_MISSED) != 0u ? 1u : 0u;
        state.balanceFailureCount +=
            transition.termination.x != 0u &&
            (transition.termination.w == MR_TASK_TERMINATION_HEIGHT ||
             transition.termination.w == MR_TASK_TERMINATION_TILT)
            ? 1u
            : 0u;
        if (transition.termination.x != 0u &&
            transition.termination.z == 0u) {
            state.trackingScoreSum +=
                transition.episodeTrackingScore;
            ++state.completedEpisodeCount;
            if (transition.termination.y != 0u) {
                ++state.timeoutEpisodeCount;
            }
        }
    }
    if (completedSteps % ulong(program.schedule.x) == 0ul) {
        const float completed =
            float(state.completedEpisodeCount);
        const ulong exposure =
            ulong(program.schedule.x) * ulong(dispatch.counts.x);
        const float rateScale = 1000000.0f /
            max(float(exposure), 1.0f);
        state.lastCompletedEpisodeCount = state.completedEpisodeCount;
        state.lastWindow = uint4(
            uint(round(float(state.impactContactCount) * rateScale)),
            uint(round(float(state.impactCleanMissCount) * rateScale)),
            uint(round(float(state.balanceFailureCount) * rateScale)),
            uint(round(
                state.trackingScoreSum /
                max(completed, 1.0f) * 1000000.0f
            ))
        );
        ++state.evidenceWindows;
        state.completedEpisodeCount = 0u;
        state.timeoutEpisodeCount = 0u;
        state.impactContactCount = 0u;
        state.impactCleanMissCount = 0u;
        state.balanceFailureCount = 0u;
        state.trackingScoreSum = 0.0f;
    }
    state.controlSteps = completedSteps;
    evidenceState[0] = state;
}
