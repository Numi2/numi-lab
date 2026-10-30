#pragma once
#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace metalrobo::human::coarse_tissue {
struct PhysicalSkinFractionalSemanticRoleSupportAsset {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::filesystem::path binaryPath,receiptPath,admissibilityProducerReceiptPath,admissibilityVerificationReceiptPath;
    std::string binarySHA256,receiptSHA256,admissibilityProducerReceiptSHA256,admissibilityVerificationReceiptSHA256;
};
struct PhysicalSkinFractionalSemanticSupportAssets { std::array<PhysicalSkinFractionalSemanticRoleSupportAsset,2> roles{}; };
struct PhysicalSkinFractionalSemanticSupportImportReport {
    std::array<std::string,2> binarySHA256{},receiptSHA256{},typedSupportSHA256{},
        admissibilitySourceSupportSHA256{},admissibilityProducerReceiptSHA256{},admissibilityVerificationReceiptSHA256{};
    std::array<std::size_t,2> childRows{},parentRows{};
    std::string compositeTypedSupportSHA256,compositeAdmissibilitySourceSHA256;
    bool verified=false;
};
namespace fractional_semantic_support_import_detail {
struct Payload {
    ConservativeSpatialChildSemanticSupportData data;
    std::vector<std::uint32_t> childMaterial,childDonor,childRegion;
    std::size_t childCount=0,parentCount=0;
    std::uint32_t material=NM_INVALID_INDEX;
    std::string binarySHA,receiptSHA,typedSHA,sourceSHA,producerSHA,verifierSHA;
};
inline bool validSHA(const std::string& x){return physical_skin_child_mass_detail::validSHA256(x);}
inline bool readText(const std::filesystem::path& p,std::size_t lim,std::string& x,std::string& e){
    std::error_code ec;auto n=std::filesystem::file_size(p,ec);if(ec||n>lim){e="receipt is unreadable or oversized";return false;}
    std::ifstream f(p,std::ios::binary);if(!f){e="cannot open receipt";return false;}
    x.assign(static_cast<std::size_t>(n),'\0');if(!f.read(x.data(),std::streamsize(n))||f.peek()!=EOF){e="receipt read failed";return false;}return true;
}
inline bool str(std::string_view j,std::string_view k,std::string& v){
    std::string t="\""+std::string(k)+"\"";auto p=j.find(t);if(p==j.npos||j.find(t,p+t.size())!=j.npos)return false;
    p+=t.size();while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;if(p==j.size()||j[p++]!=':')return false;
    while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;if(p==j.size()||j[p++]!='"')return false;
    auto q=j.find('"',p);if(q==j.npos)return false;v.assign(j.substr(p,q-p));return true;
}
inline bool num(std::string_view j,std::string_view k,std::uint64_t& v){
    std::string t="\""+std::string(k)+"\"";auto p=j.find(t);if(p==j.npos||j.find(t,p+t.size())!=j.npos)return false;
    p+=t.size();while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;if(p==j.size()||j[p++]!=':')return false;
    while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;auto b=p;while(p<j.size()&&j[p]>='0'&&j[p]<='9')++p;
    if(b==p)return false;try{v=std::stoull(std::string(j.substr(b,p-b)));}catch(...){return false;}
    return p==j.size()||j[p]==','||j[p]=='}'||std::isspace((unsigned char)j[p]);
}
inline bool noAdmission(std::string_view j){
    std::string t="\"physical_admission\"";auto p=j.find(t);if(p==j.npos||j.find(t,p+t.size())!=j.npos)return false;
    p+=t.size();while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;if(p==j.size()||j[p++]!=':')return false;
    while(p<j.size()&&std::isspace((unsigned char)j[p]))++p;
    return j.substr(p,5)=="false"&&(p+5==j.size()||j[p+5]==','||j[p+5]=='}'||std::isspace((unsigned char)j[p+5]));
}
inline bool equals(std::string_view j,std::string_view k,std::string_view e){std::string v;return str(j,k,v)&&v==e;}
inline bool objectForRole(std::string_view j,std::string_view role,std::string& out){
    std::string key="\"role\"";std::size_t p=0,n=0;
    while((p=j.find(key,p))!=j.npos){
        auto q=p+key.size();while(q<j.size()&&std::isspace((unsigned char)j[q]))++q;
        if(q==j.size()||j[q++]!=':')return false;while(q<j.size()&&std::isspace((unsigned char)j[q]))++q;
        if(q==j.size()||j[q++]!='"')return false;auto e=j.find('"',q);if(e==j.npos)return false;
        auto val=j.substr(q,e-q);p=e+1;if(val!=role)continue;
        auto begin=j.rfind('{',q);if(begin==j.npos)return false;
        unsigned depth=0;bool quote=false,escape=false;std::size_t end=j.npos;
        for(std::size_t i=begin;i<j.size();++i){char c=j[i];if(quote){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quote=false;continue;}
            if(c=='"'){quote=true;continue;}if(c=='{')++depth;else if(c=='}'&&--depth==0){end=i+1;break;}}
        if(end==j.npos||++n!=1)return false;out.assign(j.substr(begin,end-begin));
    }return n==1;
}
template<class T> inline bool vector(std::ifstream& f,std::vector<T>& x,std::size_t n){
    if(n>std::size_t(std::numeric_limits<std::streamsize>::max())/sizeof(T))return false;
    x.resize(n);return n==0||bool(f.read(reinterpret_cast<char*>(x.data()),std::streamsize(n*sizeof(T))));
}
template<class T> inline bool one(std::ifstream& f,T& x){return bool(f.read(reinterpret_cast<char*>(&x),sizeof(T)));}
inline bool addBytes(std::uint64_t& total,std::uint64_t n,std::uint64_t width){
    if(width&&n>(UINT64_MAX-total)/width)return false;total+=n*width;return true;
}
inline bool pinnedJSON(const std::filesystem::path& p,const std::string& expected,std::string& text,std::string& error){
    if(p.empty()||!validSHA(expected)){error="source receipt path/hash absent";return false;}
    std::string h1,h2;if(!physical_skin_child_mass_detail::sha256File(p,h1,error)||h1!=expected){error="source receipt hash mismatch";return false;}
    if(!readText(p,4u*1024u*1024u,text,error))return false;
    if(!physical_skin_child_mass_detail::sha256File(p,h2,error)||h2!=h1){error="source receipt changed during read";return false;}return true;
}
inline bool readRole(const PhysicalSkinFractionalSemanticRoleSupportAsset& a,
                     const PhysicalSkinChildMassMappingBundle& b,Payload& p,std::string& error){
    auto fail=[&](std::string x){error=std::move(x);return false;};
    if((a.materialIndex!=15u&&a.materialIndex!=16u)||a.binaryPath.empty()||a.receiptPath.empty()||
       !validSHA(a.binarySHA256)||!validSHA(a.receiptSHA256)||a.admissibilityProducerReceiptPath.empty()||
       a.admissibilityVerificationReceiptPath.empty()||!validSHA(a.admissibilityProducerReceiptSHA256)||
       !validSHA(a.admissibilityVerificationReceiptSHA256))return fail("role support paths, domain, or SHA pins are incomplete");
    if(std::endian::native!=std::endian::little)return fail("FSSUP090 requires a little-endian host");
    if(!validSHA(b.childMeshSHA256)||!validSHA(b.regionLabelSHA256)||
       !validSHA(b.parent.parentMassPacketSHA256)||!validSHA(b.parent.parentSourceContributionSHA256))
        return fail("current child/parent domain SHA is missing");
    std::string h;if(!physical_skin_child_mass_detail::sha256File(a.binaryPath,h,error)||h!=a.binarySHA256)return fail("support binary SHA mismatch");
    std::string r;if(!pinnedJSON(a.receiptPath,a.receiptSHA256,r,error))return false;
    std::string typed,source;std::uint64_t mat=0,hops=0,steps=0;
    if(!equals(r,"schema","numi.human.fractional-semantic-support-role.v1")||
       !equals(r,"status","PASS_SOURCE_ONLY_FRACTIONAL_SEMANTIC_ROLE_SUPPORT")||
       !equals(r,"binary_sha256",a.binarySHA256)||!equals(r,"child_mesh_sha256",b.childMeshSHA256)||
       !equals(r,"region_label_sha256",b.regionLabelSHA256)||
       !equals(r,"parent_mass_packet_sha256",b.parent.parentMassPacketSHA256)||
       !equals(r,"parent_source_contribution_sha256",b.parent.parentSourceContributionSHA256)||
       !num(r,"role_material_index",mat)||mat!=a.materialIndex||!num(r,"admissibility_max_source_region_hops",hops)||hops!=1||
       !num(r,"native_steps",steps)||steps!=0||!noAdmission(r)||
       !str(r,"support_sha256",typed)||!validSHA(typed)||
       !str(r,"admissibility_source_support_sha256",source)||!validSHA(source)||
       !equals(r,"admissibility_producer_receipt_sha256",a.admissibilityProducerReceiptSHA256)||
       !equals(r,"admissibility_verification_receipt_sha256",a.admissibilityVerificationReceiptSHA256))
        return fail("source-only role receipt does not bind the current domains or one-hop proof");
    std::string producer,verifier;
    if(!pinnedJSON(a.admissibilityProducerReceiptPath,a.admissibilityProducerReceiptSHA256,producer,error)||
       !pinnedJSON(a.admissibilityVerificationReceiptPath,a.admissibilityVerificationReceiptSHA256,verifier,error))return false;
    const auto role=a.materialIndex==15u?std::string_view("Skin"):std::string_view("Fat");
    std::string prole,vrole;std::uint64_t n=0,edges=0;
    if(!equals(producer,"status","SOURCE_ONLY_AUTHENTICATED_ONE_HOP_SAME_TISSUE_ADMISSIBILITY")||
       !noAdmission(producer)||!num(producer,"native_steps",steps)||steps!=0||
       !objectForRole(producer,role,prole)||!equals(prole,"source_support_sha256",source)||
       !num(prole,"material",mat)||mat!=a.materialIndex)
        return fail("one-hop producer receipt does not authenticate this role's source-support bytes");
    if(!equals(verifier,"status","PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT")||
       !noAdmission(verifier)||!num(verifier,"native_steps",steps)||steps!=0||
       !equals(verifier,"producer_receipt_sha256",a.admissibilityProducerReceiptSHA256)||
       !objectForRole(verifier,role,vrole)||!num(vrole,"cells",n)||!num(vrole,"verified_source_face_adjacency_edges",edges)||!edges)
        return fail("one-hop independent verification receipt is absent, unlinked, or has no role witnesses");

    std::error_code ec;const auto fileBytes=std::filesystem::file_size(a.binaryPath,ec);
    if(ec||fileBytes>1024ull*1024ull*1024ull)return fail("support binary exceeds 1 GiB bounded format");
    std::ifstream f(a.binaryPath,std::ios::binary);if(!f)return fail("cannot open support binary");
    char magic[8];if(!f.read(magic,8)||std::string(magic,8)!="FSSUP090")return fail("FSSUP090 magic/version mismatch");
    std::uint64_t c[6]{};for(auto& x:c)if(!one(f,x))return fail("support count header truncated");
    const auto [nc,ns,ng,np,npr,nfr]=std::tuple{c[0],c[1],c[2],c[3],c[4],c[5]};
    if(!nc||nc>2000000||!ns||ns>8000000||!ng||ng>20000||!np||np>1000000||!npr||npr>8000000||!nfr||nfr>8000000)
        return fail("support counts outside role bounds");
    if(nc!=static_cast<std::uint64_t>(std::count(b.childMaterialIndices.begin(),b.childMaterialIndices.end(),a.materialIndex))||
       np!=static_cast<std::uint64_t>(std::count(b.parent.materialIndices.begin(),b.parent.materialIndices.end(),a.materialIndex)))
        return fail("support counts differ from exact role subsequences");
    if(n!=nc)return fail("independent one-hop receipt cell count differs from role binary");
    auto& s=p.data;
    if(!vector(f,s.childOffsets,nc+1)||!vector(f,s.slotGroupIndices,ns)||!vector(f,s.volumeFractions,ns)||
       !vector(f,s.groupMaterialIndices,ng)||!vector(f,s.donorBodyIndices,ng)||!vector(f,s.anatomicalRegionIndices,ng)||
       !vector(f,s.derivedGroupIdentityIDs,ng)||!vector(f,s.parentStableSourceFaceIdentities,np)||
       !vector(f,s.sourceParentTetOffsets,ng+1)||!vector(f,s.sourceParentTetRows,npr)||
       !vector(f,s.sourceStableFaceOffsets,ng+1)||!vector(f,s.sourceStableFaceRows,nfr)||
       !vector(f,p.childMaterial,nc)||!vector(f,p.childDonor,nc)||!vector(f,p.childRegion,nc))
        return fail("support primary arrays truncated");
    char digest[64];if(!f.read(digest,64))return fail("support typed digest truncated");
    s.typedSupportSHA256.assign(digest,64);
    char adj[8];std::uint64_t na=0;std::uint32_t parsedHops=0;char sh[64];
    if(!f.read(adj,8)||std::string(adj,8)!="FSSADJ01"||!one(f,na)||!one(f,parsedHops)||parsedHops!=1||!na||na>50000000||
       !f.read(sh,64))return fail("one-hop extension header invalid");
    s.admissibilitySourceSupportSHA256.assign(sh,64);s.admissibilityMaximumSourceRegionHops=parsedHops;
    if(s.typedSupportSHA256!=typed||s.admissibilitySourceSupportSHA256!=source||
       !vector(f,s.cellAdmissibleRegionOffsets,nc+1)||!vector(f,s.cellAdmissibleRegionIds,na)||f.peek()!=EOF)
        return fail("embedded digest, adjacency CSR, or file end differs from receipt");
    std::uint64_t bytes=56;
    for(auto [n,w]:std::array<std::pair<std::uint64_t,std::uint64_t>,17>{{
       {nc+1,4},{ns,4},{ns,8},{ng,12},{ng,8},{np,4},{ng+1,4},{npr,4},{ng+1,4},{nfr,4},
       {nc,12},{64,1},{8,1},{8,1},{4,1},{64,1},{nc+1,4}}})
        if(!addBytes(bytes,n,w))return fail("support expected byte count overflow");
    if(!addBytes(bytes,na,4)||bytes!=fileBytes)return fail("support binary byte length mismatch");
    p.childCount=nc;p.parentCount=np;p.material=a.materialIndex;p.binarySHA=a.binarySHA256;
    p.receiptSHA=a.receiptSHA256;p.typedSHA=typed;p.sourceSHA=source;
    p.producerSHA=a.admissibilityProducerReceiptSHA256;p.verifierSHA=a.admissibilityVerificationReceiptSHA256;
    if(s.childOffsets.front()!=0||s.childOffsets.back()!=ns||s.sourceParentTetOffsets.front()!=0||
       s.sourceParentTetOffsets.back()!=npr||s.sourceStableFaceOffsets.front()!=0||s.sourceStableFaceOffsets.back()!=nfr||
       s.cellAdmissibleRegionOffsets.front()!=0||s.cellAdmissibleRegionOffsets.back()!=na)return fail("CSR endpoints invalid");
    auto mono=[](const auto& x){return std::adjacent_find(x.begin(),x.end(),[](auto a,auto b){return a>b;})==x.end();};
    if(!mono(s.sourceParentTetOffsets)||!mono(s.sourceStableFaceOffsets)||!mono(s.cellAdmissibleRegionOffsets))
        return fail("CSR offsets decrease");
    for(std::size_t i=1;i<s.childOffsets.size();++i)if(s.childOffsets[i]<=s.childOffsets[i-1]||s.childOffsets[i]>ns)return fail("empty child support row");
    for(std::size_t i=1;i<s.sourceParentTetOffsets.size();++i)if(s.sourceParentTetOffsets[i]<=s.sourceParentTetOffsets[i-1]||s.sourceParentTetOffsets[i]>npr)return fail("empty group parent row");
    for(std::size_t i=1;i<s.sourceStableFaceOffsets.size();++i)if(s.sourceStableFaceOffsets[i]<=s.sourceStableFaceOffsets[i-1]||s.sourceStableFaceOffsets[i]>nfr)return fail("empty group face row");
    for(std::size_t i=1;i<s.cellAdmissibleRegionOffsets.size();++i)if(s.cellAdmissibleRegionOffsets[i]<=s.cellAdmissibleRegionOffsets[i-1]||s.cellAdmissibleRegionOffsets[i]>na)return fail("empty adjacency row");
    std::vector<std::uint32_t> cm,pm;
    for(std::uint32_t i=0;i<b.childMaterialIndices.size();++i)if(b.childMaterialIndices[i]==a.materialIndex)cm.push_back(i);
    for(std::uint32_t i=0;i<b.parent.materialIndices.size();++i)if(b.parent.materialIndices[i]==a.materialIndex)pm.push_back(i);
    for(std::size_t i=0;i<nc;++i)if(p.childMaterial[i]!=a.materialIndex||p.childMaterial[i]!=b.childMaterialIndices[cm[i]]||
       p.childDonor[i]!=b.childDonorBodyIndices[cm[i]]||p.childRegion[i]!=b.childAnatomicalRegionIndices[cm[i]])
        return fail("support child rows differ from ordered current role labels");
    for(std::size_t i=0;i<np;++i)if(s.parentStableSourceFaceIdentities[i]!=b.parent.stableSourceFaceIdentities[pm[i]])
        return fail("support source stable-face IDs differ from ordered parent role rows");

    std::set<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>> tuples;
    std::set<std::uint64_t> ids;std::set<std::uint32_t> registeredRegions;
    for(std::uint32_t g=0;g<ng;++g){
        if(s.groupMaterialIndices[g]!=a.materialIndex||s.donorBodyIndices[g]==NM_INVALID_INDEX||
           s.anatomicalRegionIndices[g]==NM_INVALID_INDEX||!s.derivedGroupIdentityIDs[g]||
           !tuples.emplace(a.materialIndex,s.donorBodyIndices[g],s.anatomicalRegionIndices[g]).second||
           !ids.insert(s.derivedGroupIdentityIDs[g]).second)return fail("invalid or repeated semantic group");
        const auto pb=s.sourceParentTetOffsets[g],pe=s.sourceParentTetOffsets[g+1],fb=s.sourceStableFaceOffsets[g],fe=s.sourceStableFaceOffsets[g+1];
        std::set<std::uint32_t> faces;std::uint32_t prev=0;
        for(std::uint32_t j=pb;j<pe;++j){const auto local=s.sourceParentTetRows[j];
            if(local>=np||(j>pb&&local<=prev))return fail("parent ancestry CSR out of range/noncanonical");
            prev=local;const auto global=pm[local];
            if(b.parent.materialIndices[global]!=a.materialIndex||b.parent.donorBodyIndices[global]!=s.donorBodyIndices[g]||
               b.parent.anatomicalRegionIndices[global]!=s.anatomicalRegionIndices[g])return fail("group ancestry crosses typed source labels");
            faces.insert(b.parent.stableSourceFaceIdentities[global]);}
        if(faces.size()!=fe-fb)return fail("stable-face CSR not the full group parent union");
        prev=0;auto it=faces.begin();for(std::uint32_t j=fb;j<fe;++j,++it)
            if(it==faces.end()||s.sourceStableFaceRows[j]!=*it||(j>fb&&s.sourceStableFaceRows[j]<=prev))return fail("stable-face CSR mismatch");
            else prev=s.sourceStableFaceRows[j];
        registeredRegions.insert(s.anatomicalRegionIndices[g]);
    }
    for(std::size_t c=0;c<nc;++c){
        double sum=0,max=-1;std::uint32_t maxGroup=NM_INVALID_INDEX,pd=0,pr=0;bool primary=false,have=false;
        for(std::uint32_t j=s.childOffsets[c];j<s.childOffsets[c+1];++j){
            auto g=s.slotGroupIndices[j];auto f=s.volumeFractions[j];if(g>=ng||s.groupMaterialIndices[g]!=a.materialIndex||
               !std::isfinite(f)||f<=0||f>1)return fail("invalid semantic slot");
            auto d=s.donorBodyIndices[g],r=s.anatomicalRegionIndices[g];
            if(have&&std::tie(pd,pr)>=std::tie(d,r))return fail("semantic slots not strictly tuple-sorted");
            pd=d;pr=r;have=true;sum+=f;if(f>max){max=f;maxGroup=g;}if(d==p.childDonor[c]&&r==p.childRegion[c])primary=true;}
        if(std::abs(sum-1)>1e-12||!primary||maxGroup==NM_INVALID_INDEX||
           s.donorBodyIndices[maxGroup]!=p.childDonor[c]||s.anatomicalRegionIndices[maxGroup]!=p.childRegion[c])
            return fail("scalar label is not a maximum-fraction reference group");
        auto rb=s.cellAdmissibleRegionOffsets[c],re=s.cellAdmissibleRegionOffsets[c+1];std::uint32_t prev=0;bool first=true;
        for(std::uint32_t j=rb;j<re;++j){auto r=s.cellAdmissibleRegionIds[j];
            if((!first&&r<=prev)||!registeredRegions.contains(r))return fail("one-hop region row unordered/unregistered");
            prev=r;first=false;}
        for(std::uint32_t j=s.childOffsets[c];j<s.childOffsets[c+1];++j)
            if(!std::binary_search(s.cellAdmissibleRegionIds.begin()+rb,s.cellAdmissibleRegionIds.begin()+re,
                                   s.anatomicalRegionIndices[s.slotGroupIndices[j]]))return fail("one-hop row omits P1 support");
    }
    if(!physical_skin_child_mass_detail::sha256File(a.binaryPath,h,error)||h!=a.binarySHA256)return fail("support binary changed during import");
    return true;
}
inline std::string digest(const char* domain,const std::array<Payload,2>& r,bool adjacency){
    spatial_mass_transport_detail::SHA256Builder h;h.string(domain);
    for(const auto& p:r){h.u32(p.material);h.string(p.binarySHA);h.string(p.receiptSHA);h.string(p.typedSHA);
        if(adjacency){h.string(p.sourceSHA);h.string(p.producerSHA);h.string(p.verifierSHA);h.u32(1);}}
    return h.finish();
}
} // namespace fractional_semantic_support_import_detail

inline bool loadPhysicalSkinFractionalSemanticSupport090(
    const PhysicalSkinFractionalSemanticSupportAssets& assets,
    const PhysicalSkinChildMassMappingBundle& bundle,
    ConservativeSpatialChildSemanticSupportData& output,
    PhysicalSkinFractionalSemanticSupportImportReport& report,std::string& error) {
    using namespace fractional_semantic_support_import_detail;
    auto fail=[&](std::string m){error=std::move(m);return false;};
    if(assets.roles[0].materialIndex!=15u||assets.roles[1].materialIndex!=16u)
        return fail("fractional role assets must be ordered Skin 15 then Fat 16");
    const auto nc=bundle.childTetrahedra.size(),np=bundle.parent.tetrahedra.size();
    if(!nc||!np||bundle.childMaterialIndices.size()!=nc||bundle.childDonorBodyIndices.size()!=nc||
       bundle.childAnatomicalRegionIndices.size()!=nc||bundle.childStableSourceFaceIdentities.size()!=nc||
       bundle.childDominantParentTetrahedra.size()!=nc||bundle.parent.materialIndices.size()!=np||
       bundle.parent.donorBodyIndices.size()!=np||bundle.parent.anatomicalRegionIndices.size()!=np||
       bundle.parent.stableSourceFaceIdentities.size()!=np)return fail("current combined semantic row domains are incomplete");
    std::array<Payload,2> r;
    for(unsigned i=0;i<2;++i)if(!readRole(assets.roles[i],bundle,r[i],error))return false;
    using Key=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;
    struct G {std::array<std::uint32_t,2> local{{NM_INVALID_INDEX,NM_INVALID_INDEX}};std::vector<std::uint32_t> parents,faces;};
    std::map<Key,G> groups;std::array<std::vector<std::uint32_t>,2> cm,pm,l2g;
    for(unsigned role=0;role<2;++role){
        for(std::uint32_t i=0;i<nc;++i)if(bundle.childMaterialIndices[i]==r[role].material)cm[role].push_back(i);
        for(std::uint32_t i=0;i<np;++i)if(bundle.parent.materialIndices[i]==r[role].material)pm[role].push_back(i);
        const auto& s=r[role].data;l2g[role].resize(s.groupMaterialIndices.size(),NM_INVALID_INDEX);
        for(std::uint32_t g=0;g<s.groupMaterialIndices.size();++g){
            Key key{r[role].material,s.donorBodyIndices[g],s.anatomicalRegionIndices[g]};auto& dst=groups[key];
            if(dst.local[role]!=NM_INVALID_INDEX)return fail("role repeats a semantic group");
            dst.local[role]=g;
            for(std::uint32_t j=s.sourceParentTetOffsets[g];j<s.sourceParentTetOffsets[g+1];++j)
                dst.parents.push_back(pm[role][s.sourceParentTetRows[j]]);
            dst.faces.assign(s.sourceStableFaceRows.begin()+s.sourceStableFaceOffsets[g],
                             s.sourceStableFaceRows.begin()+s.sourceStableFaceOffsets[g+1]);
        }
    }
    std::uint32_t index=0;for(const auto& [key,g]:groups){(void)key;
        for(unsigned role=0;role<2;++role)if(g.local[role]!=NM_INVALID_INDEX)l2g[role][g.local[role]]=index;
        ++index;}
    ConservativeSpatialChildSemanticSupportData out;out.childOffsets.assign(nc+1,0);
    out.parentStableSourceFaceIdentities=bundle.parent.stableSourceFaceIdentities;
    out.sourceParentTetOffsets.push_back(0);out.sourceStableFaceOffsets.push_back(0);
    for(const auto& [key,g]:groups){
        out.groupMaterialIndices.push_back(std::get<0>(key));out.donorBodyIndices.push_back(std::get<1>(key));
        out.anatomicalRegionIndices.push_back(std::get<2>(key));
        auto gid=std::uint64_t(out.derivedGroupIdentityIDs.size()+1u);
        out.derivedGroupIdentityIDs.push_back((std::uint64_t(std::get<0>(key))<<56u)|gid);
        auto parents=g.parents;std::sort(parents.begin(),parents.end());parents.erase(std::unique(parents.begin(),parents.end()),parents.end());
        out.sourceParentTetRows.insert(out.sourceParentTetRows.end(),parents.begin(),parents.end());
        out.sourceParentTetOffsets.push_back(static_cast<std::uint32_t>(out.sourceParentTetRows.size()));
        std::set<std::uint32_t> faces;for(auto p:parents)faces.insert(bundle.parent.stableSourceFaceIdentities[p]);
        if(!std::equal(faces.begin(),faces.end(),g.faces.begin(),g.faces.end()))
            return fail("combined stable-face group differs from exact role source-parent union");
        out.sourceStableFaceRows.insert(out.sourceStableFaceRows.end(),faces.begin(),faces.end());
        out.sourceStableFaceOffsets.push_back(static_cast<std::uint32_t>(out.sourceStableFaceRows.size()));
    }
    out.cellAdmissibleRegionOffsets.push_back(0);std::array<std::uint32_t,2> cursor{};
    for(std::size_t c=0;c<nc;++c){
        auto m=bundle.childMaterialIndices[c];unsigned role=m==15u?0u:(m==16u?1u:2u);
        if(role>=2)return fail("combined child has non-Skin/Fat material");
        auto local=cursor[role]++;if(local>=cm[role].size()||cm[role][local]!=c)
            return fail("role child rows are not exact ordered combined material subsequences");
        const auto& s=r[role].data;
        for(std::uint32_t j=s.childOffsets[local];j<s.childOffsets[local+1];++j){
            auto g=s.slotGroupIndices[j];if(g>=l2g[role].size()||l2g[role][g]==NM_INVALID_INDEX)return fail("unmapped role semantic slot");
            out.slotGroupIndices.push_back(l2g[role][g]);out.volumeFractions.push_back(s.volumeFractions[j]);}
        out.childOffsets[c+1]=static_cast<std::uint32_t>(out.slotGroupIndices.size());
        auto b=s.cellAdmissibleRegionOffsets[local],e=s.cellAdmissibleRegionOffsets[local+1];
        out.cellAdmissibleRegionIds.insert(out.cellAdmissibleRegionIds.end(),s.cellAdmissibleRegionIds.begin()+b,s.cellAdmissibleRegionIds.begin()+e);
        out.cellAdmissibleRegionOffsets.push_back(static_cast<std::uint32_t>(out.cellAdmissibleRegionIds.size()));
    }
    if(cursor[0]!=r[0].childCount||cursor[1]!=r[1].childCount)return fail("combined support omitted role rows");
    for(std::size_t c=0;c<nc;++c){
        auto mat=bundle.childMaterialIndices[c],donor=bundle.childDonorBodyIndices[c],region=bundle.childAnatomicalRegionIndices[c];
        auto parent=bundle.childDominantParentTetrahedra[c];if(parent>=np||bundle.parent.materialIndices[parent]!=mat)
            return fail("dominant child reference parent is outside current material domain");
        bool found=false;for(std::uint32_t j=out.childOffsets[c];j<out.childOffsets[c+1];++j){
            auto g=out.slotGroupIndices[j];if(out.groupMaterialIndices[g]==mat&&out.donorBodyIndices[g]==donor&&out.anatomicalRegionIndices[g]==region){
                auto b=out.sourceParentTetOffsets[g],e=out.sourceParentTetOffsets[g+1];
                found=std::binary_search(out.sourceParentTetRows.begin()+b,out.sourceParentTetRows.begin()+e,parent);}}
        if(!found)return fail("scalar reference anchor parent is absent from complete source-parent semantic support");
    }
    out.typedSupportSHA256=digest("numi.human.fractional-semantic-support-pair.v1",r,false);
    out.admissibilitySourceSupportSHA256=digest("numi.human.fractional-semantic-admissibility-pair.v1",r,true);
    out.admissibilityMaximumSourceRegionHops=1;
    PhysicalSkinFractionalSemanticSupportImportReport measured;
    for(unsigned i=0;i<2;++i){measured.binarySHA256[i]=r[i].binarySHA;measured.receiptSHA256[i]=r[i].receiptSHA;
        measured.typedSupportSHA256[i]=r[i].typedSHA;measured.admissibilitySourceSupportSHA256[i]=r[i].sourceSHA;
        measured.admissibilityProducerReceiptSHA256[i]=r[i].producerSHA;measured.admissibilityVerificationReceiptSHA256[i]=r[i].verifierSHA;
        measured.childRows[i]=r[i].childCount;measured.parentRows[i]=r[i].parentCount;}
    measured.compositeTypedSupportSHA256=out.typedSupportSHA256;
    measured.compositeAdmissibilitySourceSHA256=out.admissibilitySourceSupportSHA256;
    measured.verified=true;output=std::move(out);report=std::move(measured);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
